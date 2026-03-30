import asyncio
import copy
import os
import resource
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from codetiming import Timer
from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.utils.context_managers import local_profiler

from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider, get_extra_data_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.checkpoint_manager import download_model
from roll.utils.import_utils import safe_import_class


class EnvironmentWorker(Worker):
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_managers: Dict[int, BaseEnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[ProcessorMixin] = None
        self.env_configs: Dict[int, Dict] = worker_config.env_configs[self.rank]
        self.thread_lock = threading.Lock()
        self.output_queue = None
        self.mode = "train"
        self._global_step = -1
        self._stopped = False

    def _get_rss_gb(self) -> float:
        """Get current RSS in GB."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def initialize(self,
                   pipeline_config,
                   generate_scheduler,
                   output_queue,
                   collator: Optional[callable] = None,
                   mode: str = "train"):
        super().initialize(pipeline_config)

        self.logger.info(f"[mem] after super().initialize: RSS={self._get_rss_gb():.2f} GB")

        self.output_queue = output_queue
        self.mode = mode
        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(self.worker_config.model_args, model_name_or_path)
        self.processor = default_processor_provider(self.worker_config.model_args, model_name_or_path)

        self.logger.info(f"[mem] after tokenizer/processor load: RSS={self._get_rss_gb():.2f} GB")

        # Store context for lazy env_manager creation instead of creating all upfront
        self._init_context = dict(
            model_name_or_path=model_name_or_path,
            pipeline_config=pipeline_config,
            generate_scheduler=generate_scheduler,
            output_queue=output_queue,
            mode=mode,
        )

        max_concurrent = self.worker_config.max_env_num_per_worker
        if max_concurrent and max_concurrent < len(self.env_configs):
            # Lazy mode: only create env_managers on demand during run_rollout_loop
            self.logger.info(
                f"Lazy env init: {len(self.env_configs)} envs assigned, "
                f"max {max_concurrent} concurrent (will create on demand)"
            )
        else:
            # Eager mode: create all env_managers upfront (original behavior)
            self._create_env_managers(list(self.env_configs.items()))

        self.logger.info(f"[mem] after env_managers created: RSS={self._get_rss_gb():.2f} GB, "
                         f"num_env_managers={len(self.env_managers)}")

    def _create_env_manager(self, env_id: int, env_config) -> BaseEnvManager:
        ctx = self._init_context
        if env_id == 0:
            self.logger.info(f"use env_manager_cls: {env_config['env_manager_cls']}")
        env_manager_cls = safe_import_class(env_config["env_manager_cls"])
        assert env_manager_cls is not None
        tokenizer = self.tokenizer
        processor = self.processor
        extra_data_provider = None
        if processor is not None and isinstance(processor, ProcessorMixin):
            extra_data_provider = get_extra_data_provider(ctx["model_name_or_path"], processor=processor)
        return env_manager_cls(
            worker_config=self.worker_config,
            pipeline_config=ctx["pipeline_config"],
            env_config=env_config,
            tokenizer=tokenizer,
            processor=processor,
            generate_scheduler=ctx["generate_scheduler"],
            output_queue=ctx["output_queue"],
            thread_lock=self.thread_lock,
            mode=ctx["mode"],
            extra_data_provider=extra_data_provider,
        )

    def _create_env_managers(self, items: list):
        """Create env_managers in parallel for a batch of (env_id, env_config) pairs."""
        def _create(env_id, env_config):
            return env_id, self._create_env_manager(env_id, env_config)
        with ThreadPoolExecutor(max_workers=min(len(items), 64)) as executor:
            futures = [executor.submit(_create, eid, ecfg) for eid, ecfg in items]
            for future in as_completed(futures):
                try:
                    env_id, env_manager = future.result()
                    self.env_managers[env_id] = env_manager
                except Exception as e:
                    self.logger.error(f"Failed to initialize env_manager: {e}", exc_info=True)
                    raise e

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def run_rollout_loop(self, seed):
        # Set environment variables for profiler context
        os.environ["roll_EXEC_FUNC_NAME"] = "run_rollout_loop"
        os.environ["WORKER_NAME"] = f"EnvironmentWorker_{self.rank}"

        max_concurrent = self.worker_config.max_env_num_per_worker
        is_lazy = max_concurrent and max_concurrent < len(self.env_configs)

        if not is_lazy:
            # Eager mode: all env_managers already created, run them all
            await self._run_env_managers(self.env_managers, seed)
        else:
            # Streaming lazy mode: semaphore-based concurrency, one-in-one-out
            await self._run_env_managers_streaming(list(self.env_configs.items()), max_concurrent, seed)

    async def _run_env_managers(self, env_managers: dict, seed):
        loop = asyncio.get_event_loop()
        pool = ThreadPoolExecutor(max_workers=len(env_managers))

        def run_with_profiler(env_manager, data_proto):
            with local_profiler():
                return env_manager.run_rollout_loop(data_proto)

        def run_without_profiler(env_manager, data_proto):
            return env_manager.run_rollout_loop(data_proto)

        tasks = []
        for env_id, env_manager in env_managers.items():
            run_func = run_without_profiler
            if self.rank == 0 and env_id == 0:
                run_func = run_with_profiler
            tasks.append(loop.run_in_executor(pool, run_func, env_manager, DataProto(meta_info={"seed": seed})))

        await asyncio.gather(*tasks)
        pool.shutdown()

    async def _run_env_managers_streaming(self, env_items: list, max_concurrent: int, seed):
        """Streaming lazy mode: semaphore-based concurrency, one-in-one-out.

        Instead of processing envs in fixed batches (wait for entire batch to finish),
        this uses a semaphore to maintain exactly `max_concurrent` envs running at any time.
        When one env finishes, the next one starts immediately.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        loop = asyncio.get_event_loop()
        pool = ThreadPoolExecutor(max_workers=max_concurrent)

        def create_and_run(env_id: int, env_config, use_profiler: bool):
            env_manager = self._create_env_manager(env_id, env_config)
            if self._global_step >= 0:
                env_manager.update_step(self._global_step)
            self.env_managers[env_id] = env_manager
            data = DataProto(meta_info={"seed": seed})
            try:
                if use_profiler:
                    with local_profiler():
                        env_manager.run_rollout_loop(data)
                else:
                    env_manager.run_rollout_loop(data)
            finally:
                self.env_managers.pop(env_id, None)

        async def run_single_env(env_id: int, env_config):
            async with semaphore:
                if self._stopped:
                    return
                use_profiler = (self.rank == 0 and env_id == 0)
                await loop.run_in_executor(pool, create_and_run, env_id, env_config, use_profiler)

        tasks = [run_single_env(env_id, env_config) for env_id, env_config in env_items]
        await asyncio.gather(*tasks)
        pool.shutdown()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def update_step(self, global_step):
        self._global_step = global_step
        for env_manager in list(self.env_managers.values()):
            env_manager.update_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def stop(self):
        self._stopped = True
        for env_manager in list(self.env_managers.values()):
            env_manager.stop()
