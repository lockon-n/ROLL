"""
McbotsEnvManager: A thin recording proxy that bridges mcbots' OpenAI-compatible
agent loop with ROLL's RL training pipeline.

Architecture:
    McbotsEnvManager (ROLL env worker)
      ├── HTTP server (推理 + window/episode signals)
      ├── MC client process (Minecraft game client, long-lived)
      └── mcbots agent process (decision loop, disposable per episode)

    mcbots agent → HTTP → McbotsEnvManager → PolicyProxy → vLLM

See plan: /homes/junlong/.claude/plans/mutable-hatching-naur.md
"""
import base64
import copy
import io
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import PIL.Image

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterManager
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import (
    split_by_token,
    token_ids_to_assistant_mask,
)
from roll.pipeline.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger

logger = get_logger()


def _find_free_port() -> int:
    """Ask the OS for a free port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _extract_response_token_ranges(response_masks: List[int]) -> List[List[int]]:
    """Extract [start, end) ranges for contiguous 1-regions in response_masks."""
    ranges = []
    in_region = False
    start = 0
    for i, v in enumerate(response_masks):
        if v == 1 and not in_region:
            start = i
            in_region = True
        elif v == 0 and in_region:
            ranges.append([start, i])
            in_region = False
    if in_region:
        ranges.append([start, len(response_masks)])
    return ranges


class McbotsEnvManager(BaseEnvManager):
    """
    A thin recording proxy env manager for mcbots integration.

    Exposes an OpenAI-compatible HTTP endpoint. mcbots agent calls this endpoint
    instead of the OpenAI API. The manager translates requests to PolicyProxy
    (vLLM), records all messages/responses, and converts completed context
    windows to DataProto for ROLL's training pipeline.

    HTTP Endpoints:
        POST /v1/chat/completions  — OpenAI-compatible generation
        POST /window_complete      — Signal context window boundary
        POST /episode_done         — Signal episode end
        GET  /health               — Health check (returns assigned port)
    """

    def __init__(
        self,
        worker_config,
        pipeline_config,
        env_config: Dict,
        tokenizer,
        processor,
        generate_scheduler,
        output_queue,
        thread_lock,
        mode: str = "train",
        extra_data_provider=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.logger = get_logger()
        self.worker_config = worker_config
        self.pipeline_config = pipeline_config
        self.env_config = env_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.extra_data_provider = extra_data_provider
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RouterManager = generate_scheduler
        self.thread_lock = thread_lock

        # Collator for VL tokenization (Qwen3.5 vision support)
        self.collator = DataCollatorWithPaddingForMM(
            tokenizer=self.tokenizer,
            processor=self.processor,
            answer_key=None,
            image_flag_key=None,
            video_flag_key=None,
            extra_data_provider=self.extra_data_provider,
        )

        # LLM proxy (PolicyProxy → vLLM)
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=None,  # No gem.Env — mcbots env is external
        )

        # State (minimal — no conversation management)
        self.current_window_messages: List[Dict] = []
        self.episode_id: int = 0
        self.sequence_id_counter: int = 0
        self.group_seed: int = 0
        self.http_port: int = 0
        self._httpd: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        # Process management state
        self._client_java_pid: Optional[int] = None
        self._agent_proc: Optional[subprocess.Popen] = None
        self._agent_log_file = None
        self._runtime_config: Optional[Dict] = None

        # Config — top-level keys come from the entry dict, nested keys from entry["config"]
        self.tag = self.env_config.get("tag", "Minecraft")
        self.env_id = self.env_config.get("env_id", 0)
        self.group_id = self.env_config.get("group_id", 0)

        # mcbots process management config (from YAML env_config, stored under "config" key)
        cfg = dict(self.env_config.get("config", {}))
        self.mcbots_project_root = cfg.get("mcbots_project_root", "")
        self.mc_server_host = cfg.get("mc_server_host", "127.0.0.1")
        self.mc_server_port = cfg.get("mc_server_port", 25565)
        self.render_mode = cfg.get("render_mode", "cpu")
        self.client_health_check_timeout = cfg.get("client_health_check_timeout", 180)
        self.agent_max_llm_successes = cfg.get("agent_max_llm_successes", 0)
        self.bot_name = cfg.get("bot_name", f"Bot{self.env_id}")

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────

    def run_rollout_loop(self, data: DataProto):
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info["seed"] + self.env_config.get("group_seed", 0)
        self.episode_id = 0
        self.sequence_id_counter = 0
        self.current_window_messages = []

        # Start HTTP server
        self.http_port = _find_free_port()
        self._start_http_server()
        self.logger.info(
            f"McbotsEnvManager env_id={self.env_id} started HTTP server on port {self.http_port}"
        )

        # Write port to discoverable file
        self._write_port_file()

        try:
            # Start MC client and wait for readiness
            self._ensure_client_running()

            # Episode loop: start agent, wait for it to finish, repeat
            while self.running:
                self._ensure_client_running()
                self._start_agent()
                self._wait_for_agent()
        finally:
            self._kill_agent()
            self._kill_client()
            self._stop_http_server()
            self._cleanup_port_file()
            # Notify output queue that this env is done (prevents pipeline hang)
            ray.get(
                self.output_queue.put.remote(
                    self.group_id, self.episode_id, self.current_step, None, self.env_id
                )
            )

    def stop(self):
        self.running = False

    # ──────────────────────────────────────────────────────────────────────
    # HTTP Server
    # ──────────────────────────────────────────────────────────────────────

    def _start_http_server(self):
        manager = self  # capture reference for handler

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # suppress default stderr logging

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b""
                request_data = json.loads(body) if body else {}

                try:
                    if self.path == "/v1/chat/completions":
                        response = manager._handle_chat_completion(request_data)
                    elif self.path == "/window_complete":
                        response = manager._handle_window_complete(request_data)
                    elif self.path == "/episode_done":
                        response = manager._handle_episode_done(request_data)
                    else:
                        response = {"error": f"Unknown endpoint: {self.path}"}
                        self.send_response(404)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                        return
                except Exception as e:
                    manager.logger.error(f"Error handling {self.path}: {e}", exc_info=True)
                    response = {"error": str(e), "choices": []}
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def do_GET(self):
                if self.path == "/health":
                    response = {"status": "ok", "port": manager.http_port, "env_id": manager.env_id}
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = HTTPServer(("0.0.0.0", self.http_port), RequestHandler)
        self._server_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._server_thread.start()

    def _stop_http_server(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
            self._server_thread = None

    def _get_port_dir(self) -> str:
        """Get port directory from pipeline config output_dir, falling back to env var."""
        output_dir = getattr(self.pipeline_config, "output_dir", None)
        if output_dir:
            return os.path.join(output_dir, "ports")
        return os.environ.get("MCBOTS_PORT_DIR", "/tmp/roll_mcbots_ports")

    def _write_port_file(self):
        """Write assigned port to a discoverable file for mcbots to read."""
        port_dir = self._get_port_dir()
        os.makedirs(port_dir, exist_ok=True)
        port_file = os.path.join(port_dir, f"roll_{self.env_id}.port")
        with open(port_file, "w") as f:
            f.write(str(self.http_port))

    def _cleanup_port_file(self):
        port_dir = self._get_port_dir()
        port_file = os.path.join(port_dir, f"roll_{self.env_id}.port")
        try:
            os.remove(port_file)
        except OSError:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # MC Client Process Management
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_client_running(self):
        """Start MC client if not running, wait for health check."""
        # The launch script starts Java in background and exits, so we can't
        # rely on _client_proc.poll() to check liveness. Use health check instead.
        if self._runtime_config is not None and self._check_client_health():
            return

        if self._runtime_config is not None:
            self.logger.warning(f"env_id={self.env_id}: MC client health check failed, restarting")
            self._kill_client()

        self._start_client()
        self._wait_client_ready()

    def _start_client(self):
        """Launch MC client via start-single-bot-inside.sh."""
        if not self.mcbots_project_root:
            self.logger.warning("mcbots_project_root not configured, skipping MC client startup")
            return

        launch_script = os.path.join(
            self.mcbots_project_root, "scripts", "launch", "start-single-bot-inside.sh"
        )
        if not os.path.isfile(launch_script):
            raise FileNotFoundError(f"MC client launch script not found: {launch_script}")

        env = {
            **os.environ,
            "BOT_NAME": self.bot_name,
            "SERVER_HOST": str(self.mc_server_host),
            "SERVER_PORT": str(self.mc_server_port),
            "RENDER_MODE": self.render_mode,
            "CLIENT_FOREGROUND": "false",
            "MCBOTS_PROJECT_ROOT": self.mcbots_project_root,
        }

        self.logger.info(
            f"env_id={self.env_id}: Starting MC client '{self.bot_name}' "
            f"(server={self.mc_server_host}:{self.mc_server_port})"
        )
        proc = subprocess.Popen(
            ["bash", launch_script, self.bot_name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait for script to finish (it launches client in background and exits)
        proc.wait(timeout=60)
        if proc.returncode != 0:
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(
                f"MC client launch script failed (rc={proc.returncode}): {output[-500:]}"
            )

        # Read runtime config written by the launch script
        runtime_path = os.path.join(
            self.mcbots_project_root, "workspaces", "main", self.bot_name, ".mcbots_runtime.json"
        )
        if not os.path.isfile(runtime_path):
            raise FileNotFoundError(f"MC client runtime config not found: {runtime_path}")
        with open(runtime_path) as f:
            self._runtime_config = json.load(f)

        # Find the Java client PID from the PID file written by the launch script
        self._client_java_pid = self._find_client_pid()

        self.logger.info(
            f"env_id={self.env_id}: MC client launched, pid={self._client_java_pid}, "
            f"agentbridge_port={self._runtime_config['agentbridge']['port']}"
        )

    def _find_client_pid(self) -> Optional[int]:
        """Find the Java client PID from mcbots runtime PID files."""
        runtime_dir = os.path.join(self.mcbots_project_root, "runtime")
        if not os.path.isdir(runtime_dir):
            return None
        # Find the most recent PID file for this bot
        best_pid = None
        best_mtime = 0.0
        for fname in os.listdir(runtime_dir):
            if not fname.endswith(".pids"):
                continue
            fpath = os.path.join(runtime_dir, fname)
            try:
                with open(fpath) as f:
                    content = f.read()
                if f"CLIENT_NAME={self.bot_name}" not in content:
                    continue
                mtime = os.path.getmtime(fpath)
                if mtime > best_mtime:
                    best_mtime = mtime
                    for line in content.splitlines():
                        if line.startswith("CLIENT_PID="):
                            best_pid = int(line.split("=", 1)[1].strip())
            except (OSError, ValueError):
                continue
        return best_pid

    def _wait_client_ready(self):
        """Poll AgentBridge health endpoint until ready."""
        if self._runtime_config is None:
            return

        ab_host = self._runtime_config["agentbridge"]["host"]
        ab_port = self._runtime_config["agentbridge"]["port"]
        url = f"http://{ab_host}:{ab_port}/api/health"
        deadline = time.time() + self.client_health_check_timeout

        self.logger.info(f"env_id={self.env_id}: Waiting for MC client health check ({url})...")
        while time.time() < deadline:
            if self._check_client_health():
                self.logger.info(f"env_id={self.env_id}: MC client is ready")
                return
            time.sleep(2.0)

        raise TimeoutError(
            f"MC client health check timed out after {self.client_health_check_timeout}s"
        )

    def _check_client_health(self) -> bool:
        """Check AgentBridge /api/health endpoint."""
        if self._runtime_config is None:
            return False
        ab_host = self._runtime_config["agentbridge"]["host"]
        ab_port = self._runtime_config["agentbridge"]["port"]
        try:
            req = urllib.request.Request(f"http://{ab_host}:{ab_port}/api/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("success", False)
        except Exception:
            return False

    def _kill_client(self):
        """Terminate MC client Java process."""
        if self._client_java_pid is not None:
            self.logger.info(f"env_id={self.env_id}: Killing MC client pid={self._client_java_pid}")
            try:
                os.kill(self._client_java_pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            # Give it a moment, then force kill if still alive
            time.sleep(0.5)
            try:
                os.kill(self._client_java_pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            self._client_java_pid = None
        self._runtime_config = None

    # ──────────────────────────────────────────────────────────────────────
    # mcbots Agent Process Management
    # ──────────────────────────────────────────────────────────────────────

    def _start_agent(self):
        """Launch mcbots agent subprocess for one episode."""
        if not self.mcbots_project_root:
            self.logger.warning("mcbots_project_root not configured, skipping agent startup")
            # Fallback: wait for external agent to connect
            while self.running:
                time.sleep(1.0)
            return

        agent_main = os.path.join(self.mcbots_project_root, "agent", "main.py")
        if not os.path.isfile(agent_main):
            raise FileNotFoundError(f"mcbots agent main.py not found: {agent_main}")

        # Read remote_bash config from MC client's runtime config
        rb_host = "127.0.0.1"
        rb_port = "9090"
        if self._runtime_config:
            rb_host = str(self._runtime_config.get("remote_bash", {}).get("host", rb_host))
            rb_port = str(self._runtime_config.get("remote_bash", {}).get("port", rb_port))

        roll_url = f"http://127.0.0.1:{self.http_port}"

        # Record directory: {output_dir}/mcbots_records/{bot_name}_ep{episode_id}/
        record_dir = ""
        output_dir = getattr(self.pipeline_config, "output_dir", None)
        if output_dir:
            record_dir = os.path.join(output_dir, "mcbots_records", f"{self.bot_name}_ep{self.episode_id}")
            os.makedirs(record_dir, exist_ok=True)

        # Workspace and display from runtime config
        workspace_root = ""
        display = ":0"
        game_log_path = ""
        if self._runtime_config:
            workspace_root = self._runtime_config.get("workspace_root", "")
            display = self._runtime_config.get("x11", {}).get("display", ":0")
            if workspace_root:
                game_log_path = os.path.join(workspace_root, "game", "logs", "latest.log")

        env = {
            **os.environ,
            "MCBOTS_PLAYER": self.bot_name,
            "MCBOTS_BASE_URL": f"{roll_url}/v1",
            "MCBOTS_ROLL_NOTIFY_URL": roll_url,
            "MCBOTS_REMOTE_BASH_HOST": rb_host,
            "MCBOTS_REMOTE_BASH_PORT": rb_port,
            "MCBOTS_API_KEY": "roll-internal",
            "MCBOTS_WORKSPACE_ROOT": workspace_root,
            "MCBOTS_DISPLAY": display,
            "MCBOTS_LOG_FILE_PATH": game_log_path,
            "MCBOTS_RECORD_VIDEO": "false",
            "MCBOTS_DEBUG_RANDOM_REWARD": os.environ.get("MCBOTS_DEBUG_RANDOM_REWARD", "0"),
        }
        if record_dir:
            env["MCBOTS_RECORD_DIR"] = record_dir
        if self.agent_max_llm_successes > 0:
            env["MCBOTS_MAX_LLM_REQUEST_SUCCESSES"] = str(self.agent_max_llm_successes)

        # Write agent logs to file for debugging
        agent_log_path = os.path.join(record_dir, "agent.log") if record_dir else None
        self.logger.info(
            f"env_id={self.env_id}: Starting mcbots agent "
            f"(base_url={roll_url}/v1, remote_bash={rb_host}:{rb_port}, "
            f"log={agent_log_path})"
        )
        self._agent_log_file = open(agent_log_path, "w") if agent_log_path else None
        self._agent_proc = subprocess.Popen(
            ["python", "-m", "agent.main"],
            env=env,
            cwd=self.mcbots_project_root,
            stdout=self._agent_log_file or subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def _wait_for_agent(self):
        """Wait for agent subprocess to exit. Log if it crashed."""
        if self._agent_proc is None:
            return
        try:
            while self.running:
                retcode = self._agent_proc.poll()
                if retcode is not None:
                    if retcode != 0:
                        # Read last output from log file or pipe
                        output = ""
                        if self._agent_log_file is not None:
                            self._agent_log_file.flush()
                            try:
                                with open(self._agent_log_file.name) as f:
                                    output = f.read()[-1000:]
                            except OSError:
                                pass
                        elif self._agent_proc.stdout:
                            output = self._agent_proc.stdout.read().decode(errors="replace")[-1000:]
                        self.logger.warning(
                            f"env_id={self.env_id}: mcbots agent exited with code {retcode}. "
                            f"Last output: {output}"
                        )
                    else:
                        self.logger.info(f"env_id={self.env_id}: mcbots agent finished normally")
                    self._agent_proc = None
                    return
                time.sleep(1.0)
            # If self.running became False, kill the agent
            self._kill_agent()
        finally:
            if self._agent_log_file is not None:
                self._agent_log_file.close()
                self._agent_log_file = None

    def _kill_agent(self):
        """Terminate mcbots agent process."""
        if self._agent_proc is not None and self._agent_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._agent_proc.pid), signal.SIGTERM)
                self._agent_proc.wait(timeout=5)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self._agent_proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        self._agent_proc = None

    # ──────────────────────────────────────────────────────────────────────
    # HTTP Handlers
    # ──────────────────────────────────────────────────────────────────────

    def _handle_chat_completion(self, request: Dict) -> Dict:
        """Handle POST /v1/chat/completions — OpenAI-compatible generation."""
        incoming_messages = request.get("messages", [])
        if not incoming_messages:
            return {"error": "No messages provided"}

        # ── Record messages ──
        self._record_incoming_messages(incoming_messages)

        # ── Extract images and convert to Qwen3.5 VL format ──
        messages_vl, pil_images = self._extract_and_convert_images(incoming_messages)

        # ── Tokenize ──
        chat_template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=getattr(self.pipeline_config, "enable_thinking", False),
        )
        if getattr(self.pipeline_config, "chat_template", None) is not None:
            chat_template_kwargs["chat_template"] = self.pipeline_config.chat_template

        lm_input_texts = self.tokenizer.apply_chat_template(messages_vl, **chat_template_kwargs)

        feature = {self.collator.prompt_key: lm_input_texts}
        feature[self.collator.image_key] = pil_images if pil_images else None
        inputs = self.collator([feature])
        lm_input: DataProto = DataProto.from_single_dict(inputs)

        # ── Generate via PolicyProxy ──
        generation_config = {
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 1.0),
            "max_new_tokens": request.get("max_tokens", 2048),
        }
        # Merge with worker generating_args defaults
        default_config = self.worker_config.generating_args.to_dict()
        for k, v in generation_config.items():
            if v is not None:
                default_config[k] = v
        generation_config = default_config

        lm_input.meta_info["src_rank"] = self.env_id
        lm_output = self.llm_proxy.generate(
            messages=incoming_messages,
            lm_input=lm_input,
            generation_config=generation_config,
        )

        if lm_output is None:
            return {
                "error": "Generation failed (PolicyProxy returned None)",
                "choices": [],
            }

        # ── Decode response ──
        response_text = self.tokenizer.batch_decode(
            lm_output.batch["responses"], skip_special_tokens=False
        )[0]
        # Strip EOS token if present
        if response_text.endswith(self.tokenizer.eos_token):
            response_text = response_text[: -len(self.tokenizer.eos_token)]

        # ── Record assistant response ──
        self.current_window_messages.append({"role": "assistant", "content": response_text})

        # ── Return OpenAI-format response ──
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roll-policy",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": lm_input.batch["input_ids"].shape[1],
                "completion_tokens": len(lm_output.batch["responses"][0]),
                "total_tokens": lm_input.batch["input_ids"].shape[1] + len(lm_output.batch["responses"][0]),
            },
        }

    def _handle_window_complete(self, request: Dict) -> Dict:
        """Handle POST /window_complete — emit current context window as DataProto."""
        reward = request.get("reward", 0.0)

        # Check if there's anything to emit
        has_assistant = any(m.get("role") == "assistant" for m in self.current_window_messages)
        if not self.current_window_messages or not has_assistant:
            self.logger.warning("window_complete called but no assistant messages to emit")
            self.current_window_messages = []
            self.sequence_id_counter += 1
            return {"status": "skipped", "reason": "no assistant messages", "sequence_id": self.sequence_id_counter}

        # Formulate DataProto
        try:
            dataproto = self._formulate_dataproto(self.current_window_messages, reward)
        except Exception as e:
            self.logger.error(f"Failed to formulate DataProto: {e}", exc_info=True)
            self.current_window_messages = []
            self.sequence_id_counter += 1
            return {"status": "error", "reason": str(e)}

        # Set trajectory IDs
        traj_group_id = f"{self.tag}_{self.group_id}_{self.episode_id}_{self.group_seed}"
        traj_id = f"{traj_group_id}_{self.env_id}_{self.sequence_id_counter}"
        dataproto.non_tensor_batch["traj_group_id"] = np.array(
            [traj_group_id] * dataproto.batch.batch_size[0], dtype=object
        )
        dataproto.non_tensor_batch["traj_id"] = np.array(
            [traj_id] * dataproto.batch.batch_size[0], dtype=object
        )

        # Emit to output queue
        ray.get(
            self.output_queue.put.remote(
                self.group_id, self.episode_id, self.current_step, dataproto, self.env_id
            )
        )

        self.logger.info(
            f"Emitted window: env_id={self.env_id} episode={self.episode_id} "
            f"seq={self.sequence_id_counter} reward={reward} "
            f"messages={len(self.current_window_messages)}"
        )

        # Reset
        self.current_window_messages = []
        self.sequence_id_counter += 1
        return {"status": "ok", "sequence_id": self.sequence_id_counter}

    def _handle_episode_done(self, request: Dict) -> Dict:
        """Handle POST /episode_done — emit final window and reset for next episode."""
        reward = request.get("reward", 0.0)

        # Emit final window if non-empty
        result = {"status": "ok"}
        has_assistant = any(m.get("role") == "assistant" for m in self.current_window_messages)
        if self.current_window_messages and has_assistant:
            result = self._handle_window_complete({"reward": reward})

        # Reset for next episode
        self.episode_id += 1
        self.sequence_id_counter = 0
        self.current_window_messages = []

        result["episode_id"] = self.episode_id
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Message Recording
    # ──────────────────────────────────────────────────────────────────────

    def _record_incoming_messages(self, incoming_messages: List[Dict]):
        """
        Record incoming messages incrementally with consistency checks.

        OpenAI API is stateless — client sends full history each time.
        We only append new messages and verify existing ones match.
        """
        if len(self.current_window_messages) == 0:
            # First call in this window: record everything
            self.current_window_messages = copy.deepcopy(incoming_messages)
        else:
            new_start = len(self.current_window_messages)

            # Assert: incoming must be at least as long as recorded
            assert len(incoming_messages) >= new_start, (
                f"Incoming messages ({len(incoming_messages)}) shorter than recorded "
                f"({new_start}). mcbots must call /window_complete before context reset."
            )

            # Assert: existing messages must match by role
            for i in range(new_start):
                assert incoming_messages[i]["role"] == self.current_window_messages[i]["role"], (
                    f"Message role mismatch at index {i}: "
                    f"incoming={incoming_messages[i]['role']}, "
                    f"recorded={self.current_window_messages[i]['role']}"
                )

            # Append only new messages (excluding the assistant response we'll add later)
            for msg in incoming_messages[new_start:]:
                self.current_window_messages.append(copy.deepcopy(msg))

    # ──────────────────────────────────────────────────────────────────────
    # DataProto Construction
    # ──────────────────────────────────────────────────────────────────────

    def _formulate_dataproto(self, messages: List[Dict], reward: float) -> DataProto:
        """Convert a completed context window to training-ready DataProto."""
        # Extract images and convert to Qwen3.5 VL format
        messages_vl, pil_images = self._extract_and_convert_images(messages)

        # Tokenize full conversation (no generation prompt — conversation is complete)
        chat_template_kwargs = dict(
            add_generation_prompt=False,
            tokenize=False,
        )
        if getattr(self.pipeline_config, "chat_template", None) is not None:
            chat_template_kwargs["chat_template"] = self.pipeline_config.chat_template

        lm_input_texts = self.tokenizer.apply_chat_template(messages_vl, **chat_template_kwargs)

        feature = {self.collator.prompt_key: lm_input_texts}
        feature[self.collator.image_key] = pil_images if pil_images else None
        inputs = self.collator([feature])
        lm_input: DataProto = DataProto.from_single_dict(inputs)

        input_ids = lm_input.batch["input_ids"]
        attention_mask = lm_input.batch["attention_mask"]
        position_ids = lm_input.batch["position_ids"]

        # Build response mask
        token_ids = input_ids[0].tolist()
        token_ids_split = split_by_token(
            token_ids, token_ids[0], messages=messages_vl, tokenizer=self.tokenizer
        )
        response_masks_list = token_ids_to_assistant_mask(
            messages=messages_vl, input_ids_list=token_ids_split, tokenizer=self.tokenizer
        )
        response_masks = [item for items in response_masks_list for item in items]

        assert len(response_masks) == len(token_ids), (
            f"response_masks length mismatch: {len(response_masks)} != {len(token_ids)}"
        )

        # Extract response token ranges before converting to tensor
        response_token_ranges = _extract_response_token_ranges(response_masks)

        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        # Build prompt_mask and scores
        if 1 in response_masks:
            first_response_idx = response_masks.index(1)
            last_response_idx = len(response_masks) - 1 - response_masks[::-1].index(1)
        else:
            # No assistant response tokens found — edge case
            self.logger.warning("No response tokens found in window, using full sequence")
            first_response_idx = len(response_masks)
            last_response_idx = len(response_masks) - 1

        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)

        score_tensor = torch.zeros(len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][last_response_idx] = reward

        # Truncate to last response token
        truncate_len = min(last_response_idx + 1, self.pipeline_config.sequence_length)
        input_ids = input_ids[:, :truncate_len]
        attention_mask = attention_mask[:, :truncate_len]
        position_ids = (
            position_ids[:, :, :truncate_len]
            if position_ids.dim() == 3
            else position_ids[:, :truncate_len]
        )
        response_mask = response_mask[:, :truncate_len]
        prompt_mask = prompt_mask[:, :truncate_len]
        score_tensor = score_tensor[:, :truncate_len]

        # Pad to sequence_length
        seq_len = self.pipeline_config.sequence_length
        input_ids = pad_to_length(input_ids, length=seq_len, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=seq_len, pad_value=0)
        position_ids = pad_to_length(position_ids, length=seq_len, pad_value=0)
        response_mask = pad_to_length(response_mask, length=seq_len, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=seq_len, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=seq_len, pad_value=0)

        # Count assistant turns and response length for metrics
        num_assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        response_length = response_mask.sum(dim=-1).float().mean().item()

        # Pack DataProto
        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.env_id], dtype=object),
            "group_ids": np.array([self.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([self.tag], dtype=object),
            "step_scores": np.array([[reward]], dtype=object),
            "episode_scores": np.array([reward], dtype=object),
            "response_token_ranges": np.array([response_token_ranges], dtype=object),
        })
        lm_input.meta_info = {
            "metrics": {
                f"env/{self.tag}/num_actions": num_assistant_turns,
                "env/response_length": response_length,
            }
        }

        return lm_input

    # ──────────────────────────────────────────────────────────────────────
    # Image Handling
    # ──────────────────────────────────────────────────────────────────────

    def _extract_and_convert_images(
        self, messages: List[Dict]
    ) -> Tuple[List[Dict], List[PIL.Image.Image]]:
        """
        Extract base64 images from OpenAI format and convert to PIL + Qwen3.5 VL format.

        Input format (OpenAI):
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,XXX"}}

        Output format (Qwen3.5 VL):
            {"type": "image"}

        Returns:
            (messages_copy, pil_images) where messages_copy has image_url replaced
            with {"type": "image"}, and pil_images is the list of PIL images.
        """
        messages_copy = copy.deepcopy(messages)
        pil_images = []

        for msg in messages_copy:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for i, item in enumerate(content):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "")
                    if url.startswith("data:image"):
                        # Extract base64 data after the comma
                        b64_data = url.split(",", 1)[1] if "," in url else url
                        image_bytes = base64.b64decode(b64_data)
                        pil_image = PIL.Image.open(io.BytesIO(image_bytes))
                        pil_images.append(pil_image)
                    # Replace with Qwen3.5 VL format
                    content[i] = {"type": "image"}

        return messages_copy, pil_images
