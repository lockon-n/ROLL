"""
End-to-end smoke test for McbotsEnvManager.

Simulates a realistic mcbots agent interaction via HTTP:
    1. POST /v1/chat/completions  (system + user with screenshots → assistant with action)
    2. POST /v1/chat/completions  (previous + new observations → assistant)
    3. POST /window_complete      (emit DataProto to output queue)
    4. POST /episode_done         (reset for next episode)

Uses real Qwen3.5 tokenizer/processor/collator. Only llm_proxy.generate() and
output_queue are mocked.

Real mcbots messages have:
    - User content as list of interleaved text blocks + base64 screenshots
    - Assistant content as XML action blocks, optionally with <think> reasoning
    - System prompt ~19KB describing Minecraft bot capabilities

Run with:
    pytest tests/agentic/env_manager/test_mcbots_e2e.py -v

Requires:
    Local Qwen3.5-0.8B model at /homes/junlong/junlong_export_ssd/models/Qwen/Qwen3.5-0.8B
"""
import base64
import io
import json
import os
import time
import types
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest
import torch
from tensordict import TensorDict
from transformers import BatchFeature

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.mcbots_env_manager import (
    McbotsEnvManager,
    _find_free_port,
)

QWEN_MODEL_PATH = "/homes/junlong/junlong_export_ssd/models/Qwen/Qwen3.5-0.8B"
CUSTOM_CHAT_TEMPLATE_PATH = (
    "/homes/junlong/junlong_export_ssd/models/Qwen/Qwen3.5-large-nodrop-thinking/chat_template.jinja"
)


def _model_available() -> bool:
    return os.path.isdir(QWEN_MODEL_PATH)


requires_qwen = pytest.mark.skipif(not _model_available(), reason=f"Qwen model not found at {QWEN_MODEL_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def extra_data_provider():
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config = AutoConfig.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    from transformers import AutoProcessor as AP
    proc_tmp = AP.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)

    spatial_merge_size = getattr(getattr(config, "vision_config", None), "spatial_merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = getattr(getattr(proc_tmp, "image_processor", None), "merge_size", None)

    vc = {"spatial_merge_size": spatial_merge_size}
    dummy_self = BatchFeature({
        "config": BatchFeature({
            "vision_config": BatchFeature(vc),
            "image_token_id": config.image_token_id,
            "video_token_id": config.video_token_id,
            "vision_start_token_id": config.vision_start_token_id,
        })
    })
    get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, dummy_self)

    def _provider(
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        out = get_rope_index(
            input_ids, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, attention_mask=attention_mask,
        )
        rope_index = out[0]
        if rope_index.dim() == 3 and rope_index.size(0) == 3:
            text_pos = torch.clamp(attention_mask.long().cumsum(-1) - 1, min=0).unsqueeze(0)
            rope_index = torch.cat([text_pos, rope_index], dim=0)
        return {"position_ids": rope_index.transpose(0, 1)}

    return _provider


@pytest.fixture(scope="module")
def collator(tokenizer, processor, extra_data_provider):
    return DataCollatorWithPaddingForMM(
        tokenizer=tokenizer,
        processor=processor,
        answer_key=None,
        image_flag_key=None,
        video_flag_key=None,
        extra_data_provider=extra_data_provider,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Realistic mcbots message builders
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Minecraft bot controller with vision.\n\n"
    "The game is running in a full asynchronous mode which means when you are "
    "thinking and generating actions, the game is still running instead of waiting "
    "for you to do anything.\n\n"
    "You will receive:\n"
    "- Screenshots showing what the bot sees\n"
    "- Command execution results\n"
    "- Chat messages from the game\n\n"
    "You control one Minecraft bot and its machine workspace.\n\n"
    "Please output only one action block each turn, only the last one will be executed."
)

ACTION_RESPONSE_TEMPLATE = """\
<action>
  <type>exec</type>
  <content>{command}</content>
  <observe_after_sec>1.0</observe_after_sec>
</action>"""

THINKING_ACTION_RESPONSE_TEMPLATE = """\
{thinking}
</think>

<action>
  <type>exec</type>
  <content>{command}</content>
  <observe_after_sec>1.0</observe_after_sec>
</action>"""


def _make_screenshot(width: int = 64, height: int = 64, color: tuple = (100, 150, 200)) -> str:
    """Create a test JPEG screenshot and return as base64 data URL."""
    img = PIL.Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _build_initial_user_message(task: str = "Mine dirt from the ground.") -> Dict:
    """First user message: task instruction (text only, like mcbots)."""
    return {"role": "user", "content": task}


def _build_observation_message(
    num_screenshots: int = 3,
    text_blocks: Optional[List[str]] = None,
    action_id: Optional[str] = None,
) -> Dict:
    """Build a realistic mcbots observation message with interleaved text + screenshots.

    Real mcbots messages have 10-17 content items: action assignment, async notices,
    multiple timestamped screenshots, action events, etc.
    """
    if text_blocks is None:
        text_blocks = []
        if action_id:
            text_blocks.append(
                f"[Action Assignment] The action in the previous assistant response "
                f"is assigned action_id={action_id}."
            )
            text_blocks.append(
                f"[Async Notice Start at 2026-03-31 12:00:00.00] The following observations "
                f"were captured while the previous LLM request was in-flight."
            )

    content: List[Dict] = []
    for text in text_blocks:
        content.append({"type": "text", "text": text})

    # Interleave screenshots with timestamp labels
    for i in range(num_screenshots):
        ts = f"2026-03-31 12:00:{i:02d}.{i*13:02d}"
        content.append({"type": "text", "text": f"[Screenshot at {ts}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": _make_screenshot(64, 64, color=(50 + i * 30, 100, 150))},
        })

    if action_id:
        content.append({
            "type": "text",
            "text": f"[Action Event at 2026-03-31 12:00:05.00] Action {action_id} Start",
        })
        content.append({
            "type": "text",
            "text": f"[Action Event at 2026-03-31 12:00:06.50] Action {action_id} End (exit_code=0)",
        })

    return {"role": "user", "content": content}


def _build_action_response(command: str = "mcapi look --yaw 0 --pitch -20") -> str:
    """Non-thinking assistant response (plain action XML)."""
    return ACTION_RESPONSE_TEMPLATE.format(command=command)


def _build_thinking_action_response(
    thinking: str = "I need to look around to find dirt blocks. Let me look down first.",
    command: str = "mcapi look --pitch 45",
) -> str:
    """Thinking assistant response (reasoning + </think> + action)."""
    return THINKING_ACTION_RESPONSE_TEMPLATE.format(thinking=thinking, command=command)


# ──────────────────────────────────────────────────────────────────────────────
# Manager builder + HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeGeneratingArgs:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048

    def to_dict(self):
        return {"temperature": self.temperature, "top_p": self.top_p, "max_new_tokens": self.max_new_tokens}


@dataclass
class _FakePipelineConfig:
    sequence_length: int = 2048
    chat_template: Optional[str] = None


@dataclass
class _FakeWorkerConfig:
    generating_args: _FakeGeneratingArgs = field(default_factory=_FakeGeneratingArgs)
    llm_proxy: Any = None


def _make_mock_llm_proxy(tokenizer):
    """Mock LLM proxy that returns tokenized response tokens.

    Unlike the unit test mock, this encodes a real string so the decoded
    output is meaningful and the response_mask aligns correctly.
    """
    proxy = MagicMock()
    # Default response — can be overridden per-test via proxy._response_text
    proxy._response_text = _build_action_response("mcapi look --yaw 0 --pitch -20")

    def _generate(messages, lm_input, generation_config):
        text = proxy._response_text
        response_ids = tokenizer.encode(text, add_special_tokens=False)
        # Append EOS
        response_ids.append(tokenizer.eos_token_id)
        resp_tensor = torch.tensor([response_ids], dtype=torch.long)
        return DataProto(
            batch=TensorDict({"responses": resp_tensor}, batch_size=[1]),
        )

    proxy.generate = MagicMock(side_effect=_generate)
    return proxy


_ray_get_patcher = None


def _build_e2e_manager(
    tokenizer,
    collator,
    chat_template: Optional[str] = None,
    sequence_length: int = 2048,
) -> McbotsEnvManager:
    """Build a McbotsEnvManager wired with real tokenizer/collator and mock LLM proxy."""
    pipeline_config = _FakePipelineConfig(sequence_length=sequence_length, chat_template=chat_template)
    worker_config = _FakeWorkerConfig()
    env_config = {"tag": "Minecraft", "env_id": 0, "group_id": 0}

    with patch.object(McbotsEnvManager, "__init__", lambda self, *a, **kw: None):
        mgr = McbotsEnvManager.__new__(McbotsEnvManager)

    from roll.utils.logging import get_logger

    mgr.logger = get_logger()
    mgr.worker_config = worker_config
    mgr.pipeline_config = pipeline_config
    mgr.env_config = env_config
    mgr.tokenizer = tokenizer
    mgr.processor = None
    mgr.collator = collator
    mgr.llm_proxy = _make_mock_llm_proxy(tokenizer)
    mgr.output_queue = MagicMock()
    mgr.mode = "train"
    mgr.generate_scheduler = MagicMock()
    mgr.thread_lock = MagicMock()
    mgr.extra_data_provider = None
    mgr.current_window_messages = []
    mgr.episode_id = 0
    mgr.sequence_id_counter = 0
    mgr.group_seed = 42
    mgr.current_step = 0
    mgr.http_port = 0
    mgr.tag = "Minecraft"
    mgr.env_id = 0
    mgr.group_id = 0
    mgr.running = True
    mgr._httpd = None
    mgr._server_thread = None
    return mgr


def _http_post(port: int, path: str, data: dict) -> dict:
    """Send a POST request to the manager's HTTP server."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _http_get(port: int, path: str) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _wait_for_server(port: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _http_get(port, "/health")
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server on port {port} did not start within {timeout}s")


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@requires_qwen
class TestE2EFullEpisode:
    """Simulate a complete mcbots episode via HTTP."""

    def setup_method(self):
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        # Patch ray.get to be a no-op (output_queue.put.remote returns MagicMock, not ObjectRef)
        self._ray_patcher = patch("roll.pipeline.agentic.env_manager.mcbots_env_manager.ray.get", lambda x: x)
        self._ray_patcher.start()

    def teardown_method(self):
        self._ray_patcher.stop()
        os.environ.pop("no_proxy", None)

    def test_multi_turn_episode_with_images(self, tokenizer, collator):
        """Realistic 2-turn Minecraft episode: system → user(task) → obs+screenshots →
        assistant(action) → obs+screenshots → assistant(action) → window_complete → episode_done.
        """
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=4096)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            # ── Turn 1: system + task + first observation ──
            messages_turn1 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_initial_user_message("Mine dirt from the ground."),
                _build_observation_message(
                    num_screenshots=2,
                    text_blocks=[
                        "[System at 2026-03-31 12:00:00.00] [Baritone] Settings loaded.",
                    ],
                ),
            ]
            # Set the mock to return a non-thinking action
            mgr.llm_proxy._response_text = _build_action_response("mcapi look --yaw 0 --pitch -20")

            resp1 = _http_post(mgr.http_port, "/v1/chat/completions", {
                "model": "roll-policy",
                "messages": messages_turn1,
                "temperature": 0.7,
                "max_tokens": 2048,
            })

            assert "choices" in resp1, f"Expected choices in response, got: {resp1}"
            assert len(resp1["choices"]) == 1
            content1 = resp1["choices"][0]["message"]["content"]
            assert "<action>" in content1
            assert resp1["usage"]["prompt_tokens"] > 0
            assert resp1["usage"]["completion_tokens"] > 0

            # ── Turn 2: previous messages + new observations (stateless API) ──
            messages_turn2 = messages_turn1 + [
                {"role": "assistant", "content": content1},
                _build_observation_message(
                    num_screenshots=4,
                    action_id="abc123",
                ),
            ]
            mgr.llm_proxy._response_text = _build_action_response("mcapi dig --block_x 10 --block_y 64 --block_z 20")

            resp2 = _http_post(mgr.http_port, "/v1/chat/completions", {
                "model": "roll-policy",
                "messages": messages_turn2,
                "temperature": 0.7,
                "max_tokens": 2048,
            })

            assert len(resp2["choices"]) == 1
            content2 = resp2["choices"][0]["message"]["content"]
            assert "mcapi dig" in content2

            # ── Window complete ──
            wc_resp = _http_post(mgr.http_port, "/window_complete", {"reward": 1.0})
            assert wc_resp["status"] == "ok"

            # Verify output_queue.put.remote was called
            assert mgr.output_queue.put.remote.called
            call_args = mgr.output_queue.put.remote.call_args
            dataproto = call_args[0][3]  # 4th positional arg
            assert isinstance(dataproto, DataProto)
            assert dataproto.batch["response_mask"][0].sum().item() > 0
            assert dataproto.batch["scores"][0].sum().item() == pytest.approx(1.0)

            # Check response_token_ranges
            ranges = dataproto.non_tensor_batch["response_token_ranges"][0]
            assert len(ranges) == 2, f"Expected 2 response ranges (2 assistant turns), got {len(ranges)}"

            # ── Episode done ──
            ed_resp = _http_post(mgr.http_port, "/episode_done", {"reward": 0.0})
            assert ed_resp["status"] == "ok" or ed_resp.get("reason") == "no assistant messages"
            assert ed_resp["episode_id"] == 1
            assert mgr.episode_id == 1
            assert mgr.sequence_id_counter == 0
            assert mgr.current_window_messages == []

        finally:
            mgr._stop_http_server()

    def test_thinking_mode_responses(self, tokenizer, collator):
        """Test with thinking-mode assistant responses (reasoning + </think> + action)."""
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=4096)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_initial_user_message("Mine dirt from the ground."),
                _build_observation_message(num_screenshots=2),
            ]

            thinking_text = (
                "I need to find dirt blocks. Looking at the screenshot, I can see "
                "grass blocks on top which means dirt should be underneath. Let me "
                "look down to find exposed dirt."
            )
            mgr.llm_proxy._response_text = _build_thinking_action_response(
                thinking=thinking_text,
                command="mcapi look --pitch 45",
            )

            resp = _http_post(mgr.http_port, "/v1/chat/completions", {
                "model": "roll-policy",
                "messages": messages,
                "temperature": 0.7,
            })

            content = resp["choices"][0]["message"]["content"]
            # The response should contain the thinking and action
            assert "</think>" in content
            assert "<action>" in content
            assert "mcapi look --pitch 45" in content

            # Window complete
            wc_resp = _http_post(mgr.http_port, "/window_complete", {"reward": 0.5})
            assert wc_resp["status"] == "ok"

            # Verify the DataProto includes thinking tokens in response mask
            dataproto = mgr.output_queue.put.remote.call_args[0][3]
            assert dataproto.batch["response_mask"][0].sum().item() > 0

        finally:
            mgr._stop_http_server()

    def test_many_screenshots_in_single_message(self, tokenizer, collator):
        """Stress test: user message with 6 screenshots (realistic for mcbots async window)."""
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=8192)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_initial_user_message("Craft a torch."),
                _build_observation_message(
                    num_screenshots=6,
                    text_blocks=[
                        "[Action Assignment] action_id=aaa111.",
                        "[Async Notice Start] Observations captured while LLM was thinking.",
                        "[Async Notice End] In-flight segment ended.",
                        "[Action Event] Action aaa111 Start",
                        "[Action Event] Action aaa111 End (exit_code=0)",
                    ],
                ),
            ]
            mgr.llm_proxy._response_text = _build_action_response("mcapi craft torch 1")

            resp = _http_post(mgr.http_port, "/v1/chat/completions", {
                "model": "roll-policy",
                "messages": messages,
            })

            assert len(resp["choices"]) == 1
            # With 6 screenshots, token count should be high
            assert resp["usage"]["prompt_tokens"] > 100

            wc_resp = _http_post(mgr.http_port, "/window_complete", {"reward": 1.0})
            assert wc_resp["status"] == "ok"

        finally:
            mgr._stop_http_server()

    def test_multiple_windows_in_episode(self, tokenizer, collator):
        """Simulate context window reset mid-episode (mcbots calls /window_complete
        then starts new context)."""
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=4096)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            # ── Window 1 ──
            messages_w1 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_initial_user_message("Mine dirt."),
                _build_observation_message(num_screenshots=1),
            ]
            mgr.llm_proxy._response_text = _build_action_response("mcapi look --pitch -20")
            _http_post(mgr.http_port, "/v1/chat/completions", {
                "messages": messages_w1, "model": "roll-policy",
            })

            wc1 = _http_post(mgr.http_port, "/window_complete", {"reward": 0.0})
            assert wc1["status"] == "ok"
            seq1 = wc1["sequence_id"]

            # ── Window 2 (new context after reset) ──
            messages_w2 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_observation_message(num_screenshots=2),
            ]
            mgr.llm_proxy._response_text = _build_action_response("mcapi dig --block_x 5 --block_y 60 --block_z 5")
            _http_post(mgr.http_port, "/v1/chat/completions", {
                "messages": messages_w2, "model": "roll-policy",
            })

            wc2 = _http_post(mgr.http_port, "/window_complete", {"reward": 1.0})
            assert wc2["status"] == "ok"
            assert wc2["sequence_id"] > seq1

            # output_queue.put.remote should have been called twice
            assert mgr.output_queue.put.remote.call_count == 2

            # ── Episode done ──
            ed = _http_post(mgr.http_port, "/episode_done", {})
            assert ed["episode_id"] == 1

        finally:
            mgr._stop_http_server()

    def test_incremental_message_recording_via_http(self, tokenizer, collator):
        """Verify the stateless OpenAI API pattern: client sends full history each time,
        manager incrementally records only new messages."""
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=4096)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            base = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_initial_user_message("Kill a sheep."),
                _build_observation_message(num_screenshots=1),
            ]
            mgr.llm_proxy._response_text = _build_action_response("mcapi attack sheep")
            resp1 = _http_post(mgr.http_port, "/v1/chat/completions", {
                "messages": base, "model": "roll-policy",
            })
            content1 = resp1["choices"][0]["message"]["content"]

            # After turn 1: should have system + user(task) + user(obs) + assistant
            assert len(mgr.current_window_messages) == 4

            # Turn 2: client sends full history + new observation
            turn2_messages = base + [
                {"role": "assistant", "content": content1},
                _build_observation_message(num_screenshots=2, action_id="xyz789"),
            ]
            mgr.llm_proxy._response_text = _build_action_response("mcapi attack sheep")
            resp2 = _http_post(mgr.http_port, "/v1/chat/completions", {
                "messages": turn2_messages, "model": "roll-policy",
            })

            # After turn 2: 4 previous + user(obs) + assistant = 6
            assert len(mgr.current_window_messages) == 6

            # Roles should alternate correctly
            roles = [m["role"] for m in mgr.current_window_messages]
            assert roles == ["system", "user", "user", "assistant", "user", "assistant"]

        finally:
            mgr._stop_http_server()

    def test_generation_failure_returns_error(self, tokenizer, collator):
        """If llm_proxy.generate returns None, HTTP should return error gracefully."""
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=4096)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            mgr.llm_proxy.generate = MagicMock(return_value=None)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                _build_observation_message(num_screenshots=1),
            ]
            resp = _http_post(mgr.http_port, "/v1/chat/completions", {
                "messages": messages, "model": "roll-policy",
            })

            assert "error" in resp
            assert resp["choices"] == []

        finally:
            mgr._stop_http_server()

    def test_window_complete_without_assistant_skips(self, tokenizer, collator):
        """window_complete with no assistant messages should skip gracefully."""
        mgr = _build_e2e_manager(tokenizer, collator, sequence_length=4096)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            # No chat completion calls — just window_complete
            resp = _http_post(mgr.http_port, "/window_complete", {"reward": 1.0})
            assert resp["status"] == "skipped"
            assert mgr.output_queue.put.remote.call_count == 0

        finally:
            mgr._stop_http_server()

    def test_health_endpoint(self, tokenizer, collator):
        """Health endpoint should return port and env_id."""
        mgr = _build_e2e_manager(tokenizer, collator)
        mgr.http_port = _find_free_port()
        mgr._start_http_server()
        _wait_for_server(mgr.http_port)

        try:
            resp = _http_get(mgr.http_port, "/health")
            assert resp["status"] == "ok"
            assert resp["port"] == mgr.http_port
            assert resp["env_id"] == 0
        finally:
            mgr._stop_http_server()
