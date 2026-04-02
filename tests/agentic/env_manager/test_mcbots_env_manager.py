"""
Unit tests for McbotsEnvManager.

Tests cover:
    - _extract_response_token_ranges (pure logic)
    - _extract_and_convert_images (base64 → PIL + Qwen VL format)
    - _record_incoming_messages (incremental recording with consistency checks)
    - _handle_chat_completion (OpenAI-compat endpoint)
    - _handle_window_complete / _handle_episode_done (DataProto emission)
    - HTTP server lifecycle (start, health, endpoints, stop)

Run with:
    pytest tests/agentic/env_manager/test_mcbots_env_manager.py -v
"""
import base64
import copy
import io
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.mcbots_env_manager import (
    McbotsEnvManager,
    _extract_response_token_ranges,
    _find_free_port,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_tiny_jpeg_b64() -> str:
    """Create a minimal 2x2 red JPEG and return base64."""
    img = PIL.Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_mock_tokenizer(vocab_size: int = 100):
    """Create a mock tokenizer with basic encode/decode support."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token = "<|endoftext|>"
    tok.eos_token_id = 2

    # apply_chat_template returns a string like "<|im_start|>system...assistant...<|im_end|>"
    def _apply_chat_template(messages, **kwargs):
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "[image]") if isinstance(item, dict) else str(item)
                    for item in content
                )
            parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>")
        if kwargs.get("add_generation_prompt"):
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    tok.apply_chat_template = _apply_chat_template

    # batch_decode: return content from token ids (just placeholder strings)
    def _batch_decode(token_ids, **kwargs):
        results = []
        for seq in token_ids:
            results.append("Hello from policy<|endoftext|>")
        return results

    tok.batch_decode = _batch_decode
    return tok


class _MockCollator:
    """Mock DataCollatorWithPaddingForMM that returns minimal tensors."""
    prompt_key = "prompt"
    image_key = "images"

    def __call__(self, features):
        seq_len = 20
        return {
            "input_ids": torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0),
        }


def _make_mock_llm_proxy():
    """Mock LLM proxy that returns a DataProto with dummy responses."""
    proxy = MagicMock()

    def _generate(messages, lm_input, generation_config):
        resp_tokens = torch.tensor([[10, 20, 30, 2]], dtype=torch.long)  # 3 tokens + eos
        return DataProto(
            batch=TensorDict({"responses": resp_tokens}, batch_size=[1]),
        )

    proxy.generate = MagicMock(side_effect=_generate)
    return proxy


@dataclass
class _FakeGeneratingArgs:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048

    def to_dict(self):
        return {"temperature": self.temperature, "top_p": self.top_p, "max_new_tokens": self.max_new_tokens}


@dataclass
class _FakePipelineConfig:
    sequence_length: int = 32
    chat_template: Optional[str] = None


@dataclass
class _FakeWorkerConfig:
    generating_args: _FakeGeneratingArgs = field(default_factory=_FakeGeneratingArgs)
    llm_proxy: Any = None


def _build_manager(**overrides) -> McbotsEnvManager:
    """Build a McbotsEnvManager with all dependencies mocked out."""
    tokenizer = overrides.get("tokenizer", _make_mock_tokenizer())
    processor = overrides.get("processor", MagicMock())
    worker_config = overrides.get("worker_config", _FakeWorkerConfig())
    pipeline_config = overrides.get("pipeline_config", _FakePipelineConfig())
    env_config = overrides.get("env_config", {"tag": "TestMinecraft", "env_id": 7, "group_id": 1})
    output_queue = overrides.get("output_queue", MagicMock())

    # Patch create_llm_proxy and DataCollatorWithPaddingForMM to avoid real init
    with patch(
        "roll.pipeline.agentic.env_manager.mcbots_env_manager.create_llm_proxy"
    ) as mock_create_proxy, patch(
        "roll.pipeline.agentic.env_manager.mcbots_env_manager.DataCollatorWithPaddingForMM"
    ) as mock_collator_cls:
        mock_create_proxy.return_value = overrides.get("llm_proxy", _make_mock_llm_proxy())
        mock_collator_cls.return_value = _MockCollator()

        mgr = McbotsEnvManager(
            worker_config=worker_config,
            pipeline_config=pipeline_config,
            env_config=env_config,
            tokenizer=tokenizer,
            processor=processor,
            generate_scheduler=MagicMock(),
            output_queue=output_queue,
            thread_lock=threading.Lock(),
            mode="train",
        )

    # Replace collator with our mock (the __init__ already ran)
    mgr.collator = _MockCollator()
    return mgr


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _extract_response_token_ranges
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractResponseTokenRanges:
    def test_empty(self):
        assert _extract_response_token_ranges([]) == []

    def test_no_ones(self):
        assert _extract_response_token_ranges([0, 0, 0]) == []

    def test_single_region(self):
        assert _extract_response_token_ranges([0, 0, 1, 1, 1, 0]) == [[2, 5]]

    def test_multiple_regions(self):
        masks = [0, 1, 1, 0, 0, 1, 0, 1, 1]
        assert _extract_response_token_ranges(masks) == [[1, 3], [5, 6], [7, 9]]

    def test_ends_with_one(self):
        assert _extract_response_token_ranges([0, 1, 1]) == [[1, 3]]

    def test_all_ones(self):
        assert _extract_response_token_ranges([1, 1, 1]) == [[0, 3]]

    def test_single_element_one(self):
        assert _extract_response_token_ranges([1]) == [[0, 1]]

    def test_alternating(self):
        assert _extract_response_token_ranges([1, 0, 1, 0, 1]) == [[0, 1], [2, 3], [4, 5]]


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _find_free_port
# ──────────────────────────────────────────────────────────────────────────────


class TestFindFreePort:
    def test_returns_positive_int(self):
        port = _find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_returns_different_ports(self):
        """Ports should generally differ (OS doesn't reuse immediately)."""
        ports = {_find_free_port() for _ in range(5)}
        # At least 2 distinct ports out of 5
        assert len(ports) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _extract_and_convert_images
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractAndConvertImages:
    def setup_method(self):
        self.mgr = _build_manager()

    def test_no_images(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        converted, images = self.mgr._extract_and_convert_images(messages)
        assert images == []
        assert converted[0]["content"] == "You are helpful."
        assert converted[1]["content"] == "Hello"

    def test_single_base64_image(self):
        b64 = _make_tiny_jpeg_b64()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ]
        converted, images = self.mgr._extract_and_convert_images(messages)

        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)
        assert images[0].size == (2, 2)

        # image_url replaced with {"type": "image"}
        content = converted[0]["content"]
        assert content[0] == {"type": "text", "text": "What is this?"}
        assert content[1] == {"type": "image"}

    def test_multiple_images(self):
        b64 = _make_tiny_jpeg_b64()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ]
        converted, images = self.mgr._extract_and_convert_images(messages)
        assert len(images) == 2
        assert converted[0]["content"][0] == {"type": "image"}
        assert converted[0]["content"][2] == {"type": "image"}

    def test_does_not_mutate_original(self):
        b64 = _make_tiny_jpeg_b64()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ]
        original = copy.deepcopy(messages)
        self.mgr._extract_and_convert_images(messages)
        assert messages == original

    def test_string_content_passthrough(self):
        """Messages with string content (not list) should pass through unchanged."""
        messages = [{"role": "assistant", "content": "Sure, I see a red block."}]
        converted, images = self.mgr._extract_and_convert_images(messages)
        assert images == []
        assert converted[0]["content"] == "Sure, I see a red block."


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _record_incoming_messages
# ──────────────────────────────────────────────────────────────────────────────


class TestRecordIncomingMessages:
    def setup_method(self):
        self.mgr = _build_manager()

    def test_first_call_records_all(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        self.mgr._record_incoming_messages(messages)
        assert len(self.mgr.current_window_messages) == 2
        assert self.mgr.current_window_messages[0]["role"] == "system"
        assert self.mgr.current_window_messages[1]["role"] == "user"

    def test_incremental_append(self):
        """Second call with extended history only appends new messages."""
        msgs1 = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        self.mgr._record_incoming_messages(msgs1)

        # Simulate: assistant added by _handle_chat_completion
        self.mgr.current_window_messages.append({"role": "assistant", "content": "Hello!"})

        # Client sends full history + new user message
        msgs2 = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What can you do?"},
        ]
        self.mgr._record_incoming_messages(msgs2)

        assert len(self.mgr.current_window_messages) == 4
        assert self.mgr.current_window_messages[3]["content"] == "What can you do?"

    def test_role_mismatch_raises(self):
        msgs1 = [{"role": "user", "content": "Hi"}]
        self.mgr._record_incoming_messages(msgs1)

        # Mismatched role at index 0
        msgs2 = [{"role": "system", "content": "Hi"}, {"role": "user", "content": "More"}]
        with pytest.raises(AssertionError, match="role mismatch"):
            self.mgr._record_incoming_messages(msgs2)

    def test_shorter_messages_raises(self):
        msgs = [
            {"role": "system", "content": "X"},
            {"role": "user", "content": "Y"},
        ]
        self.mgr._record_incoming_messages(msgs)
        self.mgr.current_window_messages.append({"role": "assistant", "content": "Z"})

        # Incoming shorter than recorded
        with pytest.raises(AssertionError, match="shorter than recorded"):
            self.mgr._record_incoming_messages([{"role": "system", "content": "X"}])

    def test_deep_copy_isolation(self):
        """Changes to original messages after recording should not affect internal state."""
        msgs = [{"role": "user", "content": "original"}]
        self.mgr._record_incoming_messages(msgs)
        msgs[0]["content"] = "mutated"
        assert self.mgr.current_window_messages[0]["content"] == "original"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _handle_window_complete
# ──────────────────────────────────────────────────────────────────────────────


class TestHandleWindowComplete:
    def setup_method(self):
        self.output_queue = MagicMock()
        # Make output_queue.put.remote() return a mock ObjectRef
        self.output_queue.put = MagicMock()
        self.output_queue.put.remote = MagicMock(return_value=MagicMock())
        self.mgr = _build_manager(output_queue=self.output_queue)
        self.mgr.update_step(global_step=0)

    def test_skip_when_empty(self):
        result = self.mgr._handle_window_complete({"reward": 1.0})
        assert result["status"] == "skipped"
        self.output_queue.put.remote.assert_not_called()

    def test_skip_when_no_assistant(self):
        """Window with only user/system messages should be skipped."""
        self.mgr.current_window_messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = self.mgr._handle_window_complete({"reward": 1.0})
        assert result["status"] == "skipped"

    @patch("roll.pipeline.agentic.env_manager.mcbots_env_manager.ray")
    def test_emits_dataproto(self, mock_ray):
        """Window with assistant messages should emit DataProto."""
        mock_ray.get = MagicMock(return_value=None)

        self.mgr.current_window_messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Mock _formulate_dataproto to return a minimal DataProto
        dummy_proto = DataProto(
            batch=TensorDict(
                {"input_ids": torch.ones(1, 32, dtype=torch.long)},
                batch_size=[1],
            ),
            non_tensor_batch={},
        )
        self.mgr._formulate_dataproto = MagicMock(return_value=dummy_proto)

        result = self.mgr._handle_window_complete({"reward": 2.5})
        assert result["status"] == "ok"

        # Verify _formulate_dataproto was called with the messages and reward
        self.mgr._formulate_dataproto.assert_called_once()
        args = self.mgr._formulate_dataproto.call_args
        assert args[0][1] == 2.5  # reward

        # Verify output_queue.put.remote was called
        self.output_queue.put.remote.assert_called_once()

        # Verify state was reset
        assert self.mgr.current_window_messages == []

    @patch("roll.pipeline.agentic.env_manager.mcbots_env_manager.ray")
    def test_increments_sequence_id(self, mock_ray):
        mock_ray.get = MagicMock(return_value=None)

        self.mgr.current_window_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        dummy_proto = DataProto(
            batch=TensorDict({"input_ids": torch.ones(1, 32, dtype=torch.long)}, batch_size=[1]),
            non_tensor_batch={},
        )
        self.mgr._formulate_dataproto = MagicMock(return_value=dummy_proto)

        assert self.mgr.sequence_id_counter == 0
        self.mgr._handle_window_complete({"reward": 0.0})
        assert self.mgr.sequence_id_counter == 1

    def test_error_handling(self):
        """If _formulate_dataproto raises, should return error and reset."""
        self.mgr.current_window_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        self.mgr._formulate_dataproto = MagicMock(side_effect=RuntimeError("tokenization failed"))

        result = self.mgr._handle_window_complete({"reward": 1.0})
        assert result["status"] == "error"
        assert "tokenization failed" in result["reason"]
        assert self.mgr.current_window_messages == []


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _handle_episode_done
# ──────────────────────────────────────────────────────────────────────────────


class TestHandleEpisodeDone:
    def setup_method(self):
        self.mgr = _build_manager()
        self.mgr.update_step(global_step=0)

    @patch("roll.pipeline.agentic.env_manager.mcbots_env_manager.ray")
    def test_emits_final_window_and_resets(self, mock_ray):
        mock_ray.get = MagicMock(return_value=None)

        self.mgr.current_window_messages = [
            {"role": "user", "content": "Last move"},
            {"role": "assistant", "content": "Done!"},
        ]
        self.mgr.sequence_id_counter = 3
        self.mgr.episode_id = 0

        dummy_proto = DataProto(
            batch=TensorDict({"input_ids": torch.ones(1, 32, dtype=torch.long)}, batch_size=[1]),
            non_tensor_batch={},
        )
        self.mgr._formulate_dataproto = MagicMock(return_value=dummy_proto)

        # Mock output_queue
        self.mgr.output_queue.put = MagicMock()
        self.mgr.output_queue.put.remote = MagicMock(return_value=MagicMock())

        result = self.mgr._handle_episode_done({"reward": 10.0})

        assert result["episode_id"] == 1  # incremented
        assert self.mgr.episode_id == 1
        assert self.mgr.sequence_id_counter == 0  # reset
        assert self.mgr.current_window_messages == []

    def test_empty_window_episode_done(self):
        """Episode done with no messages should still bump episode_id."""
        self.mgr.episode_id = 5
        result = self.mgr._handle_episode_done({})
        assert result["episode_id"] == 6
        assert self.mgr.sequence_id_counter == 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _handle_chat_completion
# ──────────────────────────────────────────────────────────────────────────────


class TestHandleChatCompletion:
    def setup_method(self):
        self.mgr = _build_manager()

    def test_empty_messages_returns_error(self):
        result = self.mgr._handle_chat_completion({"messages": []})
        assert "error" in result

    def test_no_messages_key_returns_error(self):
        result = self.mgr._handle_chat_completion({})
        assert "error" in result

    def test_successful_completion(self):
        request = {
            "messages": [
                {"role": "system", "content": "You are a Minecraft bot."},
                {"role": "user", "content": "Mine a tree"},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }

        result = self.mgr._handle_chat_completion(request)

        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "usage" in result

        # Verify assistant message was recorded
        assert self.mgr.current_window_messages[-1]["role"] == "assistant"

    def test_generation_failure_returns_error(self):
        """If llm_proxy.generate returns None, should return error."""
        self.mgr.llm_proxy.generate = MagicMock(return_value=None)

        request = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.mgr._handle_chat_completion(request)
        assert "error" in result
        assert result["choices"] == []

    def test_strips_eos_token(self):
        """Response text should have EOS token stripped."""
        request = {"messages": [{"role": "user", "content": "Hi"}]}
        result = self.mgr._handle_chat_completion(request)

        content = result["choices"][0]["message"]["content"]
        assert not content.endswith("<|endoftext|>")


# ──────────────────────────────────────────────────────────────────────────────
# Tests: HTTP Server
# ──────────────────────────────────────────────────────────────────────────────


class TestHTTPServer:
    def setup_method(self):
        # Bypass proxy for localhost connections
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        self.mgr = _build_manager()
        self.mgr.http_port = _find_free_port()
        self.mgr._start_http_server()
        # Wait for server to be ready
        self._wait_for_server()

    def teardown_method(self):
        self.mgr._stop_http_server()
        os.environ.pop("no_proxy", None)

    def _wait_for_server(self, timeout: float = 3.0):
        """Poll health endpoint until server is ready."""
        import socket as _socket
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with _socket.create_connection(("127.0.0.1", self.mgr.http_port), timeout=0.5):
                    return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.05)
        raise RuntimeError(f"HTTP server not ready after {timeout}s")

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.mgr.http_port}{path}"

    def _post(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self._url(path), method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def test_health_endpoint(self):
        result = self._get("/health")
        assert result["status"] == "ok"
        assert result["port"] == self.mgr.http_port
        assert result["env_id"] == self.mgr.env_id

    def test_chat_completion_via_http(self):
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
        }
        result = self._post("/v1/chat/completions", request)
        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1

    def test_window_complete_via_http(self):
        # First populate some messages
        self.mgr.current_window_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        # Mock _formulate_dataproto to avoid real tokenization
        dummy_proto = DataProto(
            batch=TensorDict({"input_ids": torch.ones(1, 32, dtype=torch.long)}, batch_size=[1]),
            non_tensor_batch={},
        )
        self.mgr._formulate_dataproto = MagicMock(return_value=dummy_proto)
        self.mgr.output_queue.put = MagicMock()
        self.mgr.output_queue.put.remote = MagicMock(return_value=MagicMock())

        with patch("roll.pipeline.agentic.env_manager.mcbots_env_manager.ray") as mock_ray:
            mock_ray.get = MagicMock(return_value=None)
            result = self._post("/window_complete", {"reward": 1.0})

        assert result["status"] == "ok"

    def test_unknown_endpoint_returns_404(self):
        body = json.dumps({}).encode()
        req = urllib.request.Request(
            self._url("/nonexistent"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "Should have raised HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Port file management
# ──────────────────────────────────────────────────────────────────────────────


class TestPortFile:
    def setup_method(self):
        self.test_port_dir = f"/tmp/roll_mcbots_ports_test_{os.getpid()}"
        os.environ["MCBOTS_PORT_DIR"] = self.test_port_dir
        self.mgr = _build_manager()
        self.mgr.http_port = 12345

    def teardown_method(self):
        # Cleanup
        import shutil
        if os.path.exists(self.test_port_dir):
            shutil.rmtree(self.test_port_dir)
        os.environ.pop("MCBOTS_PORT_DIR", None)

    def test_write_and_read_port_file(self):
        self.mgr._write_port_file()
        port_file = os.path.join(self.test_port_dir, f"roll_{self.mgr.env_id}.port")
        assert os.path.exists(port_file)
        with open(port_file) as f:
            assert f.read() == "12345"

    def test_cleanup_port_file(self):
        self.mgr._write_port_file()
        port_file = os.path.join(self.test_port_dir, f"roll_{self.mgr.env_id}.port")
        assert os.path.exists(port_file)

        self.mgr._cleanup_port_file()
        assert not os.path.exists(port_file)

    def test_cleanup_nonexistent_file_no_error(self):
        """Cleaning up a non-existent file should not raise."""
        self.mgr._cleanup_port_file()  # should not raise


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Manager initialization
# ──────────────────────────────────────────────────────────────────────────────


class TestManagerInit:
    def test_default_config_values(self):
        mgr = _build_manager(env_config={"tag": "MC", "env_id": 3, "group_id": 2})
        assert mgr.tag == "MC"
        assert mgr.env_id == 3
        assert mgr.group_id == 2

    def test_default_env_config(self):
        mgr = _build_manager(env_config={})
        assert mgr.tag == "Minecraft"
        assert mgr.env_id == 0
        assert mgr.group_id == 0

    def test_initial_state(self):
        mgr = _build_manager()
        assert mgr.current_window_messages == []
        assert mgr.episode_id == 0
        assert mgr.sequence_id_counter == 0
        assert mgr.mode == "train"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
