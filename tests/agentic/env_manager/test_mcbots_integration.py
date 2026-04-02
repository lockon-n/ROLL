"""
Integration tests for McbotsEnvManager._formulate_dataproto with a real Qwen3.5 tokenizer,
processor, and collator — covering both text-only and multimodal (image) scenarios.

Tests the full chain:
    tokenize → collator (with VL processor) → split_by_token → token_ids_to_assistant_mask → DataProto

Run with:
    pytest tests/agentic/env_manager/test_mcbots_integration.py -v

Requires:
    Local Qwen3.5-0.8B model at /homes/junlong/junlong_export_ssd/models/Qwen/Qwen3.5-0.8B
"""
import io
import os
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest
import torch
from transformers import BatchFeature

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.mcbots_env_manager import (
    McbotsEnvManager,
    _extract_response_token_ranges,
)
from roll.pipeline.agentic.env_manager.token_mask_utils import (
    split_by_token,
    token_ids_to_assistant_mask,
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
    """Build the Qwen3.5 VL extra_data_provider (3D-RoPE position_ids) without Ray."""
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config = AutoConfig.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    proc_tmp = None
    try:
        from transformers import AutoProcessor as AP
        proc_tmp = AP.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    except Exception:
        pass

    spatial_merge_size = getattr(getattr(config, "vision_config", None), "spatial_merge_size", None)
    if spatial_merge_size is None and proc_tmp is not None:
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
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        rope_index = out[0]
        # Normalize to 4-channel: [text_pos, mrope_t, mrope_h, mrope_w]
        if rope_index.dim() == 3 and rope_index.size(0) == 3:
            bsz, seqlen = input_ids.shape
            text_pos = torch.clamp(attention_mask.long().cumsum(-1) - 1, min=0).unsqueeze(0)
            rope_index = torch.cat([text_pos, rope_index], dim=0)
        # (C, bsz, seqlen) -> (bsz, C, seqlen) for DataProto
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


@pytest.fixture(scope="module")
def custom_chat_template() -> Optional[str]:
    if os.path.isfile(CUSTOM_CHAT_TEMPLATE_PATH):
        with open(CUSTOM_CHAT_TEMPLATE_PATH) as f:
            return f.read()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_test_image(width: int = 64, height: int = 64, color: tuple = (255, 0, 0)) -> PIL.Image.Image:
    """Create a simple test image."""
    return PIL.Image.new("RGB", (width, height), color=color)


@dataclass
class _FakeGeneratingArgs:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048

    def to_dict(self):
        return {"temperature": self.temperature, "top_p": self.top_p, "max_new_tokens": self.max_new_tokens}


@dataclass
class _FakePipelineConfig:
    sequence_length: int = 512
    chat_template: Optional[str] = None


@dataclass
class _FakeWorkerConfig:
    generating_args: _FakeGeneratingArgs = field(default_factory=_FakeGeneratingArgs)
    llm_proxy: Any = None


def _build_manager(
    tokenizer,
    collator,
    chat_template: Optional[str] = None,
    sequence_length: int = 512,
) -> McbotsEnvManager:
    """Build a McbotsEnvManager with real tokenizer, processor, and collator."""
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
    mgr.llm_proxy = MagicMock()
    mgr.output_queue = MagicMock()
    mgr.mode = "train"
    mgr.generate_scheduler = MagicMock()
    mgr.thread_lock = MagicMock()
    mgr.extra_data_provider = None
    mgr.current_window_messages = []
    mgr.episode_id = 0
    mgr.sequence_id_counter = 0
    mgr.group_seed = 0
    mgr.http_port = 0
    mgr.tag = "Minecraft"
    mgr.env_id = 0
    mgr.group_id = 0
    return mgr


def _assert_dataproto_valid(result: DataProto, seq_len: int, num_assistant_turns: int):
    """Common assertions for DataProto output."""
    assert isinstance(result, DataProto)

    # Batch keys
    for key in ["input_ids", "attention_mask", "position_ids", "response_mask", "prompt_mask", "scores"]:
        assert key in result.batch, f"Missing batch key: {key}"

    # Shape checks (position_ids is 3D for VL: (1, C, seq_len))
    for key in ["input_ids", "attention_mask", "response_mask", "prompt_mask", "scores"]:
        assert result.batch[key].shape == (1, seq_len), (
            f"{key} shape mismatch: {result.batch[key].shape} != (1, {seq_len})"
        )

    pos_ids = result.batch["position_ids"]
    if pos_ids.dim() == 3:
        assert pos_ids.shape[0] == 1 and pos_ids.shape[2] == seq_len, (
            f"position_ids 3D shape mismatch: {pos_ids.shape}, expected (1, C, {seq_len})"
        )
    else:
        assert pos_ids.shape == (1, seq_len), (
            f"position_ids 2D shape mismatch: {pos_ids.shape} != (1, {seq_len})"
        )

    # Non-tensor keys
    for key in ["env_ids", "group_ids", "messages_list", "tags", "step_scores", "episode_scores", "response_token_ranges"]:
        assert key in result.non_tensor_batch, f"Missing non-tensor key: {key}"

    # Metrics
    assert result.meta_info["metrics"]["env/Minecraft/num_actions"] == num_assistant_turns

    # response_mask and prompt_mask should not overlap
    response_mask = result.batch["response_mask"][0]
    prompt_mask = result.batch["prompt_mask"][0]
    overlap = (response_mask.bool() & prompt_mask.bool()).sum().item()
    assert overlap == 0, f"response_mask and prompt_mask overlap at {overlap} positions"

    # response_mask should have some 1s
    assert response_mask.sum().item() > 0, "response_mask should have some 1s"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: split_by_token with real tokenizer
# ──────────────────────────────────────────────────────────────────────────────


@requires_qwen
class TestSplitByTokenReal:
    """Verify split_by_token aligns segments with messages using a real tokenizer."""

    def test_single_turn(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        bos_token = token_ids[0]

        segments = split_by_token(token_ids, bos_token, messages=messages, tokenizer=tokenizer)

        assert len(segments) == len(messages)
        reconstructed = [tid for seg in segments for tid in seg]
        assert reconstructed == token_ids

    def test_multi_turn(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are a Minecraft bot."},
            {"role": "user", "content": "What do you see?"},
            {"role": "assistant", "content": "I see a tree and some grass blocks."},
            {"role": "user", "content": "Go chop the tree."},
            {"role": "assistant", "content": "OK, moving to the tree now."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        bos_token = token_ids[0]

        segments = split_by_token(token_ids, bos_token, messages=messages, tokenizer=tokenizer)

        assert len(segments) == len(messages)
        reconstructed = [tid for seg in segments for tid in seg]
        assert reconstructed == token_ids

    def test_with_custom_chat_template(self, tokenizer, custom_chat_template):
        if custom_chat_template is None:
            pytest.skip("Custom chat template not found")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, chat_template=custom_chat_template
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        bos_token = token_ids[0]

        segments = split_by_token(token_ids, bos_token, messages=messages, tokenizer=tokenizer)

        assert len(segments) == len(messages)
        reconstructed = [tid for seg in segments for tid in seg]
        assert reconstructed == token_ids


# ──────────────────────────────────────────────────────────────────────────────
# Tests: token_ids_to_assistant_mask with real tokenizer
# ──────────────────────────────────────────────────────────────────────────────


@requires_qwen
class TestAssistantMaskReal:
    """Verify assistant mask correctly marks only assistant content tokens."""

    def test_single_turn_mask(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        bos_token = token_ids[0]

        segments = split_by_token(token_ids, bos_token, messages=messages, tokenizer=tokenizer)
        masks = token_ids_to_assistant_mask(messages=messages, input_ids_list=segments, tokenizer=tokenizer)

        assert all(v == 0 for v in masks[0]), "System message mask should be all zeros"
        assert all(v == 0 for v in masks[1]), "User message mask should be all zeros"
        assert any(v == 1 for v in masks[2]), "Assistant mask should have content tokens marked as 1"

        flat_mask = [item for m in masks for item in m]
        assert len(flat_mask) == len(token_ids)

    def test_multi_turn_mask(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are a Minecraft bot."},
            {"role": "user", "content": "What do you see?"},
            {"role": "assistant", "content": "I see a tree."},
            {"role": "user", "content": "Chop it."},
            {"role": "assistant", "content": "OK, chopping now."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        bos_token = token_ids[0]

        segments = split_by_token(token_ids, bos_token, messages=messages, tokenizer=tokenizer)
        masks = token_ids_to_assistant_mask(messages=messages, input_ids_list=segments, tokenizer=tokenizer)

        assert any(v == 1 for v in masks[2]), "First assistant turn should have 1s"
        assert any(v == 1 for v in masks[4]), "Second assistant turn should have 1s"

        for i in [0, 1, 3]:
            assert all(v == 0 for v in masks[i]), f"Message {i} (non-assistant) should be all zeros"

    def test_mask_excludes_format_tokens(self, tokenizer):
        """Format tokens like <|im_start|>assistant\\n should be 0, content should be 1."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello world"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        bos_token = token_ids[0]

        segments = split_by_token(token_ids, bos_token, messages=messages, tokenizer=tokenizer)
        masks = token_ids_to_assistant_mask(messages=messages, input_ids_list=segments, tokenizer=tokenizer)

        assistant_mask = masks[1]
        assert assistant_mask[0] == 0, "First token of assistant segment (format) should be 0"

        found_transition = False
        for i in range(1, len(assistant_mask)):
            if assistant_mask[i - 1] == 0 and assistant_mask[i] == 1:
                found_transition = True
                break
        assert found_transition, "Should find a 0→1 transition (format tokens → content)"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _formulate_dataproto — text only (with real collator + 3D-RoPE)
# ──────────────────────────────────────────────────────────────────────────────


@requires_qwen
class TestFormulateDataprotoTextOnly:
    """End-to-end _formulate_dataproto with real Qwen3.5 collator, text-only messages."""

    def test_single_turn(self, tokenizer, collator):
        mgr = _build_manager(tokenizer, collator, sequence_length=256)
        messages = [
            {"role": "system", "content": "You are a helpful Minecraft assistant."},
            {"role": "user", "content": "What blocks are nearby?"},
            {"role": "assistant", "content": "I can see stone, dirt, and oak logs nearby."},
        ]
        result = mgr._formulate_dataproto(messages, reward=1.0)
        _assert_dataproto_valid(result, seq_len=256, num_assistant_turns=1)

    def test_multi_turn(self, tokenizer, collator):
        mgr = _build_manager(tokenizer, collator, sequence_length=512)
        messages = [
            {"role": "system", "content": "You are a Minecraft bot."},
            {"role": "user", "content": "What do you see?"},
            {"role": "assistant", "content": "I see a tree and some grass."},
            {"role": "user", "content": "Go chop the tree."},
            {"role": "assistant", "content": "Moving to the tree. I'll use my axe."},
        ]
        result = mgr._formulate_dataproto(messages, reward=2.5)
        _assert_dataproto_valid(result, seq_len=512, num_assistant_turns=2)

    def test_position_ids_are_3d_rope(self, tokenizer, collator):
        """Real Qwen3.5 collator should produce 3D position_ids (bsz, C, seqlen)."""
        mgr = _build_manager(tokenizer, collator, sequence_length=256)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = mgr._formulate_dataproto(messages, reward=1.0)
        pos_ids = result.batch["position_ids"]
        assert pos_ids.dim() == 3, f"Expected 3D position_ids, got {pos_ids.dim()}D"
        assert pos_ids.shape[0] == 1  # batch size
        assert pos_ids.shape[1] == 4  # 4 channels: text + mrope_t + mrope_h + mrope_w
        assert pos_ids.shape[2] == 256  # sequence_length

    def test_reward_at_last_response_token(self, tokenizer, collator):
        mgr = _build_manager(tokenizer, collator, sequence_length=256)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        reward = 3.14
        result = mgr._formulate_dataproto(messages, reward=reward)

        scores = result.batch["scores"][0]
        response_mask = result.batch["response_mask"][0]
        last_response_idx = torch.nonzero(response_mask).squeeze(-1)[-1].item()

        assert scores[last_response_idx].item() == pytest.approx(reward)

        other_scores = torch.cat([scores[:last_response_idx], scores[last_response_idx + 1:]])
        assert other_scores.sum().item() == 0.0

    def test_multi_turn_response_token_ranges(self, tokenizer, collator):
        mgr = _build_manager(tokenizer, collator, sequence_length=512)
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        result = mgr._formulate_dataproto(messages, reward=1.0)

        ranges = result.non_tensor_batch["response_token_ranges"][0]
        assert len(ranges) == 3, f"Expected 3 response ranges, got {len(ranges)}"

        for i in range(len(ranges) - 1):
            assert ranges[i][1] <= ranges[i + 1][0], (
                f"Range {i} end ({ranges[i][1]}) > range {i+1} start ({ranges[i+1][0]})"
            )
        for i, (start, end) in enumerate(ranges):
            assert start < end, f"Range {i}: start ({start}) >= end ({end})"

    def test_truncation(self, tokenizer, collator):
        short_seq_len = 32
        mgr = _build_manager(tokenizer, collator, sequence_length=short_seq_len)
        messages = [
            {"role": "user", "content": "Tell me a long story about Minecraft."},
            {"role": "assistant", "content": "Once upon a time in a vast blocky world, "
             "there was a brave adventurer who explored deep caves, "
             "fought creepers, and built magnificent castles."},
        ]
        result = mgr._formulate_dataproto(messages, reward=1.0)

        for key in ["input_ids", "attention_mask", "response_mask", "prompt_mask", "scores"]:
            assert result.batch[key].shape[1] == short_seq_len

    def test_with_custom_chat_template(self, tokenizer, collator, custom_chat_template):
        if custom_chat_template is None:
            pytest.skip("Custom chat template not found")

        mgr = _build_manager(tokenizer, collator, chat_template=custom_chat_template, sequence_length=256)
        messages = [
            {"role": "system", "content": "You are a Minecraft bot."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        result = mgr._formulate_dataproto(messages, reward=1.0)
        _assert_dataproto_valid(result, seq_len=256, num_assistant_turns=1)

    def test_negative_reward(self, tokenizer, collator):
        mgr = _build_manager(tokenizer, collator, sequence_length=256)
        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "I failed."},
        ]
        result = mgr._formulate_dataproto(messages, reward=-1.0)

        scores = result.batch["scores"][0]
        response_mask = result.batch["response_mask"][0]
        last_response_idx = torch.nonzero(response_mask).squeeze(-1)[-1].item()
        assert scores[last_response_idx].item() == pytest.approx(-1.0)

    def test_zero_reward(self, tokenizer, collator):
        mgr = _build_manager(tokenizer, collator, sequence_length=256)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = mgr._formulate_dataproto(messages, reward=0.0)
        assert result.batch["scores"][0].sum().item() == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _formulate_dataproto — with images (multimodal)
# ──────────────────────────────────────────────────────────────────────────────


@requires_qwen
class TestFormulateDataprotoWithImages:
    """End-to-end _formulate_dataproto with real Qwen3.5 VL processor and images."""

    def test_single_image_single_turn(self, tokenizer, collator):
        """User sends one image, assistant responds — basic multimodal flow."""
        mgr = _build_manager(tokenizer, collator, sequence_length=512)

        # OpenAI format: image_url with base64 → _extract_and_convert_images → Qwen VL format
        img = _make_test_image(64, 64, (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = __import__("base64").b64encode(buf.getvalue()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "What do you see in this screenshot?"},
                ],
            },
            {"role": "assistant", "content": "I see a red square image."},
        ]

        result = mgr._formulate_dataproto(messages, reward=1.0)
        _assert_dataproto_valid(result, seq_len=512, num_assistant_turns=1)

        # Verify image tokens expanded input_ids (should be longer than text-only)
        attention_mask = result.batch["attention_mask"][0]
        num_active = attention_mask.sum().item()
        assert num_active > 30, f"With image, should have many tokens, got {num_active}"

    def test_multi_image_multi_turn(self, tokenizer, collator):
        """Multiple images across turns — verify all are processed."""
        mgr = _build_manager(tokenizer, collator, sequence_length=1024)

        img1 = _make_test_image(32, 32, (255, 0, 0))
        img2 = _make_test_image(32, 32, (0, 255, 0))

        def _to_b64(img):
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return __import__("base64").b64encode(buf.getvalue()).decode()

        messages = [
            {"role": "system", "content": "You are a Minecraft vision assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(img1)}"}},
                    {"type": "text", "text": "What's in the first screenshot?"},
                ],
            },
            {"role": "assistant", "content": "I see a red area."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(img2)}"}},
                    {"type": "text", "text": "And this one?"},
                ],
            },
            {"role": "assistant", "content": "This shows a green area."},
        ]

        result = mgr._formulate_dataproto(messages, reward=2.0)
        _assert_dataproto_valid(result, seq_len=1024, num_assistant_turns=2)

        # Should have 2 response ranges
        ranges = result.non_tensor_batch["response_token_ranges"][0]
        assert len(ranges) == 2

    def test_mixed_text_and_image_turns(self, tokenizer, collator):
        """Mix of pure text and image turns in the same conversation."""
        mgr = _build_manager(tokenizer, collator, sequence_length=512)

        img = _make_test_image(32, 32, (0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = __import__("base64").b64encode(buf.getvalue()).decode()

        messages = [
            {"role": "system", "content": "You are a Minecraft bot with vision."},
            {"role": "user", "content": "What's your status?"},  # text-only turn
            {"role": "assistant", "content": "I'm ready to help."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Now look at this."},
                ],
            },
            {"role": "assistant", "content": "I see a blue square."},
        ]

        result = mgr._formulate_dataproto(messages, reward=1.5)
        _assert_dataproto_valid(result, seq_len=512, num_assistant_turns=2)

    def test_image_position_ids_differ_from_text(self, tokenizer, collator):
        """Image tokens should cause 3D-RoPE position_ids to diverge from text-only."""
        seq_len = 512

        # Text-only
        mgr_text = _build_manager(tokenizer, collator, sequence_length=seq_len)
        text_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result_text = mgr_text._formulate_dataproto(text_messages, reward=1.0)

        # With image
        mgr_img = _build_manager(tokenizer, collator, sequence_length=seq_len)
        img = _make_test_image(32, 32, (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = __import__("base64").b64encode(buf.getvalue()).decode()

        img_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Hello"},
                ],
            },
            {"role": "assistant", "content": "Hi!"},
        ]
        result_img = mgr_img._formulate_dataproto(img_messages, reward=1.0)

        # Both should be 3D
        assert result_text.batch["position_ids"].dim() == 3
        assert result_img.batch["position_ids"].dim() == 3

        # Image version should have more active tokens
        text_active = result_text.batch["attention_mask"][0].sum().item()
        img_active = result_img.batch["attention_mask"][0].sum().item()
        assert img_active > text_active, (
            f"Image should have more tokens ({img_active}) than text-only ({text_active})"
        )

    def test_image_reward_placement(self, tokenizer, collator):
        """Reward should still be at the last response token even with images."""
        mgr = _build_manager(tokenizer, collator, sequence_length=512)

        img = _make_test_image(32, 32)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = __import__("base64").b64encode(buf.getvalue()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Describe this."},
                ],
            },
            {"role": "assistant", "content": "A small red square."},
        ]
        reward = 5.0
        result = mgr._formulate_dataproto(messages, reward=reward)

        scores = result.batch["scores"][0]
        response_mask = result.batch["response_mask"][0]
        last_response_idx = torch.nonzero(response_mask).squeeze(-1)[-1].item()

        assert scores[last_response_idx].item() == pytest.approx(reward)
        other_scores = torch.cat([scores[:last_response_idx], scores[last_response_idx + 1:]])
        assert other_scores.sum().item() == 0.0
