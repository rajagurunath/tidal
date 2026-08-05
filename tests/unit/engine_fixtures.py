"""Scheduler-construction helpers for the engine tests.

These mirror vLLM's own unit-test harness so our subclass is exercised exactly
the way upstream exercises the base class:

* ``create_scheduler`` <- ``vllm/tests/v1/core/test_scheduler.py::create_scheduler_with_priority``
  (priority policy, watermark 0.0, synthetic ``KVCacheConfig`` with a single
  ``FullAttentionSpec`` group, ``cache_config.num_gpu_blocks`` set by hand)
* ``create_requests`` <- ``vllm/tests/v1/core/test_scheduler.py::create_requests_with_priority``

No weights are ever loaded: ``ModelConfig`` only reads the HF *config* (and we
skip tokenizer init), and the KV "cache" is a block bookkeeping structure with
no tensors behind it. The default model is the one Tidal serves on the Mac dev
box, so it is already in the local HF cache; override with ``TIDAL_TEST_MODEL``.

Only importable where vLLM is installed — the engine venv
(``vllm-experiments/.venv``), not tidal's own.
"""

from __future__ import annotations

import os

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

DEFAULT_MODEL = os.environ.get("TIDAL_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
EOS_TOKEN_ID = 50256

_none_hash_initialized = False


def create_scheduler(
    scheduler_cls: type[Scheduler] = Scheduler,
    *,
    model: str = DEFAULT_MODEL,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 256,
    max_model_len: int = 2048,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    num_blocks: int = 1000,
    block_size: int = 16,
    policy: str = "priority",
) -> Scheduler:
    """Build a real vLLM V1 scheduler (or any subclass) with no model weights."""
    model_config = ModelConfig(
        model=model,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy=policy,
        # Deterministic admission/preemption mechanics, as upstream does.
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
    )
    scheduler.use_v2_model_runner = False
    return scheduler


def create_requests(
    num_requests: int,
    *,
    priorities: list[int] | None = None,
    num_tokens: int = 32,
    max_tokens: int = 16,
    arrival_times: list[float] | None = None,
    block_size: int = 16,
    starting_idx: int = 0,
    same_prompt: bool = False,
) -> list[Request]:
    """Synthetic requests with explicit priorities (0 = online, >=10 = batch)."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    if priorities is None:
        priorities = [0] * num_requests
    assert len(priorities) == num_requests
    if arrival_times is None:
        arrival_times = [float(i) for i in range(num_requests)]
    assert len(arrival_times) == num_requests

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    requests = []
    for i in range(num_requests):
        token = starting_idx if same_prompt else i + starting_idx
        requests.append(
            Request(
                request_id=f"{i + starting_idx}",
                prompt_token_ids=[token] * num_tokens,
                sampling_params=sampling_params,
                pooling_params=None,
                arrival_time=arrival_times[i],
                priority=priorities[i],
                block_hasher=block_hasher,
            )
        )
    return requests


def make_model_runner_output(scheduler: Scheduler) -> ModelRunnerOutput:
    """A one-token-per-running-request output, enough to advance a step."""
    req_ids = [r.request_id for r in scheduler.running]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
