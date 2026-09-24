# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Engram with readonly accepted history and global ceil-row TP shards."""

import ctypes
import os
import time
from functools import lru_cache

import torch
import torch.nn as nn
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall, require_prepared
from b12x.sequence import engram as native
from b12x.sequence._shared.disk_table import MappedHostAllocation

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_transfer import copy_weight, get_file_tensor_source
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    set_b12x_preparation_provider,
)
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource
from vllm.utils.torch_utils import direct_register_custom_op


_FP8_BLOCK_ROWS = 32


class _PaddedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear whose output need not divide by the TP size.

    The output is padded to a multiple of tp_size x 32 rows so every rank owns
    whole block-FP8 scale blocks. The last rank's missing checkpoint rows are
    zero-filled through the loader's existing ``allow_tp_padding`` path (a
    narrowed copy the B12X checkpoint loader supports), so the padded rows
    compute exactly zero, and the gathered output is sliced back to the
    logical width.
    """

    def __init__(self, input_size, output_size, **kwargs):
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        quantum = tp_size * _FP8_BLOCK_ROWS
        padded = -(-output_size // quantum) * quantum
        self.logical_output_size = output_size
        self.padded_output_size = padded
        super().__init__(input_size, padded, **kwargs)
        if padded != output_size:
            for name in ("weight", "weight_scale_inv", "weight_scale"):
                param = getattr(self, name, None)
                if param is not None:
                    param.allow_tp_padding = True

    def forward(self, input_):
        output = super().forward(input_)
        if self.padded_output_size == self.logical_output_size:
            return output
        return output[..., : self.logical_output_size].contiguous()

logger = init_logger(__name__)
DEAD_ID = -1

# SPARK3_ENGRAM_ASYNC=1: read disk Engram rows on a reader thread while the
# forward graph launches, and gate each Engram layer on a device-visible flag
# instead of finishing every read before the launch (experiment overlay).
ENGRAM_ASYNC = os.environ.get("SPARK3_ENGRAM_ASYNC") == "1"
# Debug only: delay each reader-thread read, to show the graph really waits.
ENGRAM_ASYNC_DELAY = float(os.environ.get("SPARK3_ENGRAM_ASYNC_DELAY_MS", "0")) / 1000
# Debug only: log every stage, read, and wait of the asynchronous path.
ENGRAM_ASYNC_TRACE = os.environ.get("SPARK3_ENGRAM_ASYNC_TRACE") == "1"
# A reader that cannot see its row IDs within this long raises instead of
# spinning forever: the device is blocked upstream of the Engram staging.
ENGRAM_ASYNC_TIMEOUT_S = float(os.environ.get("SPARK3_ENGRAM_ASYNC_TIMEOUT_S", "20"))


def wait_for_event(event, what: str) -> None:
    """Poll a CUDA event with a deadline instead of spinning in synchronize."""
    deadline = time.monotonic() + ENGRAM_ASYNC_TIMEOUT_S
    while not event.query():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"asynchronous Engram rows: {what} not reached within "
                f"{ENGRAM_ASYNC_TIMEOUT_S:g} s; the device is blocked upstream of "
                "the Engram staging"
            )
        time.sleep(50e-6)


def _cuda_check(result):
    status, *values = result
    from cuda.bindings import driver

    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Engram row flag: CUDA driver call failed: {status}")
    return values[0] if values else None


class RowsReadyFlag:
    """A 32-bit flag in mapped host memory that device streams wait on.

    The staging stream writes 0, the reader's side stream writes 1 after the
    row decode, and the forward waits for 1 before reading the rows. The host
    can write 1 directly to release a waiting forward if the reader fails.
    Starts at 1 so profiling and warmup forwards never wait.
    """

    def __init__(self):
        from cuda.bindings import driver

        self.driver = driver
        self.pointer = _cuda_check(driver.cuMemHostAlloc(4, driver.CU_MEMHOSTALLOC_DEVICEMAP))
        self.device_pointer = _cuda_check(driver.cuMemHostGetDevicePointer(self.pointer, 0))
        self.host = ctypes.c_uint32.from_address(int(self.pointer))
        self.host.value = 1

    def write(self, stream, value: int) -> None:
        _cuda_check(self.driver.cuStreamWriteValue32(stream.cuda_stream, self.device_pointer, value, 0))

    def wait(self, stream) -> None:
        _cuda_check(self.driver.cuStreamWaitValue32(
            stream.cuda_stream, self.device_pointer, 1,
            int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        ))

    def release_from_host(self) -> None:
        self.host.value = 1


class HostGate:
    """Device streams wait until the host has written a sequence number.

    The reader thread releases each step's gate after its disk read, so the
    decode kernel, queued earlier by the main thread, runs only on complete
    rows. Only the main thread launches kernels.
    """

    def __init__(self):
        from cuda.bindings import driver

        self.driver = driver
        self.pointer = _cuda_check(driver.cuMemHostAlloc(4, driver.CU_MEMHOSTALLOC_DEVICEMAP))
        self.device_pointer = _cuda_check(driver.cuMemHostGetDevicePointer(self.pointer, 0))
        self.host = ctypes.c_uint32.from_address(int(self.pointer))
        self.host.value = 0
        self.sequence = 0

    def next(self) -> int:
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return self.sequence

    def wait(self, stream, sequence: int) -> None:
        _cuda_check(self.driver.cuStreamWaitValue32(
            stream.cuda_stream, self.device_pointer, sequence,
            int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        ))

    def release(self, sequence: int) -> None:
        self.host.value = sequence


_ROWS_READY_FLAGS: dict[int, RowsReadyFlag] = {}


def _engram_wait_rows(rows: torch.Tensor, key: int) -> None:
    stream = torch.cuda.current_stream(rows.device)
    if ENGRAM_ASYNC_TRACE:
        logger.info(
            "engram-async wait key=%x capturing=%s flag=%d",
            key, torch.cuda.is_current_stream_capturing(), _ROWS_READY_FLAGS[key].host.value,
        )
    _ROWS_READY_FLAGS[key].wait(stream)


def _engram_wait_rows_fake(rows: torch.Tensor, key: int) -> None:
    return None


direct_register_custom_op(
    op_name="spark3_engram_wait_rows",
    op_func=_engram_wait_rows,
    mutates_args=["rows"],
    fake_impl=_engram_wait_rows_fake,
)


@lru_cache(maxsize=2)
def _token_map(path, revision, trust_remote_code):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, revision=revision, trust_remote_code=trust_remote_code
    )
    return native.build_compressed_token_map(tokenizer)


class EngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.geometry = native.build_geometry(
            layer_ids=self.layer_ids,
            base_table_size=config.engram_vocab_size,
            compressed_vocab_size=config.engram_compressed_vocab_size,
        )
        if (self.max_ngram_size, self.n_heads, self.head_dim) != (4, 8, 256):
            raise ValueError(
                "V4.1 Engram requires four-gram/eight-head/dim256 geometry"
            )
        if self.geometry.num_embeddings != self.num_embeddings:
            raise ValueError(
                "Engram checkpoint table rows do not match global hash geometry"
            )
        vc = get_current_vllm_config()
        mc = vc.model_config
        engram_config = vc.engram_config
        self.table_memory = (
            engram_config.table_memory if engram_config is not None else "device"
        )
        self.disk_resident_scales = (
            engram_config.disk_resident_scales if engram_config is not None else False
        )
        self.projection_tp = (
            engram_config.projection_tp if engram_config is not None else False
        )
        if (
            self.table_memory == "disk"
            and vc.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Disk Engram requires pipeline_parallel_size=1")
        token_map, compressed_size = _token_map(
            mc.tokenizer, mc.revision, mc.trust_remote_code
        )
        if compressed_size != config.engram_compressed_vocab_size:
            raise ValueError("Engram tokenizer compressed vocabulary mismatch")
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.caps = tuple(
            native.Caps(
                device=device,
                max_tokens=vc.scheduler_config.max_num_batched_tokens,
                max_seqs=vc.scheduler_config.max_num_seqs,
                max_requests=vc.scheduler_config.max_num_seqs,
                vocab_size=config.vocab_size,
                layer_id=layer,
                tp_size=get_tensor_model_parallel_world_size(),
                tp_rank=get_tensor_model_parallel_rank(),
            )
            for layer in self.layer_ids
        )
        self.hash_plans = tuple(
            native.plan(caps, token_map=token_map, geometry=self.geometry)
            for caps in self.caps
        )
        self.lookup_plans = tuple(
            native.plan(
                caps,
                token_map=token_map,
                geometry=self.geometry,
                invocation={
                    "operation": "lookup",
                    "compact_rows": self.table_memory == "disk",
                    "resident_scales": self.disk_resident_scales,
                },
            )
            for caps in self.caps
        )

    @classmethod
    def from_config(cls, config):
        return cls(config) if getattr(config, "engram_layer_ids", None) else None


@triton.jit(do_not_specialize=["tokens", "seqs"])
def _prepare_metadata(
    ids,
    mask,
    starts,
    history,
    token_map,
    ids_out,
    mask_out,
    starts_out,
    history_out,
    slots,
    num_seqs_out,
    num_tokens_out,
    tokens,
    seqs,
    max_tokens: tl.constexpr,
    max_seqs: tl.constexpr,
    history_stride: tl.constexpr,
    vocab: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # Captured input tensors include padding; starts carries the live count.
    live_tokens = tl.load(starts + seqs)
    value = tl.load(ids + i, (i < tokens) & (i < live_tokens), other=0)
    keep = tl.load(mask + i, (i < tokens) & (i < live_tokens), other=False)
    tl.store(ids_out + i, value, i < max_tokens)
    tl.store(mask_out + i, keep, i < max_tokens)
    start = tl.load(starts + i, i <= seqs, other=live_tokens)
    tl.store(starts_out + i, start, i <= max_seqs)
    tl.store(slots + i, i.to(tl.int32), i < max_seqs)
    r, col = i // 3, i % 3
    raw = tl.load(history + r * history_stride + col, r < seqs, other=-1)
    valid = (r < seqs) & (raw >= 0) & (raw < vocab) & (raw != 129264)
    compressed = tl.load(token_map + raw, valid, other=-1)
    tl.store(history_out + i, compressed, i < max_seqs * 3)
    if tl.program_id(0) == 0:
        tl.store(num_seqs_out, seqs)
        tl.store(num_tokens_out, live_tokens)


class NgramHashState(nn.Module):
    def __init__(self, vllm_config, layout, swa_cache_module):
        super().__init__()
        self.layout = layout
        self.lookback_depth = 3
        self.use_slot_cache = False
        c = layout.caps[0]
        self.bindings = []
        self._scratch = []
        self._hashes = []
        self._token_map = None
        for i in range(len(layout.hash_plans)):
            self.register_buffer(
                f"hashes_{i}",
                torch.empty((0,), dtype=torch.int64, device=c.device),
                persistent=False,
            )
        for name, shape, dtype in (
            ("ids", (c.max_tokens,), torch.int64),
            ("mask", (c.max_tokens,), torch.bool),
            ("starts", (c.max_seqs + 1,), torch.int32),
            ("history", (c.max_requests, 3), torch.int64),
            ("slots", (c.max_seqs,), torch.int32),
            ("num_seqs", (1,), torch.int32),
            ("num_tokens", (1,), torch.int32),
        ):
            self.register_buffer(
                name, torch.empty(shape, dtype=dtype, device=c.device), persistent=False
            )
        set_b12x_preparation_provider(self, self)

    def _ensure_bindings(self) -> None:
        """Bind the runner-owned buffers to each prepared hash plan, once."""
        if self.bindings:
            return
        bindings, scratch, hashes = [], [], []
        token_map = None
        for plan in self.layout.hash_plans:
            state = require_prepared(plan, "sequence.engram")
            if token_map is None:
                token_map = state.token_map
            (spec,) = state.scratch_specs()
            buffer = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            hash_ids = torch.empty(
                (state.caps.max_tokens, 24), dtype=torch.int64, device=state.caps.device
            )
            scratch.append(buffer)
            hashes.append(hash_ids)
            bindings.append(
                native.bind(
                    plan,
                    scratch=buffer,
                    token_ids=self.ids,
                    token_mask=self.mask,
                    query_start_loc=self.starts,
                    request_slots=self.slots,
                    committed_history=self.history,
                    num_seqs=self.num_seqs,
                    num_tokens=self.num_tokens,
                    hash_ids=hash_ids,
                )
            )
        self.bindings, self._scratch, self._hashes = bindings, scratch, hashes
        self._token_map = token_map

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if workload.max_tokens > self.layout.caps[0].max_tokens:
            return ()

        def make_call(state):
            from b12x.sequence.engram._impl import _bind_state

            (spec,) = state.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            hashes = torch.empty(
                (state.caps.max_tokens, 24), dtype=torch.int64, device=state.caps.device
            )
            ids = torch.full(
                (state.caps.max_tokens,), 2, dtype=torch.int64, device=state.caps.device
            )
            mask = torch.ones(
                (state.caps.max_tokens,), dtype=torch.bool, device=state.caps.device
            )
            starts = torch.zeros(
                (state.caps.max_seqs + 1,), dtype=torch.int32, device=state.caps.device
            )
            starts[1] = 1
            slots = torch.zeros(
                (state.caps.max_seqs,), dtype=torch.int32, device=state.caps.device
            )
            num_seqs = torch.ones((1,), dtype=torch.int32, device=state.caps.device)
            num_tokens = torch.ones((1,), dtype=torch.int32, device=state.caps.device)
            binding = _bind_state(
                state,
                scratch=scratch,
                token_ids=ids,
                token_mask=mask,
                query_start_loc=starts,
                request_slots=slots,
                committed_history=self.history,
                num_seqs=num_seqs,
                num_tokens=num_tokens,
                hash_ids=hashes,
            )
            history_before = self.history[0].clone()

            return PreparedCall(
                run=lambda: state.run(binding, 1),
                produce=lambda: self.history[0].fill_(DEAD_ID),
                restore=lambda: self.history[0].copy_(history_before),
                owners=(self.history,),
            )

        requests = tuple(
            plan.request(
                name=f"engram/hash/{index}",
                prepare_call=make_call,
                benchmark_call=make_call,
            )
            for index, plan in enumerate(self.layout.hash_plans)
        )
        return (
            B12xPreparationUnit(
                name="EngramHash",
                key=(id(self.layout), workload.max_tokens),
                requests=requests,
                stage="weights",
            ),
        )

    def ensure_cache(self):
        return True  # Hash history belongs to the runner, never the KV slot pool.

    def run_native(self, ids, mask, starts, history, out):
        retain_cuda_graph_capture_resource(self)
        self._ensure_bindings()
        c = self.layout.caps[0]
        seqs = starts.numel() - 1
        if ids.numel() > c.max_tokens or seqs > c.max_seqs:
            raise ValueError("Engram live rows exceed preallocated capacity")
        if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] < seqs:
            raise ValueError(
                "Engram requires chronological accepted history [requests,3]"
            )
        work = max(c.max_tokens, c.max_seqs * 3, c.max_seqs + 1)
        _prepare_metadata[(triton.cdiv(work, 256),)](
            ids,
            mask,
            starts,
            history,
            self._token_map,
            self.ids,
            self.mask,
            self.starts,
            self.history,
            self.slots,
            self.num_seqs,
            self.num_tokens,
            ids.numel(),
            seqs,
            c.max_tokens,
            c.max_seqs,
            history.stride(0),
            c.vocab_size,
            256,
        )
        for i, binding in enumerate(self.bindings):
            native.run(binding, token_count=ids.numel())
            out[:, i].copy_(binding.hash_ids[: ids.numel()])

    def forward(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask=None,
        slot_mapping=None,
        block_table=None,
    ):
        out = torch.empty(
            (input_ids.numel(), len(self.layout.hash_plans), 24),
            dtype=torch.int64,
            device=input_ids.device,
        )
        self.run_native(input_ids, ~dead_mask, query_start_loc, lookback_token_ids, out)
        return out


def _read_table_rows(destination, source, start):
    """Read only this shard, straight into its final mapped CPU allocation."""
    if destination.device.type != "cpu" or not destination.is_contiguous():
        raise ValueError("File-backed Engram loading requires contiguous CPU storage")
    row_bytes = destination.shape[1] * destination.element_size()
    data = memoryview(destination.view(torch.uint8).numpy()).cast("B")
    # Unbuffered readinto has no payload-sized Python/NumPy allocation, and a
    # bounded syscall size also handles tables larger than Linux's read limit.
    with open(source.path, "rb", buffering=0) as checkpoint:
        checkpoint.seek(source.offset + start * row_bytes)
        offset = 0
        while offset < len(data):
            count = checkpoint.readinto(data[offset : offset + (8 << 20)])
            if not count:
                raise OSError(
                    f"Short read loading Engram table from {source.path}: "
                    f"{offset} of {len(data)} local bytes"
                )
            offset += count


def _load_table(param, loaded_weight):
    source = get_file_tensor_source(loaded_weight)
    shape = source.shape if source is not None else loaded_weight.shape
    dtype = source.dtype if source is not None else loaded_weight.dtype
    if shape != (param.global_rows, *param.shape[1:]):
        raise ValueError("Engram source must be the unpadded global checkpoint table")
    dtypes = (
        (torch.uint8, torch.float8_e8m0fnu)
        if param.dtype == torch.uint8
        else (torch.float8_e4m3fn,)
    )
    if dtype not in dtypes:
        raise TypeError(f"Invalid Engram table dtype: {dtype}")
    destination = getattr(param, "load_view", param.data)
    start = param.shard_start
    count = max(0, min(param.shape[0], param.global_rows - start))
    if count < param.shape[0]:
        destination[count:].zero_()
    if count:
        if source is not None:
            _read_table_rows(destination[:count], source, start)
        else:
            copy_weight(
                destination[:count].view(torch.uint8),
                loaded_weight[start : start + count].view(torch.uint8),
            )


class ParallelEngramEmbedding(nn.Module):
    def __init__(
        self, plan, caps, geometry, table_memory="device", *, resident_scales=False
    ):
        super().__init__()
        self.plan = plan
        self.caps = caps
        self.table_rows = geometry.num_embeddings[
            geometry.layer_ids.index(caps.layer_id)
        ]
        self.shard_rows = (self.table_rows + caps.tp_size - 1) // caps.tp_size
        self.shard_start = caps.tp_rank * self.shard_rows
        self.shard_end = (caps.tp_rank + 1) * self.shard_rows
        self.weight_shape = (self.shard_rows, 256)
        self.scale_shape = (self.shard_rows, 8)
        self.tp_size = caps.tp_size
        if table_memory not in ("device", "ram", "disk"):
            raise ValueError("Engram table_memory must be device, ram or disk")
        self.table_memory = table_memory
        self.disk_table = None
        self._disk_sources = []
        self.resident_scales = resident_scales
        self._disk_binding = None
        self.mapped_host_nbytes = 0
        if table_memory == "ram":
            nbytes = (
                self.weight_shape[0] * self.weight_shape[1]
                + self.scale_shape[0] * self.scale_shape[1]
            )
            logger.info(
                "Engram layer %d TP rank %d: allocating %.2f GiB mapped-host "
                "RAM (packed E4M3 weights and E8M0 scales)",
                caps.layer_id,
                caps.tp_rank,
                nbytes / (1 << 30),
            )
            self._weight_allocation = None
            try:
                self._weight_allocation = MappedHostAllocation(
                    self.weight_shape, torch.float8_e4m3fn, caps.device
                )
                self._scale_allocation = MappedHostAllocation(
                    self.scale_shape, torch.uint8, caps.device
                )
            except Exception as exc:
                if self._weight_allocation is not None:
                    self._weight_allocation.close()
                raise RuntimeError(
                    f"Engram mapped-host RAM allocation failed for "
                    f"{nbytes / (1 << 30):.2f} GiB on {caps.device}; "
                    "no disk or device fallback is permitted"
                ) from exc
            self.mapped_host_nbytes = nbytes
            self.weight_load_view = self._weight_allocation.host_view
            self.weight_scale_load_view = self._scale_allocation.host_view
            self.weight = nn.Parameter(
                self._weight_allocation.device_view, requires_grad=False
            )
            self.weight_scale_inv = nn.Parameter(
                self._scale_allocation.device_view, requires_grad=False
            )
            set_weight_attrs(self.weight, {"load_view": self.weight_load_view})
            set_weight_attrs(
                self.weight_scale_inv, {"load_view": self.weight_scale_load_view}
            )
        elif table_memory != "disk":
            self.weight = nn.Parameter(
                torch.empty(
                    self.weight_shape,
                    dtype=torch.float8_e4m3fn,
                    device=caps.device,
                ),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(self.scale_shape, dtype=torch.uint8, device=caps.device),
                requires_grad=False,
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("weight_scale_inv", None)
        if table_memory != "disk":
            for param in (self.weight, self.weight_scale_inv):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": _load_table,
                        "global_rows": self.table_rows,
                        "shard_start": self.shard_start,
                    },
                )
        self.register_buffer(
            "hashes",
            torch.empty((caps.max_tokens, 24), dtype=torch.int64, device=caps.device),
            persistent=False,
        )
        self.register_buffer(
            "num_tokens",
            torch.empty((1,), dtype=torch.int32, device=caps.device),
            persistent=False,
        )

    def close(self):
        """Release file-backed staging only after graph consumers are gone."""
        if self.disk_table is not None:
            self.disk_table.close()
            self.disk_table = None
            self._disk_binding = None
        for allocation in (
            getattr(self, "_scale_allocation", None),
            getattr(self, "_weight_allocation", None),
        ):
            if allocation is not None:
                allocation.close()

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            if name not in ("weight", "weight_scale_inv"):
                raise ValueError(f"Unknown Engram table weight: {name}")
            if self.table_memory != "disk":
                _load_table(getattr(self, name), value)
            else:
                source = get_file_tensor_source(value)
                if source is None:
                    raise ValueError(
                        "Disk Engram requires a file-backed checkpoint table"
                    )
                scale = name == "weight_scale_inv"
                width = self.scale_shape[1] if scale else self.weight_shape[1]
                if source.shape != (self.table_rows, width):
                    raise ValueError("Engram source must be the unpadded global table")
                dtypes = (
                    (torch.uint8, torch.float8_e8m0fnu)
                    if scale
                    else (torch.float8_e4m3fn,)
                )
                if source.dtype not in dtypes:
                    raise TypeError(f"Invalid Engram {name} dtype: {source.dtype}")
                self._disk_sources.append((source.path, source.offset, scale))
            loaded.add(name)
        return loaded

    def _ensure_disk_table(self) -> None:
        """Build the disk-backed lookup table from the prepared plan, once."""
        if self.table_memory != "disk" or self.disk_table is not None:
            return
        state = require_prepared(self.plan, "sequence.engram")
        table = native.DiskTable(state, resident_scales=self.resident_scales)
        for path, offset, scale in self._disk_sources:
            table.add_shard(0, path, offset, scale=scale)
        table.require_complete()
        self.disk_table = table

    def prepare_disk(self, indices, out, num_tokens):
        if self.table_memory != "disk":
            raise RuntimeError("Engram table is not disk-backed")
        self._ensure_disk_table()
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.copy_(num_tokens)
        if self._disk_binding is None or self._disk_binding.out is not out:
            self._disk_binding = native.bind_lookup(
                self.plan,
                weight=None,
                scales=None,
                hash_ids=self.hashes,
                num_tokens=self.num_tokens,
                out=out,
                disk_table=self.disk_table,
            )
            out.zero_()
            self._disk_prepared_rows = 0
        if indices.shape[0] < self._disk_prepared_rows:
            out[indices.shape[0] : self._disk_prepared_rows].zero_()
        native.run_lookup(
            self._disk_binding, token_count=indices.shape[0], clear_tail=False
        )
        self._disk_prepared_rows = indices.shape[0]

    def stage_disk(self, indices, out, num_tokens, flag, side_stream):
        """Main thread: stage the IDs and queue the gated decode; no disk I/O."""
        from b12x.sequence.engram._kernels import lookup_op

        if self.table_memory != "disk":
            raise RuntimeError("Engram table is not disk-backed")
        self._ensure_disk_table()
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        cache = self.disk_table._cache
        if cache._gds is not None:
            raise RuntimeError("asynchronous Engram rows need the io_uring backend")
        if getattr(self, "_read_gate", None) is None:
            self._read_gate = HostGate()
        stream = torch.cuda.current_stream(out.device)
        flag.write(stream, 0)
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.copy_(num_tokens)
        if self._disk_binding is None or self._disk_binding.out is not out:
            self._disk_binding = native.bind_lookup(
                self.plan,
                weight=None,
                scales=None,
                hash_ids=self.hashes,
                num_tokens=self.num_tokens,
                out=out,
                disk_table=self.disk_table,
            )
            out.zero_()
            self._disk_prepared_rows = 0
        if indices.shape[0] < self._disk_prepared_rows:
            out[indices.shape[0] : self._disk_prepared_rows].zero_()
        # The previous decode must be done with the mapped rows before the
        # reader overwrites them; the ID copy below is ordered after it.
        if cache._cache_used:
            stream.wait_event(cache._cache_done)
        count = indices.shape[0] * 24
        cache.ids_host[:count].copy_(self.hashes.view(-1)[:count], non_blocking=True)
        cache._ids_ready.record(stream)
        # The decode waits on the side stream for the host read, then marks
        # the rows ready for the forward's Engram layer.
        sequence = self._read_gate.next()
        side_stream.wait_event(cache._ids_ready)
        self._read_gate.wait(side_stream, sequence)
        binding = self._disk_binding
        with torch.cuda.stream(side_stream):
            lookup_op(
                binding.plan.handle,
                binding.weight,
                binding.scale_bytes,
                binding.hash_ids,
                binding.num_tokens,
                binding.out,
                indices.shape[0],
                False,
            )
            cache._cache_done.record(side_stream)
            cache._cache_used = True
            flag.write(side_stream, 1)
        self._disk_prepared_rows = indices.shape[0]
        if ENGRAM_ASYNC_TRACE:
            logger.info(
                "engram-async stage table=%x tokens=%d lookups=%d sequence=%d",
                id(self), indices.shape[0], count, sequence,
            )
        return indices.shape[0], count, sequence

    def finish_disk(self, job):
        """Reader thread, host work only: wait for the IDs, read the rows."""
        tokens, count, sequence = job
        cache = self.disk_table._cache
        try:
            wait_for_event(cache._ids_ready, f"row IDs for table {id(self):x} ({tokens} tokens)")
            if ENGRAM_ASYNC_DELAY:
                time.sleep(ENGRAM_ASYNC_DELAY)
            cache._native.ple_reader_run(
                cache._reader,
                cache._ids_buffer,
                cache._weight_buffer,
                cache._scale_buffer,
                count,
            )
        finally:
            # Release even on failure so no stream waits forever; the caller
            # re-raises the failure before the next step.
            self._read_gate.release(sequence)
        if ENGRAM_ASYNC_TRACE:
            logger.info("engram-async rows read table=%x sequence=%d", id(self), sequence)

    def lookup_native(self, indices, out):
        if self.plan.prepared is None:
            raise PreparationResourceUnavailableError(
                "Engram lookup plan must be prepared before forwarding"
            )
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.fill_(indices.shape[0])
        binding = native.bind_lookup(
            self.plan,
            weight=self.weight,
            scales=self.weight_scale_inv,
            hash_ids=self.hashes,
            num_tokens=self.num_tokens,
            out=out,
        )
        retain_cuda_graph_capture_resource(binding)
        if self.mapped_host_nbytes:
            retain_cuda_graph_capture_resource(self)
        native.run_lookup(binding)

    def lookup(self, indices, out):
        self.lookup_native(indices, out)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        layout,
        layer_hash_index,
        use_sequence_parallel,
        prefix,
    ):
        super().__init__()
        if use_sequence_parallel:
            raise ValueError("V4.1 Engram consumes replicated token rows")
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        plan = layout.lookup_plans[layer_hash_index]
        caps = layout.caps[layer_hash_index]
        self.embed_tokens = ParallelEngramEmbedding(
            plan,
            caps,
            layout.geometry,
            layout.table_memory,
            resident_scales=getattr(layout, "disk_resident_scales", False),
        )
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        projection_tp = getattr(layout, "projection_tp", False)
        projection_cls = (
            _PaddedColumnParallelLinear if projection_tp else ReplicatedLinear
        )
        self.wkv = projection_cls(
            6144,
            self.dim * (self.hc_mult + 1),
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            **({"gather_output": True} if projection_tp else {}),
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty_like(self.q_weight), requires_grad=False
        )
        self.register_buffer(
            "norm_weights",
            torch.empty(self.hc_mult * self.dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "staged_rows",
            torch.empty(
                caps.max_tokens,
                6144,
                dtype=torch.bfloat16,
                device=caps.device,
            ),
            persistent=False,
        )
        mix_caps = hyperconnection.Caps(
            device=caps.device,
            max_tokens=caps.max_tokens,
            hidden_size=self.dim,
            streams=self.hc_mult,
        )
        self.mix_plans = {
            masked: hyperconnection.plan(
                mix_caps,
                invocation={
                    "operation": "engram_mix",
                    "eps": self.eps,
                    "token_mask": masked,
                },
            )
            for masked in (False, True)
        }
        set_b12x_preparation_provider(self, self)

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        caps = self.embed_tokens.caps
        if workload.max_tokens > caps.max_tokens:
            return ()

        def make_call(state):
            from b12x.sequence.engram._impl import _bind_lookup_state

            embed = self.embed_tokens
            hash_ids = torch.full(
                (embed.caps.max_tokens, 24),
                -1,
                dtype=torch.int64,
                device=embed.caps.device,
            )
            prime_row = min(embed.shard_start + 1, embed.shard_end - 1)
            if prime_row <= 0:
                raise PreparationResourceUnavailableError(
                    "Engram shard has no nonzero local lookup row"
                )
            hash_ids[0, 0] = prime_row
            num_tokens = torch.ones((1,), dtype=torch.int32, device=embed.caps.device)
            out = torch.empty(
                (embed.caps.max_tokens, 6144),
                dtype=torch.bfloat16,
                device=embed.caps.device,
            )
            table = None
            if embed.table_memory == "disk":
                table = native.DiskTable(state, resident_scales=embed.resident_scales)
                for path, offset, scale in embed._disk_sources:
                    table.add_shard(0, path, offset, scale=scale)
                table.require_complete()
            binding = _bind_lookup_state(
                state,
                weight=embed.weight if table is None else None,
                scales=embed.weight_scale_inv if table is None else None,
                hash_ids=hash_ids,
                num_tokens=num_tokens,
                out=out,
                disk_table=table,
            )

            def run():
                if table is None:
                    return state.run_lookup(binding, 1, clear_tail=True)
                with table._cache.transaction():
                    table._cache.read_rows(hash_ids, 24)
                    return state.run_lookup(binding, 1, clear_tail=True)

            return PreparedCall(
                run=run,
                capture_safe=table is None,
                owners=(embed.weight, embed.weight_scale_inv),
                restore=table.close if table is not None else None,
            )

        request = self.embed_tokens.plan.request(
            name=f"{id(self)}/engram-lookup",
            prepare_call=make_call,
            benchmark_call=make_call,
        )

        def make_mix_call(state):
            from b12x.norm.hyperconnection._impl import run_engram_mix_impl

            rows = state.query.max_tokens
            width = self.hc_mult * self.dim
            residual = torch.empty(
                (rows, width),
                dtype=torch.bfloat16,
                device=caps.device,
            )
            projected = torch.empty(
                (rows, width + self.dim),
                dtype=torch.bfloat16,
                device=caps.device,
            )
            out = torch.empty_like(residual)
            mask = (
                torch.ones(rows, dtype=torch.bool, device=caps.device)
                if state.query.token_mask
                else None
            )
            if mask is not None:
                mask[::2] = False

            def produce():
                residual.fill_(0.125)
                projected.fill_(0.25)

            return PreparedCall(
                run=lambda: run_engram_mix_impl(
                    residual,
                    projected,
                    self.norm_weights,
                    eps=self.eps,
                    plan=state,
                    out=out,
                    token_mask=mask,
                ),
                produce=produce,
                output=out,
                owners=(self.norm_weights,),
            )

        mix_requests = tuple(
            plan.request(
                name=f"{id(self)}/engram-mix/{masked}",
                prepare_call=make_mix_call,
                benchmark_call=make_mix_call,
            )
            for masked, plan in self.mix_plans.items()
        )
        return (
            B12xPreparationUnit(
                name="EngramLookup",
                key=(id(self), workload.max_tokens),
                requests=(request,),
                stage="weights",
            ),
            B12xPreparationUnit(
                name="EngramMix",
                key=(id(self), workload.max_tokens),
                requests=mix_requests,
                stage="weights",
            ),
        )

    def process_weights_after_loading(self):
        self.norm_weights.copy_(
            (self.q_weight.float() * self.k_weight.float()).flatten()
        )
        self._rows_flag = None
        if ENGRAM_ASYNC and self.embed_tokens.table_memory == "disk":
            self._rows_flag = RowsReadyFlag()
            self._rows_flag_key = id(self)
            _ROWS_READY_FLAGS[self._rows_flag_key] = self._rows_flag
            logger.info("asynchronous disk Engram rows enabled (flag key %x)", self._rows_flag_key)

    @property
    def async_rows(self) -> bool:
        return getattr(self, "_rows_flag", None) is not None

    def stage_disk(self, hash_ids, num_tokens, side_stream):
        self.invalidate_disk_output()
        job = self.embed_tokens.stage_disk(
            hash_ids, self.staged_rows, num_tokens, self._rows_flag, side_stream
        )
        self._disk_prepared_tokens = hash_ids.shape[0]
        self._disk_prepared = True
        return job

    def finish_disk(self, job):
        self.embed_tokens.finish_disk(job)

    def prepare_embeddings(self, hash_ids):
        self.embed_tokens.lookup(hash_ids, self.staged_rows)

    def invalidate_disk_output(self, *, clear=False):
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        if clear:
            self.staged_rows.zero_()

    def prepare_disk(self, hash_ids, num_tokens):
        self.invalidate_disk_output()
        try:
            self.embed_tokens.prepare_disk(hash_ids, self.staged_rows, num_tokens)
        except BaseException:
            self.invalidate_disk_output(clear=True)
            raise
        self._disk_prepared_tokens = hash_ids.shape[0]
        self._disk_prepared = True

    def prepare_dummy_output(self, num_tokens):
        self.invalidate_disk_output(clear=True)
        self._disk_prepared_tokens = num_tokens
        self._disk_prepared = True
        if self.async_rows:
            self._rows_flag.write(torch.cuda.current_stream(self.staged_rows.device), 1)
            if ENGRAM_ASYNC_TRACE:
                logger.info("engram-async dummy tokens=%d", num_tokens)

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if (
            self.embed_tokens.table_memory == "disk"
            and not torch.compiler.is_compiling()
            and (
                not self._disk_prepared
                or self._disk_prepared_tokens < hash_ids.shape[0]
            )
        ):
            raise RuntimeError("Disk Engram output is not prepared")
        if getattr(self, "_rows_flag", None) is not None:
            torch.ops.vllm.spark3_engram_wait_rows(self.staged_rows, self._rows_flag_key)
        rows = tensor_model_parallel_all_reduce(self.staged_rows[: hash_ids.shape[0]])
        kv = self.wkv(rows)
        state = hidden_states.flatten(1)
        out = torch.empty_like(state)
        hyperconnection.run_engram_mix(
            state,
            kv,
            self.norm_weights,
            eps=self.eps,
            plan=self.mix_plans[token_mask is not None],
            out=out,
            token_mask=token_mask,
        )
        return out.view_as(hidden_states)
