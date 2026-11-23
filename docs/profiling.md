# Profiling notes

> **Status: template.** No profiler data has been collected yet. Every `—` below
> must be filled from a real run of `docs/profile.sh`; the "Hypothesis" column is
> what the design predicts, *not* a result. If a measurement contradicts the
> hypothesis, say so here — that is the interesting part.

Setup (fill in): GPU `—`, driver `—`, CUDA `—`, Nsight Compute `—`, PyTorch `—`.
Workload: B=4, H=8, N=2048, D=64, fp16 (naive/tiled: fp32 internally).

Reproduce: `bash docs/profile.sh` (raw CSVs + `.ncu-rep` in `docs/prof/`),
and `python benchmarks/ablate.py` for the latency column.

## 1. Naive vs fused (the N×N traffic)

| Metric (ncu) | naive qk | naive softmax | naive pv | fused |
|---|---|---|---|---|
| Duration (µs) | — | — | — | — |
| DRAM bytes read | — | — | — | — |
| DRAM bytes written | — | — | — | — |
| DRAM throughput (% peak) | — | — | — | — |
| Achieved occupancy (%) | — | — | — | — |
| Registers / thread | — | — | — | — |

Hypothesis: naive writes S once (qk), reads+writes it (softmax), reads it again
(pv): ≈ 4·B·H·N²·4 bytes of DRAM traffic, dominated by S. Fused should have DRAM
traffic ≈ reads of Q,K,V + write of O only.

## 2. Optimisation log (fused kernel, cumulative)

Each row toggles one knob (see `csrc/fused.cu`, `benchmarks/ablate.py`).
Correctness (`pytest tests`) must pass for every row.

| # | Change | Latency (ms) | Shared ld bank conflicts | Shared ld wavefronts | Occupancy (%) | Regs/thr | Other |
|---|---|---|---|---|---|---|---|
| 0 | Stage-3 baseline (`pad=0 vec=0 fast_exp=0 warps=8`) | — | — | — | — | — | — |
| 1 | Pad K tile row by 1 float (`pad=1`) | — | — | — | — | — | — |
| 2 | 16-byte vectorised K/V loads (`vec=1`) | — | — | — | — | — | global ld instr: — |
| 3 | `exp2f` with log2e folded into Q (`fast_exp=1`) | — | — | — | — | — | inst executed: — |
| 4 | 4 warps / block instead of 8 (`warps=4`) | — | — | — | — | — | — |

What each change should do, and what to check in the report:

1. **Padding.** In the score phase lane `l` reads `Ks[l*D + d]`; with `D` a
   multiple of 32 all lanes map to one bank → 32-way conflict on every K read.
   With stride `D+1` the lanes hit 32 different banks. Check
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` drops by ~an order of
   magnitude. (Note: the Q reads are broadcasts and the V reads are stride-1, so
   they are already conflict-free.)
2. **Vector loads.** Tile loads go from 1 element to 16 bytes per instruction
   (4× fewer for fp32, 8× for fp16). Expect fewer LSU instructions; the gain in
   wall time depends on whether the kernel is load-issue bound at all (it may not
   be — the score loop is FMA/shared-memory bound).
3. **exp2.** Removes the range-reduction multiply in `expf`. Expect a small
   instruction-count decrease; may be within noise.
4. **Warps per block.** Trades K/V-tile reuse (more rows per tile load) against
   occupancy / shared memory per block. Read `launch__occupancy_limit_*` to see
   which resource binds.

## 3. Why SDPA is (probably) faster — to be confirmed with profiler data

`F.scaled_dot_product_attention` in fp16 dispatches to FlashAttention / a
cuDNN or mem-efficient kernel that uses Tensor Core MMA for both QKᵀ and PV with
double-buffered async copies. This kernel does all math on the FP32 CUDA cores
(FMA). Compare `sm__inst_executed_pipe_tensor*` (SDPA) vs
`smsp__sass_thread_inst_executed_op_ffma_pred_on` (ours) and the achieved
FLOP/s of each in the SDPA vs fused ncu reports. Record the SDPA kernel name
from `nsys stats` here: `—`.

## 4. Nsight Systems (timeline)

Fill in from `docs/prof/{naive,fused}_kernels.txt`: number of kernel launches
per call (naive: 3, fused: 1), total GPU time, any gaps. `—`
