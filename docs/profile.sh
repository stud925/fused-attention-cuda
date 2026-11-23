#!/usr/bin/env bash
# Collects the Nsight data used in docs/profiling.md. Run from the repo root.
# Requires ncu / nsys on PATH (CUDA toolkit) and permission to read GPU counters
# (on a DGX Spark running as a normal user this normally works; otherwise use sudo).
set -euo pipefail
mkdir -p docs/prof
SEQ=${SEQ:-2048}

# Metrics that map to the numbers asked for in the spec:
#  - memory throughput:  dram__throughput, l1tex__throughput, lts__throughput
#  - occupancy:          sm__warps_active (achieved), launch__occupancy_limit_*, launch__registers_per_thread
#  - achieved FLOPs:     sm__sass_thread_inst_executed_op_{ffma,fadd,fmul}_pred_on, hmul/hfma
#  - bank conflicts:     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}
METRICS=gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
launch__registers_per_thread,launch__shared_mem_per_block_static,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__inst_executed.sum

ncu_run () {  # name, kernel-regex, args...
  local name=$1 kern=$2; shift 2
  ncu --kernel-name "regex:$kern" --launch-skip 0 --launch-count 1 \
      --metrics "$METRICS" --csv --page raw \
      python benchmarks/profile_target.py "$@" > "docs/prof/${name}.csv"
  # Full report (open in ncu-ui) for source-level / roofline views:
  ncu --kernel-name "regex:$kern" --launch-count 1 --set full -f -o "docs/prof/${name}" \
      python benchmarks/profile_target.py "$@" > /dev/null
}

# --- naive: all three kernels, shows the N*N traffic ---------------------------
ncu_run naive 'qk_naive|softmax_rows|pv_naive' --impl naive --dtype fp32 --seq "$SEQ"
# (use --launch-count 3 manually if you want all three kernels in one CSV)

# --- fused: each optimisation step in isolation (cumulative) --------------------
F="--impl fused --seq $SEQ"
ncu_run fused_0_baseline  fused_fwd $F --no-pad --no-vec --no-fast-exp
ncu_run fused_1_pad       fused_fwd $F           --no-vec --no-fast-exp
ncu_run fused_2_vec       fused_fwd $F                    --no-fast-exp
ncu_run fused_3_exp2      fused_fwd $F
ncu_run fused_4_warps4    fused_fwd $F --warps 4

# --- Nsight Systems timeline (launch gaps, memcpys, naive vs fused) -------------
nsys profile -o docs/prof/naive_timeline -f true python benchmarks/profile_target.py --impl naive --dtype fp32 --seq "$SEQ"
nsys profile -o docs/prof/fused_timeline -f true python benchmarks/profile_target.py --impl fused --seq "$SEQ"
nsys stats --report cuda_gpu_kern_sum docs/prof/naive_timeline.nsys-rep > docs/prof/naive_kernels.txt
nsys stats --report cuda_gpu_kern_sum docs/prof/fused_timeline.nsys-rep > docs/prof/fused_kernels.txt
echo "done; fill in docs/profiling.md from docs/prof/*"
