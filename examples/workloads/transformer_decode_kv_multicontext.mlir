// Single-request Transformer decode multicontext workload (GQA 16:4, BF16 history).
//
// Shapes: history=2048, KV_BLOCK=256 -> 8 blocks; hidden=1024; head_dim=64.
// K/V packets flatten [kv_head,token,dim] into a contiguous BF16 payload.
// Each packet has 64 untransferred padding bytes after its 128 KiB payload.
// Default globals [8,65568]xbf16 occupy 1,049,088 B each but transfer exactly
// 1 MiB each. With a 64-byte gap, first-block K channels are 0,2,4,6;
// binding V_CACHE_mc at base+64 puts its first blocks on channels 1,3,5,7.
// This channel-placement optimization is distinct from UCE multicontext:
// compare R1/R2/R4 with the same padding/bindings to isolate UCE gains.
// Q_IN_mc [4,4,64] is bf16, and OUT_mc [4,4,64] is f32.
// S_INIT_STATE_mc [4,4,4,2] and S_INIT_OUT_mc [4,4,4,64] are f32.
// Append inputs H_T_mc [1024], WK_A_mc/WV_A_mc [4,1024,64] are bf16;
// K_APPEND_mc/V_APPEND_mc [4,1,64] are written at static position 2048.
// Each of the 8 attention-block dispatches fans out to all four tiles exactly once.
// Tile t owns KV head t and query heads 4t..4t+3. Partition p owns blocks
// 2p and 2p+1; its first block pins context p%R and its continuation uses
// context (p+1)%R. output_ready orders each recurrence across that handoff.
// K/V append are two four-task dispatches; K pins context 0 and V pins 1%R.
// Partition state and partial-output L2 allocations are distinct.
// Context resources count 44 logical tasks: 32 attention, 8 append-phase, and 4 merge tasks.
//
// R=4 requested Tile Tasks per tile; continuations rotate modulo R.
// Default R=4 explores four UCE contexts; R=1/2 remain available as ablations.
// This source needs sim context_count >= 4; one Device root is submitted.
// The compiler-derived L1 contracts use conservative_arena_bytes and admit
// only profiles where R times every per-program per-bank envelope fits.
// All task contexts share one context-local L2 Arena; R does not multiply L2.
// L2 mode 0 is the baseline; allowed L2 profiles are derived from the full
// conservative allocation list and must cover the one shared root Arena.
//
// Each partition has one block-sized K/V staging slot, so all four first-block
// prefetch/dispatch pairs are independent. Its second-block prefetch waits
// for that partition's first input_released before overwriting its slot; the
// second dispatch waits for its own prior output_ready. State/output buffers
// are distinct across partitions; there is no whole-KV prefetch barrier.
// K append depends only on H_T/WK and may run while WV is still transferring;
// V append depends on H_T/WV. R>=2 pins these projection grids to contexts 0/1.
// The merge dispatch depends on all four final partition output_ready events.
// The 8 attention, 2 append, and 1 merge dispatches need at most 11 live
// Grid routes, below the default group.dispatch_capacity=16.
//
// Each partition has one S_INIT_STATE row (m=-inf, l=0) and one S_INIT_OUT
// row (unnormalized o=0). They are split into contiguous global arrays to
// preserve legal row-major views. Stable merge uses m=max(m_i),
// alpha_i=exp(m_i-m), l=sum(alpha_i*l_i), o=sum(alpha_i*o_i), then y=o/l.
// The merge EVU count is 8,480 ops (16 query heads x [18+8*64]); local
// attention EVU work is 49,408 ops across all blocks/tiles, including the
// per-block output recurrence. Attention BOA remains 8,388,608 FLOPs, the
// same useful QK+PV work as the one-request software-pipeline reference.
// The fixed-position-2048 H_T[1024] -> K/V append remains 1,048,576 BOA FLOPs.
// Splitting projections rereads H_T once per phase, adding 8,192 L2/local-DMA bytes.
// HBM/global payload bytes, output bytes, and all BOA/EVU work remain unchanged.
// S_INIT adds 16,896 input bytes; OUT plus K_APPEND/V_APPEND write 5,120 bytes.
//
// TIMING/RESOURCE MODEL ONLY: BOA and EVU ops carry no tensor values and do
// not establish numerical correctness. S_INIT bindings must contain the
// documented identity states; runtime data values are not executed here.
// No token-generation loop or dynamic append address is modeled.
//
// Regenerate from the repository root:
//   PYTHONPATH=. conda run -n elenor-validator python \
//     examples/generators/generate_transformer_decode_multicontext.py \
//     --context-mode 4
// Run after run.sh integration: bash examples/run.sh transformer-decode-kv-multicontext

builtin.module {
  tile.program @decode_partition_block(
    %task: !nest.task, %q_l2: !nest.l2_buffer<4x4x64xbf16>, %k_l2: !nest.l2_buffer<4x256x64xbf16>,
    %v_l2: !nest.l2_buffer<4x256x64xbf16>, %state_l2: !nest.l2_buffer<4x4x2xf32>,
    %partial_out_l2: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 86016> {
    %0 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %1 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x256x64xbf16>
    %2 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x256x64xbf16>
    %3 = tile.subview %state_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %4 = tile.subview %partial_out_l2 task = %task task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %5 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %6 = tile.alloc shape = [256, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<256x64xbf16>
    %7 = tile.alloc shape = [256, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<256x64xbf16>
    %8 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %9 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %10 = tile.alloc shape = [4, 256] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x256xf32>
    %11 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %q_loaded = tile.load.async %0 into %5 : !tile.event<"q_loaded">
    %k_loaded = tile.load.async %1 into %6 : !tile.event<"k_loaded">
    %v_loaded = tile.load.async %2 into %7 : !tile.event<"v_loaded">
    %state_loaded = tile.load.async %3 into %8 : !tile.event<"state_loaded">
    %partial_out_loaded = tile.load.async %4 into %9 : !tile.event<"partial_out_loaded">
    tile.await %q_loaded, %k_loaded, %v_loaded, %state_loaded, %partial_out_loaded
    tile.signal input_released(%task)
    %qk_boa = tile.boa.async "matmul" m = 4 n = 256 k = 64 ops = 131072 : !tile.event<"qk_boa">
    tile.await %qk_boa
    %score_update = tile.evu.async "online_softmax_score_update" ops = 1032
      : !tile.event<"score_update">
    tile.await %score_update
    %pv_boa = tile.boa.async "matmul" m = 4 n = 64 k = 256 ops = 131072 : !tile.event<"pv_boa">
    tile.await %pv_boa
    %output_accumulate = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"output_accumulate">
    tile.await %output_accumulate
    %state_stored = tile.store.async %8 into %3 : !tile.event<"state_stored">
    %partial_out_stored = tile.store.async %9 into %4 : !tile.event<"partial_out_stored">
    tile.await %state_stored, %partial_out_stored
    tile.signal output_ready(%task)
    tile.free %5
    tile.free %6
    tile.free %7
    tile.free %8
    tile.free %9
    tile.free %10
    tile.free %11
    tile.return
  }
  tile.program @decode_stable_partition_merge(
    %task_1: !nest.task, %state_pa: !nest.l2_buffer<4x4x2xf32>,
    %partial_out_pa: !nest.l2_buffer<4x4x64xf32>, %state_pb: !nest.l2_buffer<4x4x2xf32>,
    %partial_out_pb: !nest.l2_buffer<4x4x64xf32>, %state_pc: !nest.l2_buffer<4x4x2xf32>,
    %partial_out_pc: !nest.l2_buffer<4x4x64xf32>, %state_pd: !nest.l2_buffer<4x4x2xf32>,
    %partial_out_pd: !nest.l2_buffer<4x4x64xf32>, %merged_out_l2: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 11264> {
    %12 = tile.subview %state_pa task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %13 = tile.subview %state_pb task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %14 = tile.subview %state_pc task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %15 = tile.subview %state_pd task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %16 = tile.subview %partial_out_pa task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %17 = tile.subview %partial_out_pb task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %18 = tile.subview %partial_out_pc task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %19 = tile.subview %partial_out_pd task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %20 = tile.subview %merged_out_l2 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %21 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %22 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %23 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %24 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %25 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %26 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %27 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %28 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %29 = tile.alloc shape = [4, 4] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x4xf32>
    %30 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %31 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %state_pa_loaded = tile.load.async %12 into %21 : !tile.event<"state_pa_loaded">
    %out_pa_loaded = tile.load.async %16 into %25 : !tile.event<"out_pa_loaded">
    %state_pb_loaded = tile.load.async %13 into %22 : !tile.event<"state_pb_loaded">
    %out_pb_loaded = tile.load.async %17 into %26 : !tile.event<"out_pb_loaded">
    %state_pc_loaded = tile.load.async %14 into %23 : !tile.event<"state_pc_loaded">
    %out_pc_loaded = tile.load.async %18 into %27 : !tile.event<"out_pc_loaded">
    %state_pd_loaded = tile.load.async %15 into %24 : !tile.event<"state_pd_loaded">
    %out_pd_loaded = tile.load.async %19 into %28 : !tile.event<"out_pd_loaded">
    tile.await %state_pa_loaded, %out_pa_loaded, %state_pb_loaded, %out_pb_loaded, %state_pc_loaded,
      %out_pc_loaded, %state_pd_loaded, %out_pd_loaded
    tile.signal input_released(%task_1)
    %stable_merge = tile.evu.async "stable_online_softmax_merge" ops = 2120
      : !tile.event<"stable_merge">
    tile.await %stable_merge
    %merged_out_stored = tile.store.async %31 into %20 : !tile.event<"merged_out_stored">
    tile.await %merged_out_stored
    tile.signal output_ready(%task_1)
    tile.free %21
    tile.free %22
    tile.free %23
    tile.free %24
    tile.free %25
    tile.free %26
    tile.free %27
    tile.free %28
    tile.free %29
    tile.free %30
    tile.free %31
    tile.return
  }
  tile.program @decode_kv_append_k(
    %task_2: !nest.task, %h_l2: !nest.l2_buffer<1024xbf16>,
    %weight_l2: !nest.l2_buffer<4x1024x64xbf16>, %result_l2: !nest.l2_buffer<4x1x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 139264> {
    %32 = tile.subview %h_l2 offsets = [0] sizes = [1024] strides = [1] : !nest.l2_view<1024xbf16>
    %33 = tile.subview %weight_l2 task = %task_2 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1024, 64] strides = [1, 1, 1] : !nest.l2_view<1x1024x64xbf16>
    %34 = tile.subview %result_l2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %35 = tile.alloc shape = [1024] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1024xbf16>
    %36 = tile.alloc shape = [1024, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<1024x64xbf16>
    %37 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %h_loaded = tile.load.async %32 into %35 : !tile.event<"h_loaded">
    %weight_loaded = tile.load.async %33 into %36 : !tile.event<"weight_loaded">
    tile.await %h_loaded, %weight_loaded
    %k_new_boa = tile.boa.async "matmul" m = 1 n = 64 k = 1024 ops = 131072
      : !tile.event<"k_new_boa">
    tile.await %k_new_boa
    %k_append_stored = tile.store.async %37 into %34 : !tile.event<"k_append_stored">
    tile.await %k_append_stored
    tile.signal input_released(%task_2)
    tile.signal output_ready(%task_2)
    tile.free %35
    tile.free %36
    tile.free %37
    tile.return
  }
  tile.program @decode_kv_append_v(
    %task_3: !nest.task, %h_l2_1: !nest.l2_buffer<1024xbf16>,
    %weight_l2_1: !nest.l2_buffer<4x1024x64xbf16>, %result_l2_1: !nest.l2_buffer<4x1x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 139264> {
    %38 = tile.subview %h_l2_1 offsets = [0] sizes = [1024] strides = [1] : !nest.l2_view<1024xbf16>
    %39 = tile.subview %weight_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1024, 64] strides = [1, 1, 1] : !nest.l2_view<1x1024x64xbf16>
    %40 = tile.subview %result_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %41 = tile.alloc shape = [1024] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1024xbf16>
    %42 = tile.alloc shape = [1024, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<1024x64xbf16>
    %43 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %h_loaded_1 = tile.load.async %38 into %41 : !tile.event<"h_loaded">
    %weight_loaded_1 = tile.load.async %39 into %42 : !tile.event<"weight_loaded">
    tile.await %h_loaded_1, %weight_loaded_1
    %v_new_boa = tile.boa.async "matmul" m = 1 n = 64 k = 1024 ops = 131072
      : !tile.event<"v_new_boa">
    tile.await %v_new_boa
    %v_append_stored = tile.store.async %43 into %40 : !tile.event<"v_append_stored">
    tile.await %v_append_stored
    tile.signal input_released(%task_3)
    tile.signal output_ready(%task_3)
    tile.free %41
    tile.free %42
    tile.free %43
    tile.return
  }
  nest.context @decode_multicontext(
    %K_CACHE_ctx: !nest.global_memref<8x65568xbf16>,
    %V_CACHE_ctx: !nest.global_memref<8x65568xbf16>, %Q_IN_ctx: !nest.global_memref<4x4x64xbf16>,
    %S_INIT_STATE_ctx: !nest.global_memref<4x4x4x2xf32>,
    %S_INIT_OUT_ctx: !nest.global_memref<4x4x4x64xf32>, %OUT_ctx: !nest.global_memref<4x4x64xf32>,
    %H_T_ctx: !nest.global_memref<1024xbf16>, %WK_A_ctx: !nest.global_memref<4x1024x64xbf16>,
    %WV_A_ctx: !nest.global_memref<4x1024x64xbf16>,
    %K_APPEND_ctx: !nest.global_memref<4x1x64xbf16>,
    %V_APPEND_ctx: !nest.global_memref<4x1x64xbf16>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
      logical_tasks = 44, l2_spm_bytes = 2150400, requested_contexts_per_tile = 4> {
    %k_pa = nest.alloc slot = "k_pa" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %k_pb = nest.alloc slot = "k_pb" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %k_pc = nest.alloc slot = "k_pc" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %k_pd = nest.alloc slot = "k_pd" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %v_pa = nest.alloc slot = "v_pa" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %v_pb = nest.alloc slot = "v_pb" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %v_pc = nest.alloc slot = "v_pc" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %v_pd = nest.alloc slot = "v_pd" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %q_l2_1 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x4x64xbf16>
    %state_pa_1 = nest.alloc slot = "state_pa" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_pb_1 = nest.alloc slot = "state_pb" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_pc_1 = nest.alloc slot = "state_pc" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_pd_1 = nest.alloc slot = "state_pd" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %partial_out_pa_1 = nest.alloc slot = "partial_out_pa" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %partial_out_pb_1 = nest.alloc slot = "partial_out_pb" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %partial_out_pc_1 = nest.alloc slot = "partial_out_pc" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %partial_out_pd_1 = nest.alloc slot = "partial_out_pd" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %merged_out_l2_1 = nest.alloc slot = "merged_out_l2" role = "out" shape = [4, 4, 64]
      dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %h_l2_2 = nest.alloc slot = "h_l2" role = "in" shape = [1024] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1024xbf16>
    %wk_l2 = nest.alloc slot = "wk_l2" role = "in" shape = [4, 1024, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x64xbf16>
    %wv_l2 = nest.alloc slot = "wv_l2" role = "in" shape = [4, 1024, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x64xbf16>
    %kn_l2 = nest.alloc slot = "kn_l2" role = "out" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %vn_l2 = nest.alloc slot = "vn_l2" role = "out" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %44 = nest.subview %Q_IN_ctx offsets = [0, 0, 0] sizes = [4, 4, 64] strides = [1, 1, 1]
      : !nest.global_view<4x4x64xbf16>
    %45 = nest.subview %S_INIT_STATE_ctx offsets = [0, 0, 0, 0] sizes = [1, 4, 4, 2]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x2xf32>
    %46 = nest.subview %S_INIT_STATE_ctx offsets = [1, 0, 0, 0] sizes = [1, 4, 4, 2]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x2xf32>
    %47 = nest.subview %S_INIT_STATE_ctx offsets = [2, 0, 0, 0] sizes = [1, 4, 4, 2]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x2xf32>
    %48 = nest.subview %S_INIT_STATE_ctx offsets = [3, 0, 0, 0] sizes = [1, 4, 4, 2]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x2xf32>
    %49 = nest.subview %S_INIT_OUT_ctx offsets = [0, 0, 0, 0] sizes = [1, 4, 4, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x64xf32>
    %50 = nest.subview %S_INIT_OUT_ctx offsets = [1, 0, 0, 0] sizes = [1, 4, 4, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x64xf32>
    %51 = nest.subview %S_INIT_OUT_ctx offsets = [2, 0, 0, 0] sizes = [1, 4, 4, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x64xf32>
    %52 = nest.subview %S_INIT_OUT_ctx offsets = [3, 0, 0, 0] sizes = [1, 4, 4, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x4x64xf32>
    %53 = nest.subview %OUT_ctx offsets = [0, 0, 0] sizes = [4, 4, 64] strides = [1, 1, 1]
      : !nest.global_view<4x4x64xf32>
    %54 = nest.subview %H_T_ctx offsets = [0] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %55 = nest.subview %WK_A_ctx offsets = [0, 0, 0] sizes = [4, 1024, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1024x64xbf16>
    %56 = nest.subview %WV_A_ctx offsets = [0, 0, 0] sizes = [4, 1024, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1024x64xbf16>
    %57 = nest.subview %K_APPEND_ctx offsets = [0, 0, 0] sizes = [4, 1, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1x64xbf16>
    %58 = nest.subview %V_APPEND_ctx offsets = [0, 0, 0] sizes = [4, 1, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1x64xbf16>
    %59 = nest.task.range from = 0 to = 4 : !nest.task_range
    %pre_q = nest.dma.prefetch.async %44 into %q_l2_1 : !nest.event<"pre_q">
    %pre_state_pa = nest.dma.prefetch.async %45 into %state_pa_1 : !nest.event<"pre_state_pa">
    %pre_out_pa = nest.dma.prefetch.async %49 into %partial_out_pa_1 : !nest.event<"pre_out_pa">
    %pre_state_pb = nest.dma.prefetch.async %46 into %state_pb_1 : !nest.event<"pre_state_pb">
    %pre_out_pb = nest.dma.prefetch.async %50 into %partial_out_pb_1 : !nest.event<"pre_out_pb">
    %pre_state_pc = nest.dma.prefetch.async %47 into %state_pc_1 : !nest.event<"pre_state_pc">
    %pre_out_pc = nest.dma.prefetch.async %51 into %partial_out_pc_1 : !nest.event<"pre_out_pc">
    %pre_state_pd = nest.dma.prefetch.async %48 into %state_pd_1 : !nest.event<"pre_state_pd">
    %pre_out_pd = nest.dma.prefetch.async %52 into %partial_out_pd_1 : !nest.event<"pre_out_pd">
    %60 = nest.subview %K_CACHE_ctx offsets = [0, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %61 = nest.subview %V_CACHE_ctx offsets = [0, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_0_event = nest.dma.prefetch.async %60 into %k_pa
      : !nest.event<"pre_k_block_0_event">
    %pre_v_block_0_event = nest.dma.prefetch.async %61 into %v_pa
      : !nest.event<"pre_v_block_0_event">
    %attn_block_0_grid, %attn_block_0_inrel, %attn_block_0_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 0 tasks(%59) globals()
      bindings(%q_l2_1, %k_pa, %v_pa, %state_pa_1, %partial_out_pa_1)
      ins(%q_l2_1, %k_pa, %v_pa, %state_pa_1, %partial_out_pa_1)
      outs(%state_pa_1, %partial_out_pa_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_block_0_event, %pre_v_block_0_event, %pre_q, %pre_state_pa, %pre_out_pa)
      : (
        !nest.event<"attn_block_0_grid">, !nest.event<"attn_block_0_inrel">,
        !nest.event<"attn_block_0_out">)
    %62 = nest.subview %K_CACHE_ctx offsets = [2, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %63 = nest.subview %V_CACHE_ctx offsets = [2, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_2_event = nest.dma.prefetch.async %62 into %k_pb
      : !nest.event<"pre_k_block_2_event">
    %pre_v_block_2_event = nest.dma.prefetch.async %63 into %v_pb
      : !nest.event<"pre_v_block_2_event">
    %attn_block_2_grid, %attn_block_2_inrel, %attn_block_2_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 1 tasks(%59) globals()
      bindings(%q_l2_1, %k_pb, %v_pb, %state_pb_1, %partial_out_pb_1)
      ins(%q_l2_1, %k_pb, %v_pb, %state_pb_1, %partial_out_pb_1)
      outs(%state_pb_1, %partial_out_pb_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_block_2_event, %pre_v_block_2_event, %pre_q, %pre_state_pb, %pre_out_pb)
      : (
        !nest.event<"attn_block_2_grid">, !nest.event<"attn_block_2_inrel">,
        !nest.event<"attn_block_2_out">)
    %64 = nest.subview %K_CACHE_ctx offsets = [4, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %65 = nest.subview %V_CACHE_ctx offsets = [4, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_4_event = nest.dma.prefetch.async %64 into %k_pc
      : !nest.event<"pre_k_block_4_event">
    %pre_v_block_4_event = nest.dma.prefetch.async %65 into %v_pc
      : !nest.event<"pre_v_block_4_event">
    %attn_block_4_grid, %attn_block_4_inrel, %attn_block_4_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 2 tasks(%59) globals()
      bindings(%q_l2_1, %k_pc, %v_pc, %state_pc_1, %partial_out_pc_1)
      ins(%q_l2_1, %k_pc, %v_pc, %state_pc_1, %partial_out_pc_1)
      outs(%state_pc_1, %partial_out_pc_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_block_4_event, %pre_v_block_4_event, %pre_q, %pre_state_pc, %pre_out_pc)
      : (
        !nest.event<"attn_block_4_grid">, !nest.event<"attn_block_4_inrel">,
        !nest.event<"attn_block_4_out">)
    %66 = nest.subview %K_CACHE_ctx offsets = [6, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %67 = nest.subview %V_CACHE_ctx offsets = [6, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_6_event = nest.dma.prefetch.async %66 into %k_pd
      : !nest.event<"pre_k_block_6_event">
    %pre_v_block_6_event = nest.dma.prefetch.async %67 into %v_pd
      : !nest.event<"pre_v_block_6_event">
    %attn_block_6_grid, %attn_block_6_inrel, %attn_block_6_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 3 tasks(%59) globals()
      bindings(%q_l2_1, %k_pd, %v_pd, %state_pd_1, %partial_out_pd_1)
      ins(%q_l2_1, %k_pd, %v_pd, %state_pd_1, %partial_out_pd_1)
      outs(%state_pd_1, %partial_out_pd_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_block_6_event, %pre_v_block_6_event, %pre_q, %pre_state_pd, %pre_out_pd)
      : (
        !nest.event<"attn_block_6_grid">, !nest.event<"attn_block_6_inrel">,
        !nest.event<"attn_block_6_out">)
    %68 = nest.subview %K_CACHE_ctx offsets = [1, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %69 = nest.subview %V_CACHE_ctx offsets = [1, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_1_event = nest.dma.prefetch.async %68 into %k_pa depends_on(%attn_block_0_inrel)
      : !nest.event<"pre_k_block_1_event">
    %pre_v_block_1_event = nest.dma.prefetch.async %69 into %v_pa depends_on(%attn_block_0_inrel)
      : !nest.event<"pre_v_block_1_event">
    %attn_block_1_grid, %attn_block_1_inrel, %attn_block_1_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 1 tasks(%59) globals()
      bindings(%q_l2_1, %k_pa, %v_pa, %state_pa_1, %partial_out_pa_1)
      ins(%q_l2_1, %k_pa, %v_pa, %state_pa_1, %partial_out_pa_1)
      outs(%state_pa_1, %partial_out_pa_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pre_k_block_1_event, %pre_v_block_1_event, %pre_q, %pre_state_pa, %pre_out_pa,
        %attn_block_0_out)
      : (
        !nest.event<"attn_block_1_grid">, !nest.event<"attn_block_1_inrel">,
        !nest.event<"attn_block_1_out">)
    %70 = nest.subview %K_CACHE_ctx offsets = [3, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %71 = nest.subview %V_CACHE_ctx offsets = [3, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_3_event = nest.dma.prefetch.async %70 into %k_pb depends_on(%attn_block_2_inrel)
      : !nest.event<"pre_k_block_3_event">
    %pre_v_block_3_event = nest.dma.prefetch.async %71 into %v_pb depends_on(%attn_block_2_inrel)
      : !nest.event<"pre_v_block_3_event">
    %attn_block_3_grid, %attn_block_3_inrel, %attn_block_3_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 2 tasks(%59) globals()
      bindings(%q_l2_1, %k_pb, %v_pb, %state_pb_1, %partial_out_pb_1)
      ins(%q_l2_1, %k_pb, %v_pb, %state_pb_1, %partial_out_pb_1)
      outs(%state_pb_1, %partial_out_pb_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pre_k_block_3_event, %pre_v_block_3_event, %pre_q, %pre_state_pb, %pre_out_pb,
        %attn_block_2_out)
      : (
        !nest.event<"attn_block_3_grid">, !nest.event<"attn_block_3_inrel">,
        !nest.event<"attn_block_3_out">)
    %72 = nest.subview %K_CACHE_ctx offsets = [5, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %73 = nest.subview %V_CACHE_ctx offsets = [5, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_5_event = nest.dma.prefetch.async %72 into %k_pc depends_on(%attn_block_4_inrel)
      : !nest.event<"pre_k_block_5_event">
    %pre_v_block_5_event = nest.dma.prefetch.async %73 into %v_pc depends_on(%attn_block_4_inrel)
      : !nest.event<"pre_v_block_5_event">
    %attn_block_5_grid, %attn_block_5_inrel, %attn_block_5_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 3 tasks(%59) globals()
      bindings(%q_l2_1, %k_pc, %v_pc, %state_pc_1, %partial_out_pc_1)
      ins(%q_l2_1, %k_pc, %v_pc, %state_pc_1, %partial_out_pc_1)
      outs(%state_pc_1, %partial_out_pc_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pre_k_block_5_event, %pre_v_block_5_event, %pre_q, %pre_state_pc, %pre_out_pc,
        %attn_block_4_out)
      : (
        !nest.event<"attn_block_5_grid">, !nest.event<"attn_block_5_inrel">,
        !nest.event<"attn_block_5_out">)
    %74 = nest.subview %K_CACHE_ctx offsets = [7, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %75 = nest.subview %V_CACHE_ctx offsets = [7, 0] sizes = [1, 65536] strides = [1, 1]
      : !nest.global_view<1x65536xbf16>
    %pre_k_block_7_event = nest.dma.prefetch.async %74 into %k_pd depends_on(%attn_block_6_inrel)
      : !nest.event<"pre_k_block_7_event">
    %pre_v_block_7_event = nest.dma.prefetch.async %75 into %v_pd depends_on(%attn_block_6_inrel)
      : !nest.event<"pre_v_block_7_event">
    %attn_block_7_grid, %attn_block_7_inrel, %attn_block_7_out =
      nest.dispatch.tasks.async @decode_partition_block l1_mode = 0 context = 0 tasks(%59) globals()
      bindings(%q_l2_1, %k_pd, %v_pd, %state_pd_1, %partial_out_pd_1)
      ins(%q_l2_1, %k_pd, %v_pd, %state_pd_1, %partial_out_pd_1)
      outs(%state_pd_1, %partial_out_pd_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pre_k_block_7_event, %pre_v_block_7_event, %pre_q, %pre_state_pd, %pre_out_pd,
        %attn_block_6_out)
      : (
        !nest.event<"attn_block_7_grid">, !nest.event<"attn_block_7_inrel">,
        !nest.event<"attn_block_7_out">)
    %pre_h = nest.dma.prefetch.async %54 into %h_l2_2 : !nest.event<"pre_h">
    %pre_wk = nest.dma.prefetch.async %55 into %wk_l2 : !nest.event<"pre_wk">
    %pre_wv = nest.dma.prefetch.async %56 into %wv_l2 : !nest.event<"pre_wv">
    %append_k_grid, %append_k_inrel, %append_k_out = nest.dispatch.tasks.async @decode_kv_append_k
      l1_mode = 0 context = 0 tasks(%59) globals() bindings(%h_l2_2, %wk_l2, %kn_l2)
      ins(%h_l2_2, %wk_l2) outs(%kn_l2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_h, %pre_wk)
      : (!nest.event<"append_k_grid">, !nest.event<"append_k_inrel">, !nest.event<"append_k_out">)
    %append_v_grid, %append_v_inrel, %append_v_out = nest.dispatch.tasks.async @decode_kv_append_v
      l1_mode = 0 context = 1 tasks(%59) globals() bindings(%h_l2_2, %wv_l2, %vn_l2)
      ins(%h_l2_2, %wv_l2) outs(%vn_l2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_h, %pre_wv)
      : (!nest.event<"append_v_grid">, !nest.event<"append_v_inrel">, !nest.event<"append_v_out">)
    %k_append_store = nest.dma.store.async %kn_l2 into %57 depends_on(%append_k_out)
      : !nest.event<"k_append_store">
    %v_append_store = nest.dma.store.async %vn_l2 into %58 depends_on(%append_v_out)
      : !nest.event<"v_append_store">
    %merge_grid, %merge_inrel, %merge_out =
      nest.dispatch.tasks.async @decode_stable_partition_merge l1_mode = 0 tasks(%59) globals()
      bindings(
        %state_pa_1, %partial_out_pa_1, %state_pb_1, %partial_out_pb_1, %state_pc_1,
        %partial_out_pc_1, %state_pd_1, %partial_out_pd_1, %merged_out_l2_1)
      ins(
        %state_pa_1, %state_pb_1, %state_pc_1, %state_pd_1, %partial_out_pa_1, %partial_out_pb_1,
        %partial_out_pc_1, %partial_out_pd_1)
      outs(%merged_out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%attn_block_1_out, %attn_block_3_out, %attn_block_5_out, %attn_block_7_out)
      : (!nest.event<"merge_grid">, !nest.event<"merge_inrel">, !nest.event<"merge_out">)
    %merged_output_store = nest.dma.store.async %merged_out_l2_1 into %53 depends_on(%merge_out)
      : !nest.event<"merged_output_store">
    nest.release %k_pa depends_on(
      %pre_k_block_0_event, %pre_k_block_1_event, %attn_block_0_inrel, %attn_block_1_inrel)
    nest.release %v_pa depends_on(
      %pre_v_block_0_event, %pre_v_block_1_event, %attn_block_0_inrel, %attn_block_1_inrel)
    nest.release %k_pb depends_on(
      %pre_k_block_2_event, %pre_k_block_3_event, %attn_block_2_inrel, %attn_block_3_inrel)
    nest.release %v_pb depends_on(
      %pre_v_block_2_event, %pre_v_block_3_event, %attn_block_2_inrel, %attn_block_3_inrel)
    nest.release %k_pc depends_on(
      %pre_k_block_4_event, %pre_k_block_5_event, %attn_block_4_inrel, %attn_block_5_inrel)
    nest.release %v_pc depends_on(
      %pre_v_block_4_event, %pre_v_block_5_event, %attn_block_4_inrel, %attn_block_5_inrel)
    nest.release %k_pd depends_on(
      %pre_k_block_6_event, %pre_k_block_7_event, %attn_block_6_inrel, %attn_block_7_inrel)
    nest.release %v_pd depends_on(
      %pre_v_block_6_event, %pre_v_block_7_event, %attn_block_6_inrel, %attn_block_7_inrel)
    nest.release %q_l2_1 depends_on(
      %pre_q, %attn_block_0_inrel, %attn_block_1_inrel, %attn_block_2_inrel, %attn_block_3_inrel,
      %attn_block_4_inrel, %attn_block_5_inrel, %attn_block_6_inrel, %attn_block_7_inrel)
    nest.release %state_pa_1 depends_on(
      %pre_state_pa, %attn_block_0_inrel, %attn_block_1_inrel, %attn_block_0_out,
      %attn_block_1_out, %merge_inrel)
    nest.release %partial_out_pa_1 depends_on(
      %pre_out_pa, %attn_block_0_inrel, %attn_block_1_inrel, %attn_block_0_out, %attn_block_1_out,
      %merge_inrel)
    nest.release %state_pb_1 depends_on(
      %pre_state_pb, %attn_block_2_inrel, %attn_block_3_inrel, %attn_block_2_out,
      %attn_block_3_out, %merge_inrel)
    nest.release %partial_out_pb_1 depends_on(
      %pre_out_pb, %attn_block_2_inrel, %attn_block_3_inrel, %attn_block_2_out, %attn_block_3_out,
      %merge_inrel)
    nest.release %state_pc_1 depends_on(
      %pre_state_pc, %attn_block_4_inrel, %attn_block_5_inrel, %attn_block_4_out,
      %attn_block_5_out, %merge_inrel)
    nest.release %partial_out_pc_1 depends_on(
      %pre_out_pc, %attn_block_4_inrel, %attn_block_5_inrel, %attn_block_4_out, %attn_block_5_out,
      %merge_inrel)
    nest.release %state_pd_1 depends_on(
      %pre_state_pd, %attn_block_6_inrel, %attn_block_7_inrel, %attn_block_6_out,
      %attn_block_7_out, %merge_inrel)
    nest.release %partial_out_pd_1 depends_on(
      %pre_out_pd, %attn_block_6_inrel, %attn_block_7_inrel, %attn_block_6_out, %attn_block_7_out,
      %merge_inrel)
    nest.release %merged_out_l2_1 depends_on(%merged_output_store)
    nest.release %h_l2_2 depends_on(%pre_h, %append_k_inrel, %append_v_inrel)
    nest.release %wk_l2 depends_on(%pre_wk, %append_k_inrel)
    nest.release %wv_l2 depends_on(%pre_wv, %append_v_inrel)
    nest.release %kn_l2 depends_on(%k_append_store)
    nest.release %vn_l2 depends_on(%v_append_store)
    nest.await %attn_block_0_grid, %attn_block_2_grid, %attn_block_4_grid, %attn_block_6_grid,
      %attn_block_1_grid, %attn_block_3_grid, %attn_block_5_grid, %attn_block_7_grid, %merge_grid,
      %merged_output_store, %append_k_grid, %append_v_grid, %k_append_store, %v_append_store
    nest.return
  }
  nexus.program @transformer_decode_kv_multicontext(
    %K_CACHE_mc: !nest.global_memref<8x65568xbf16>,
    %V_CACHE_mc: !nest.global_memref<8x65568xbf16>, %Q_IN_mc: !nest.global_memref<4x4x64xbf16>,
    %S_INIT_STATE_mc: !nest.global_memref<4x4x4x2xf32>,
    %S_INIT_OUT_mc: !nest.global_memref<4x4x4x64xf32>, %OUT_mc: !nest.global_memref<4x4x64xf32>,
    %H_T_mc: !nest.global_memref<1024xbf16>, %WK_A_mc: !nest.global_memref<4x1024x64xbf16>,
    %WV_A_mc: !nest.global_memref<4x1024x64xbf16>, %K_APPEND_mc: !nest.global_memref<4x1x64xbf16>,
    %V_APPEND_mc: !nest.global_memref<4x1x64xbf16>) {
    %decode_multicontext_done =
      nexus.submit_context.async @decode_multicontext(
        %K_CACHE_mc, %V_CACHE_mc, %Q_IN_mc, %S_INIT_STATE_mc, %S_INIT_OUT_mc, %OUT_mc, %H_T_mc,
        %WK_A_mc, %WV_A_mc, %K_APPEND_mc, %V_APPEND_mc)
      : !nexus.event<"decode_multicontext_done">
    nexus.await %decode_multicontext_done
    nexus.return
  }
}
