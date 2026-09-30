// Transformer decode step + KV cache block pipeline (GQA 16:4, BF16).
//
// Shapes: valid_sequence=2048, KV_BLOCK=256 -> 8 KV blocks; head_dim=64;
// K_CACHE/V_CACHE [kv_block][tile][token][dim] block-packed so each
// HBM->L2 block fetch is one contiguous transfer; minimum useful KV
// traffic per scan = 2 MiB (K 1 MiB + V 1 MiB).
// Tile t (placement 15) owns KV head t and Q heads 4t..4t+3; one
// dispatch processes exactly one KV block (task -> KV head).
//
// This example represents one decoder iteration at
// valid_sequence_length = 2048.  The K/V append for the current token
// is written to the static block-packed append location
// K_APPEND/V_APPEND[tile][1][dim] (position 2048) by a timing-only
// kv_append_tile dispatch (QKV projection BOA, no numerics).
// Future device-side loop support may turn the append offset into a
// runtime loop-carried value; the current IR has no dynamic addresses
// and no token-generation loop.
//
// TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor
// operands and execute no numerics.  The workload measures KV-block
// initiation interval, buffer lifetimes and dependency stalls -- NOT
// numerical correctness.
//
// Pipelined variant: L2 ping/pong buffer sets; prefetch block b+2 gates
// on block b input_released (tile has copied the block into L1), while
// dispatch b+1 also waits block b output_ready (online softmax state is
// loop-carried through the context-local L2 state buffer).  Baseline
// variant: prefetch -> await -> dispatch -> await per block, single
// buffer set, no overlap.
//
// Run: bash examples/run.sh transformer-decode-kv

builtin.module {
  tile.program @decode_attention_block(
    %task: !nest.task, %q_l2: !nest.l2_buffer<4x4x64xbf16>, %k_l2: !nest.l2_buffer<4x256x64xbf16>,
    %v_l2: !nest.l2_buffer<4x256x64xbf16>, %state: !nest.l2_buffer<4x4x66xf32>,
    %out_l2: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 77824> {
    %0 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %1 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x256x64xbf16>
    %2 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x256x64xbf16>
    %3 = tile.subview %state task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 66]
      strides = [1, 1, 1] : !nest.l2_view<1x4x66xf32>
    %4 = tile.subview %out_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %5 = tile.alloc shape = [256, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<256x64xbf16>
    %6 = tile.alloc shape = [256, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<256x64xbf16>
    %7 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %8 = tile.alloc shape = [4, 66] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x66xf32>
    %9 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %q_loaded = tile.load.async %0 into %7 : !tile.event<"q_loaded">
    %k_loaded = tile.load.async %1 into %5 : !tile.event<"k_loaded">
    %v_loaded = tile.load.async %2 into %6 : !tile.event<"v_loaded">
    %state_loaded = tile.load.async %3 into %8 : !tile.event<"state_loaded">
    tile.await %q_loaded, %k_loaded, %v_loaded, %state_loaded
    tile.signal input_released(%task)
    %qk_boa = tile.boa.async "matmul" m = 4 n = 256 k = 64 ops = 131072 : !tile.event<"qk_boa">
    tile.await %qk_boa
    %sm_update = tile.evu.async "online_softmax_update" ops = 1288 : !tile.event<"sm_update">
    tile.await %sm_update
    %pv_boa = tile.boa.async "matmul" m = 4 n = 64 k = 256 ops = 131072 : !tile.event<"pv_boa">
    tile.await %pv_boa
    %state_stored = tile.store.async %8 into %3 : !tile.event<"state_stored">
    tile.await %state_stored
    %out_stored = tile.store.async %9 into %4 : !tile.event<"out_stored">
    tile.await %out_stored
    tile.signal output_ready(%task)
    tile.free %5
    tile.free %6
    tile.free %7
    tile.free %8
    tile.free %9
    tile.return
  }
  tile.program @kv_append_tile(
    %task_1: !nest.task, %h_l2: !nest.l2_buffer<1024xbf16>,
    %wk_l2: !nest.l2_buffer<4x1024x64xbf16>, %wv_l2: !nest.l2_buffer<4x1024x64xbf16>,
    %kn_l2: !nest.l2_buffer<4x1x64xbf16>, %vn_l2: !nest.l2_buffer<4x1x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 274432> {
    %10 = tile.subview %h_l2 offsets = [0] sizes = [1024] strides = [1] : !nest.l2_view<1024xbf16>
    %11 = tile.subview %wk_l2 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1024, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1024x64xbf16>
    %12 = tile.subview %wv_l2 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1024, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1024x64xbf16>
    %13 = tile.subview %kn_l2 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %14 = tile.subview %vn_l2 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %15 = tile.alloc shape = [1024] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1024xbf16>
    %16 = tile.alloc shape = [1024, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<1024x64xbf16>
    %17 = tile.alloc shape = [1024, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<1024x64xbf16>
    %18 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %19 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %h_loaded = tile.load.async %10 into %15 : !tile.event<"h_loaded">
    %wk_loaded = tile.load.async %11 into %16 : !tile.event<"wk_loaded">
    %wv_loaded = tile.load.async %12 into %17 : !tile.event<"wv_loaded">
    tile.await %h_loaded, %wk_loaded, %wv_loaded
    tile.signal input_released(%task_1)
    %k_new_boa = tile.boa.async "matmul" m = 1 n = 64 k = 1024 ops = 131072
      : !tile.event<"k_new_boa">
    tile.await %k_new_boa
    %v_new_boa = tile.boa.async "matmul" m = 1 n = 64 k = 1024 ops = 131072
      : !tile.event<"v_new_boa">
    tile.await %v_new_boa
    %k_new_stored = tile.store.async %18 into %13 : !tile.event<"k_new_stored">
    tile.await %k_new_stored
    %v_new_stored = tile.store.async %19 into %14 : !tile.event<"v_new_stored">
    tile.await %v_new_stored
    tile.signal output_ready(%task_1)
    tile.free %15
    tile.free %16
    tile.free %17
    tile.free %18
    tile.free %19
    tile.return
  }
  nest.context @decode_req(
    %K_CACHE: !nest.global_memref<8x4x256x64xbf16>,
    %V_CACHE: !nest.global_memref<8x4x256x64xbf16>, %Q_IN: !nest.global_memref<4x4x64xbf16>,
    %S_INIT: !nest.global_memref<4x4x66xf32>, %OUT: !nest.global_memref<4x4x64xf32>,
    %H_T: !nest.global_memref<1024xbf16>, %WK_A: !nest.global_memref<4x1024x64xbf16>,
    %WV_A: !nest.global_memref<4x1024x64xbf16>, %K_APPEND: !nest.global_memref<4x1x64xbf16>,
    %V_APPEND: !nest.global_memref<4x1x64xbf16>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
      logical_tasks = 36, l2_spm_bytes = 1601536, requested_contexts_per_tile = 1> {
    %k_p0 = nest.alloc slot = "k_p0" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %k_p1 = nest.alloc slot = "k_p1" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %v_p0 = nest.alloc slot = "v_p0" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %v_p1 = nest.alloc slot = "v_p1" role = "in" shape = [4, 256, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x256x64xbf16>
    %q_l2_1 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x4x64xbf16>
    %state_l2 = nest.alloc slot = "state_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 66] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x66xf32>
    %out_l2_1 = nest.alloc slot = "out_l2" role = "out" shape = [4, 4, 64] dtype = "f32"
      alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %20 = nest.subview %Q_IN offsets = [0, 0, 0] sizes = [4, 4, 64] strides = [1, 1, 1]
      : !nest.global_view<4x4x64xbf16>
    %21 = nest.subview %S_INIT offsets = [0, 0, 0] sizes = [4, 4, 66] strides = [1, 1, 1]
      : !nest.global_view<4x4x66xf32>
    %22 = nest.subview %OUT offsets = [0, 0, 0] sizes = [4, 4, 64] strides = [1, 1, 1]
      : !nest.global_view<4x4x64xf32>
    %23 = nest.task.range from = 0 to = 4 : !nest.task_range
    %pre_q = nest.dma.prefetch.async %20 into %q_l2_1 : !nest.event<"pre_q">
    %pre_state = nest.dma.prefetch.async %21 into %state_l2 : !nest.event<"pre_state">
    %24 = nest.subview %K_CACHE offsets = [0, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %25 = nest.subview %V_CACHE offsets = [0, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k = nest.dma.prefetch.async %24 into %k_p0 : !nest.event<"pre_k_0">
    %pre_v = nest.dma.prefetch.async %25 into %v_p0 : !nest.event<"pre_v_0">
    %d0_grid, %d0_inrel, %d0_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p0, %v_p0, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p0, %v_p0, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k, %pre_v, %pre_q, %pre_state)
      : (!nest.event<"d0_grid">, !nest.event<"d0_inrel">, !nest.event<"d0_out">)
    %26 = nest.subview %K_CACHE offsets = [1, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %27 = nest.subview %V_CACHE offsets = [1, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_1 = nest.dma.prefetch.async %26 into %k_p1 : !nest.event<"pre_k_1">
    %pre_v_1 = nest.dma.prefetch.async %27 into %v_p1 : !nest.event<"pre_v_1">
    %d1_grid, %d1_inrel, %d1_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p1, %v_p1, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p1, %v_p1, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_1, %pre_v_1, %pre_q, %pre_state, %d0_out)
      : (!nest.event<"d1_grid">, !nest.event<"d1_inrel">, !nest.event<"d1_out">)
    %28 = nest.subview %K_CACHE offsets = [2, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %29 = nest.subview %V_CACHE offsets = [2, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_2 = nest.dma.prefetch.async %28 into %k_p0 depends_on(%d0_inrel) : !nest.event<"pre_k_2">
    %pre_v_2 = nest.dma.prefetch.async %29 into %v_p0 depends_on(%d0_inrel) : !nest.event<"pre_v_2">
    %d2_grid, %d2_inrel, %d2_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p0, %v_p0, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p0, %v_p0, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_2, %pre_v_2, %pre_q, %pre_state, %d1_out)
      : (!nest.event<"d2_grid">, !nest.event<"d2_inrel">, !nest.event<"d2_out">)
    %30 = nest.subview %K_CACHE offsets = [3, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %31 = nest.subview %V_CACHE offsets = [3, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_3 = nest.dma.prefetch.async %30 into %k_p1 depends_on(%d1_inrel) : !nest.event<"pre_k_3">
    %pre_v_3 = nest.dma.prefetch.async %31 into %v_p1 depends_on(%d1_inrel) : !nest.event<"pre_v_3">
    %d3_grid, %d3_inrel, %d3_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p1, %v_p1, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p1, %v_p1, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_3, %pre_v_3, %pre_q, %pre_state, %d2_out)
      : (!nest.event<"d3_grid">, !nest.event<"d3_inrel">, !nest.event<"d3_out">)
    %32 = nest.subview %K_CACHE offsets = [4, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %33 = nest.subview %V_CACHE offsets = [4, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_4 = nest.dma.prefetch.async %32 into %k_p0 depends_on(%d2_inrel) : !nest.event<"pre_k_4">
    %pre_v_4 = nest.dma.prefetch.async %33 into %v_p0 depends_on(%d2_inrel) : !nest.event<"pre_v_4">
    %d4_grid, %d4_inrel, %d4_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p0, %v_p0, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p0, %v_p0, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_4, %pre_v_4, %pre_q, %pre_state, %d3_out)
      : (!nest.event<"d4_grid">, !nest.event<"d4_inrel">, !nest.event<"d4_out">)
    %34 = nest.subview %K_CACHE offsets = [5, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %35 = nest.subview %V_CACHE offsets = [5, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_5 = nest.dma.prefetch.async %34 into %k_p1 depends_on(%d3_inrel) : !nest.event<"pre_k_5">
    %pre_v_5 = nest.dma.prefetch.async %35 into %v_p1 depends_on(%d3_inrel) : !nest.event<"pre_v_5">
    %d5_grid, %d5_inrel, %d5_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p1, %v_p1, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p1, %v_p1, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_5, %pre_v_5, %pre_q, %pre_state, %d4_out)
      : (!nest.event<"d5_grid">, !nest.event<"d5_inrel">, !nest.event<"d5_out">)
    %36 = nest.subview %K_CACHE offsets = [6, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %37 = nest.subview %V_CACHE offsets = [6, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_6 = nest.dma.prefetch.async %36 into %k_p0 depends_on(%d4_inrel) : !nest.event<"pre_k_6">
    %pre_v_6 = nest.dma.prefetch.async %37 into %v_p0 depends_on(%d4_inrel) : !nest.event<"pre_v_6">
    %d6_grid, %d6_inrel, %d6_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p0, %v_p0, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p0, %v_p0, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_6, %pre_v_6, %pre_q, %pre_state, %d5_out)
      : (!nest.event<"d6_grid">, !nest.event<"d6_inrel">, !nest.event<"d6_out">)
    %38 = nest.subview %K_CACHE offsets = [7, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %39 = nest.subview %V_CACHE offsets = [7, 0, 0, 0] sizes = [1, 4, 256, 64]
      strides = [1, 1, 1, 1] : !nest.global_view<1x4x256x64xbf16>
    %pre_k_7 = nest.dma.prefetch.async %38 into %k_p1 depends_on(%d5_inrel) : !nest.event<"pre_k_7">
    %pre_v_7 = nest.dma.prefetch.async %39 into %v_p1 depends_on(%d5_inrel) : !nest.event<"pre_v_7">
    %d7_grid, %d7_inrel, %d7_out = nest.dispatch.tasks.async @decode_attention_block l1_mode = 0
      tasks(%23) globals() bindings(%q_l2_1, %k_p1, %v_p1, %state_l2, %out_l2_1)
      ins(%q_l2_1, %k_p1, %v_p1, %state_l2) outs(%state_l2, %out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_k_7, %pre_v_7, %pre_q, %pre_state, %d6_out)
      : (!nest.event<"d7_grid">, !nest.event<"d7_inrel">, !nest.event<"d7_out">)
    %out_store = nest.dma.store.async %out_l2_1 into %22
      depends_on(%d0_out, %d1_out, %d2_out, %d3_out, %d4_out, %d5_out, %d6_out, %d7_out)
      : !nest.event<"out_store">
    %h_l2_1 = nest.alloc slot = "h_l2" role = "in" shape = [1024] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1024xbf16>
    %wk_l2_1 = nest.alloc slot = "wk_l2" role = "in" shape = [4, 1024, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x64xbf16>
    %wv_l2_1 = nest.alloc slot = "wv_l2" role = "in" shape = [4, 1024, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x64xbf16>
    %kn_l2_1 = nest.alloc slot = "kn_l2" role = "out" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %vn_l2_1 = nest.alloc slot = "vn_l2" role = "out" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %40 = nest.subview %H_T offsets = [0] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %41 = nest.subview %WK_A offsets = [0, 0, 0] sizes = [4, 1024, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1024x64xbf16>
    %42 = nest.subview %WV_A offsets = [0, 0, 0] sizes = [4, 1024, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1024x64xbf16>
    %43 = nest.subview %K_APPEND offsets = [0, 0, 0] sizes = [4, 1, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1x64xbf16>
    %44 = nest.subview %V_APPEND offsets = [0, 0, 0] sizes = [4, 1, 64] strides = [1, 1, 1]
      : !nest.global_view<4x1x64xbf16>
    %pre_qt = nest.dma.prefetch.async %40 into %h_l2_1 : !nest.event<"pre_qt">
    %pre_wk = nest.dma.prefetch.async %41 into %wk_l2_1 : !nest.event<"pre_wk">
    %pre_wv = nest.dma.prefetch.async %42 into %wv_l2_1 : !nest.event<"pre_wv">
    %app_grid, %app_inrel, %app_out = nest.dispatch.tasks.async @kv_append_tile l1_mode = 0
      tasks(%23) globals() bindings(%h_l2_1, %wk_l2_1, %wv_l2_1, %kn_l2_1, %vn_l2_1)
      ins(%h_l2_1, %wk_l2_1, %wv_l2_1) outs(%kn_l2_1, %vn_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_qt, %pre_wk, %pre_wv)
      : (!nest.event<"app_grid">, !nest.event<"app_inrel">, !nest.event<"app_out">)
    %k_append_store = nest.dma.store.async %kn_l2_1 into %43 depends_on(%app_out)
      : !nest.event<"k_append_store">
    %v_append_store = nest.dma.store.async %vn_l2_1 into %44 depends_on(%app_out)
      : !nest.event<"v_append_store">
    nest.release %k_p0
      depends_on(%pre_k, %pre_k_2, %pre_k_4, %pre_k_6, %d0_inrel, %d2_inrel, %d4_inrel, %d6_inrel)
    nest.release %v_p0
      depends_on(%pre_v, %pre_v_2, %pre_v_4, %pre_v_6, %d0_inrel, %d2_inrel, %d4_inrel, %d6_inrel)
    nest.release %k_p1
      depends_on(%pre_k_1, %pre_k_3, %pre_k_5, %pre_k_7, %d1_inrel, %d3_inrel, %d5_inrel, %d7_inrel)
    nest.release %v_p1
      depends_on(%pre_v_1, %pre_v_3, %pre_v_5, %pre_v_7, %d1_inrel, %d3_inrel, %d5_inrel, %d7_inrel)
    nest.release %q_l2_1 depends_on(
      %pre_q, %d0_inrel, %d1_inrel, %d2_inrel, %d3_inrel, %d4_inrel, %d5_inrel, %d6_inrel,
      %d7_inrel)
    nest.release %state_l2 depends_on(
      %d0_inrel, %d1_inrel, %d2_inrel, %d3_inrel, %d4_inrel, %d5_inrel, %d6_inrel, %d7_inrel,
      %d0_out, %d1_out, %d2_out, %d3_out, %d4_out, %d5_out, %d6_out, %d7_out, %pre_state)
    nest.release %out_l2_1 depends_on(%out_store)
    nest.release %h_l2_1 depends_on(%pre_qt, %app_inrel)
    nest.release %wk_l2_1 depends_on(%pre_wk, %app_inrel)
    nest.release %wv_l2_1 depends_on(%pre_wv, %app_inrel)
    nest.release %kn_l2_1 depends_on(%k_append_store)
    nest.release %vn_l2_1 depends_on(%v_append_store)
    nest.await %d7_grid, %out_store, %app_grid, %k_append_store, %v_append_store
    nest.return
  }
  nexus.program @transformer_decode_kv(
    %K_CACHE_1: !nest.global_memref<8x4x256x64xbf16>,
    %V_CACHE_1: !nest.global_memref<8x4x256x64xbf16>, %Q_IN_1: !nest.global_memref<4x4x64xbf16>,
    %S_INIT_1: !nest.global_memref<4x4x66xf32>, %OUT_1: !nest.global_memref<4x4x64xf32>,
    %H_T_1: !nest.global_memref<1024xbf16>, %WK_A_1: !nest.global_memref<4x1024x64xbf16>,
    %WV_A_1: !nest.global_memref<4x1024x64xbf16>, %K_APPEND_1: !nest.global_memref<4x1x64xbf16>,
    %V_APPEND_1: !nest.global_memref<4x1x64xbf16>) {
    %decode_done =
      nexus.submit_context.async @decode_req(
        %K_CACHE_1, %V_CACHE_1, %Q_IN_1, %S_INIT_1, %OUT_1, %H_T_1, %WK_A_1, %WV_A_1, %K_APPEND_1,
        %V_APPEND_1)
      : !nexus.event<"decode_done">
    nexus.await %decode_done
    nexus.return
  }
}
