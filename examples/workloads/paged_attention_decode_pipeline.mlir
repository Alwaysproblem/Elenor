// PagedAttention decode (pipeline variant, plan §6) -- timing model, no numerics.
// R=3 initial=[255, 511, 767] steps=4 page_tokens=16 kv_heads=4 heads_per_kv=4 head_dim=64 pages=128 page_stride=16448B
// Pool pages are host-managed (HostAllocPages/HostFreePages); the block table is
// host-written between steps; attention gathers K/V rows straight into L1 with
// #tile.indexed_map<index_scale=page_stride, task_stride=B*D, segment=valid*D>.

builtin.module {
  tile.program @paged_attention_append_r0_tip0(
    %task: !nest.task, %pool: !nest.global_view<128x8224xbf16>,
    %k_new_l2: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %0 = tile.subview %k_new_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %1 = tile.subview %v_new_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %2 = tile.subview %append_idx_l2 offsets = [0] sizes = [1] strides = [1] : !nest.l2_view<1xi32>
    %3 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %4 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %5 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %6 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r0_tip0 = tile.load.async %2 into %5 : !tile.event<"aidx_k_r0_tip0">
    %aidx_v_r0_tip0 = tile.load.async %2 into %6 : !tile.event<"aidx_v_r0_tip0">
    %k_row_r0_tip0 = tile.load.async %0 into %3 : !tile.event<"k_row_r0_tip0">
    %v_row_r0_tip0 = tile.load.async %1 into %4 : !tile.event<"v_row_r0_tip0">
    tile.await %aidx_k_r0_tip0, %aidx_v_r0_tip0, %k_row_r0_tip0, %v_row_r0_tip0
    tile.signal input_released(%task)
    %scatter_k_r0_tip0 = tile.scatter.global.async %3 indices(%5) into %pool
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_k_r0_tip0">
    %scatter_v_r0_tip0 = tile.scatter.global.async %4 indices(%6) into %pool
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_v_r0_tip0">
    tile.await %scatter_k_r0_tip0, %scatter_v_r0_tip0
    tile.free %3
    tile.free %4
    tile.free %5
    tile.free %6
    tile.return
  }
  tile.program @paged_attention_append_r0_tip1(
    %task_1: !nest.task, %pool_1: !nest.global_view<128x8224xbf16>,
    %k_new_l2_1: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_1: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_1: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %7 = tile.subview %k_new_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %8 = tile.subview %v_new_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 1, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %9 = tile.subview %append_idx_l2_1 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %10 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %11 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %12 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %13 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r0_tip1 = tile.load.async %9 into %12 : !tile.event<"aidx_k_r0_tip1">
    %aidx_v_r0_tip1 = tile.load.async %9 into %13 : !tile.event<"aidx_v_r0_tip1">
    %k_row_r0_tip1 = tile.load.async %7 into %10 : !tile.event<"k_row_r0_tip1">
    %v_row_r0_tip1 = tile.load.async %8 into %11 : !tile.event<"v_row_r0_tip1">
    tile.await %aidx_k_r0_tip1, %aidx_v_r0_tip1, %k_row_r0_tip1, %v_row_r0_tip1
    tile.signal input_released(%task_1)
    %scatter_k_r0_tip1 = tile.scatter.global.async %10 indices(%12) into %pool_1
      map = #tile.indexed_map< index_scale = 8224 offset = 256 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_k_r0_tip1">
    %scatter_v_r0_tip1 = tile.scatter.global.async %11 indices(%13) into %pool_1
      map = #tile.indexed_map< index_scale = 8224 offset = 4352 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_v_r0_tip1">
    tile.await %scatter_k_r0_tip1, %scatter_v_r0_tip1
    tile.free %10
    tile.free %11
    tile.free %12
    tile.free %13
    tile.return
  }
  tile.program @paged_attention_append_r0_tip2(
    %task_2: !nest.task, %pool_2: !nest.global_view<128x8224xbf16>,
    %k_new_l2_2: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_2: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_2: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %14 = tile.subview %k_new_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %15 = tile.subview %v_new_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %16 = tile.subview %append_idx_l2_2 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %17 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %18 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %19 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %20 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r0_tip2 = tile.load.async %16 into %19 : !tile.event<"aidx_k_r0_tip2">
    %aidx_v_r0_tip2 = tile.load.async %16 into %20 : !tile.event<"aidx_v_r0_tip2">
    %k_row_r0_tip2 = tile.load.async %14 into %17 : !tile.event<"k_row_r0_tip2">
    %v_row_r0_tip2 = tile.load.async %15 into %18 : !tile.event<"v_row_r0_tip2">
    tile.await %aidx_k_r0_tip2, %aidx_v_r0_tip2, %k_row_r0_tip2, %v_row_r0_tip2
    tile.signal input_released(%task_2)
    %scatter_k_r0_tip2 = tile.scatter.global.async %17 indices(%19) into %pool_2
      map = #tile.indexed_map< index_scale = 8224 offset = 512 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_k_r0_tip2">
    %scatter_v_r0_tip2 = tile.scatter.global.async %18 indices(%20) into %pool_2
      map = #tile.indexed_map< index_scale = 8224 offset = 4608 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_v_r0_tip2">
    tile.await %scatter_k_r0_tip2, %scatter_v_r0_tip2
    tile.free %17
    tile.free %18
    tile.free %19
    tile.free %20
    tile.return
  }
  tile.program @paged_attention_append_r0_tip15(
    %task_3: !nest.task, %pool_3: !nest.global_view<128x8224xbf16>,
    %k_new_l2_3: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_3: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_3: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %21 = tile.subview %k_new_l2_3 task = %task_3 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %22 = tile.subview %v_new_l2_3 task = %task_3 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %23 = tile.subview %append_idx_l2_3 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %24 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %25 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %26 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %27 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r0_tip15 = tile.load.async %23 into %26 : !tile.event<"aidx_k_r0_tip15">
    %aidx_v_r0_tip15 = tile.load.async %23 into %27 : !tile.event<"aidx_v_r0_tip15">
    %k_row_r0_tip15 = tile.load.async %21 into %24 : !tile.event<"k_row_r0_tip15">
    %v_row_r0_tip15 = tile.load.async %22 into %25 : !tile.event<"v_row_r0_tip15">
    tile.await %aidx_k_r0_tip15, %aidx_v_r0_tip15, %k_row_r0_tip15, %v_row_r0_tip15
    tile.signal input_released(%task_3)
    %scatter_k_r0_tip15 = tile.scatter.global.async %24 indices(%26) into %pool_3
      map = #tile.indexed_map< index_scale = 8224 offset = 3840 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_k_r0_tip15">
    %scatter_v_r0_tip15 = tile.scatter.global.async %25 indices(%27) into %pool_3
      map = #tile.indexed_map< index_scale = 8224 offset = 7936 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"scatter_v_r0_tip15">
    tile.await %scatter_k_r0_tip15, %scatter_v_r0_tip15
    tile.free %24
    tile.free %25
    tile.free %26
    tile.free %27
    tile.return
  }
  tile.program @paged_attention_t16_step_r0(
    %task_4: !nest.task, %pool_4: !nest.global_view<128x8224xbf16>,
    %block_idx_l2: !nest.l2_buffer<1xi32>, %q_l2: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2: !nest.l2_buffer<4x4x2xf32>, %acc_l2: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %28 = tile.subview %block_idx_l2 offsets = [0] sizes = [1] strides = [1] : !nest.l2_view<1xi32>
    %29 = tile.subview %q_l2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %30 = tile.subview %state_l2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %31 = tile.subview %acc_l2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %32 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %33 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %34 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %35 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %36 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %37 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %38 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r0_t16 = tile.load.async %28 into %32 : !tile.event<"bidx_k_r0_t16">
    %bidx_v_r0_t16 = tile.load.async %28 into %33 : !tile.event<"bidx_v_r0_t16">
    %q_r0_t16 = tile.load.async %29 into %36 : !tile.event<"q_r0_t16">
    %state_r0_t16 = tile.load.async %30 into %37 : !tile.event<"state_r0_t16">
    %acc_r0_t16 = tile.load.async %31 into %38 : !tile.event<"acc_r0_t16">
    tile.await %bidx_k_r0_t16, %bidx_v_r0_t16
    %gather_k_r0_t16 = tile.gather.global.async %pool_4 indices(%32) into %34
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_k_r0_t16">
    %gather_v_r0_t16 = tile.gather.global.async %pool_4 indices(%33) into %35
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_v_r0_t16">
    tile.await %gather_k_r0_t16, %gather_v_r0_t16, %q_r0_t16, %state_r0_t16, %acc_r0_t16
    tile.signal input_released(%task_4)
    %qk_r0_t16 = tile.boa.async "matmul" m = 4 n = 16 k = 64 ops = 8192 : !tile.event<"qk_r0_t16">
    tile.await %qk_r0_t16
    %sm_r0_t16 = tile.evu.async "online_softmax_update" ops = 72 : !tile.event<"sm_r0_t16">
    tile.await %sm_r0_t16
    %pv_r0_t16 = tile.boa.async "matmul" m = 4 n = 64 k = 16 ops = 8192 : !tile.event<"pv_r0_t16">
    tile.await %pv_r0_t16
    %acc_evu_r0_t16 = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r0_t16">
    tile.await %acc_evu_r0_t16
    %state_store_r0_t16 = tile.store.async %37 into %30 : !tile.event<"state_store_r0_t16">
    %acc_store_r0_t16 = tile.store.async %38 into %31 : !tile.event<"acc_store_r0_t16">
    tile.await %state_store_r0_t16, %acc_store_r0_t16
    tile.signal output_ready(%task_4)
    tile.free %32
    tile.free %33
    tile.free %34
    tile.free %35
    tile.free %36
    tile.free %37
    tile.free %38
    tile.return
  }
  tile.program @paged_attention_t16_final_r0(
    %task_5: !nest.task, %pool_5: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_1: !nest.l2_buffer<1xi32>, %q_l2_1: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_1: !nest.l2_buffer<4x4x2xf32>, %acc_l2_1: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %39 = tile.subview %block_idx_l2_1 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %40 = tile.subview %q_l2_1 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %41 = tile.subview %state_l2_1 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %42 = tile.subview %acc_l2_1 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %43 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %44 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %45 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %46 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %47 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %48 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %49 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r0_t16_final = tile.load.async %39 into %43 : !tile.event<"bidx_k_r0_t16_final">
    %bidx_v_r0_t16_final = tile.load.async %39 into %44 : !tile.event<"bidx_v_r0_t16_final">
    %q_r0_t16_final = tile.load.async %40 into %47 : !tile.event<"q_r0_t16_final">
    %state_r0_t16_final = tile.load.async %41 into %48 : !tile.event<"state_r0_t16_final">
    %acc_r0_t16_final = tile.load.async %42 into %49 : !tile.event<"acc_r0_t16_final">
    tile.await %bidx_k_r0_t16_final, %bidx_v_r0_t16_final
    %gather_k_r0_t16_final = tile.gather.global.async %pool_5 indices(%43) into %45
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_k_r0_t16_final">
    %gather_v_r0_t16_final = tile.gather.global.async %pool_5 indices(%44) into %46
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_v_r0_t16_final">
    tile.await %gather_k_r0_t16_final, %gather_v_r0_t16_final, %q_r0_t16_final, %state_r0_t16_final,
      %acc_r0_t16_final
    tile.signal input_released(%task_5)
    %qk_r0_t16_final = tile.boa.async "matmul" m = 4 n = 16 k = 64 ops = 8192
      : !tile.event<"qk_r0_t16_final">
    tile.await %qk_r0_t16_final
    %sm_r0_t16_final = tile.evu.async "online_softmax_update" ops = 72
      : !tile.event<"sm_r0_t16_final">
    tile.await %sm_r0_t16_final
    %pv_r0_t16_final = tile.boa.async "matmul" m = 4 n = 64 k = 16 ops = 8192
      : !tile.event<"pv_r0_t16_final">
    tile.await %pv_r0_t16_final
    %acc_evu_r0_t16_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r0_t16_final">
    tile.await %acc_evu_r0_t16_final
    %norm_r0_t16_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r0_t16_final">
    tile.await %norm_r0_t16_final
    %state_store_r0_t16_final = tile.store.async %48 into %41
      : !tile.event<"state_store_r0_t16_final">
    %acc_store_r0_t16_final = tile.store.async %49 into %42 : !tile.event<"acc_store_r0_t16_final">
    tile.await %state_store_r0_t16_final, %acc_store_r0_t16_final
    tile.signal output_ready(%task_5)
    tile.free %43
    tile.free %44
    tile.free %45
    tile.free %46
    tile.free %47
    tile.free %48
    tile.free %49
    tile.return
  }
  tile.program @paged_attention_t1_final_r0(
    %task_6: !nest.task, %pool_6: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_2: !nest.l2_buffer<1xi32>, %q_l2_2: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_2: !nest.l2_buffer<4x4x2xf32>, %acc_l2_2: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %50 = tile.subview %block_idx_l2_2 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %51 = tile.subview %q_l2_2 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %52 = tile.subview %state_l2_2 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %53 = tile.subview %acc_l2_2 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %54 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %55 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %56 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %57 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %58 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %59 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %60 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r0_t1_final = tile.load.async %50 into %54 : !tile.event<"bidx_k_r0_t1_final">
    %bidx_v_r0_t1_final = tile.load.async %50 into %55 : !tile.event<"bidx_v_r0_t1_final">
    %q_r0_t1_final = tile.load.async %51 into %58 : !tile.event<"q_r0_t1_final">
    %state_r0_t1_final = tile.load.async %52 into %59 : !tile.event<"state_r0_t1_final">
    %acc_r0_t1_final = tile.load.async %53 into %60 : !tile.event<"acc_r0_t1_final">
    tile.await %bidx_k_r0_t1_final, %bidx_v_r0_t1_final
    %gather_k_r0_t1_final = tile.gather.global.async %pool_6 indices(%54) into %56
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_k_r0_t1_final">
    %gather_v_r0_t1_final = tile.gather.global.async %pool_6 indices(%55) into %57
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_v_r0_t1_final">
    tile.await %gather_k_r0_t1_final, %gather_v_r0_t1_final, %q_r0_t1_final, %state_r0_t1_final,
      %acc_r0_t1_final
    tile.signal input_released(%task_6)
    %qk_r0_t1_final = tile.boa.async "matmul" m = 4 n = 1 k = 64 ops = 512
      : !tile.event<"qk_r0_t1_final">
    tile.await %qk_r0_t1_final
    %sm_r0_t1_final = tile.evu.async "online_softmax_update" ops = 12
      : !tile.event<"sm_r0_t1_final">
    tile.await %sm_r0_t1_final
    %pv_r0_t1_final = tile.boa.async "matmul" m = 4 n = 64 k = 1 ops = 512
      : !tile.event<"pv_r0_t1_final">
    tile.await %pv_r0_t1_final
    %acc_evu_r0_t1_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r0_t1_final">
    tile.await %acc_evu_r0_t1_final
    %norm_r0_t1_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r0_t1_final">
    tile.await %norm_r0_t1_final
    %state_store_r0_t1_final = tile.store.async %59 into %52
      : !tile.event<"state_store_r0_t1_final">
    %acc_store_r0_t1_final = tile.store.async %60 into %53 : !tile.event<"acc_store_r0_t1_final">
    tile.await %state_store_r0_t1_final, %acc_store_r0_t1_final
    tile.signal output_ready(%task_6)
    tile.free %54
    tile.free %55
    tile.free %56
    tile.free %57
    tile.free %58
    tile.free %59
    tile.free %60
    tile.return
  }
  tile.program @paged_attention_t2_final_r0(
    %task_7: !nest.task, %pool_7: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_3: !nest.l2_buffer<1xi32>, %q_l2_3: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_3: !nest.l2_buffer<4x4x2xf32>, %acc_l2_3: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %61 = tile.subview %block_idx_l2_3 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %62 = tile.subview %q_l2_3 task = %task_7 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %63 = tile.subview %state_l2_3 task = %task_7 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %64 = tile.subview %acc_l2_3 task = %task_7 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %65 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %66 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %67 = tile.alloc shape = [2, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x64xbf16>
    %68 = tile.alloc shape = [2, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x64xbf16>
    %69 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %70 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %71 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r0_t2_final = tile.load.async %61 into %65 : !tile.event<"bidx_k_r0_t2_final">
    %bidx_v_r0_t2_final = tile.load.async %61 into %66 : !tile.event<"bidx_v_r0_t2_final">
    %q_r0_t2_final = tile.load.async %62 into %69 : !tile.event<"q_r0_t2_final">
    %state_r0_t2_final = tile.load.async %63 into %70 : !tile.event<"state_r0_t2_final">
    %acc_r0_t2_final = tile.load.async %64 into %71 : !tile.event<"acc_r0_t2_final">
    tile.await %bidx_k_r0_t2_final, %bidx_v_r0_t2_final
    %gather_k_r0_t2_final = tile.gather.global.async %pool_7 indices(%65) into %67
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 128>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_k_r0_t2_final">
    %gather_v_r0_t2_final = tile.gather.global.async %pool_7 indices(%66) into %68
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 128>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_v_r0_t2_final">
    tile.await %gather_k_r0_t2_final, %gather_v_r0_t2_final, %q_r0_t2_final, %state_r0_t2_final,
      %acc_r0_t2_final
    tile.signal input_released(%task_7)
    %qk_r0_t2_final = tile.boa.async "matmul" m = 4 n = 2 k = 64 ops = 1024
      : !tile.event<"qk_r0_t2_final">
    tile.await %qk_r0_t2_final
    %sm_r0_t2_final = tile.evu.async "online_softmax_update" ops = 16
      : !tile.event<"sm_r0_t2_final">
    tile.await %sm_r0_t2_final
    %pv_r0_t2_final = tile.boa.async "matmul" m = 4 n = 64 k = 2 ops = 1024
      : !tile.event<"pv_r0_t2_final">
    tile.await %pv_r0_t2_final
    %acc_evu_r0_t2_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r0_t2_final">
    tile.await %acc_evu_r0_t2_final
    %norm_r0_t2_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r0_t2_final">
    tile.await %norm_r0_t2_final
    %state_store_r0_t2_final = tile.store.async %70 into %63
      : !tile.event<"state_store_r0_t2_final">
    %acc_store_r0_t2_final = tile.store.async %71 into %64 : !tile.event<"acc_store_r0_t2_final">
    tile.await %state_store_r0_t2_final, %acc_store_r0_t2_final
    tile.signal output_ready(%task_7)
    tile.free %65
    tile.free %66
    tile.free %67
    tile.free %68
    tile.free %69
    tile.free %70
    tile.free %71
    tile.return
  }
  tile.program @paged_attention_t3_final_r0(
    %task_8: !nest.task, %pool_8: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_4: !nest.l2_buffer<1xi32>, %q_l2_4: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_4: !nest.l2_buffer<4x4x2xf32>, %acc_l2_4: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %72 = tile.subview %block_idx_l2_4 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %73 = tile.subview %q_l2_4 task = %task_8 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %74 = tile.subview %state_l2_4 task = %task_8 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %75 = tile.subview %acc_l2_4 task = %task_8 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %76 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %77 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %78 = tile.alloc shape = [3, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x64xbf16>
    %79 = tile.alloc shape = [3, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x64xbf16>
    %80 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %81 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %82 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r0_t3_final = tile.load.async %72 into %76 : !tile.event<"bidx_k_r0_t3_final">
    %bidx_v_r0_t3_final = tile.load.async %72 into %77 : !tile.event<"bidx_v_r0_t3_final">
    %q_r0_t3_final = tile.load.async %73 into %80 : !tile.event<"q_r0_t3_final">
    %state_r0_t3_final = tile.load.async %74 into %81 : !tile.event<"state_r0_t3_final">
    %acc_r0_t3_final = tile.load.async %75 into %82 : !tile.event<"acc_r0_t3_final">
    tile.await %bidx_k_r0_t3_final, %bidx_v_r0_t3_final
    %gather_k_r0_t3_final = tile.gather.global.async %pool_8 indices(%76) into %78
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 192>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_k_r0_t3_final">
    %gather_v_r0_t3_final = tile.gather.global.async %pool_8 indices(%77) into %79
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 192>
      window_entries = 1 scope = "owner_0" : !tile.event<"gather_v_r0_t3_final">
    tile.await %gather_k_r0_t3_final, %gather_v_r0_t3_final, %q_r0_t3_final, %state_r0_t3_final,
      %acc_r0_t3_final
    tile.signal input_released(%task_8)
    %qk_r0_t3_final = tile.boa.async "matmul" m = 4 n = 3 k = 64 ops = 1536
      : !tile.event<"qk_r0_t3_final">
    tile.await %qk_r0_t3_final
    %sm_r0_t3_final = tile.evu.async "online_softmax_update" ops = 20
      : !tile.event<"sm_r0_t3_final">
    tile.await %sm_r0_t3_final
    %pv_r0_t3_final = tile.boa.async "matmul" m = 4 n = 64 k = 3 ops = 1536
      : !tile.event<"pv_r0_t3_final">
    tile.await %pv_r0_t3_final
    %acc_evu_r0_t3_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r0_t3_final">
    tile.await %acc_evu_r0_t3_final
    %norm_r0_t3_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r0_t3_final">
    tile.await %norm_r0_t3_final
    %state_store_r0_t3_final = tile.store.async %81 into %74
      : !tile.event<"state_store_r0_t3_final">
    %acc_store_r0_t3_final = tile.store.async %82 into %75 : !tile.event<"acc_store_r0_t3_final">
    tile.await %state_store_r0_t3_final, %acc_store_r0_t3_final
    tile.signal output_ready(%task_8)
    tile.free %76
    tile.free %77
    tile.free %78
    tile.free %79
    tile.free %80
    tile.free %81
    tile.free %82
    tile.return
  }
  tile.program @paged_attention_append_r1_tip0(
    %task_9: !nest.task, %pool_9: !nest.global_view<128x8224xbf16>,
    %k_new_l2_4: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_4: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_4: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %83 = tile.subview %k_new_l2_4 task = %task_9 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %84 = tile.subview %v_new_l2_4 task = %task_9 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %85 = tile.subview %append_idx_l2_4 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %86 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %87 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %88 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %89 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r1_tip0 = tile.load.async %85 into %88 : !tile.event<"aidx_k_r1_tip0">
    %aidx_v_r1_tip0 = tile.load.async %85 into %89 : !tile.event<"aidx_v_r1_tip0">
    %k_row_r1_tip0 = tile.load.async %83 into %86 : !tile.event<"k_row_r1_tip0">
    %v_row_r1_tip0 = tile.load.async %84 into %87 : !tile.event<"v_row_r1_tip0">
    tile.await %aidx_k_r1_tip0, %aidx_v_r1_tip0, %k_row_r1_tip0, %v_row_r1_tip0
    tile.signal input_released(%task_9)
    %scatter_k_r1_tip0 = tile.scatter.global.async %86 indices(%88) into %pool_9
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_k_r1_tip0">
    %scatter_v_r1_tip0 = tile.scatter.global.async %87 indices(%89) into %pool_9
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_v_r1_tip0">
    tile.await %scatter_k_r1_tip0, %scatter_v_r1_tip0
    tile.free %86
    tile.free %87
    tile.free %88
    tile.free %89
    tile.return
  }
  tile.program @paged_attention_append_r1_tip1(
    %task_10: !nest.task, %pool_10: !nest.global_view<128x8224xbf16>,
    %k_new_l2_5: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_5: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_5: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %90 = tile.subview %k_new_l2_5 task = %task_10 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %91 = tile.subview %v_new_l2_5 task = %task_10 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %92 = tile.subview %append_idx_l2_5 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %93 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %94 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %95 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %96 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r1_tip1 = tile.load.async %92 into %95 : !tile.event<"aidx_k_r1_tip1">
    %aidx_v_r1_tip1 = tile.load.async %92 into %96 : !tile.event<"aidx_v_r1_tip1">
    %k_row_r1_tip1 = tile.load.async %90 into %93 : !tile.event<"k_row_r1_tip1">
    %v_row_r1_tip1 = tile.load.async %91 into %94 : !tile.event<"v_row_r1_tip1">
    tile.await %aidx_k_r1_tip1, %aidx_v_r1_tip1, %k_row_r1_tip1, %v_row_r1_tip1
    tile.signal input_released(%task_10)
    %scatter_k_r1_tip1 = tile.scatter.global.async %93 indices(%95) into %pool_10
      map = #tile.indexed_map< index_scale = 8224 offset = 256 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_k_r1_tip1">
    %scatter_v_r1_tip1 = tile.scatter.global.async %94 indices(%96) into %pool_10
      map = #tile.indexed_map< index_scale = 8224 offset = 4352 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_v_r1_tip1">
    tile.await %scatter_k_r1_tip1, %scatter_v_r1_tip1
    tile.free %93
    tile.free %94
    tile.free %95
    tile.free %96
    tile.return
  }
  tile.program @paged_attention_append_r1_tip2(
    %task_11: !nest.task, %pool_11: !nest.global_view<128x8224xbf16>,
    %k_new_l2_6: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_6: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_6: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %97 = tile.subview %k_new_l2_6 task = %task_11 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %98 = tile.subview %v_new_l2_6 task = %task_11 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %99 = tile.subview %append_idx_l2_6 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %100 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %101 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %102 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %103 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r1_tip2 = tile.load.async %99 into %102 : !tile.event<"aidx_k_r1_tip2">
    %aidx_v_r1_tip2 = tile.load.async %99 into %103 : !tile.event<"aidx_v_r1_tip2">
    %k_row_r1_tip2 = tile.load.async %97 into %100 : !tile.event<"k_row_r1_tip2">
    %v_row_r1_tip2 = tile.load.async %98 into %101 : !tile.event<"v_row_r1_tip2">
    tile.await %aidx_k_r1_tip2, %aidx_v_r1_tip2, %k_row_r1_tip2, %v_row_r1_tip2
    tile.signal input_released(%task_11)
    %scatter_k_r1_tip2 = tile.scatter.global.async %100 indices(%102) into %pool_11
      map = #tile.indexed_map< index_scale = 8224 offset = 512 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_k_r1_tip2">
    %scatter_v_r1_tip2 = tile.scatter.global.async %101 indices(%103) into %pool_11
      map = #tile.indexed_map< index_scale = 8224 offset = 4608 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_v_r1_tip2">
    tile.await %scatter_k_r1_tip2, %scatter_v_r1_tip2
    tile.free %100
    tile.free %101
    tile.free %102
    tile.free %103
    tile.return
  }
  tile.program @paged_attention_append_r1_tip15(
    %task_12: !nest.task, %pool_12: !nest.global_view<128x8224xbf16>,
    %k_new_l2_7: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_7: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_7: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %104 = tile.subview %k_new_l2_7 task = %task_12 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %105 = tile.subview %v_new_l2_7 task = %task_12 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %106 = tile.subview %append_idx_l2_7 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %107 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %108 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %109 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %110 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r1_tip15 = tile.load.async %106 into %109 : !tile.event<"aidx_k_r1_tip15">
    %aidx_v_r1_tip15 = tile.load.async %106 into %110 : !tile.event<"aidx_v_r1_tip15">
    %k_row_r1_tip15 = tile.load.async %104 into %107 : !tile.event<"k_row_r1_tip15">
    %v_row_r1_tip15 = tile.load.async %105 into %108 : !tile.event<"v_row_r1_tip15">
    tile.await %aidx_k_r1_tip15, %aidx_v_r1_tip15, %k_row_r1_tip15, %v_row_r1_tip15
    tile.signal input_released(%task_12)
    %scatter_k_r1_tip15 = tile.scatter.global.async %107 indices(%109) into %pool_12
      map = #tile.indexed_map< index_scale = 8224 offset = 3840 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_k_r1_tip15">
    %scatter_v_r1_tip15 = tile.scatter.global.async %108 indices(%110) into %pool_12
      map = #tile.indexed_map< index_scale = 8224 offset = 7936 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"scatter_v_r1_tip15">
    tile.await %scatter_k_r1_tip15, %scatter_v_r1_tip15
    tile.free %107
    tile.free %108
    tile.free %109
    tile.free %110
    tile.return
  }
  tile.program @paged_attention_t16_step_r1(
    %task_13: !nest.task, %pool_13: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_5: !nest.l2_buffer<1xi32>, %q_l2_5: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_5: !nest.l2_buffer<4x4x2xf32>, %acc_l2_5: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %111 = tile.subview %block_idx_l2_5 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %112 = tile.subview %q_l2_5 task = %task_13 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %113 = tile.subview %state_l2_5 task = %task_13 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %114 = tile.subview %acc_l2_5 task = %task_13 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %115 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %116 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %117 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %118 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %119 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %120 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %121 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r1_t16 = tile.load.async %111 into %115 : !tile.event<"bidx_k_r1_t16">
    %bidx_v_r1_t16 = tile.load.async %111 into %116 : !tile.event<"bidx_v_r1_t16">
    %q_r1_t16 = tile.load.async %112 into %119 : !tile.event<"q_r1_t16">
    %state_r1_t16 = tile.load.async %113 into %120 : !tile.event<"state_r1_t16">
    %acc_r1_t16 = tile.load.async %114 into %121 : !tile.event<"acc_r1_t16">
    tile.await %bidx_k_r1_t16, %bidx_v_r1_t16
    %gather_k_r1_t16 = tile.gather.global.async %pool_13 indices(%115) into %117
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_k_r1_t16">
    %gather_v_r1_t16 = tile.gather.global.async %pool_13 indices(%116) into %118
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_v_r1_t16">
    tile.await %gather_k_r1_t16, %gather_v_r1_t16, %q_r1_t16, %state_r1_t16, %acc_r1_t16
    tile.signal input_released(%task_13)
    %qk_r1_t16 = tile.boa.async "matmul" m = 4 n = 16 k = 64 ops = 8192 : !tile.event<"qk_r1_t16">
    tile.await %qk_r1_t16
    %sm_r1_t16 = tile.evu.async "online_softmax_update" ops = 72 : !tile.event<"sm_r1_t16">
    tile.await %sm_r1_t16
    %pv_r1_t16 = tile.boa.async "matmul" m = 4 n = 64 k = 16 ops = 8192 : !tile.event<"pv_r1_t16">
    tile.await %pv_r1_t16
    %acc_evu_r1_t16 = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r1_t16">
    tile.await %acc_evu_r1_t16
    %state_store_r1_t16 = tile.store.async %120 into %113 : !tile.event<"state_store_r1_t16">
    %acc_store_r1_t16 = tile.store.async %121 into %114 : !tile.event<"acc_store_r1_t16">
    tile.await %state_store_r1_t16, %acc_store_r1_t16
    tile.signal output_ready(%task_13)
    tile.free %115
    tile.free %116
    tile.free %117
    tile.free %118
    tile.free %119
    tile.free %120
    tile.free %121
    tile.return
  }
  tile.program @paged_attention_t16_final_r1(
    %task_14: !nest.task, %pool_14: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_6: !nest.l2_buffer<1xi32>, %q_l2_6: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_6: !nest.l2_buffer<4x4x2xf32>, %acc_l2_6: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %122 = tile.subview %block_idx_l2_6 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %123 = tile.subview %q_l2_6 task = %task_14 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %124 = tile.subview %state_l2_6 task = %task_14 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %125 = tile.subview %acc_l2_6 task = %task_14 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %126 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %127 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %128 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %129 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %130 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %131 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %132 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r1_t16_final = tile.load.async %122 into %126 : !tile.event<"bidx_k_r1_t16_final">
    %bidx_v_r1_t16_final = tile.load.async %122 into %127 : !tile.event<"bidx_v_r1_t16_final">
    %q_r1_t16_final = tile.load.async %123 into %130 : !tile.event<"q_r1_t16_final">
    %state_r1_t16_final = tile.load.async %124 into %131 : !tile.event<"state_r1_t16_final">
    %acc_r1_t16_final = tile.load.async %125 into %132 : !tile.event<"acc_r1_t16_final">
    tile.await %bidx_k_r1_t16_final, %bidx_v_r1_t16_final
    %gather_k_r1_t16_final = tile.gather.global.async %pool_14 indices(%126) into %128
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_k_r1_t16_final">
    %gather_v_r1_t16_final = tile.gather.global.async %pool_14 indices(%127) into %129
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_v_r1_t16_final">
    tile.await %gather_k_r1_t16_final, %gather_v_r1_t16_final, %q_r1_t16_final, %state_r1_t16_final,
      %acc_r1_t16_final
    tile.signal input_released(%task_14)
    %qk_r1_t16_final = tile.boa.async "matmul" m = 4 n = 16 k = 64 ops = 8192
      : !tile.event<"qk_r1_t16_final">
    tile.await %qk_r1_t16_final
    %sm_r1_t16_final = tile.evu.async "online_softmax_update" ops = 72
      : !tile.event<"sm_r1_t16_final">
    tile.await %sm_r1_t16_final
    %pv_r1_t16_final = tile.boa.async "matmul" m = 4 n = 64 k = 16 ops = 8192
      : !tile.event<"pv_r1_t16_final">
    tile.await %pv_r1_t16_final
    %acc_evu_r1_t16_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r1_t16_final">
    tile.await %acc_evu_r1_t16_final
    %norm_r1_t16_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r1_t16_final">
    tile.await %norm_r1_t16_final
    %state_store_r1_t16_final = tile.store.async %131 into %124
      : !tile.event<"state_store_r1_t16_final">
    %acc_store_r1_t16_final = tile.store.async %132 into %125
      : !tile.event<"acc_store_r1_t16_final">
    tile.await %state_store_r1_t16_final, %acc_store_r1_t16_final
    tile.signal output_ready(%task_14)
    tile.free %126
    tile.free %127
    tile.free %128
    tile.free %129
    tile.free %130
    tile.free %131
    tile.free %132
    tile.return
  }
  tile.program @paged_attention_t1_final_r1(
    %task_15: !nest.task, %pool_15: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_7: !nest.l2_buffer<1xi32>, %q_l2_7: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_7: !nest.l2_buffer<4x4x2xf32>, %acc_l2_7: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %133 = tile.subview %block_idx_l2_7 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %134 = tile.subview %q_l2_7 task = %task_15 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %135 = tile.subview %state_l2_7 task = %task_15 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %136 = tile.subview %acc_l2_7 task = %task_15 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %137 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %138 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %139 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %140 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %141 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %142 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %143 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r1_t1_final = tile.load.async %133 into %137 : !tile.event<"bidx_k_r1_t1_final">
    %bidx_v_r1_t1_final = tile.load.async %133 into %138 : !tile.event<"bidx_v_r1_t1_final">
    %q_r1_t1_final = tile.load.async %134 into %141 : !tile.event<"q_r1_t1_final">
    %state_r1_t1_final = tile.load.async %135 into %142 : !tile.event<"state_r1_t1_final">
    %acc_r1_t1_final = tile.load.async %136 into %143 : !tile.event<"acc_r1_t1_final">
    tile.await %bidx_k_r1_t1_final, %bidx_v_r1_t1_final
    %gather_k_r1_t1_final = tile.gather.global.async %pool_15 indices(%137) into %139
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_k_r1_t1_final">
    %gather_v_r1_t1_final = tile.gather.global.async %pool_15 indices(%138) into %140
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_v_r1_t1_final">
    tile.await %gather_k_r1_t1_final, %gather_v_r1_t1_final, %q_r1_t1_final, %state_r1_t1_final,
      %acc_r1_t1_final
    tile.signal input_released(%task_15)
    %qk_r1_t1_final = tile.boa.async "matmul" m = 4 n = 1 k = 64 ops = 512
      : !tile.event<"qk_r1_t1_final">
    tile.await %qk_r1_t1_final
    %sm_r1_t1_final = tile.evu.async "online_softmax_update" ops = 12
      : !tile.event<"sm_r1_t1_final">
    tile.await %sm_r1_t1_final
    %pv_r1_t1_final = tile.boa.async "matmul" m = 4 n = 64 k = 1 ops = 512
      : !tile.event<"pv_r1_t1_final">
    tile.await %pv_r1_t1_final
    %acc_evu_r1_t1_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r1_t1_final">
    tile.await %acc_evu_r1_t1_final
    %norm_r1_t1_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r1_t1_final">
    tile.await %norm_r1_t1_final
    %state_store_r1_t1_final = tile.store.async %142 into %135
      : !tile.event<"state_store_r1_t1_final">
    %acc_store_r1_t1_final = tile.store.async %143 into %136 : !tile.event<"acc_store_r1_t1_final">
    tile.await %state_store_r1_t1_final, %acc_store_r1_t1_final
    tile.signal output_ready(%task_15)
    tile.free %137
    tile.free %138
    tile.free %139
    tile.free %140
    tile.free %141
    tile.free %142
    tile.free %143
    tile.return
  }
  tile.program @paged_attention_t2_final_r1(
    %task_16: !nest.task, %pool_16: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_8: !nest.l2_buffer<1xi32>, %q_l2_8: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_8: !nest.l2_buffer<4x4x2xf32>, %acc_l2_8: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %144 = tile.subview %block_idx_l2_8 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %145 = tile.subview %q_l2_8 task = %task_16 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %146 = tile.subview %state_l2_8 task = %task_16 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %147 = tile.subview %acc_l2_8 task = %task_16 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %148 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %149 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %150 = tile.alloc shape = [2, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x64xbf16>
    %151 = tile.alloc shape = [2, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x64xbf16>
    %152 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %153 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %154 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r1_t2_final = tile.load.async %144 into %148 : !tile.event<"bidx_k_r1_t2_final">
    %bidx_v_r1_t2_final = tile.load.async %144 into %149 : !tile.event<"bidx_v_r1_t2_final">
    %q_r1_t2_final = tile.load.async %145 into %152 : !tile.event<"q_r1_t2_final">
    %state_r1_t2_final = tile.load.async %146 into %153 : !tile.event<"state_r1_t2_final">
    %acc_r1_t2_final = tile.load.async %147 into %154 : !tile.event<"acc_r1_t2_final">
    tile.await %bidx_k_r1_t2_final, %bidx_v_r1_t2_final
    %gather_k_r1_t2_final = tile.gather.global.async %pool_16 indices(%148) into %150
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 128>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_k_r1_t2_final">
    %gather_v_r1_t2_final = tile.gather.global.async %pool_16 indices(%149) into %151
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 128>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_v_r1_t2_final">
    tile.await %gather_k_r1_t2_final, %gather_v_r1_t2_final, %q_r1_t2_final, %state_r1_t2_final,
      %acc_r1_t2_final
    tile.signal input_released(%task_16)
    %qk_r1_t2_final = tile.boa.async "matmul" m = 4 n = 2 k = 64 ops = 1024
      : !tile.event<"qk_r1_t2_final">
    tile.await %qk_r1_t2_final
    %sm_r1_t2_final = tile.evu.async "online_softmax_update" ops = 16
      : !tile.event<"sm_r1_t2_final">
    tile.await %sm_r1_t2_final
    %pv_r1_t2_final = tile.boa.async "matmul" m = 4 n = 64 k = 2 ops = 1024
      : !tile.event<"pv_r1_t2_final">
    tile.await %pv_r1_t2_final
    %acc_evu_r1_t2_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r1_t2_final">
    tile.await %acc_evu_r1_t2_final
    %norm_r1_t2_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r1_t2_final">
    tile.await %norm_r1_t2_final
    %state_store_r1_t2_final = tile.store.async %153 into %146
      : !tile.event<"state_store_r1_t2_final">
    %acc_store_r1_t2_final = tile.store.async %154 into %147 : !tile.event<"acc_store_r1_t2_final">
    tile.await %state_store_r1_t2_final, %acc_store_r1_t2_final
    tile.signal output_ready(%task_16)
    tile.free %148
    tile.free %149
    tile.free %150
    tile.free %151
    tile.free %152
    tile.free %153
    tile.free %154
    tile.return
  }
  tile.program @paged_attention_t3_final_r1(
    %task_17: !nest.task, %pool_17: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_9: !nest.l2_buffer<1xi32>, %q_l2_9: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_9: !nest.l2_buffer<4x4x2xf32>, %acc_l2_9: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %155 = tile.subview %block_idx_l2_9 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %156 = tile.subview %q_l2_9 task = %task_17 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %157 = tile.subview %state_l2_9 task = %task_17 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %158 = tile.subview %acc_l2_9 task = %task_17 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %159 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %160 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %161 = tile.alloc shape = [3, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x64xbf16>
    %162 = tile.alloc shape = [3, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x64xbf16>
    %163 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %164 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %165 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r1_t3_final = tile.load.async %155 into %159 : !tile.event<"bidx_k_r1_t3_final">
    %bidx_v_r1_t3_final = tile.load.async %155 into %160 : !tile.event<"bidx_v_r1_t3_final">
    %q_r1_t3_final = tile.load.async %156 into %163 : !tile.event<"q_r1_t3_final">
    %state_r1_t3_final = tile.load.async %157 into %164 : !tile.event<"state_r1_t3_final">
    %acc_r1_t3_final = tile.load.async %158 into %165 : !tile.event<"acc_r1_t3_final">
    tile.await %bidx_k_r1_t3_final, %bidx_v_r1_t3_final
    %gather_k_r1_t3_final = tile.gather.global.async %pool_17 indices(%159) into %161
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 192>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_k_r1_t3_final">
    %gather_v_r1_t3_final = tile.gather.global.async %pool_17 indices(%160) into %162
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 192>
      window_entries = 1 scope = "owner_1" : !tile.event<"gather_v_r1_t3_final">
    tile.await %gather_k_r1_t3_final, %gather_v_r1_t3_final, %q_r1_t3_final, %state_r1_t3_final,
      %acc_r1_t3_final
    tile.signal input_released(%task_17)
    %qk_r1_t3_final = tile.boa.async "matmul" m = 4 n = 3 k = 64 ops = 1536
      : !tile.event<"qk_r1_t3_final">
    tile.await %qk_r1_t3_final
    %sm_r1_t3_final = tile.evu.async "online_softmax_update" ops = 20
      : !tile.event<"sm_r1_t3_final">
    tile.await %sm_r1_t3_final
    %pv_r1_t3_final = tile.boa.async "matmul" m = 4 n = 64 k = 3 ops = 1536
      : !tile.event<"pv_r1_t3_final">
    tile.await %pv_r1_t3_final
    %acc_evu_r1_t3_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r1_t3_final">
    tile.await %acc_evu_r1_t3_final
    %norm_r1_t3_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r1_t3_final">
    tile.await %norm_r1_t3_final
    %state_store_r1_t3_final = tile.store.async %164 into %157
      : !tile.event<"state_store_r1_t3_final">
    %acc_store_r1_t3_final = tile.store.async %165 into %158 : !tile.event<"acc_store_r1_t3_final">
    tile.await %state_store_r1_t3_final, %acc_store_r1_t3_final
    tile.signal output_ready(%task_17)
    tile.free %159
    tile.free %160
    tile.free %161
    tile.free %162
    tile.free %163
    tile.free %164
    tile.free %165
    tile.return
  }
  tile.program @paged_attention_append_r2_tip0(
    %task_18: !nest.task, %pool_18: !nest.global_view<128x8224xbf16>,
    %k_new_l2_8: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_8: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_8: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %166 = tile.subview %k_new_l2_8 task = %task_18 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %167 = tile.subview %v_new_l2_8 task = %task_18 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %168 = tile.subview %append_idx_l2_8 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %169 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %170 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %171 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %172 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r2_tip0 = tile.load.async %168 into %171 : !tile.event<"aidx_k_r2_tip0">
    %aidx_v_r2_tip0 = tile.load.async %168 into %172 : !tile.event<"aidx_v_r2_tip0">
    %k_row_r2_tip0 = tile.load.async %166 into %169 : !tile.event<"k_row_r2_tip0">
    %v_row_r2_tip0 = tile.load.async %167 into %170 : !tile.event<"v_row_r2_tip0">
    tile.await %aidx_k_r2_tip0, %aidx_v_r2_tip0, %k_row_r2_tip0, %v_row_r2_tip0
    tile.signal input_released(%task_18)
    %scatter_k_r2_tip0 = tile.scatter.global.async %169 indices(%171) into %pool_18
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_k_r2_tip0">
    %scatter_v_r2_tip0 = tile.scatter.global.async %170 indices(%172) into %pool_18
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_v_r2_tip0">
    tile.await %scatter_k_r2_tip0, %scatter_v_r2_tip0
    tile.free %169
    tile.free %170
    tile.free %171
    tile.free %172
    tile.return
  }
  tile.program @paged_attention_append_r2_tip1(
    %task_19: !nest.task, %pool_19: !nest.global_view<128x8224xbf16>,
    %k_new_l2_9: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_9: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_9: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %173 = tile.subview %k_new_l2_9 task = %task_19 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %174 = tile.subview %v_new_l2_9 task = %task_19 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %175 = tile.subview %append_idx_l2_9 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %176 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %177 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %178 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %179 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r2_tip1 = tile.load.async %175 into %178 : !tile.event<"aidx_k_r2_tip1">
    %aidx_v_r2_tip1 = tile.load.async %175 into %179 : !tile.event<"aidx_v_r2_tip1">
    %k_row_r2_tip1 = tile.load.async %173 into %176 : !tile.event<"k_row_r2_tip1">
    %v_row_r2_tip1 = tile.load.async %174 into %177 : !tile.event<"v_row_r2_tip1">
    tile.await %aidx_k_r2_tip1, %aidx_v_r2_tip1, %k_row_r2_tip1, %v_row_r2_tip1
    tile.signal input_released(%task_19)
    %scatter_k_r2_tip1 = tile.scatter.global.async %176 indices(%178) into %pool_19
      map = #tile.indexed_map< index_scale = 8224 offset = 256 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_k_r2_tip1">
    %scatter_v_r2_tip1 = tile.scatter.global.async %177 indices(%179) into %pool_19
      map = #tile.indexed_map< index_scale = 8224 offset = 4352 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_v_r2_tip1">
    tile.await %scatter_k_r2_tip1, %scatter_v_r2_tip1
    tile.free %176
    tile.free %177
    tile.free %178
    tile.free %179
    tile.return
  }
  tile.program @paged_attention_append_r2_tip2(
    %task_20: !nest.task, %pool_20: !nest.global_view<128x8224xbf16>,
    %k_new_l2_10: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_10: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_10: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %180 = tile.subview %k_new_l2_10 task = %task_20 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %181 = tile.subview %v_new_l2_10 task = %task_20 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %182 = tile.subview %append_idx_l2_10 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %183 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %184 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %185 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %186 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r2_tip2 = tile.load.async %182 into %185 : !tile.event<"aidx_k_r2_tip2">
    %aidx_v_r2_tip2 = tile.load.async %182 into %186 : !tile.event<"aidx_v_r2_tip2">
    %k_row_r2_tip2 = tile.load.async %180 into %183 : !tile.event<"k_row_r2_tip2">
    %v_row_r2_tip2 = tile.load.async %181 into %184 : !tile.event<"v_row_r2_tip2">
    tile.await %aidx_k_r2_tip2, %aidx_v_r2_tip2, %k_row_r2_tip2, %v_row_r2_tip2
    tile.signal input_released(%task_20)
    %scatter_k_r2_tip2 = tile.scatter.global.async %183 indices(%185) into %pool_20
      map = #tile.indexed_map< index_scale = 8224 offset = 512 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_k_r2_tip2">
    %scatter_v_r2_tip2 = tile.scatter.global.async %184 indices(%186) into %pool_20
      map = #tile.indexed_map< index_scale = 8224 offset = 4608 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_v_r2_tip2">
    tile.await %scatter_k_r2_tip2, %scatter_v_r2_tip2
    tile.free %183
    tile.free %184
    tile.free %185
    tile.free %186
    tile.return
  }
  tile.program @paged_attention_append_r2_tip15(
    %task_21: !nest.task, %pool_21: !nest.global_view<128x8224xbf16>,
    %k_new_l2_11: !nest.l2_buffer<4x1x64xbf16>, %v_new_l2_11: !nest.l2_buffer<4x1x64xbf16>,
    %append_idx_l2_11: !nest.l2_buffer<1xi32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 16384,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %187 = tile.subview %k_new_l2_11 task = %task_21 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %188 = tile.subview %v_new_l2_11 task = %task_21 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 1, 64] strides = [1, 1, 1] : !nest.l2_view<1x1x64xbf16>
    %189 = tile.subview %append_idx_l2_11 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %190 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %191 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %192 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %193 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %aidx_k_r2_tip15 = tile.load.async %189 into %192 : !tile.event<"aidx_k_r2_tip15">
    %aidx_v_r2_tip15 = tile.load.async %189 into %193 : !tile.event<"aidx_v_r2_tip15">
    %k_row_r2_tip15 = tile.load.async %187 into %190 : !tile.event<"k_row_r2_tip15">
    %v_row_r2_tip15 = tile.load.async %188 into %191 : !tile.event<"v_row_r2_tip15">
    tile.await %aidx_k_r2_tip15, %aidx_v_r2_tip15, %k_row_r2_tip15, %v_row_r2_tip15
    tile.signal input_released(%task_21)
    %scatter_k_r2_tip15 = tile.scatter.global.async %190 indices(%192) into %pool_21
      map = #tile.indexed_map< index_scale = 8224 offset = 3840 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_k_r2_tip15">
    %scatter_v_r2_tip15 = tile.scatter.global.async %191 indices(%193) into %pool_21
      map = #tile.indexed_map< index_scale = 8224 offset = 7936 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"scatter_v_r2_tip15">
    tile.await %scatter_k_r2_tip15, %scatter_v_r2_tip15
    tile.free %190
    tile.free %191
    tile.free %192
    tile.free %193
    tile.return
  }
  tile.program @paged_attention_t16_step_r2(
    %task_22: !nest.task, %pool_22: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_10: !nest.l2_buffer<1xi32>, %q_l2_10: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_10: !nest.l2_buffer<4x4x2xf32>, %acc_l2_10: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %194 = tile.subview %block_idx_l2_10 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %195 = tile.subview %q_l2_10 task = %task_22 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %196 = tile.subview %state_l2_10 task = %task_22 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %197 = tile.subview %acc_l2_10 task = %task_22 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %198 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %199 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %200 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %201 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %202 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %203 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %204 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r2_t16 = tile.load.async %194 into %198 : !tile.event<"bidx_k_r2_t16">
    %bidx_v_r2_t16 = tile.load.async %194 into %199 : !tile.event<"bidx_v_r2_t16">
    %q_r2_t16 = tile.load.async %195 into %202 : !tile.event<"q_r2_t16">
    %state_r2_t16 = tile.load.async %196 into %203 : !tile.event<"state_r2_t16">
    %acc_r2_t16 = tile.load.async %197 into %204 : !tile.event<"acc_r2_t16">
    tile.await %bidx_k_r2_t16, %bidx_v_r2_t16
    %gather_k_r2_t16 = tile.gather.global.async %pool_22 indices(%198) into %200
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_k_r2_t16">
    %gather_v_r2_t16 = tile.gather.global.async %pool_22 indices(%199) into %201
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_v_r2_t16">
    tile.await %gather_k_r2_t16, %gather_v_r2_t16, %q_r2_t16, %state_r2_t16, %acc_r2_t16
    tile.signal input_released(%task_22)
    %qk_r2_t16 = tile.boa.async "matmul" m = 4 n = 16 k = 64 ops = 8192 : !tile.event<"qk_r2_t16">
    tile.await %qk_r2_t16
    %sm_r2_t16 = tile.evu.async "online_softmax_update" ops = 72 : !tile.event<"sm_r2_t16">
    tile.await %sm_r2_t16
    %pv_r2_t16 = tile.boa.async "matmul" m = 4 n = 64 k = 16 ops = 8192 : !tile.event<"pv_r2_t16">
    tile.await %pv_r2_t16
    %acc_evu_r2_t16 = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r2_t16">
    tile.await %acc_evu_r2_t16
    %state_store_r2_t16 = tile.store.async %203 into %196 : !tile.event<"state_store_r2_t16">
    %acc_store_r2_t16 = tile.store.async %204 into %197 : !tile.event<"acc_store_r2_t16">
    tile.await %state_store_r2_t16, %acc_store_r2_t16
    tile.signal output_ready(%task_22)
    tile.free %198
    tile.free %199
    tile.free %200
    tile.free %201
    tile.free %202
    tile.free %203
    tile.free %204
    tile.return
  }
  tile.program @paged_attention_t16_final_r2(
    %task_23: !nest.task, %pool_23: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_11: !nest.l2_buffer<1xi32>, %q_l2_11: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_11: !nest.l2_buffer<4x4x2xf32>, %acc_l2_11: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %205 = tile.subview %block_idx_l2_11 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %206 = tile.subview %q_l2_11 task = %task_23 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %207 = tile.subview %state_l2_11 task = %task_23 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %208 = tile.subview %acc_l2_11 task = %task_23 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %209 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %210 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %211 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %212 = tile.alloc shape = [16, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x64xbf16>
    %213 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %214 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %215 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r2_t16_final = tile.load.async %205 into %209 : !tile.event<"bidx_k_r2_t16_final">
    %bidx_v_r2_t16_final = tile.load.async %205 into %210 : !tile.event<"bidx_v_r2_t16_final">
    %q_r2_t16_final = tile.load.async %206 into %213 : !tile.event<"q_r2_t16_final">
    %state_r2_t16_final = tile.load.async %207 into %214 : !tile.event<"state_r2_t16_final">
    %acc_r2_t16_final = tile.load.async %208 into %215 : !tile.event<"acc_r2_t16_final">
    tile.await %bidx_k_r2_t16_final, %bidx_v_r2_t16_final
    %gather_k_r2_t16_final = tile.gather.global.async %pool_23 indices(%209) into %211
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_k_r2_t16_final">
    %gather_v_r2_t16_final = tile.gather.global.async %pool_23 indices(%210) into %212
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 1024>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_v_r2_t16_final">
    tile.await %gather_k_r2_t16_final, %gather_v_r2_t16_final, %q_r2_t16_final, %state_r2_t16_final,
      %acc_r2_t16_final
    tile.signal input_released(%task_23)
    %qk_r2_t16_final = tile.boa.async "matmul" m = 4 n = 16 k = 64 ops = 8192
      : !tile.event<"qk_r2_t16_final">
    tile.await %qk_r2_t16_final
    %sm_r2_t16_final = tile.evu.async "online_softmax_update" ops = 72
      : !tile.event<"sm_r2_t16_final">
    tile.await %sm_r2_t16_final
    %pv_r2_t16_final = tile.boa.async "matmul" m = 4 n = 64 k = 16 ops = 8192
      : !tile.event<"pv_r2_t16_final">
    tile.await %pv_r2_t16_final
    %acc_evu_r2_t16_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r2_t16_final">
    tile.await %acc_evu_r2_t16_final
    %norm_r2_t16_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r2_t16_final">
    tile.await %norm_r2_t16_final
    %state_store_r2_t16_final = tile.store.async %214 into %207
      : !tile.event<"state_store_r2_t16_final">
    %acc_store_r2_t16_final = tile.store.async %215 into %208
      : !tile.event<"acc_store_r2_t16_final">
    tile.await %state_store_r2_t16_final, %acc_store_r2_t16_final
    tile.signal output_ready(%task_23)
    tile.free %209
    tile.free %210
    tile.free %211
    tile.free %212
    tile.free %213
    tile.free %214
    tile.free %215
    tile.return
  }
  tile.program @paged_attention_t1_final_r2(
    %task_24: !nest.task, %pool_24: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_12: !nest.l2_buffer<1xi32>, %q_l2_12: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_12: !nest.l2_buffer<4x4x2xf32>, %acc_l2_12: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %216 = tile.subview %block_idx_l2_12 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %217 = tile.subview %q_l2_12 task = %task_24 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %218 = tile.subview %state_l2_12 task = %task_24 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %219 = tile.subview %acc_l2_12 task = %task_24 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %220 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %221 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %222 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %223 = tile.alloc shape = [1, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<1x64xbf16>
    %224 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %225 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %226 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r2_t1_final = tile.load.async %216 into %220 : !tile.event<"bidx_k_r2_t1_final">
    %bidx_v_r2_t1_final = tile.load.async %216 into %221 : !tile.event<"bidx_v_r2_t1_final">
    %q_r2_t1_final = tile.load.async %217 into %224 : !tile.event<"q_r2_t1_final">
    %state_r2_t1_final = tile.load.async %218 into %225 : !tile.event<"state_r2_t1_final">
    %acc_r2_t1_final = tile.load.async %219 into %226 : !tile.event<"acc_r2_t1_final">
    tile.await %bidx_k_r2_t1_final, %bidx_v_r2_t1_final
    %gather_k_r2_t1_final = tile.gather.global.async %pool_24 indices(%220) into %222
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_k_r2_t1_final">
    %gather_v_r2_t1_final = tile.gather.global.async %pool_24 indices(%221) into %223
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 64>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_v_r2_t1_final">
    tile.await %gather_k_r2_t1_final, %gather_v_r2_t1_final, %q_r2_t1_final, %state_r2_t1_final,
      %acc_r2_t1_final
    tile.signal input_released(%task_24)
    %qk_r2_t1_final = tile.boa.async "matmul" m = 4 n = 1 k = 64 ops = 512
      : !tile.event<"qk_r2_t1_final">
    tile.await %qk_r2_t1_final
    %sm_r2_t1_final = tile.evu.async "online_softmax_update" ops = 12
      : !tile.event<"sm_r2_t1_final">
    tile.await %sm_r2_t1_final
    %pv_r2_t1_final = tile.boa.async "matmul" m = 4 n = 64 k = 1 ops = 512
      : !tile.event<"pv_r2_t1_final">
    tile.await %pv_r2_t1_final
    %acc_evu_r2_t1_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r2_t1_final">
    tile.await %acc_evu_r2_t1_final
    %norm_r2_t1_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r2_t1_final">
    tile.await %norm_r2_t1_final
    %state_store_r2_t1_final = tile.store.async %225 into %218
      : !tile.event<"state_store_r2_t1_final">
    %acc_store_r2_t1_final = tile.store.async %226 into %219 : !tile.event<"acc_store_r2_t1_final">
    tile.await %state_store_r2_t1_final, %acc_store_r2_t1_final
    tile.signal output_ready(%task_24)
    tile.free %220
    tile.free %221
    tile.free %222
    tile.free %223
    tile.free %224
    tile.free %225
    tile.free %226
    tile.return
  }
  tile.program @paged_attention_t2_final_r2(
    %task_25: !nest.task, %pool_25: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_13: !nest.l2_buffer<1xi32>, %q_l2_13: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_13: !nest.l2_buffer<4x4x2xf32>, %acc_l2_13: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %227 = tile.subview %block_idx_l2_13 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %228 = tile.subview %q_l2_13 task = %task_25 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %229 = tile.subview %state_l2_13 task = %task_25 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %230 = tile.subview %acc_l2_13 task = %task_25 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %231 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %232 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %233 = tile.alloc shape = [2, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x64xbf16>
    %234 = tile.alloc shape = [2, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x64xbf16>
    %235 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %236 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %237 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r2_t2_final = tile.load.async %227 into %231 : !tile.event<"bidx_k_r2_t2_final">
    %bidx_v_r2_t2_final = tile.load.async %227 into %232 : !tile.event<"bidx_v_r2_t2_final">
    %q_r2_t2_final = tile.load.async %228 into %235 : !tile.event<"q_r2_t2_final">
    %state_r2_t2_final = tile.load.async %229 into %236 : !tile.event<"state_r2_t2_final">
    %acc_r2_t2_final = tile.load.async %230 into %237 : !tile.event<"acc_r2_t2_final">
    tile.await %bidx_k_r2_t2_final, %bidx_v_r2_t2_final
    %gather_k_r2_t2_final = tile.gather.global.async %pool_25 indices(%231) into %233
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 128>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_k_r2_t2_final">
    %gather_v_r2_t2_final = tile.gather.global.async %pool_25 indices(%232) into %234
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 128>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_v_r2_t2_final">
    tile.await %gather_k_r2_t2_final, %gather_v_r2_t2_final, %q_r2_t2_final, %state_r2_t2_final,
      %acc_r2_t2_final
    tile.signal input_released(%task_25)
    %qk_r2_t2_final = tile.boa.async "matmul" m = 4 n = 2 k = 64 ops = 1024
      : !tile.event<"qk_r2_t2_final">
    tile.await %qk_r2_t2_final
    %sm_r2_t2_final = tile.evu.async "online_softmax_update" ops = 16
      : !tile.event<"sm_r2_t2_final">
    tile.await %sm_r2_t2_final
    %pv_r2_t2_final = tile.boa.async "matmul" m = 4 n = 64 k = 2 ops = 1024
      : !tile.event<"pv_r2_t2_final">
    tile.await %pv_r2_t2_final
    %acc_evu_r2_t2_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r2_t2_final">
    tile.await %acc_evu_r2_t2_final
    %norm_r2_t2_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r2_t2_final">
    tile.await %norm_r2_t2_final
    %state_store_r2_t2_final = tile.store.async %236 into %229
      : !tile.event<"state_store_r2_t2_final">
    %acc_store_r2_t2_final = tile.store.async %237 into %230 : !tile.event<"acc_store_r2_t2_final">
    tile.await %state_store_r2_t2_final, %acc_store_r2_t2_final
    tile.signal output_ready(%task_25)
    tile.free %231
    tile.free %232
    tile.free %233
    tile.free %234
    tile.free %235
    tile.free %236
    tile.free %237
    tile.return
  }
  tile.program @paged_attention_t3_final_r2(
    %task_26: !nest.task, %pool_26: !nest.global_view<128x8224xbf16>,
    %block_idx_l2_14: !nest.l2_buffer<1xi32>, %q_l2_14: !nest.l2_buffer<4x4x64xbf16>,
    %state_l2_14: !nest.l2_buffer<4x4x2xf32>, %acc_l2_14: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 28672,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %238 = tile.subview %block_idx_l2_14 offsets = [0] sizes = [1] strides = [1]
      : !nest.l2_view<1xi32>
    %239 = tile.subview %q_l2_14 task = %task_26 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xbf16>
    %240 = tile.subview %state_l2_14 task = %task_26 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 2] strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %241 = tile.subview %acc_l2_14 task = %task_26 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 4, 64] strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %242 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %243 = tile.alloc shape = [1] dtype = "i32" alignment = 64 : !tile.l1_buffer<1xi32>
    %244 = tile.alloc shape = [3, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x64xbf16>
    %245 = tile.alloc shape = [3, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x64xbf16>
    %246 = tile.alloc shape = [4, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<4x64xbf16>
    %247 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %248 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %bidx_k_r2_t3_final = tile.load.async %238 into %242 : !tile.event<"bidx_k_r2_t3_final">
    %bidx_v_r2_t3_final = tile.load.async %238 into %243 : !tile.event<"bidx_v_r2_t3_final">
    %q_r2_t3_final = tile.load.async %239 into %246 : !tile.event<"q_r2_t3_final">
    %state_r2_t3_final = tile.load.async %240 into %247 : !tile.event<"state_r2_t3_final">
    %acc_r2_t3_final = tile.load.async %241 into %248 : !tile.event<"acc_r2_t3_final">
    tile.await %bidx_k_r2_t3_final, %bidx_v_r2_t3_final
    %gather_k_r2_t3_final = tile.gather.global.async %pool_26 indices(%242) into %244
      map = #tile.indexed_map< index_scale = 8224 offset = 0 task_stride = 64 repeat = 1 stride = 0 segment = 192>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_k_r2_t3_final">
    %gather_v_r2_t3_final = tile.gather.global.async %pool_26 indices(%243) into %245
      map = #tile.indexed_map< index_scale = 8224 offset = 4096 task_stride = 64 repeat = 1 stride = 0 segment = 192>
      window_entries = 1 scope = "owner_2" : !tile.event<"gather_v_r2_t3_final">
    tile.await %gather_k_r2_t3_final, %gather_v_r2_t3_final, %q_r2_t3_final, %state_r2_t3_final,
      %acc_r2_t3_final
    tile.signal input_released(%task_26)
    %qk_r2_t3_final = tile.boa.async "matmul" m = 4 n = 3 k = 64 ops = 1536
      : !tile.event<"qk_r2_t3_final">
    tile.await %qk_r2_t3_final
    %sm_r2_t3_final = tile.evu.async "online_softmax_update" ops = 20
      : !tile.event<"sm_r2_t3_final">
    tile.await %sm_r2_t3_final
    %pv_r2_t3_final = tile.boa.async "matmul" m = 4 n = 64 k = 3 ops = 1536
      : !tile.event<"pv_r2_t3_final">
    tile.await %pv_r2_t3_final
    %acc_evu_r2_t3_final = tile.evu.async "online_softmax_output_accumulate" ops = 512
      : !tile.event<"acc_evu_r2_t3_final">
    tile.await %acc_evu_r2_t3_final
    %norm_r2_t3_final = tile.evu.async "normalize_attention_output" ops = 256
      : !tile.event<"norm_r2_t3_final">
    tile.await %norm_r2_t3_final
    %state_store_r2_t3_final = tile.store.async %247 into %240
      : !tile.event<"state_store_r2_t3_final">
    %acc_store_r2_t3_final = tile.store.async %248 into %241 : !tile.event<"acc_store_r2_t3_final">
    tile.await %state_store_r2_t3_final, %acc_store_r2_t3_final
    tile.signal output_ready(%task_26)
    tile.free %242
    tile.free %243
    tile.free %244
    tile.free %245
    tile.free %246
    tile.free %247
    tile.free %248
    tile.return
  }
  tile.program @paged_attention_merge_p4(
    %task_27: !nest.task, %state_p0: !nest.l2_buffer<4x4x2xf32>,
    %state_p1: !nest.l2_buffer<4x4x2xf32>, %state_p2: !nest.l2_buffer<4x4x2xf32>,
    %state_p3: !nest.l2_buffer<4x4x2xf32>, %acc_p0: !nest.l2_buffer<4x4x64xf32>,
    %acc_p1: !nest.l2_buffer<4x4x64xf32>, %acc_p2: !nest.l2_buffer<4x4x64xf32>,
    %acc_p3: !nest.l2_buffer<4x4x64xf32>, %out_l2: !nest.l2_buffer<4x4x64xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 9216,
      l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %249 = tile.subview %state_p0 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %250 = tile.subview %state_p1 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %251 = tile.subview %state_p2 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %252 = tile.subview %state_p3 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 2]
      strides = [1, 1, 1] : !nest.l2_view<1x4x2xf32>
    %253 = tile.subview %acc_p0 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %254 = tile.subview %acc_p1 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %255 = tile.subview %acc_p2 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %256 = tile.subview %acc_p3 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %257 = tile.subview %out_l2 task = %task_27 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 4, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x4x64xf32>
    %258 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %259 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %260 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %261 = tile.alloc shape = [4, 2] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x2xf32>
    %262 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %263 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %264 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %265 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %266 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 64 : !tile.l1_buffer<4x64xf32>
    %mstate_p0_merge = tile.load.async %249 into %258 : !tile.event<"mstate_p0_merge">
    %macc_p0_merge = tile.load.async %253 into %262 : !tile.event<"macc_p0_merge">
    %mstate_p1_merge = tile.load.async %250 into %259 : !tile.event<"mstate_p1_merge">
    %macc_p1_merge = tile.load.async %254 into %263 : !tile.event<"macc_p1_merge">
    %mstate_p2_merge = tile.load.async %251 into %260 : !tile.event<"mstate_p2_merge">
    %macc_p2_merge = tile.load.async %255 into %264 : !tile.event<"macc_p2_merge">
    %mstate_p3_merge = tile.load.async %252 into %261 : !tile.event<"mstate_p3_merge">
    %macc_p3_merge = tile.load.async %256 into %265 : !tile.event<"macc_p3_merge">
    %mout_merge = tile.load.async %256 into %266 : !tile.event<"mout_merge">
    tile.await %mstate_p0_merge, %macc_p0_merge, %mstate_p1_merge, %macc_p1_merge, %mstate_p2_merge,
      %macc_p2_merge, %mstate_p3_merge, %macc_p3_merge, %mout_merge
    tile.signal input_released(%task_27)
    %merge_merge = tile.evu.async "online_softmax_merge" ops = 2120 : !tile.event<"merge_merge">
    tile.await %merge_merge
    %norm_merge = tile.evu.async "normalize" ops = 256 : !tile.event<"norm_merge">
    tile.await %norm_merge
    %out_merge = tile.store.async %266 into %257 : !tile.event<"out_merge">
    tile.await %out_merge
    tile.signal output_ready(%task_27)
    tile.free %258
    tile.free %259
    tile.free %260
    tile.free %261
    tile.free %262
    tile.free %263
    tile.free %264
    tile.free %265
    tile.free %266
    tile.return
  }
  nest.context @step_r0_s0(
    %POOL: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE: !nest.global_memref<147xi32>,
    %APPEND_IDS: !nest.global_memref<12xi32>, %Q_IN: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW: !nest.global_memref<3x4x4x1x64xbf16>, %V_NEW: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT: !nest.global_memref<3x4x4x4x4x2xf32>, %O_INIT: !nest.global_memref<3x4x4x4x4x64xf32>,
    %OUT: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 72, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x1x64xbf16>
    %v_new = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x1x64xbf16>
    %append_idx = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_15 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_1 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_1 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_1 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_1 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_1 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_1 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_1 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_1 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_1 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %267 = nest.subview %POOL offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %268 = nest.subview %K_NEW offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %269 = nest.subview %V_NEW offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %270 = nest.subview %APPEND_IDS offsets = [0] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %271 = nest.subview %Q_IN offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %272 = nest.subview %S_INIT offsets = [0, 0, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %273 = nest.subview %S_INIT offsets = [0, 0, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %274 = nest.subview %S_INIT offsets = [0, 0, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %275 = nest.subview %S_INIT offsets = [0, 0, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %276 = nest.subview %O_INIT offsets = [0, 0, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %277 = nest.subview %O_INIT offsets = [0, 0, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %278 = nest.subview %O_INIT offsets = [0, 0, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %279 = nest.subview %O_INIT offsets = [0, 0, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %280 = nest.subview %OUT offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %281 = nest.subview %BLOCK_TABLE offsets = [0] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %282 = nest.subview %BLOCK_TABLE offsets = [1] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %283 = nest.subview %BLOCK_TABLE offsets = [2] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %284 = nest.subview %BLOCK_TABLE offsets = [3] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %285 = nest.subview %BLOCK_TABLE offsets = [4] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %286 = nest.subview %BLOCK_TABLE offsets = [5] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %287 = nest.subview %BLOCK_TABLE offsets = [6] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %288 = nest.subview %BLOCK_TABLE offsets = [7] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %289 = nest.subview %BLOCK_TABLE offsets = [8] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %290 = nest.subview %BLOCK_TABLE offsets = [9] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %291 = nest.subview %BLOCK_TABLE offsets = [10] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %292 = nest.subview %BLOCK_TABLE offsets = [11] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %293 = nest.subview %BLOCK_TABLE offsets = [12] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %294 = nest.subview %BLOCK_TABLE offsets = [13] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %295 = nest.subview %BLOCK_TABLE offsets = [14] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r0_s0 = nest.dma.prefetch.async %268 into %k_new : !nest.event<"pf_k_r0_s0">
    %pf_v_r0_s0 = nest.dma.prefetch.async %269 into %v_new : !nest.event<"pf_v_r0_s0">
    %pf_aidx_r0_s0 = nest.dma.prefetch.async %270 into %append_idx : !nest.event<"pf_aidx_r0_s0">
    %pf_q_r0_s0 = nest.dma.prefetch.async %271 into %q_l2_15 : !nest.event<"pf_q_r0_s0">
    %pf_state_p0_r0_s0 = nest.dma.prefetch.async %272 into %state_p0_1
      : !nest.event<"pf_state_p0_r0_s0">
    %pf_state_p1_r0_s0 = nest.dma.prefetch.async %273 into %state_p1_1
      : !nest.event<"pf_state_p1_r0_s0">
    %pf_state_p2_r0_s0 = nest.dma.prefetch.async %274 into %state_p2_1
      : !nest.event<"pf_state_p2_r0_s0">
    %pf_state_p3_r0_s0 = nest.dma.prefetch.async %275 into %state_p3_1
      : !nest.event<"pf_state_p3_r0_s0">
    %pf_acc_p0_r0_s0 = nest.dma.prefetch.async %276 into %acc_p0_1 : !nest.event<"pf_acc_p0_r0_s0">
    %pf_acc_p1_r0_s0 = nest.dma.prefetch.async %277 into %acc_p1_1 : !nest.event<"pf_acc_p1_r0_s0">
    %pf_acc_p2_r0_s0 = nest.dma.prefetch.async %278 into %acc_p2_1 : !nest.event<"pf_acc_p2_r0_s0">
    %pf_acc_p3_r0_s0 = nest.dma.prefetch.async %279 into %acc_p3_1 : !nest.event<"pf_acc_p3_r0_s0">
    %296 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r0_s0, %append_inrel_r0_s0, %297 =
      nest.dispatch.tasks.async @paged_attention_append_r0_tip15 l1_mode = 1 tasks(%296)
      globals(%267) bindings(%k_new, %v_new, %append_idx) ins(%k_new, %v_new, %append_idx) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r0_s0, %pf_v_r0_s0, %pf_aidx_r0_s0)
      : (!nest.event<"append_grid_r0_s0">, !nest.event<"append_inrel_r0_s0">, !nest.event<"">)
    %pf_bidx0_r0_s0 = nest.dma.prefetch.async %281 into %block_idx_p0
      : !nest.event<"pf_bidx0_r0_s0">
    %att0_grid_r0_s0, %att0_inrel_r0_s0, %att0_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1)
      ins(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1) outs(%state_p0_1, %acc_p0_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s0, %pf_state_p0_r0_s0, %pf_acc_p0_r0_s0, %pf_bidx0_r0_s0) : (
        !nest.event<"att0_grid_r0_s0">, !nest.event<"att0_inrel_r0_s0">,
        !nest.event<"att0_out_r0_s0">)
    %pf_bidx1_r0_s0 = nest.dma.prefetch.async %282 into %block_idx_p1
      : !nest.event<"pf_bidx1_r0_s0">
    %att1_grid_r0_s0, %att1_inrel_r0_s0, %att1_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1)
      ins(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1) outs(%state_p1_1, %acc_p1_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s0, %pf_state_p1_r0_s0, %pf_acc_p1_r0_s0, %pf_bidx1_r0_s0) : (
        !nest.event<"att1_grid_r0_s0">, !nest.event<"att1_inrel_r0_s0">,
        !nest.event<"att1_out_r0_s0">)
    %pf_bidx2_r0_s0 = nest.dma.prefetch.async %283 into %block_idx_p0 depends_on(%att0_inrel_r0_s0)
      : !nest.event<"pf_bidx2_r0_s0">
    %att2_grid_r0_s0, %att2_inrel_r0_s0, %att2_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1)
      ins(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1) outs(%state_p2_1, %acc_p2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s0, %pf_state_p2_r0_s0, %pf_acc_p2_r0_s0, %pf_bidx2_r0_s0) : (
        !nest.event<"att2_grid_r0_s0">, !nest.event<"att2_inrel_r0_s0">,
        !nest.event<"att2_out_r0_s0">)
    %pf_bidx3_r0_s0 = nest.dma.prefetch.async %284 into %block_idx_p1 depends_on(%att1_inrel_r0_s0)
      : !nest.event<"pf_bidx3_r0_s0">
    %att3_grid_r0_s0, %att3_inrel_r0_s0, %att3_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p3_1, %acc_p3_1)
      ins(%block_idx_p1, %q_l2_15, %state_p3_1, %acc_p3_1) outs(%state_p3_1, %acc_p3_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s0, %pf_state_p3_r0_s0, %pf_acc_p3_r0_s0, %pf_bidx3_r0_s0) : (
        !nest.event<"att3_grid_r0_s0">, !nest.event<"att3_inrel_r0_s0">,
        !nest.event<"att3_out_r0_s0">)
    %pf_bidx4_r0_s0 = nest.dma.prefetch.async %285 into %block_idx_p0 depends_on(%att2_inrel_r0_s0)
      : !nest.event<"pf_bidx4_r0_s0">
    %att4_grid_r0_s0, %att4_inrel_r0_s0, %att4_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1)
      ins(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1) outs(%state_p0_1, %acc_p0_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p0_r0_s0, %pf_acc_p0_r0_s0, %pf_bidx4_r0_s0, %att0_out_r0_s0)
      : (
        !nest.event<"att4_grid_r0_s0">, !nest.event<"att4_inrel_r0_s0">,
        !nest.event<"att4_out_r0_s0">)
    %pf_bidx5_r0_s0 = nest.dma.prefetch.async %286 into %block_idx_p1 depends_on(%att3_inrel_r0_s0)
      : !nest.event<"pf_bidx5_r0_s0">
    %att5_grid_r0_s0, %att5_inrel_r0_s0, %att5_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1)
      ins(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1) outs(%state_p1_1, %acc_p1_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p1_r0_s0, %pf_acc_p1_r0_s0, %pf_bidx5_r0_s0, %att1_out_r0_s0)
      : (
        !nest.event<"att5_grid_r0_s0">, !nest.event<"att5_inrel_r0_s0">,
        !nest.event<"att5_out_r0_s0">)
    %pf_bidx6_r0_s0 = nest.dma.prefetch.async %287 into %block_idx_p0 depends_on(%att4_inrel_r0_s0)
      : !nest.event<"pf_bidx6_r0_s0">
    %att6_grid_r0_s0, %att6_inrel_r0_s0, %att6_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1)
      ins(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1) outs(%state_p2_1, %acc_p2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p2_r0_s0, %pf_acc_p2_r0_s0, %pf_bidx6_r0_s0, %att2_out_r0_s0)
      : (
        !nest.event<"att6_grid_r0_s0">, !nest.event<"att6_inrel_r0_s0">,
        !nest.event<"att6_out_r0_s0">)
    %pf_bidx7_r0_s0 = nest.dma.prefetch.async %288 into %block_idx_p1 depends_on(%att5_inrel_r0_s0)
      : !nest.event<"pf_bidx7_r0_s0">
    %att7_grid_r0_s0, %att7_inrel_r0_s0, %att7_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p3_1, %acc_p3_1)
      ins(%block_idx_p1, %q_l2_15, %state_p3_1, %acc_p3_1) outs(%state_p3_1, %acc_p3_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p3_r0_s0, %pf_acc_p3_r0_s0, %pf_bidx7_r0_s0, %att3_out_r0_s0)
      : (
        !nest.event<"att7_grid_r0_s0">, !nest.event<"att7_inrel_r0_s0">,
        !nest.event<"att7_out_r0_s0">)
    %pf_bidx8_r0_s0 = nest.dma.prefetch.async %289 into %block_idx_p0 depends_on(%att6_inrel_r0_s0)
      : !nest.event<"pf_bidx8_r0_s0">
    %att8_grid_r0_s0, %att8_inrel_r0_s0, %att8_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1)
      ins(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1) outs(%state_p0_1, %acc_p0_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p0_r0_s0, %pf_acc_p0_r0_s0, %pf_bidx8_r0_s0, %att4_out_r0_s0)
      : (
        !nest.event<"att8_grid_r0_s0">, !nest.event<"att8_inrel_r0_s0">,
        !nest.event<"att8_out_r0_s0">)
    %pf_bidx9_r0_s0 = nest.dma.prefetch.async %290 into %block_idx_p1 depends_on(%att7_inrel_r0_s0)
      : !nest.event<"pf_bidx9_r0_s0">
    %att9_grid_r0_s0, %att9_inrel_r0_s0, %att9_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1)
      ins(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1) outs(%state_p1_1, %acc_p1_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p1_r0_s0, %pf_acc_p1_r0_s0, %pf_bidx9_r0_s0, %att5_out_r0_s0)
      : (
        !nest.event<"att9_grid_r0_s0">, !nest.event<"att9_inrel_r0_s0">,
        !nest.event<"att9_out_r0_s0">)
    %pf_bidx10_r0_s0 = nest.dma.prefetch.async %291 into %block_idx_p0 depends_on(%att8_inrel_r0_s0)
      : !nest.event<"pf_bidx10_r0_s0">
    %att10_grid_r0_s0, %att10_inrel_r0_s0, %att10_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1)
      ins(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1) outs(%state_p2_1, %acc_p2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p2_r0_s0, %pf_acc_p2_r0_s0, %pf_bidx10_r0_s0, %att6_out_r0_s0)
      : (
        !nest.event<"att10_grid_r0_s0">, !nest.event<"att10_inrel_r0_s0">,
        !nest.event<"att10_out_r0_s0">)
    %pf_bidx11_r0_s0 = nest.dma.prefetch.async %292 into %block_idx_p1 depends_on(%att9_inrel_r0_s0)
      : !nest.event<"pf_bidx11_r0_s0">
    %att11_grid_r0_s0, %att11_inrel_r0_s0, %att11_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p3_1, %acc_p3_1)
      ins(%block_idx_p1, %q_l2_15, %state_p3_1, %acc_p3_1) outs(%state_p3_1, %acc_p3_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p3_r0_s0, %pf_acc_p3_r0_s0, %pf_bidx11_r0_s0, %att7_out_r0_s0)
      : (
        !nest.event<"att11_grid_r0_s0">, !nest.event<"att11_inrel_r0_s0">,
        !nest.event<"att11_out_r0_s0">)
    %pf_bidx12_r0_s0 = nest.dma.prefetch.async %293 into %block_idx_p0
      depends_on(%att10_inrel_r0_s0) : !nest.event<"pf_bidx12_r0_s0">
    %att12_grid_r0_s0, %att12_inrel_r0_s0, %att12_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1)
      ins(%block_idx_p0, %q_l2_15, %state_p0_1, %acc_p0_1) outs(%state_p0_1, %acc_p0_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p0_r0_s0, %pf_acc_p0_r0_s0, %pf_bidx12_r0_s0, %att8_out_r0_s0)
      : (
        !nest.event<"att12_grid_r0_s0">, !nest.event<"att12_inrel_r0_s0">,
        !nest.event<"att12_out_r0_s0">)
    %pf_bidx13_r0_s0 = nest.dma.prefetch.async %294 into %block_idx_p1
      depends_on(%att11_inrel_r0_s0) : !nest.event<"pf_bidx13_r0_s0">
    %att13_grid_r0_s0, %att13_inrel_r0_s0, %att13_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1)
      ins(%block_idx_p1, %q_l2_15, %state_p1_1, %acc_p1_1) outs(%state_p1_1, %acc_p1_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p1_r0_s0, %pf_acc_p1_r0_s0, %pf_bidx13_r0_s0, %att9_out_r0_s0)
      : (
        !nest.event<"att13_grid_r0_s0">, !nest.event<"att13_inrel_r0_s0">,
        !nest.event<"att13_out_r0_s0">)
    %pf_bidx14_r0_s0 = nest.dma.prefetch.async %295 into %block_idx_p0
      depends_on(%att12_inrel_r0_s0) : !nest.event<"pf_bidx14_r0_s0">
    %att14_grid_r0_s0, %att14_inrel_r0_s0, %att14_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1)
      ins(%block_idx_p0, %q_l2_15, %state_p2_1, %acc_p2_1) outs(%state_p2_1, %acc_p2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p2_r0_s0, %pf_acc_p2_r0_s0, %pf_bidx14_r0_s0, %att10_out_r0_s0)
      : (
        !nest.event<"att14_grid_r0_s0">, !nest.event<"att14_inrel_r0_s0">,
        !nest.event<"att14_out_r0_s0">)
    %att15_grid_r0_s0, %att15_inrel_r0_s0, %att15_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_final_r0 l1_mode = 1 tasks(%296) globals(%267)
      bindings(%append_idx, %q_l2_15, %state_p3_1, %acc_p3_1)
      ins(%append_idx, %q_l2_15, %state_p3_1, %acc_p3_1) outs(%state_p3_1, %acc_p3_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s0, %pf_state_p3_r0_s0, %pf_acc_p3_r0_s0, %pf_aidx_r0_s0, %append_grid_r0_s0,
        %att11_out_r0_s0)
      : (
        !nest.event<"att15_grid_r0_s0">, !nest.event<"att15_inrel_r0_s0">,
        !nest.event<"att15_out_r0_s0">)
    %merge_grid_r0_s0, %merge_inrel_r0_s0, %merge_out_r0_s0 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%296) globals()
      bindings(
        %state_p0_1, %state_p1_1, %state_p2_1, %state_p3_1, %acc_p0_1, %acc_p1_1, %acc_p2_1,
        %acc_p3_1, %out_l2_1)
      ins(
        %state_p0_1, %state_p1_1, %state_p2_1, %state_p3_1, %acc_p0_1, %acc_p1_1, %acc_p2_1,
        %acc_p3_1)
      outs(%out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att12_out_r0_s0, %att13_out_r0_s0, %att14_out_r0_s0, %att15_out_r0_s0) : (
        !nest.event<"merge_grid_r0_s0">, !nest.event<"merge_inrel_r0_s0">,
        !nest.event<"merge_out_r0_s0">)
    %out_store_r0_s0 = nest.dma.store.async %out_l2_1 into %280 depends_on(%merge_out_r0_s0)
      : !nest.event<"out_store_r0_s0">
    nest.await %append_grid_r0_s0, %att15_grid_r0_s0, %merge_grid_r0_s0, %out_store_r0_s0
    nest.release %block_idx_p0 depends_on(
      %pf_bidx0_r0_s0, %pf_bidx2_r0_s0, %pf_bidx4_r0_s0, %pf_bidx6_r0_s0, %pf_bidx8_r0_s0,
      %pf_bidx10_r0_s0, %pf_bidx12_r0_s0, %pf_bidx14_r0_s0, %att0_inrel_r0_s0, %att2_inrel_r0_s0,
      %att4_inrel_r0_s0, %att6_inrel_r0_s0, %att8_inrel_r0_s0, %att10_inrel_r0_s0,
      %att12_inrel_r0_s0, %att14_inrel_r0_s0)
    nest.release %block_idx_p1 depends_on(
      %pf_bidx1_r0_s0, %pf_bidx3_r0_s0, %pf_bidx5_r0_s0, %pf_bidx7_r0_s0, %pf_bidx9_r0_s0,
      %pf_bidx11_r0_s0, %pf_bidx13_r0_s0, %att1_inrel_r0_s0, %att3_inrel_r0_s0, %att5_inrel_r0_s0,
      %att7_inrel_r0_s0, %att9_inrel_r0_s0, %att11_inrel_r0_s0, %att13_inrel_r0_s0)
    nest.release %k_new depends_on(%pf_k_r0_s0, %append_inrel_r0_s0)
    nest.release %v_new depends_on(%pf_v_r0_s0, %append_inrel_r0_s0)
    nest.release %append_idx depends_on(%pf_aidx_r0_s0, %append_inrel_r0_s0, %att15_inrel_r0_s0)
    nest.release %q_l2_15 depends_on(
      %pf_q_r0_s0, %att0_inrel_r0_s0, %att1_inrel_r0_s0, %att2_inrel_r0_s0, %att3_inrel_r0_s0,
      %att4_inrel_r0_s0, %att5_inrel_r0_s0, %att6_inrel_r0_s0, %att7_inrel_r0_s0,
      %att8_inrel_r0_s0, %att9_inrel_r0_s0, %att10_inrel_r0_s0, %att11_inrel_r0_s0,
      %att12_inrel_r0_s0, %att13_inrel_r0_s0, %att14_inrel_r0_s0, %att15_inrel_r0_s0)
    nest.release %state_p0_1 depends_on(
      %pf_state_p0_r0_s0, %att0_inrel_r0_s0, %att4_inrel_r0_s0, %att8_inrel_r0_s0,
      %att12_inrel_r0_s0, %att0_out_r0_s0, %att4_out_r0_s0, %att8_out_r0_s0, %att12_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %acc_p0_1 depends_on(
      %pf_acc_p0_r0_s0, %att0_inrel_r0_s0, %att4_inrel_r0_s0, %att8_inrel_r0_s0,
      %att12_inrel_r0_s0, %att0_out_r0_s0, %att4_out_r0_s0, %att8_out_r0_s0, %att12_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %state_p1_1 depends_on(
      %pf_state_p1_r0_s0, %att1_inrel_r0_s0, %att5_inrel_r0_s0, %att9_inrel_r0_s0,
      %att13_inrel_r0_s0, %att1_out_r0_s0, %att5_out_r0_s0, %att9_out_r0_s0, %att13_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %acc_p1_1 depends_on(
      %pf_acc_p1_r0_s0, %att1_inrel_r0_s0, %att5_inrel_r0_s0, %att9_inrel_r0_s0,
      %att13_inrel_r0_s0, %att1_out_r0_s0, %att5_out_r0_s0, %att9_out_r0_s0, %att13_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %state_p2_1 depends_on(
      %pf_state_p2_r0_s0, %att2_inrel_r0_s0, %att6_inrel_r0_s0, %att10_inrel_r0_s0,
      %att14_inrel_r0_s0, %att2_out_r0_s0, %att6_out_r0_s0, %att10_out_r0_s0, %att14_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %acc_p2_1 depends_on(
      %pf_acc_p2_r0_s0, %att2_inrel_r0_s0, %att6_inrel_r0_s0, %att10_inrel_r0_s0,
      %att14_inrel_r0_s0, %att2_out_r0_s0, %att6_out_r0_s0, %att10_out_r0_s0, %att14_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %state_p3_1 depends_on(
      %pf_state_p3_r0_s0, %att3_inrel_r0_s0, %att7_inrel_r0_s0, %att11_inrel_r0_s0,
      %att15_inrel_r0_s0, %att3_out_r0_s0, %att7_out_r0_s0, %att11_out_r0_s0, %att15_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %acc_p3_1 depends_on(
      %pf_acc_p3_r0_s0, %att3_inrel_r0_s0, %att7_inrel_r0_s0, %att11_inrel_r0_s0,
      %att15_inrel_r0_s0, %att3_out_r0_s0, %att7_out_r0_s0, %att11_out_r0_s0, %att15_out_r0_s0,
      %merge_inrel_r0_s0)
    nest.release %out_l2_1 depends_on(%merge_out_r0_s0, %out_store_r0_s0)
    nest.return
  }
  nest.context @step_r0_s1(
    %POOL_1: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_1: !nest.global_memref<147xi32>,
    %APPEND_IDS_1: !nest.global_memref<12xi32>, %Q_IN_1: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_1: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_1: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_1: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_1: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_1: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 76, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_1 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_1 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_1 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_16 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_2 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_2 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_2 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_2 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_2 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_2 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_2 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_2 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_1 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_1 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_2 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %298 = nest.subview %POOL_1 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %299 = nest.subview %K_NEW_1 offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %300 = nest.subview %V_NEW_1 offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %301 = nest.subview %APPEND_IDS_1 offsets = [1] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %302 = nest.subview %Q_IN_1 offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %303 = nest.subview %S_INIT_1 offsets = [0, 1, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %304 = nest.subview %S_INIT_1 offsets = [0, 1, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %305 = nest.subview %S_INIT_1 offsets = [0, 1, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %306 = nest.subview %S_INIT_1 offsets = [0, 1, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %307 = nest.subview %O_INIT_1 offsets = [0, 1, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %308 = nest.subview %O_INIT_1 offsets = [0, 1, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %309 = nest.subview %O_INIT_1 offsets = [0, 1, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %310 = nest.subview %O_INIT_1 offsets = [0, 1, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %311 = nest.subview %OUT_1 offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %312 = nest.subview %BLOCK_TABLE_1 offsets = [0] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %313 = nest.subview %BLOCK_TABLE_1 offsets = [1] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %314 = nest.subview %BLOCK_TABLE_1 offsets = [2] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %315 = nest.subview %BLOCK_TABLE_1 offsets = [3] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %316 = nest.subview %BLOCK_TABLE_1 offsets = [4] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %317 = nest.subview %BLOCK_TABLE_1 offsets = [5] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %318 = nest.subview %BLOCK_TABLE_1 offsets = [6] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %319 = nest.subview %BLOCK_TABLE_1 offsets = [7] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %320 = nest.subview %BLOCK_TABLE_1 offsets = [8] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %321 = nest.subview %BLOCK_TABLE_1 offsets = [9] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %322 = nest.subview %BLOCK_TABLE_1 offsets = [10] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %323 = nest.subview %BLOCK_TABLE_1 offsets = [11] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %324 = nest.subview %BLOCK_TABLE_1 offsets = [12] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %325 = nest.subview %BLOCK_TABLE_1 offsets = [13] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %326 = nest.subview %BLOCK_TABLE_1 offsets = [14] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %327 = nest.subview %BLOCK_TABLE_1 offsets = [15] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r0_s1 = nest.dma.prefetch.async %299 into %k_new_1 : !nest.event<"pf_k_r0_s1">
    %pf_v_r0_s1 = nest.dma.prefetch.async %300 into %v_new_1 : !nest.event<"pf_v_r0_s1">
    %pf_aidx_r0_s1 = nest.dma.prefetch.async %301 into %append_idx_1 : !nest.event<"pf_aidx_r0_s1">
    %pf_q_r0_s1 = nest.dma.prefetch.async %302 into %q_l2_16 : !nest.event<"pf_q_r0_s1">
    %pf_state_p0_r0_s1 = nest.dma.prefetch.async %303 into %state_p0_2
      : !nest.event<"pf_state_p0_r0_s1">
    %pf_state_p1_r0_s1 = nest.dma.prefetch.async %304 into %state_p1_2
      : !nest.event<"pf_state_p1_r0_s1">
    %pf_state_p2_r0_s1 = nest.dma.prefetch.async %305 into %state_p2_2
      : !nest.event<"pf_state_p2_r0_s1">
    %pf_state_p3_r0_s1 = nest.dma.prefetch.async %306 into %state_p3_2
      : !nest.event<"pf_state_p3_r0_s1">
    %pf_acc_p0_r0_s1 = nest.dma.prefetch.async %307 into %acc_p0_2 : !nest.event<"pf_acc_p0_r0_s1">
    %pf_acc_p1_r0_s1 = nest.dma.prefetch.async %308 into %acc_p1_2 : !nest.event<"pf_acc_p1_r0_s1">
    %pf_acc_p2_r0_s1 = nest.dma.prefetch.async %309 into %acc_p2_2 : !nest.event<"pf_acc_p2_r0_s1">
    %pf_acc_p3_r0_s1 = nest.dma.prefetch.async %310 into %acc_p3_2 : !nest.event<"pf_acc_p3_r0_s1">
    %328 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r0_s1, %append_inrel_r0_s1, %329 =
      nest.dispatch.tasks.async @paged_attention_append_r0_tip0 l1_mode = 1 tasks(%328)
      globals(%298) bindings(%k_new_1, %v_new_1, %append_idx_1)
      ins(%k_new_1, %v_new_1, %append_idx_1) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r0_s1, %pf_v_r0_s1, %pf_aidx_r0_s1)
      : (!nest.event<"append_grid_r0_s1">, !nest.event<"append_inrel_r0_s1">, !nest.event<"">)
    %pf_bidx0_r0_s1 = nest.dma.prefetch.async %312 into %block_idx_p0_1
      : !nest.event<"pf_bidx0_r0_s1">
    %att0_grid_r0_s1, %att0_inrel_r0_s1, %att0_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2) outs(%state_p0_2, %acc_p0_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s1, %pf_state_p0_r0_s1, %pf_acc_p0_r0_s1, %pf_bidx0_r0_s1) : (
        !nest.event<"att0_grid_r0_s1">, !nest.event<"att0_inrel_r0_s1">,
        !nest.event<"att0_out_r0_s1">)
    %pf_bidx1_r0_s1 = nest.dma.prefetch.async %313 into %block_idx_p1_1
      : !nest.event<"pf_bidx1_r0_s1">
    %att1_grid_r0_s1, %att1_inrel_r0_s1, %att1_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2) outs(%state_p1_2, %acc_p1_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s1, %pf_state_p1_r0_s1, %pf_acc_p1_r0_s1, %pf_bidx1_r0_s1) : (
        !nest.event<"att1_grid_r0_s1">, !nest.event<"att1_inrel_r0_s1">,
        !nest.event<"att1_out_r0_s1">)
    %pf_bidx2_r0_s1 = nest.dma.prefetch.async %314 into %block_idx_p0_1
      depends_on(%att0_inrel_r0_s1) : !nest.event<"pf_bidx2_r0_s1">
    %att2_grid_r0_s1, %att2_inrel_r0_s1, %att2_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2) outs(%state_p2_2, %acc_p2_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s1, %pf_state_p2_r0_s1, %pf_acc_p2_r0_s1, %pf_bidx2_r0_s1) : (
        !nest.event<"att2_grid_r0_s1">, !nest.event<"att2_inrel_r0_s1">,
        !nest.event<"att2_out_r0_s1">)
    %pf_bidx3_r0_s1 = nest.dma.prefetch.async %315 into %block_idx_p1_1
      depends_on(%att1_inrel_r0_s1) : !nest.event<"pf_bidx3_r0_s1">
    %att3_grid_r0_s1, %att3_inrel_r0_s1, %att3_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2) outs(%state_p3_2, %acc_p3_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s1, %pf_state_p3_r0_s1, %pf_acc_p3_r0_s1, %pf_bidx3_r0_s1) : (
        !nest.event<"att3_grid_r0_s1">, !nest.event<"att3_inrel_r0_s1">,
        !nest.event<"att3_out_r0_s1">)
    %pf_bidx4_r0_s1 = nest.dma.prefetch.async %316 into %block_idx_p0_1
      depends_on(%att2_inrel_r0_s1) : !nest.event<"pf_bidx4_r0_s1">
    %att4_grid_r0_s1, %att4_inrel_r0_s1, %att4_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2) outs(%state_p0_2, %acc_p0_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p0_r0_s1, %pf_acc_p0_r0_s1, %pf_bidx4_r0_s1, %att0_out_r0_s1)
      : (
        !nest.event<"att4_grid_r0_s1">, !nest.event<"att4_inrel_r0_s1">,
        !nest.event<"att4_out_r0_s1">)
    %pf_bidx5_r0_s1 = nest.dma.prefetch.async %317 into %block_idx_p1_1
      depends_on(%att3_inrel_r0_s1) : !nest.event<"pf_bidx5_r0_s1">
    %att5_grid_r0_s1, %att5_inrel_r0_s1, %att5_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2) outs(%state_p1_2, %acc_p1_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p1_r0_s1, %pf_acc_p1_r0_s1, %pf_bidx5_r0_s1, %att1_out_r0_s1)
      : (
        !nest.event<"att5_grid_r0_s1">, !nest.event<"att5_inrel_r0_s1">,
        !nest.event<"att5_out_r0_s1">)
    %pf_bidx6_r0_s1 = nest.dma.prefetch.async %318 into %block_idx_p0_1
      depends_on(%att4_inrel_r0_s1) : !nest.event<"pf_bidx6_r0_s1">
    %att6_grid_r0_s1, %att6_inrel_r0_s1, %att6_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2) outs(%state_p2_2, %acc_p2_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p2_r0_s1, %pf_acc_p2_r0_s1, %pf_bidx6_r0_s1, %att2_out_r0_s1)
      : (
        !nest.event<"att6_grid_r0_s1">, !nest.event<"att6_inrel_r0_s1">,
        !nest.event<"att6_out_r0_s1">)
    %pf_bidx7_r0_s1 = nest.dma.prefetch.async %319 into %block_idx_p1_1
      depends_on(%att5_inrel_r0_s1) : !nest.event<"pf_bidx7_r0_s1">
    %att7_grid_r0_s1, %att7_inrel_r0_s1, %att7_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2) outs(%state_p3_2, %acc_p3_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p3_r0_s1, %pf_acc_p3_r0_s1, %pf_bidx7_r0_s1, %att3_out_r0_s1)
      : (
        !nest.event<"att7_grid_r0_s1">, !nest.event<"att7_inrel_r0_s1">,
        !nest.event<"att7_out_r0_s1">)
    %pf_bidx8_r0_s1 = nest.dma.prefetch.async %320 into %block_idx_p0_1
      depends_on(%att6_inrel_r0_s1) : !nest.event<"pf_bidx8_r0_s1">
    %att8_grid_r0_s1, %att8_inrel_r0_s1, %att8_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2) outs(%state_p0_2, %acc_p0_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p0_r0_s1, %pf_acc_p0_r0_s1, %pf_bidx8_r0_s1, %att4_out_r0_s1)
      : (
        !nest.event<"att8_grid_r0_s1">, !nest.event<"att8_inrel_r0_s1">,
        !nest.event<"att8_out_r0_s1">)
    %pf_bidx9_r0_s1 = nest.dma.prefetch.async %321 into %block_idx_p1_1
      depends_on(%att7_inrel_r0_s1) : !nest.event<"pf_bidx9_r0_s1">
    %att9_grid_r0_s1, %att9_inrel_r0_s1, %att9_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2) outs(%state_p1_2, %acc_p1_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p1_r0_s1, %pf_acc_p1_r0_s1, %pf_bidx9_r0_s1, %att5_out_r0_s1)
      : (
        !nest.event<"att9_grid_r0_s1">, !nest.event<"att9_inrel_r0_s1">,
        !nest.event<"att9_out_r0_s1">)
    %pf_bidx10_r0_s1 = nest.dma.prefetch.async %322 into %block_idx_p0_1
      depends_on(%att8_inrel_r0_s1) : !nest.event<"pf_bidx10_r0_s1">
    %att10_grid_r0_s1, %att10_inrel_r0_s1, %att10_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2) outs(%state_p2_2, %acc_p2_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p2_r0_s1, %pf_acc_p2_r0_s1, %pf_bidx10_r0_s1, %att6_out_r0_s1)
      : (
        !nest.event<"att10_grid_r0_s1">, !nest.event<"att10_inrel_r0_s1">,
        !nest.event<"att10_out_r0_s1">)
    %pf_bidx11_r0_s1 = nest.dma.prefetch.async %323 into %block_idx_p1_1
      depends_on(%att9_inrel_r0_s1) : !nest.event<"pf_bidx11_r0_s1">
    %att11_grid_r0_s1, %att11_inrel_r0_s1, %att11_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2) outs(%state_p3_2, %acc_p3_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p3_r0_s1, %pf_acc_p3_r0_s1, %pf_bidx11_r0_s1, %att7_out_r0_s1)
      : (
        !nest.event<"att11_grid_r0_s1">, !nest.event<"att11_inrel_r0_s1">,
        !nest.event<"att11_out_r0_s1">)
    %pf_bidx12_r0_s1 = nest.dma.prefetch.async %324 into %block_idx_p0_1
      depends_on(%att10_inrel_r0_s1) : !nest.event<"pf_bidx12_r0_s1">
    %att12_grid_r0_s1, %att12_inrel_r0_s1, %att12_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p0_2, %acc_p0_2) outs(%state_p0_2, %acc_p0_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p0_r0_s1, %pf_acc_p0_r0_s1, %pf_bidx12_r0_s1, %att8_out_r0_s1)
      : (
        !nest.event<"att12_grid_r0_s1">, !nest.event<"att12_inrel_r0_s1">,
        !nest.event<"att12_out_r0_s1">)
    %pf_bidx13_r0_s1 = nest.dma.prefetch.async %325 into %block_idx_p1_1
      depends_on(%att11_inrel_r0_s1) : !nest.event<"pf_bidx13_r0_s1">
    %att13_grid_r0_s1, %att13_inrel_r0_s1, %att13_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p1_2, %acc_p1_2) outs(%state_p1_2, %acc_p1_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p1_r0_s1, %pf_acc_p1_r0_s1, %pf_bidx13_r0_s1, %att9_out_r0_s1)
      : (
        !nest.event<"att13_grid_r0_s1">, !nest.event<"att13_inrel_r0_s1">,
        !nest.event<"att13_out_r0_s1">)
    %pf_bidx14_r0_s1 = nest.dma.prefetch.async %326 into %block_idx_p0_1
      depends_on(%att12_inrel_r0_s1) : !nest.event<"pf_bidx14_r0_s1">
    %att14_grid_r0_s1, %att14_inrel_r0_s1, %att14_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2)
      ins(%block_idx_p0_1, %q_l2_16, %state_p2_2, %acc_p2_2) outs(%state_p2_2, %acc_p2_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p2_r0_s1, %pf_acc_p2_r0_s1, %pf_bidx14_r0_s1, %att10_out_r0_s1)
      : (
        !nest.event<"att14_grid_r0_s1">, !nest.event<"att14_inrel_r0_s1">,
        !nest.event<"att14_out_r0_s1">)
    %pf_bidx15_r0_s1 = nest.dma.prefetch.async %327 into %block_idx_p1_1
      depends_on(%att13_inrel_r0_s1) : !nest.event<"pf_bidx15_r0_s1">
    %att15_grid_r0_s1, %att15_inrel_r0_s1, %att15_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2)
      ins(%block_idx_p1_1, %q_l2_16, %state_p3_2, %acc_p3_2) outs(%state_p3_2, %acc_p3_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p3_r0_s1, %pf_acc_p3_r0_s1, %pf_bidx15_r0_s1, %att11_out_r0_s1)
      : (
        !nest.event<"att15_grid_r0_s1">, !nest.event<"att15_inrel_r0_s1">,
        !nest.event<"att15_out_r0_s1">)
    %att16_grid_r0_s1, %att16_inrel_r0_s1, %att16_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_t1_final_r0 l1_mode = 1 tasks(%328) globals(%298)
      bindings(%append_idx_1, %q_l2_16, %state_p0_2, %acc_p0_2)
      ins(%append_idx_1, %q_l2_16, %state_p0_2, %acc_p0_2) outs(%state_p0_2, %acc_p0_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s1, %pf_state_p0_r0_s1, %pf_acc_p0_r0_s1, %pf_aidx_r0_s1, %append_grid_r0_s1,
        %att12_out_r0_s1)
      : (
        !nest.event<"att16_grid_r0_s1">, !nest.event<"att16_inrel_r0_s1">,
        !nest.event<"att16_out_r0_s1">)
    %merge_grid_r0_s1, %merge_inrel_r0_s1, %merge_out_r0_s1 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%328) globals()
      bindings(
        %state_p0_2, %state_p1_2, %state_p2_2, %state_p3_2, %acc_p0_2, %acc_p1_2, %acc_p2_2,
        %acc_p3_2, %out_l2_2)
      ins(
        %state_p0_2, %state_p1_2, %state_p2_2, %state_p3_2, %acc_p0_2, %acc_p1_2, %acc_p2_2,
        %acc_p3_2)
      outs(%out_l2_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att16_out_r0_s1, %att13_out_r0_s1, %att14_out_r0_s1, %att15_out_r0_s1) : (
        !nest.event<"merge_grid_r0_s1">, !nest.event<"merge_inrel_r0_s1">,
        !nest.event<"merge_out_r0_s1">)
    %out_store_r0_s1 = nest.dma.store.async %out_l2_2 into %311 depends_on(%merge_out_r0_s1)
      : !nest.event<"out_store_r0_s1">
    nest.await %append_grid_r0_s1, %att16_grid_r0_s1, %merge_grid_r0_s1, %out_store_r0_s1
    nest.release %block_idx_p0_1 depends_on(
      %pf_bidx0_r0_s1, %pf_bidx2_r0_s1, %pf_bidx4_r0_s1, %pf_bidx6_r0_s1, %pf_bidx8_r0_s1,
      %pf_bidx10_r0_s1, %pf_bidx12_r0_s1, %pf_bidx14_r0_s1, %att0_inrel_r0_s1, %att2_inrel_r0_s1,
      %att4_inrel_r0_s1, %att6_inrel_r0_s1, %att8_inrel_r0_s1, %att10_inrel_r0_s1,
      %att12_inrel_r0_s1, %att14_inrel_r0_s1)
    nest.release %block_idx_p1_1 depends_on(
      %pf_bidx1_r0_s1, %pf_bidx3_r0_s1, %pf_bidx5_r0_s1, %pf_bidx7_r0_s1, %pf_bidx9_r0_s1,
      %pf_bidx11_r0_s1, %pf_bidx13_r0_s1, %pf_bidx15_r0_s1, %att1_inrel_r0_s1, %att3_inrel_r0_s1,
      %att5_inrel_r0_s1, %att7_inrel_r0_s1, %att9_inrel_r0_s1, %att11_inrel_r0_s1,
      %att13_inrel_r0_s1, %att15_inrel_r0_s1)
    nest.release %k_new_1 depends_on(%pf_k_r0_s1, %append_inrel_r0_s1)
    nest.release %v_new_1 depends_on(%pf_v_r0_s1, %append_inrel_r0_s1)
    nest.release %append_idx_1 depends_on(%pf_aidx_r0_s1, %append_inrel_r0_s1, %att16_inrel_r0_s1)
    nest.release %q_l2_16 depends_on(
      %pf_q_r0_s1, %att0_inrel_r0_s1, %att1_inrel_r0_s1, %att2_inrel_r0_s1, %att3_inrel_r0_s1,
      %att4_inrel_r0_s1, %att5_inrel_r0_s1, %att6_inrel_r0_s1, %att7_inrel_r0_s1,
      %att8_inrel_r0_s1, %att9_inrel_r0_s1, %att10_inrel_r0_s1, %att11_inrel_r0_s1,
      %att12_inrel_r0_s1, %att13_inrel_r0_s1, %att14_inrel_r0_s1, %att15_inrel_r0_s1,
      %att16_inrel_r0_s1)
    nest.release %state_p0_2 depends_on(
      %pf_state_p0_r0_s1, %att0_inrel_r0_s1, %att4_inrel_r0_s1, %att8_inrel_r0_s1,
      %att12_inrel_r0_s1, %att16_inrel_r0_s1, %att0_out_r0_s1, %att4_out_r0_s1, %att8_out_r0_s1,
      %att12_out_r0_s1, %att16_out_r0_s1, %merge_inrel_r0_s1)
    nest.release %acc_p0_2 depends_on(
      %pf_acc_p0_r0_s1, %att0_inrel_r0_s1, %att4_inrel_r0_s1, %att8_inrel_r0_s1,
      %att12_inrel_r0_s1, %att16_inrel_r0_s1, %att0_out_r0_s1, %att4_out_r0_s1, %att8_out_r0_s1,
      %att12_out_r0_s1, %att16_out_r0_s1, %merge_inrel_r0_s1)
    nest.release %state_p1_2 depends_on(
      %pf_state_p1_r0_s1, %att1_inrel_r0_s1, %att5_inrel_r0_s1, %att9_inrel_r0_s1,
      %att13_inrel_r0_s1, %att1_out_r0_s1, %att5_out_r0_s1, %att9_out_r0_s1, %att13_out_r0_s1,
      %merge_inrel_r0_s1)
    nest.release %acc_p1_2 depends_on(
      %pf_acc_p1_r0_s1, %att1_inrel_r0_s1, %att5_inrel_r0_s1, %att9_inrel_r0_s1,
      %att13_inrel_r0_s1, %att1_out_r0_s1, %att5_out_r0_s1, %att9_out_r0_s1, %att13_out_r0_s1,
      %merge_inrel_r0_s1)
    nest.release %state_p2_2 depends_on(
      %pf_state_p2_r0_s1, %att2_inrel_r0_s1, %att6_inrel_r0_s1, %att10_inrel_r0_s1,
      %att14_inrel_r0_s1, %att2_out_r0_s1, %att6_out_r0_s1, %att10_out_r0_s1, %att14_out_r0_s1,
      %merge_inrel_r0_s1)
    nest.release %acc_p2_2 depends_on(
      %pf_acc_p2_r0_s1, %att2_inrel_r0_s1, %att6_inrel_r0_s1, %att10_inrel_r0_s1,
      %att14_inrel_r0_s1, %att2_out_r0_s1, %att6_out_r0_s1, %att10_out_r0_s1, %att14_out_r0_s1,
      %merge_inrel_r0_s1)
    nest.release %state_p3_2 depends_on(
      %pf_state_p3_r0_s1, %att3_inrel_r0_s1, %att7_inrel_r0_s1, %att11_inrel_r0_s1,
      %att15_inrel_r0_s1, %att3_out_r0_s1, %att7_out_r0_s1, %att11_out_r0_s1, %att15_out_r0_s1,
      %merge_inrel_r0_s1)
    nest.release %acc_p3_2 depends_on(
      %pf_acc_p3_r0_s1, %att3_inrel_r0_s1, %att7_inrel_r0_s1, %att11_inrel_r0_s1,
      %att15_inrel_r0_s1, %att3_out_r0_s1, %att7_out_r0_s1, %att11_out_r0_s1, %att15_out_r0_s1,
      %merge_inrel_r0_s1)
    nest.release %out_l2_2 depends_on(%merge_out_r0_s1, %out_store_r0_s1)
    nest.return
  }
  nest.context @step_r0_s2(
    %POOL_2: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_2: !nest.global_memref<147xi32>,
    %APPEND_IDS_2: !nest.global_memref<12xi32>, %Q_IN_2: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_2: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_2: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_2: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_2: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_2: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 76, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_2 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_2 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_2 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_17 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_3 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_3 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_3 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_3 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_3 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_3 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_3 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_3 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_2 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_2 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_3 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %330 = nest.subview %POOL_2 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %331 = nest.subview %K_NEW_2 offsets = [0, 2, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %332 = nest.subview %V_NEW_2 offsets = [0, 2, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %333 = nest.subview %APPEND_IDS_2 offsets = [2] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %334 = nest.subview %Q_IN_2 offsets = [0, 2, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %335 = nest.subview %S_INIT_2 offsets = [0, 2, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %336 = nest.subview %S_INIT_2 offsets = [0, 2, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %337 = nest.subview %S_INIT_2 offsets = [0, 2, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %338 = nest.subview %S_INIT_2 offsets = [0, 2, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %339 = nest.subview %O_INIT_2 offsets = [0, 2, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %340 = nest.subview %O_INIT_2 offsets = [0, 2, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %341 = nest.subview %O_INIT_2 offsets = [0, 2, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %342 = nest.subview %O_INIT_2 offsets = [0, 2, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %343 = nest.subview %OUT_2 offsets = [0, 2, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %344 = nest.subview %BLOCK_TABLE_2 offsets = [0] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %345 = nest.subview %BLOCK_TABLE_2 offsets = [1] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %346 = nest.subview %BLOCK_TABLE_2 offsets = [2] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %347 = nest.subview %BLOCK_TABLE_2 offsets = [3] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %348 = nest.subview %BLOCK_TABLE_2 offsets = [4] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %349 = nest.subview %BLOCK_TABLE_2 offsets = [5] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %350 = nest.subview %BLOCK_TABLE_2 offsets = [6] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %351 = nest.subview %BLOCK_TABLE_2 offsets = [7] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %352 = nest.subview %BLOCK_TABLE_2 offsets = [8] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %353 = nest.subview %BLOCK_TABLE_2 offsets = [9] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %354 = nest.subview %BLOCK_TABLE_2 offsets = [10] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %355 = nest.subview %BLOCK_TABLE_2 offsets = [11] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %356 = nest.subview %BLOCK_TABLE_2 offsets = [12] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %357 = nest.subview %BLOCK_TABLE_2 offsets = [13] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %358 = nest.subview %BLOCK_TABLE_2 offsets = [14] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %359 = nest.subview %BLOCK_TABLE_2 offsets = [15] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r0_s2 = nest.dma.prefetch.async %331 into %k_new_2 : !nest.event<"pf_k_r0_s2">
    %pf_v_r0_s2 = nest.dma.prefetch.async %332 into %v_new_2 : !nest.event<"pf_v_r0_s2">
    %pf_aidx_r0_s2 = nest.dma.prefetch.async %333 into %append_idx_2 : !nest.event<"pf_aidx_r0_s2">
    %pf_q_r0_s2 = nest.dma.prefetch.async %334 into %q_l2_17 : !nest.event<"pf_q_r0_s2">
    %pf_state_p0_r0_s2 = nest.dma.prefetch.async %335 into %state_p0_3
      : !nest.event<"pf_state_p0_r0_s2">
    %pf_state_p1_r0_s2 = nest.dma.prefetch.async %336 into %state_p1_3
      : !nest.event<"pf_state_p1_r0_s2">
    %pf_state_p2_r0_s2 = nest.dma.prefetch.async %337 into %state_p2_3
      : !nest.event<"pf_state_p2_r0_s2">
    %pf_state_p3_r0_s2 = nest.dma.prefetch.async %338 into %state_p3_3
      : !nest.event<"pf_state_p3_r0_s2">
    %pf_acc_p0_r0_s2 = nest.dma.prefetch.async %339 into %acc_p0_3 : !nest.event<"pf_acc_p0_r0_s2">
    %pf_acc_p1_r0_s2 = nest.dma.prefetch.async %340 into %acc_p1_3 : !nest.event<"pf_acc_p1_r0_s2">
    %pf_acc_p2_r0_s2 = nest.dma.prefetch.async %341 into %acc_p2_3 : !nest.event<"pf_acc_p2_r0_s2">
    %pf_acc_p3_r0_s2 = nest.dma.prefetch.async %342 into %acc_p3_3 : !nest.event<"pf_acc_p3_r0_s2">
    %360 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r0_s2, %append_inrel_r0_s2, %361 =
      nest.dispatch.tasks.async @paged_attention_append_r0_tip1 l1_mode = 1 tasks(%360)
      globals(%330) bindings(%k_new_2, %v_new_2, %append_idx_2)
      ins(%k_new_2, %v_new_2, %append_idx_2) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r0_s2, %pf_v_r0_s2, %pf_aidx_r0_s2)
      : (!nest.event<"append_grid_r0_s2">, !nest.event<"append_inrel_r0_s2">, !nest.event<"">)
    %pf_bidx0_r0_s2 = nest.dma.prefetch.async %344 into %block_idx_p0_2
      : !nest.event<"pf_bidx0_r0_s2">
    %att0_grid_r0_s2, %att0_inrel_r0_s2, %att0_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3) outs(%state_p0_3, %acc_p0_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s2, %pf_state_p0_r0_s2, %pf_acc_p0_r0_s2, %pf_bidx0_r0_s2) : (
        !nest.event<"att0_grid_r0_s2">, !nest.event<"att0_inrel_r0_s2">,
        !nest.event<"att0_out_r0_s2">)
    %pf_bidx1_r0_s2 = nest.dma.prefetch.async %345 into %block_idx_p1_2
      : !nest.event<"pf_bidx1_r0_s2">
    %att1_grid_r0_s2, %att1_inrel_r0_s2, %att1_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3) outs(%state_p1_3, %acc_p1_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s2, %pf_state_p1_r0_s2, %pf_acc_p1_r0_s2, %pf_bidx1_r0_s2) : (
        !nest.event<"att1_grid_r0_s2">, !nest.event<"att1_inrel_r0_s2">,
        !nest.event<"att1_out_r0_s2">)
    %pf_bidx2_r0_s2 = nest.dma.prefetch.async %346 into %block_idx_p0_2
      depends_on(%att0_inrel_r0_s2) : !nest.event<"pf_bidx2_r0_s2">
    %att2_grid_r0_s2, %att2_inrel_r0_s2, %att2_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3) outs(%state_p2_3, %acc_p2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s2, %pf_state_p2_r0_s2, %pf_acc_p2_r0_s2, %pf_bidx2_r0_s2) : (
        !nest.event<"att2_grid_r0_s2">, !nest.event<"att2_inrel_r0_s2">,
        !nest.event<"att2_out_r0_s2">)
    %pf_bidx3_r0_s2 = nest.dma.prefetch.async %347 into %block_idx_p1_2
      depends_on(%att1_inrel_r0_s2) : !nest.event<"pf_bidx3_r0_s2">
    %att3_grid_r0_s2, %att3_inrel_r0_s2, %att3_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3) outs(%state_p3_3, %acc_p3_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s2, %pf_state_p3_r0_s2, %pf_acc_p3_r0_s2, %pf_bidx3_r0_s2) : (
        !nest.event<"att3_grid_r0_s2">, !nest.event<"att3_inrel_r0_s2">,
        !nest.event<"att3_out_r0_s2">)
    %pf_bidx4_r0_s2 = nest.dma.prefetch.async %348 into %block_idx_p0_2
      depends_on(%att2_inrel_r0_s2) : !nest.event<"pf_bidx4_r0_s2">
    %att4_grid_r0_s2, %att4_inrel_r0_s2, %att4_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3) outs(%state_p0_3, %acc_p0_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p0_r0_s2, %pf_acc_p0_r0_s2, %pf_bidx4_r0_s2, %att0_out_r0_s2)
      : (
        !nest.event<"att4_grid_r0_s2">, !nest.event<"att4_inrel_r0_s2">,
        !nest.event<"att4_out_r0_s2">)
    %pf_bidx5_r0_s2 = nest.dma.prefetch.async %349 into %block_idx_p1_2
      depends_on(%att3_inrel_r0_s2) : !nest.event<"pf_bidx5_r0_s2">
    %att5_grid_r0_s2, %att5_inrel_r0_s2, %att5_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3) outs(%state_p1_3, %acc_p1_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p1_r0_s2, %pf_acc_p1_r0_s2, %pf_bidx5_r0_s2, %att1_out_r0_s2)
      : (
        !nest.event<"att5_grid_r0_s2">, !nest.event<"att5_inrel_r0_s2">,
        !nest.event<"att5_out_r0_s2">)
    %pf_bidx6_r0_s2 = nest.dma.prefetch.async %350 into %block_idx_p0_2
      depends_on(%att4_inrel_r0_s2) : !nest.event<"pf_bidx6_r0_s2">
    %att6_grid_r0_s2, %att6_inrel_r0_s2, %att6_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3) outs(%state_p2_3, %acc_p2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p2_r0_s2, %pf_acc_p2_r0_s2, %pf_bidx6_r0_s2, %att2_out_r0_s2)
      : (
        !nest.event<"att6_grid_r0_s2">, !nest.event<"att6_inrel_r0_s2">,
        !nest.event<"att6_out_r0_s2">)
    %pf_bidx7_r0_s2 = nest.dma.prefetch.async %351 into %block_idx_p1_2
      depends_on(%att5_inrel_r0_s2) : !nest.event<"pf_bidx7_r0_s2">
    %att7_grid_r0_s2, %att7_inrel_r0_s2, %att7_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3) outs(%state_p3_3, %acc_p3_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p3_r0_s2, %pf_acc_p3_r0_s2, %pf_bidx7_r0_s2, %att3_out_r0_s2)
      : (
        !nest.event<"att7_grid_r0_s2">, !nest.event<"att7_inrel_r0_s2">,
        !nest.event<"att7_out_r0_s2">)
    %pf_bidx8_r0_s2 = nest.dma.prefetch.async %352 into %block_idx_p0_2
      depends_on(%att6_inrel_r0_s2) : !nest.event<"pf_bidx8_r0_s2">
    %att8_grid_r0_s2, %att8_inrel_r0_s2, %att8_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3) outs(%state_p0_3, %acc_p0_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p0_r0_s2, %pf_acc_p0_r0_s2, %pf_bidx8_r0_s2, %att4_out_r0_s2)
      : (
        !nest.event<"att8_grid_r0_s2">, !nest.event<"att8_inrel_r0_s2">,
        !nest.event<"att8_out_r0_s2">)
    %pf_bidx9_r0_s2 = nest.dma.prefetch.async %353 into %block_idx_p1_2
      depends_on(%att7_inrel_r0_s2) : !nest.event<"pf_bidx9_r0_s2">
    %att9_grid_r0_s2, %att9_inrel_r0_s2, %att9_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3) outs(%state_p1_3, %acc_p1_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p1_r0_s2, %pf_acc_p1_r0_s2, %pf_bidx9_r0_s2, %att5_out_r0_s2)
      : (
        !nest.event<"att9_grid_r0_s2">, !nest.event<"att9_inrel_r0_s2">,
        !nest.event<"att9_out_r0_s2">)
    %pf_bidx10_r0_s2 = nest.dma.prefetch.async %354 into %block_idx_p0_2
      depends_on(%att8_inrel_r0_s2) : !nest.event<"pf_bidx10_r0_s2">
    %att10_grid_r0_s2, %att10_inrel_r0_s2, %att10_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3) outs(%state_p2_3, %acc_p2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p2_r0_s2, %pf_acc_p2_r0_s2, %pf_bidx10_r0_s2, %att6_out_r0_s2)
      : (
        !nest.event<"att10_grid_r0_s2">, !nest.event<"att10_inrel_r0_s2">,
        !nest.event<"att10_out_r0_s2">)
    %pf_bidx11_r0_s2 = nest.dma.prefetch.async %355 into %block_idx_p1_2
      depends_on(%att9_inrel_r0_s2) : !nest.event<"pf_bidx11_r0_s2">
    %att11_grid_r0_s2, %att11_inrel_r0_s2, %att11_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3) outs(%state_p3_3, %acc_p3_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p3_r0_s2, %pf_acc_p3_r0_s2, %pf_bidx11_r0_s2, %att7_out_r0_s2)
      : (
        !nest.event<"att11_grid_r0_s2">, !nest.event<"att11_inrel_r0_s2">,
        !nest.event<"att11_out_r0_s2">)
    %pf_bidx12_r0_s2 = nest.dma.prefetch.async %356 into %block_idx_p0_2
      depends_on(%att10_inrel_r0_s2) : !nest.event<"pf_bidx12_r0_s2">
    %att12_grid_r0_s2, %att12_inrel_r0_s2, %att12_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p0_3, %acc_p0_3) outs(%state_p0_3, %acc_p0_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p0_r0_s2, %pf_acc_p0_r0_s2, %pf_bidx12_r0_s2, %att8_out_r0_s2)
      : (
        !nest.event<"att12_grid_r0_s2">, !nest.event<"att12_inrel_r0_s2">,
        !nest.event<"att12_out_r0_s2">)
    %pf_bidx13_r0_s2 = nest.dma.prefetch.async %357 into %block_idx_p1_2
      depends_on(%att11_inrel_r0_s2) : !nest.event<"pf_bidx13_r0_s2">
    %att13_grid_r0_s2, %att13_inrel_r0_s2, %att13_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p1_3, %acc_p1_3) outs(%state_p1_3, %acc_p1_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p1_r0_s2, %pf_acc_p1_r0_s2, %pf_bidx13_r0_s2, %att9_out_r0_s2)
      : (
        !nest.event<"att13_grid_r0_s2">, !nest.event<"att13_inrel_r0_s2">,
        !nest.event<"att13_out_r0_s2">)
    %pf_bidx14_r0_s2 = nest.dma.prefetch.async %358 into %block_idx_p0_2
      depends_on(%att12_inrel_r0_s2) : !nest.event<"pf_bidx14_r0_s2">
    %att14_grid_r0_s2, %att14_inrel_r0_s2, %att14_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3)
      ins(%block_idx_p0_2, %q_l2_17, %state_p2_3, %acc_p2_3) outs(%state_p2_3, %acc_p2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p2_r0_s2, %pf_acc_p2_r0_s2, %pf_bidx14_r0_s2, %att10_out_r0_s2)
      : (
        !nest.event<"att14_grid_r0_s2">, !nest.event<"att14_inrel_r0_s2">,
        !nest.event<"att14_out_r0_s2">)
    %pf_bidx15_r0_s2 = nest.dma.prefetch.async %359 into %block_idx_p1_2
      depends_on(%att13_inrel_r0_s2) : !nest.event<"pf_bidx15_r0_s2">
    %att15_grid_r0_s2, %att15_inrel_r0_s2, %att15_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3)
      ins(%block_idx_p1_2, %q_l2_17, %state_p3_3, %acc_p3_3) outs(%state_p3_3, %acc_p3_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p3_r0_s2, %pf_acc_p3_r0_s2, %pf_bidx15_r0_s2, %att11_out_r0_s2)
      : (
        !nest.event<"att15_grid_r0_s2">, !nest.event<"att15_inrel_r0_s2">,
        !nest.event<"att15_out_r0_s2">)
    %att16_grid_r0_s2, %att16_inrel_r0_s2, %att16_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_t2_final_r0 l1_mode = 1 tasks(%360) globals(%330)
      bindings(%append_idx_2, %q_l2_17, %state_p0_3, %acc_p0_3)
      ins(%append_idx_2, %q_l2_17, %state_p0_3, %acc_p0_3) outs(%state_p0_3, %acc_p0_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s2, %pf_state_p0_r0_s2, %pf_acc_p0_r0_s2, %pf_aidx_r0_s2, %append_grid_r0_s2,
        %att12_out_r0_s2)
      : (
        !nest.event<"att16_grid_r0_s2">, !nest.event<"att16_inrel_r0_s2">,
        !nest.event<"att16_out_r0_s2">)
    %merge_grid_r0_s2, %merge_inrel_r0_s2, %merge_out_r0_s2 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%360) globals()
      bindings(
        %state_p0_3, %state_p1_3, %state_p2_3, %state_p3_3, %acc_p0_3, %acc_p1_3, %acc_p2_3,
        %acc_p3_3, %out_l2_3)
      ins(
        %state_p0_3, %state_p1_3, %state_p2_3, %state_p3_3, %acc_p0_3, %acc_p1_3, %acc_p2_3,
        %acc_p3_3)
      outs(%out_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att16_out_r0_s2, %att13_out_r0_s2, %att14_out_r0_s2, %att15_out_r0_s2) : (
        !nest.event<"merge_grid_r0_s2">, !nest.event<"merge_inrel_r0_s2">,
        !nest.event<"merge_out_r0_s2">)
    %out_store_r0_s2 = nest.dma.store.async %out_l2_3 into %343 depends_on(%merge_out_r0_s2)
      : !nest.event<"out_store_r0_s2">
    nest.await %append_grid_r0_s2, %att16_grid_r0_s2, %merge_grid_r0_s2, %out_store_r0_s2
    nest.release %block_idx_p0_2 depends_on(
      %pf_bidx0_r0_s2, %pf_bidx2_r0_s2, %pf_bidx4_r0_s2, %pf_bidx6_r0_s2, %pf_bidx8_r0_s2,
      %pf_bidx10_r0_s2, %pf_bidx12_r0_s2, %pf_bidx14_r0_s2, %att0_inrel_r0_s2, %att2_inrel_r0_s2,
      %att4_inrel_r0_s2, %att6_inrel_r0_s2, %att8_inrel_r0_s2, %att10_inrel_r0_s2,
      %att12_inrel_r0_s2, %att14_inrel_r0_s2)
    nest.release %block_idx_p1_2 depends_on(
      %pf_bidx1_r0_s2, %pf_bidx3_r0_s2, %pf_bidx5_r0_s2, %pf_bidx7_r0_s2, %pf_bidx9_r0_s2,
      %pf_bidx11_r0_s2, %pf_bidx13_r0_s2, %pf_bidx15_r0_s2, %att1_inrel_r0_s2, %att3_inrel_r0_s2,
      %att5_inrel_r0_s2, %att7_inrel_r0_s2, %att9_inrel_r0_s2, %att11_inrel_r0_s2,
      %att13_inrel_r0_s2, %att15_inrel_r0_s2)
    nest.release %k_new_2 depends_on(%pf_k_r0_s2, %append_inrel_r0_s2)
    nest.release %v_new_2 depends_on(%pf_v_r0_s2, %append_inrel_r0_s2)
    nest.release %append_idx_2 depends_on(%pf_aidx_r0_s2, %append_inrel_r0_s2, %att16_inrel_r0_s2)
    nest.release %q_l2_17 depends_on(
      %pf_q_r0_s2, %att0_inrel_r0_s2, %att1_inrel_r0_s2, %att2_inrel_r0_s2, %att3_inrel_r0_s2,
      %att4_inrel_r0_s2, %att5_inrel_r0_s2, %att6_inrel_r0_s2, %att7_inrel_r0_s2,
      %att8_inrel_r0_s2, %att9_inrel_r0_s2, %att10_inrel_r0_s2, %att11_inrel_r0_s2,
      %att12_inrel_r0_s2, %att13_inrel_r0_s2, %att14_inrel_r0_s2, %att15_inrel_r0_s2,
      %att16_inrel_r0_s2)
    nest.release %state_p0_3 depends_on(
      %pf_state_p0_r0_s2, %att0_inrel_r0_s2, %att4_inrel_r0_s2, %att8_inrel_r0_s2,
      %att12_inrel_r0_s2, %att16_inrel_r0_s2, %att0_out_r0_s2, %att4_out_r0_s2, %att8_out_r0_s2,
      %att12_out_r0_s2, %att16_out_r0_s2, %merge_inrel_r0_s2)
    nest.release %acc_p0_3 depends_on(
      %pf_acc_p0_r0_s2, %att0_inrel_r0_s2, %att4_inrel_r0_s2, %att8_inrel_r0_s2,
      %att12_inrel_r0_s2, %att16_inrel_r0_s2, %att0_out_r0_s2, %att4_out_r0_s2, %att8_out_r0_s2,
      %att12_out_r0_s2, %att16_out_r0_s2, %merge_inrel_r0_s2)
    nest.release %state_p1_3 depends_on(
      %pf_state_p1_r0_s2, %att1_inrel_r0_s2, %att5_inrel_r0_s2, %att9_inrel_r0_s2,
      %att13_inrel_r0_s2, %att1_out_r0_s2, %att5_out_r0_s2, %att9_out_r0_s2, %att13_out_r0_s2,
      %merge_inrel_r0_s2)
    nest.release %acc_p1_3 depends_on(
      %pf_acc_p1_r0_s2, %att1_inrel_r0_s2, %att5_inrel_r0_s2, %att9_inrel_r0_s2,
      %att13_inrel_r0_s2, %att1_out_r0_s2, %att5_out_r0_s2, %att9_out_r0_s2, %att13_out_r0_s2,
      %merge_inrel_r0_s2)
    nest.release %state_p2_3 depends_on(
      %pf_state_p2_r0_s2, %att2_inrel_r0_s2, %att6_inrel_r0_s2, %att10_inrel_r0_s2,
      %att14_inrel_r0_s2, %att2_out_r0_s2, %att6_out_r0_s2, %att10_out_r0_s2, %att14_out_r0_s2,
      %merge_inrel_r0_s2)
    nest.release %acc_p2_3 depends_on(
      %pf_acc_p2_r0_s2, %att2_inrel_r0_s2, %att6_inrel_r0_s2, %att10_inrel_r0_s2,
      %att14_inrel_r0_s2, %att2_out_r0_s2, %att6_out_r0_s2, %att10_out_r0_s2, %att14_out_r0_s2,
      %merge_inrel_r0_s2)
    nest.release %state_p3_3 depends_on(
      %pf_state_p3_r0_s2, %att3_inrel_r0_s2, %att7_inrel_r0_s2, %att11_inrel_r0_s2,
      %att15_inrel_r0_s2, %att3_out_r0_s2, %att7_out_r0_s2, %att11_out_r0_s2, %att15_out_r0_s2,
      %merge_inrel_r0_s2)
    nest.release %acc_p3_3 depends_on(
      %pf_acc_p3_r0_s2, %att3_inrel_r0_s2, %att7_inrel_r0_s2, %att11_inrel_r0_s2,
      %att15_inrel_r0_s2, %att3_out_r0_s2, %att7_out_r0_s2, %att11_out_r0_s2, %att15_out_r0_s2,
      %merge_inrel_r0_s2)
    nest.release %out_l2_3 depends_on(%merge_out_r0_s2, %out_store_r0_s2)
    nest.return
  }
  nest.context @step_r0_s3(
    %POOL_3: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_3: !nest.global_memref<147xi32>,
    %APPEND_IDS_3: !nest.global_memref<12xi32>, %Q_IN_3: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_3: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_3: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_3: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_3: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_3: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 76, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_3 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_3 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_3 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_18 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_4 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_4 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_4 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_4 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_4 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_4 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_4 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_4 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_3 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_3 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_4 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %362 = nest.subview %POOL_3 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %363 = nest.subview %K_NEW_3 offsets = [0, 3, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %364 = nest.subview %V_NEW_3 offsets = [0, 3, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %365 = nest.subview %APPEND_IDS_3 offsets = [3] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %366 = nest.subview %Q_IN_3 offsets = [0, 3, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %367 = nest.subview %S_INIT_3 offsets = [0, 3, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %368 = nest.subview %S_INIT_3 offsets = [0, 3, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %369 = nest.subview %S_INIT_3 offsets = [0, 3, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %370 = nest.subview %S_INIT_3 offsets = [0, 3, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %371 = nest.subview %O_INIT_3 offsets = [0, 3, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %372 = nest.subview %O_INIT_3 offsets = [0, 3, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %373 = nest.subview %O_INIT_3 offsets = [0, 3, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %374 = nest.subview %O_INIT_3 offsets = [0, 3, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %375 = nest.subview %OUT_3 offsets = [0, 3, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %376 = nest.subview %BLOCK_TABLE_3 offsets = [0] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %377 = nest.subview %BLOCK_TABLE_3 offsets = [1] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %378 = nest.subview %BLOCK_TABLE_3 offsets = [2] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %379 = nest.subview %BLOCK_TABLE_3 offsets = [3] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %380 = nest.subview %BLOCK_TABLE_3 offsets = [4] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %381 = nest.subview %BLOCK_TABLE_3 offsets = [5] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %382 = nest.subview %BLOCK_TABLE_3 offsets = [6] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %383 = nest.subview %BLOCK_TABLE_3 offsets = [7] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %384 = nest.subview %BLOCK_TABLE_3 offsets = [8] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %385 = nest.subview %BLOCK_TABLE_3 offsets = [9] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %386 = nest.subview %BLOCK_TABLE_3 offsets = [10] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %387 = nest.subview %BLOCK_TABLE_3 offsets = [11] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %388 = nest.subview %BLOCK_TABLE_3 offsets = [12] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %389 = nest.subview %BLOCK_TABLE_3 offsets = [13] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %390 = nest.subview %BLOCK_TABLE_3 offsets = [14] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %391 = nest.subview %BLOCK_TABLE_3 offsets = [15] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r0_s3 = nest.dma.prefetch.async %363 into %k_new_3 : !nest.event<"pf_k_r0_s3">
    %pf_v_r0_s3 = nest.dma.prefetch.async %364 into %v_new_3 : !nest.event<"pf_v_r0_s3">
    %pf_aidx_r0_s3 = nest.dma.prefetch.async %365 into %append_idx_3 : !nest.event<"pf_aidx_r0_s3">
    %pf_q_r0_s3 = nest.dma.prefetch.async %366 into %q_l2_18 : !nest.event<"pf_q_r0_s3">
    %pf_state_p0_r0_s3 = nest.dma.prefetch.async %367 into %state_p0_4
      : !nest.event<"pf_state_p0_r0_s3">
    %pf_state_p1_r0_s3 = nest.dma.prefetch.async %368 into %state_p1_4
      : !nest.event<"pf_state_p1_r0_s3">
    %pf_state_p2_r0_s3 = nest.dma.prefetch.async %369 into %state_p2_4
      : !nest.event<"pf_state_p2_r0_s3">
    %pf_state_p3_r0_s3 = nest.dma.prefetch.async %370 into %state_p3_4
      : !nest.event<"pf_state_p3_r0_s3">
    %pf_acc_p0_r0_s3 = nest.dma.prefetch.async %371 into %acc_p0_4 : !nest.event<"pf_acc_p0_r0_s3">
    %pf_acc_p1_r0_s3 = nest.dma.prefetch.async %372 into %acc_p1_4 : !nest.event<"pf_acc_p1_r0_s3">
    %pf_acc_p2_r0_s3 = nest.dma.prefetch.async %373 into %acc_p2_4 : !nest.event<"pf_acc_p2_r0_s3">
    %pf_acc_p3_r0_s3 = nest.dma.prefetch.async %374 into %acc_p3_4 : !nest.event<"pf_acc_p3_r0_s3">
    %392 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r0_s3, %append_inrel_r0_s3, %393 =
      nest.dispatch.tasks.async @paged_attention_append_r0_tip2 l1_mode = 1 tasks(%392)
      globals(%362) bindings(%k_new_3, %v_new_3, %append_idx_3)
      ins(%k_new_3, %v_new_3, %append_idx_3) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r0_s3, %pf_v_r0_s3, %pf_aidx_r0_s3)
      : (!nest.event<"append_grid_r0_s3">, !nest.event<"append_inrel_r0_s3">, !nest.event<"">)
    %pf_bidx0_r0_s3 = nest.dma.prefetch.async %376 into %block_idx_p0_3
      : !nest.event<"pf_bidx0_r0_s3">
    %att0_grid_r0_s3, %att0_inrel_r0_s3, %att0_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4) outs(%state_p0_4, %acc_p0_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s3, %pf_state_p0_r0_s3, %pf_acc_p0_r0_s3, %pf_bidx0_r0_s3) : (
        !nest.event<"att0_grid_r0_s3">, !nest.event<"att0_inrel_r0_s3">,
        !nest.event<"att0_out_r0_s3">)
    %pf_bidx1_r0_s3 = nest.dma.prefetch.async %377 into %block_idx_p1_3
      : !nest.event<"pf_bidx1_r0_s3">
    %att1_grid_r0_s3, %att1_inrel_r0_s3, %att1_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4) outs(%state_p1_4, %acc_p1_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s3, %pf_state_p1_r0_s3, %pf_acc_p1_r0_s3, %pf_bidx1_r0_s3) : (
        !nest.event<"att1_grid_r0_s3">, !nest.event<"att1_inrel_r0_s3">,
        !nest.event<"att1_out_r0_s3">)
    %pf_bidx2_r0_s3 = nest.dma.prefetch.async %378 into %block_idx_p0_3
      depends_on(%att0_inrel_r0_s3) : !nest.event<"pf_bidx2_r0_s3">
    %att2_grid_r0_s3, %att2_inrel_r0_s3, %att2_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4) outs(%state_p2_4, %acc_p2_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s3, %pf_state_p2_r0_s3, %pf_acc_p2_r0_s3, %pf_bidx2_r0_s3) : (
        !nest.event<"att2_grid_r0_s3">, !nest.event<"att2_inrel_r0_s3">,
        !nest.event<"att2_out_r0_s3">)
    %pf_bidx3_r0_s3 = nest.dma.prefetch.async %379 into %block_idx_p1_3
      depends_on(%att1_inrel_r0_s3) : !nest.event<"pf_bidx3_r0_s3">
    %att3_grid_r0_s3, %att3_inrel_r0_s3, %att3_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4) outs(%state_p3_4, %acc_p3_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r0_s3, %pf_state_p3_r0_s3, %pf_acc_p3_r0_s3, %pf_bidx3_r0_s3) : (
        !nest.event<"att3_grid_r0_s3">, !nest.event<"att3_inrel_r0_s3">,
        !nest.event<"att3_out_r0_s3">)
    %pf_bidx4_r0_s3 = nest.dma.prefetch.async %380 into %block_idx_p0_3
      depends_on(%att2_inrel_r0_s3) : !nest.event<"pf_bidx4_r0_s3">
    %att4_grid_r0_s3, %att4_inrel_r0_s3, %att4_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4) outs(%state_p0_4, %acc_p0_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p0_r0_s3, %pf_acc_p0_r0_s3, %pf_bidx4_r0_s3, %att0_out_r0_s3)
      : (
        !nest.event<"att4_grid_r0_s3">, !nest.event<"att4_inrel_r0_s3">,
        !nest.event<"att4_out_r0_s3">)
    %pf_bidx5_r0_s3 = nest.dma.prefetch.async %381 into %block_idx_p1_3
      depends_on(%att3_inrel_r0_s3) : !nest.event<"pf_bidx5_r0_s3">
    %att5_grid_r0_s3, %att5_inrel_r0_s3, %att5_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4) outs(%state_p1_4, %acc_p1_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p1_r0_s3, %pf_acc_p1_r0_s3, %pf_bidx5_r0_s3, %att1_out_r0_s3)
      : (
        !nest.event<"att5_grid_r0_s3">, !nest.event<"att5_inrel_r0_s3">,
        !nest.event<"att5_out_r0_s3">)
    %pf_bidx6_r0_s3 = nest.dma.prefetch.async %382 into %block_idx_p0_3
      depends_on(%att4_inrel_r0_s3) : !nest.event<"pf_bidx6_r0_s3">
    %att6_grid_r0_s3, %att6_inrel_r0_s3, %att6_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4) outs(%state_p2_4, %acc_p2_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p2_r0_s3, %pf_acc_p2_r0_s3, %pf_bidx6_r0_s3, %att2_out_r0_s3)
      : (
        !nest.event<"att6_grid_r0_s3">, !nest.event<"att6_inrel_r0_s3">,
        !nest.event<"att6_out_r0_s3">)
    %pf_bidx7_r0_s3 = nest.dma.prefetch.async %383 into %block_idx_p1_3
      depends_on(%att5_inrel_r0_s3) : !nest.event<"pf_bidx7_r0_s3">
    %att7_grid_r0_s3, %att7_inrel_r0_s3, %att7_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4) outs(%state_p3_4, %acc_p3_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p3_r0_s3, %pf_acc_p3_r0_s3, %pf_bidx7_r0_s3, %att3_out_r0_s3)
      : (
        !nest.event<"att7_grid_r0_s3">, !nest.event<"att7_inrel_r0_s3">,
        !nest.event<"att7_out_r0_s3">)
    %pf_bidx8_r0_s3 = nest.dma.prefetch.async %384 into %block_idx_p0_3
      depends_on(%att6_inrel_r0_s3) : !nest.event<"pf_bidx8_r0_s3">
    %att8_grid_r0_s3, %att8_inrel_r0_s3, %att8_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4) outs(%state_p0_4, %acc_p0_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p0_r0_s3, %pf_acc_p0_r0_s3, %pf_bidx8_r0_s3, %att4_out_r0_s3)
      : (
        !nest.event<"att8_grid_r0_s3">, !nest.event<"att8_inrel_r0_s3">,
        !nest.event<"att8_out_r0_s3">)
    %pf_bidx9_r0_s3 = nest.dma.prefetch.async %385 into %block_idx_p1_3
      depends_on(%att7_inrel_r0_s3) : !nest.event<"pf_bidx9_r0_s3">
    %att9_grid_r0_s3, %att9_inrel_r0_s3, %att9_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4) outs(%state_p1_4, %acc_p1_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p1_r0_s3, %pf_acc_p1_r0_s3, %pf_bidx9_r0_s3, %att5_out_r0_s3)
      : (
        !nest.event<"att9_grid_r0_s3">, !nest.event<"att9_inrel_r0_s3">,
        !nest.event<"att9_out_r0_s3">)
    %pf_bidx10_r0_s3 = nest.dma.prefetch.async %386 into %block_idx_p0_3
      depends_on(%att8_inrel_r0_s3) : !nest.event<"pf_bidx10_r0_s3">
    %att10_grid_r0_s3, %att10_inrel_r0_s3, %att10_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4) outs(%state_p2_4, %acc_p2_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p2_r0_s3, %pf_acc_p2_r0_s3, %pf_bidx10_r0_s3, %att6_out_r0_s3)
      : (
        !nest.event<"att10_grid_r0_s3">, !nest.event<"att10_inrel_r0_s3">,
        !nest.event<"att10_out_r0_s3">)
    %pf_bidx11_r0_s3 = nest.dma.prefetch.async %387 into %block_idx_p1_3
      depends_on(%att9_inrel_r0_s3) : !nest.event<"pf_bidx11_r0_s3">
    %att11_grid_r0_s3, %att11_inrel_r0_s3, %att11_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4) outs(%state_p3_4, %acc_p3_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p3_r0_s3, %pf_acc_p3_r0_s3, %pf_bidx11_r0_s3, %att7_out_r0_s3)
      : (
        !nest.event<"att11_grid_r0_s3">, !nest.event<"att11_inrel_r0_s3">,
        !nest.event<"att11_out_r0_s3">)
    %pf_bidx12_r0_s3 = nest.dma.prefetch.async %388 into %block_idx_p0_3
      depends_on(%att10_inrel_r0_s3) : !nest.event<"pf_bidx12_r0_s3">
    %att12_grid_r0_s3, %att12_inrel_r0_s3, %att12_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p0_4, %acc_p0_4) outs(%state_p0_4, %acc_p0_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p0_r0_s3, %pf_acc_p0_r0_s3, %pf_bidx12_r0_s3, %att8_out_r0_s3)
      : (
        !nest.event<"att12_grid_r0_s3">, !nest.event<"att12_inrel_r0_s3">,
        !nest.event<"att12_out_r0_s3">)
    %pf_bidx13_r0_s3 = nest.dma.prefetch.async %389 into %block_idx_p1_3
      depends_on(%att11_inrel_r0_s3) : !nest.event<"pf_bidx13_r0_s3">
    %att13_grid_r0_s3, %att13_inrel_r0_s3, %att13_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p1_4, %acc_p1_4) outs(%state_p1_4, %acc_p1_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p1_r0_s3, %pf_acc_p1_r0_s3, %pf_bidx13_r0_s3, %att9_out_r0_s3)
      : (
        !nest.event<"att13_grid_r0_s3">, !nest.event<"att13_inrel_r0_s3">,
        !nest.event<"att13_out_r0_s3">)
    %pf_bidx14_r0_s3 = nest.dma.prefetch.async %390 into %block_idx_p0_3
      depends_on(%att12_inrel_r0_s3) : !nest.event<"pf_bidx14_r0_s3">
    %att14_grid_r0_s3, %att14_inrel_r0_s3, %att14_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4)
      ins(%block_idx_p0_3, %q_l2_18, %state_p2_4, %acc_p2_4) outs(%state_p2_4, %acc_p2_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p2_r0_s3, %pf_acc_p2_r0_s3, %pf_bidx14_r0_s3, %att10_out_r0_s3)
      : (
        !nest.event<"att14_grid_r0_s3">, !nest.event<"att14_inrel_r0_s3">,
        !nest.event<"att14_out_r0_s3">)
    %pf_bidx15_r0_s3 = nest.dma.prefetch.async %391 into %block_idx_p1_3
      depends_on(%att13_inrel_r0_s3) : !nest.event<"pf_bidx15_r0_s3">
    %att15_grid_r0_s3, %att15_inrel_r0_s3, %att15_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4)
      ins(%block_idx_p1_3, %q_l2_18, %state_p3_4, %acc_p3_4) outs(%state_p3_4, %acc_p3_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p3_r0_s3, %pf_acc_p3_r0_s3, %pf_bidx15_r0_s3, %att11_out_r0_s3)
      : (
        !nest.event<"att15_grid_r0_s3">, !nest.event<"att15_inrel_r0_s3">,
        !nest.event<"att15_out_r0_s3">)
    %att16_grid_r0_s3, %att16_inrel_r0_s3, %att16_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_t3_final_r0 l1_mode = 1 tasks(%392) globals(%362)
      bindings(%append_idx_3, %q_l2_18, %state_p0_4, %acc_p0_4)
      ins(%append_idx_3, %q_l2_18, %state_p0_4, %acc_p0_4) outs(%state_p0_4, %acc_p0_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r0_s3, %pf_state_p0_r0_s3, %pf_acc_p0_r0_s3, %pf_aidx_r0_s3, %append_grid_r0_s3,
        %att12_out_r0_s3)
      : (
        !nest.event<"att16_grid_r0_s3">, !nest.event<"att16_inrel_r0_s3">,
        !nest.event<"att16_out_r0_s3">)
    %merge_grid_r0_s3, %merge_inrel_r0_s3, %merge_out_r0_s3 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%392) globals()
      bindings(
        %state_p0_4, %state_p1_4, %state_p2_4, %state_p3_4, %acc_p0_4, %acc_p1_4, %acc_p2_4,
        %acc_p3_4, %out_l2_4)
      ins(
        %state_p0_4, %state_p1_4, %state_p2_4, %state_p3_4, %acc_p0_4, %acc_p1_4, %acc_p2_4,
        %acc_p3_4)
      outs(%out_l2_4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att16_out_r0_s3, %att13_out_r0_s3, %att14_out_r0_s3, %att15_out_r0_s3) : (
        !nest.event<"merge_grid_r0_s3">, !nest.event<"merge_inrel_r0_s3">,
        !nest.event<"merge_out_r0_s3">)
    %out_store_r0_s3 = nest.dma.store.async %out_l2_4 into %375 depends_on(%merge_out_r0_s3)
      : !nest.event<"out_store_r0_s3">
    nest.await %append_grid_r0_s3, %att16_grid_r0_s3, %merge_grid_r0_s3, %out_store_r0_s3
    nest.release %block_idx_p0_3 depends_on(
      %pf_bidx0_r0_s3, %pf_bidx2_r0_s3, %pf_bidx4_r0_s3, %pf_bidx6_r0_s3, %pf_bidx8_r0_s3,
      %pf_bidx10_r0_s3, %pf_bidx12_r0_s3, %pf_bidx14_r0_s3, %att0_inrel_r0_s3, %att2_inrel_r0_s3,
      %att4_inrel_r0_s3, %att6_inrel_r0_s3, %att8_inrel_r0_s3, %att10_inrel_r0_s3,
      %att12_inrel_r0_s3, %att14_inrel_r0_s3)
    nest.release %block_idx_p1_3 depends_on(
      %pf_bidx1_r0_s3, %pf_bidx3_r0_s3, %pf_bidx5_r0_s3, %pf_bidx7_r0_s3, %pf_bidx9_r0_s3,
      %pf_bidx11_r0_s3, %pf_bidx13_r0_s3, %pf_bidx15_r0_s3, %att1_inrel_r0_s3, %att3_inrel_r0_s3,
      %att5_inrel_r0_s3, %att7_inrel_r0_s3, %att9_inrel_r0_s3, %att11_inrel_r0_s3,
      %att13_inrel_r0_s3, %att15_inrel_r0_s3)
    nest.release %k_new_3 depends_on(%pf_k_r0_s3, %append_inrel_r0_s3)
    nest.release %v_new_3 depends_on(%pf_v_r0_s3, %append_inrel_r0_s3)
    nest.release %append_idx_3 depends_on(%pf_aidx_r0_s3, %append_inrel_r0_s3, %att16_inrel_r0_s3)
    nest.release %q_l2_18 depends_on(
      %pf_q_r0_s3, %att0_inrel_r0_s3, %att1_inrel_r0_s3, %att2_inrel_r0_s3, %att3_inrel_r0_s3,
      %att4_inrel_r0_s3, %att5_inrel_r0_s3, %att6_inrel_r0_s3, %att7_inrel_r0_s3,
      %att8_inrel_r0_s3, %att9_inrel_r0_s3, %att10_inrel_r0_s3, %att11_inrel_r0_s3,
      %att12_inrel_r0_s3, %att13_inrel_r0_s3, %att14_inrel_r0_s3, %att15_inrel_r0_s3,
      %att16_inrel_r0_s3)
    nest.release %state_p0_4 depends_on(
      %pf_state_p0_r0_s3, %att0_inrel_r0_s3, %att4_inrel_r0_s3, %att8_inrel_r0_s3,
      %att12_inrel_r0_s3, %att16_inrel_r0_s3, %att0_out_r0_s3, %att4_out_r0_s3, %att8_out_r0_s3,
      %att12_out_r0_s3, %att16_out_r0_s3, %merge_inrel_r0_s3)
    nest.release %acc_p0_4 depends_on(
      %pf_acc_p0_r0_s3, %att0_inrel_r0_s3, %att4_inrel_r0_s3, %att8_inrel_r0_s3,
      %att12_inrel_r0_s3, %att16_inrel_r0_s3, %att0_out_r0_s3, %att4_out_r0_s3, %att8_out_r0_s3,
      %att12_out_r0_s3, %att16_out_r0_s3, %merge_inrel_r0_s3)
    nest.release %state_p1_4 depends_on(
      %pf_state_p1_r0_s3, %att1_inrel_r0_s3, %att5_inrel_r0_s3, %att9_inrel_r0_s3,
      %att13_inrel_r0_s3, %att1_out_r0_s3, %att5_out_r0_s3, %att9_out_r0_s3, %att13_out_r0_s3,
      %merge_inrel_r0_s3)
    nest.release %acc_p1_4 depends_on(
      %pf_acc_p1_r0_s3, %att1_inrel_r0_s3, %att5_inrel_r0_s3, %att9_inrel_r0_s3,
      %att13_inrel_r0_s3, %att1_out_r0_s3, %att5_out_r0_s3, %att9_out_r0_s3, %att13_out_r0_s3,
      %merge_inrel_r0_s3)
    nest.release %state_p2_4 depends_on(
      %pf_state_p2_r0_s3, %att2_inrel_r0_s3, %att6_inrel_r0_s3, %att10_inrel_r0_s3,
      %att14_inrel_r0_s3, %att2_out_r0_s3, %att6_out_r0_s3, %att10_out_r0_s3, %att14_out_r0_s3,
      %merge_inrel_r0_s3)
    nest.release %acc_p2_4 depends_on(
      %pf_acc_p2_r0_s3, %att2_inrel_r0_s3, %att6_inrel_r0_s3, %att10_inrel_r0_s3,
      %att14_inrel_r0_s3, %att2_out_r0_s3, %att6_out_r0_s3, %att10_out_r0_s3, %att14_out_r0_s3,
      %merge_inrel_r0_s3)
    nest.release %state_p3_4 depends_on(
      %pf_state_p3_r0_s3, %att3_inrel_r0_s3, %att7_inrel_r0_s3, %att11_inrel_r0_s3,
      %att15_inrel_r0_s3, %att3_out_r0_s3, %att7_out_r0_s3, %att11_out_r0_s3, %att15_out_r0_s3,
      %merge_inrel_r0_s3)
    nest.release %acc_p3_4 depends_on(
      %pf_acc_p3_r0_s3, %att3_inrel_r0_s3, %att7_inrel_r0_s3, %att11_inrel_r0_s3,
      %att15_inrel_r0_s3, %att3_out_r0_s3, %att7_out_r0_s3, %att11_out_r0_s3, %att15_out_r0_s3,
      %merge_inrel_r0_s3)
    nest.release %out_l2_4 depends_on(%merge_out_r0_s3, %out_store_r0_s3)
    nest.return
  }
  nest.context @step_r1_s0(
    %POOL_4: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_4: !nest.global_memref<147xi32>,
    %APPEND_IDS_4: !nest.global_memref<12xi32>, %Q_IN_4: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_4: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_4: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_4: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_4: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_4: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 136, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_4 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_4 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_4 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_19 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_5 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_5 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_5 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_5 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_5 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_5 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_5 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_5 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_4 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_4 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_5 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %394 = nest.subview %POOL_4 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %395 = nest.subview %K_NEW_4 offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %396 = nest.subview %V_NEW_4 offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %397 = nest.subview %APPEND_IDS_4 offsets = [4] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %398 = nest.subview %Q_IN_4 offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %399 = nest.subview %S_INIT_4 offsets = [1, 0, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %400 = nest.subview %S_INIT_4 offsets = [1, 0, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %401 = nest.subview %S_INIT_4 offsets = [1, 0, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %402 = nest.subview %S_INIT_4 offsets = [1, 0, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %403 = nest.subview %O_INIT_4 offsets = [1, 0, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %404 = nest.subview %O_INIT_4 offsets = [1, 0, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %405 = nest.subview %O_INIT_4 offsets = [1, 0, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %406 = nest.subview %O_INIT_4 offsets = [1, 0, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %407 = nest.subview %OUT_4 offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %408 = nest.subview %BLOCK_TABLE_4 offsets = [49] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %409 = nest.subview %BLOCK_TABLE_4 offsets = [50] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %410 = nest.subview %BLOCK_TABLE_4 offsets = [51] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %411 = nest.subview %BLOCK_TABLE_4 offsets = [52] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %412 = nest.subview %BLOCK_TABLE_4 offsets = [53] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %413 = nest.subview %BLOCK_TABLE_4 offsets = [54] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %414 = nest.subview %BLOCK_TABLE_4 offsets = [55] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %415 = nest.subview %BLOCK_TABLE_4 offsets = [56] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %416 = nest.subview %BLOCK_TABLE_4 offsets = [57] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %417 = nest.subview %BLOCK_TABLE_4 offsets = [58] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %418 = nest.subview %BLOCK_TABLE_4 offsets = [59] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %419 = nest.subview %BLOCK_TABLE_4 offsets = [60] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %420 = nest.subview %BLOCK_TABLE_4 offsets = [61] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %421 = nest.subview %BLOCK_TABLE_4 offsets = [62] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %422 = nest.subview %BLOCK_TABLE_4 offsets = [63] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %423 = nest.subview %BLOCK_TABLE_4 offsets = [64] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %424 = nest.subview %BLOCK_TABLE_4 offsets = [65] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %425 = nest.subview %BLOCK_TABLE_4 offsets = [66] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %426 = nest.subview %BLOCK_TABLE_4 offsets = [67] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %427 = nest.subview %BLOCK_TABLE_4 offsets = [68] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %428 = nest.subview %BLOCK_TABLE_4 offsets = [69] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %429 = nest.subview %BLOCK_TABLE_4 offsets = [70] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %430 = nest.subview %BLOCK_TABLE_4 offsets = [71] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %431 = nest.subview %BLOCK_TABLE_4 offsets = [72] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %432 = nest.subview %BLOCK_TABLE_4 offsets = [73] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %433 = nest.subview %BLOCK_TABLE_4 offsets = [74] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %434 = nest.subview %BLOCK_TABLE_4 offsets = [75] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %435 = nest.subview %BLOCK_TABLE_4 offsets = [76] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %436 = nest.subview %BLOCK_TABLE_4 offsets = [77] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %437 = nest.subview %BLOCK_TABLE_4 offsets = [78] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %438 = nest.subview %BLOCK_TABLE_4 offsets = [79] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r1_s0 = nest.dma.prefetch.async %395 into %k_new_4 : !nest.event<"pf_k_r1_s0">
    %pf_v_r1_s0 = nest.dma.prefetch.async %396 into %v_new_4 : !nest.event<"pf_v_r1_s0">
    %pf_aidx_r1_s0 = nest.dma.prefetch.async %397 into %append_idx_4 : !nest.event<"pf_aidx_r1_s0">
    %pf_q_r1_s0 = nest.dma.prefetch.async %398 into %q_l2_19 : !nest.event<"pf_q_r1_s0">
    %pf_state_p0_r1_s0 = nest.dma.prefetch.async %399 into %state_p0_5
      : !nest.event<"pf_state_p0_r1_s0">
    %pf_state_p1_r1_s0 = nest.dma.prefetch.async %400 into %state_p1_5
      : !nest.event<"pf_state_p1_r1_s0">
    %pf_state_p2_r1_s0 = nest.dma.prefetch.async %401 into %state_p2_5
      : !nest.event<"pf_state_p2_r1_s0">
    %pf_state_p3_r1_s0 = nest.dma.prefetch.async %402 into %state_p3_5
      : !nest.event<"pf_state_p3_r1_s0">
    %pf_acc_p0_r1_s0 = nest.dma.prefetch.async %403 into %acc_p0_5 : !nest.event<"pf_acc_p0_r1_s0">
    %pf_acc_p1_r1_s0 = nest.dma.prefetch.async %404 into %acc_p1_5 : !nest.event<"pf_acc_p1_r1_s0">
    %pf_acc_p2_r1_s0 = nest.dma.prefetch.async %405 into %acc_p2_5 : !nest.event<"pf_acc_p2_r1_s0">
    %pf_acc_p3_r1_s0 = nest.dma.prefetch.async %406 into %acc_p3_5 : !nest.event<"pf_acc_p3_r1_s0">
    %439 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r1_s0, %append_inrel_r1_s0, %440 =
      nest.dispatch.tasks.async @paged_attention_append_r1_tip15 l1_mode = 1 tasks(%439)
      globals(%394) bindings(%k_new_4, %v_new_4, %append_idx_4)
      ins(%k_new_4, %v_new_4, %append_idx_4) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r1_s0, %pf_v_r1_s0, %pf_aidx_r1_s0)
      : (!nest.event<"append_grid_r1_s0">, !nest.event<"append_inrel_r1_s0">, !nest.event<"">)
    %pf_bidx0_r1_s0 = nest.dma.prefetch.async %408 into %block_idx_p0_4
      : !nest.event<"pf_bidx0_r1_s0">
    %att0_grid_r1_s0, %att0_inrel_r1_s0, %att0_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx0_r1_s0) : (
        !nest.event<"att0_grid_r1_s0">, !nest.event<"att0_inrel_r1_s0">,
        !nest.event<"att0_out_r1_s0">)
    %pf_bidx1_r1_s0 = nest.dma.prefetch.async %409 into %block_idx_p1_4
      : !nest.event<"pf_bidx1_r1_s0">
    %att1_grid_r1_s0, %att1_inrel_r1_s0, %att1_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx1_r1_s0) : (
        !nest.event<"att1_grid_r1_s0">, !nest.event<"att1_inrel_r1_s0">,
        !nest.event<"att1_out_r1_s0">)
    %pf_bidx2_r1_s0 = nest.dma.prefetch.async %410 into %block_idx_p0_4
      depends_on(%att0_inrel_r1_s0) : !nest.event<"pf_bidx2_r1_s0">
    %att2_grid_r1_s0, %att2_inrel_r1_s0, %att2_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx2_r1_s0) : (
        !nest.event<"att2_grid_r1_s0">, !nest.event<"att2_inrel_r1_s0">,
        !nest.event<"att2_out_r1_s0">)
    %pf_bidx3_r1_s0 = nest.dma.prefetch.async %411 into %block_idx_p1_4
      depends_on(%att1_inrel_r1_s0) : !nest.event<"pf_bidx3_r1_s0">
    %att3_grid_r1_s0, %att3_inrel_r1_s0, %att3_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx3_r1_s0) : (
        !nest.event<"att3_grid_r1_s0">, !nest.event<"att3_inrel_r1_s0">,
        !nest.event<"att3_out_r1_s0">)
    %pf_bidx4_r1_s0 = nest.dma.prefetch.async %412 into %block_idx_p0_4
      depends_on(%att2_inrel_r1_s0) : !nest.event<"pf_bidx4_r1_s0">
    %att4_grid_r1_s0, %att4_inrel_r1_s0, %att4_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx4_r1_s0, %att0_out_r1_s0)
      : (
        !nest.event<"att4_grid_r1_s0">, !nest.event<"att4_inrel_r1_s0">,
        !nest.event<"att4_out_r1_s0">)
    %pf_bidx5_r1_s0 = nest.dma.prefetch.async %413 into %block_idx_p1_4
      depends_on(%att3_inrel_r1_s0) : !nest.event<"pf_bidx5_r1_s0">
    %att5_grid_r1_s0, %att5_inrel_r1_s0, %att5_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx5_r1_s0, %att1_out_r1_s0)
      : (
        !nest.event<"att5_grid_r1_s0">, !nest.event<"att5_inrel_r1_s0">,
        !nest.event<"att5_out_r1_s0">)
    %pf_bidx6_r1_s0 = nest.dma.prefetch.async %414 into %block_idx_p0_4
      depends_on(%att4_inrel_r1_s0) : !nest.event<"pf_bidx6_r1_s0">
    %att6_grid_r1_s0, %att6_inrel_r1_s0, %att6_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx6_r1_s0, %att2_out_r1_s0)
      : (
        !nest.event<"att6_grid_r1_s0">, !nest.event<"att6_inrel_r1_s0">,
        !nest.event<"att6_out_r1_s0">)
    %pf_bidx7_r1_s0 = nest.dma.prefetch.async %415 into %block_idx_p1_4
      depends_on(%att5_inrel_r1_s0) : !nest.event<"pf_bidx7_r1_s0">
    %att7_grid_r1_s0, %att7_inrel_r1_s0, %att7_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx7_r1_s0, %att3_out_r1_s0)
      : (
        !nest.event<"att7_grid_r1_s0">, !nest.event<"att7_inrel_r1_s0">,
        !nest.event<"att7_out_r1_s0">)
    %pf_bidx8_r1_s0 = nest.dma.prefetch.async %416 into %block_idx_p0_4
      depends_on(%att6_inrel_r1_s0) : !nest.event<"pf_bidx8_r1_s0">
    %att8_grid_r1_s0, %att8_inrel_r1_s0, %att8_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx8_r1_s0, %att4_out_r1_s0)
      : (
        !nest.event<"att8_grid_r1_s0">, !nest.event<"att8_inrel_r1_s0">,
        !nest.event<"att8_out_r1_s0">)
    %pf_bidx9_r1_s0 = nest.dma.prefetch.async %417 into %block_idx_p1_4
      depends_on(%att7_inrel_r1_s0) : !nest.event<"pf_bidx9_r1_s0">
    %att9_grid_r1_s0, %att9_inrel_r1_s0, %att9_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx9_r1_s0, %att5_out_r1_s0)
      : (
        !nest.event<"att9_grid_r1_s0">, !nest.event<"att9_inrel_r1_s0">,
        !nest.event<"att9_out_r1_s0">)
    %pf_bidx10_r1_s0 = nest.dma.prefetch.async %418 into %block_idx_p0_4
      depends_on(%att8_inrel_r1_s0) : !nest.event<"pf_bidx10_r1_s0">
    %att10_grid_r1_s0, %att10_inrel_r1_s0, %att10_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx10_r1_s0, %att6_out_r1_s0)
      : (
        !nest.event<"att10_grid_r1_s0">, !nest.event<"att10_inrel_r1_s0">,
        !nest.event<"att10_out_r1_s0">)
    %pf_bidx11_r1_s0 = nest.dma.prefetch.async %419 into %block_idx_p1_4
      depends_on(%att9_inrel_r1_s0) : !nest.event<"pf_bidx11_r1_s0">
    %att11_grid_r1_s0, %att11_inrel_r1_s0, %att11_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx11_r1_s0, %att7_out_r1_s0)
      : (
        !nest.event<"att11_grid_r1_s0">, !nest.event<"att11_inrel_r1_s0">,
        !nest.event<"att11_out_r1_s0">)
    %pf_bidx12_r1_s0 = nest.dma.prefetch.async %420 into %block_idx_p0_4
      depends_on(%att10_inrel_r1_s0) : !nest.event<"pf_bidx12_r1_s0">
    %att12_grid_r1_s0, %att12_inrel_r1_s0, %att12_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx12_r1_s0, %att8_out_r1_s0)
      : (
        !nest.event<"att12_grid_r1_s0">, !nest.event<"att12_inrel_r1_s0">,
        !nest.event<"att12_out_r1_s0">)
    %pf_bidx13_r1_s0 = nest.dma.prefetch.async %421 into %block_idx_p1_4
      depends_on(%att11_inrel_r1_s0) : !nest.event<"pf_bidx13_r1_s0">
    %att13_grid_r1_s0, %att13_inrel_r1_s0, %att13_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx13_r1_s0, %att9_out_r1_s0)
      : (
        !nest.event<"att13_grid_r1_s0">, !nest.event<"att13_inrel_r1_s0">,
        !nest.event<"att13_out_r1_s0">)
    %pf_bidx14_r1_s0 = nest.dma.prefetch.async %422 into %block_idx_p0_4
      depends_on(%att12_inrel_r1_s0) : !nest.event<"pf_bidx14_r1_s0">
    %att14_grid_r1_s0, %att14_inrel_r1_s0, %att14_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx14_r1_s0, %att10_out_r1_s0)
      : (
        !nest.event<"att14_grid_r1_s0">, !nest.event<"att14_inrel_r1_s0">,
        !nest.event<"att14_out_r1_s0">)
    %pf_bidx15_r1_s0 = nest.dma.prefetch.async %423 into %block_idx_p1_4
      depends_on(%att13_inrel_r1_s0) : !nest.event<"pf_bidx15_r1_s0">
    %att15_grid_r1_s0, %att15_inrel_r1_s0, %att15_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx15_r1_s0, %att11_out_r1_s0)
      : (
        !nest.event<"att15_grid_r1_s0">, !nest.event<"att15_inrel_r1_s0">,
        !nest.event<"att15_out_r1_s0">)
    %pf_bidx16_r1_s0 = nest.dma.prefetch.async %424 into %block_idx_p0_4
      depends_on(%att14_inrel_r1_s0) : !nest.event<"pf_bidx16_r1_s0">
    %att16_grid_r1_s0, %att16_inrel_r1_s0, %att16_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx16_r1_s0, %att12_out_r1_s0)
      : (
        !nest.event<"att16_grid_r1_s0">, !nest.event<"att16_inrel_r1_s0">,
        !nest.event<"att16_out_r1_s0">)
    %pf_bidx17_r1_s0 = nest.dma.prefetch.async %425 into %block_idx_p1_4
      depends_on(%att15_inrel_r1_s0) : !nest.event<"pf_bidx17_r1_s0">
    %att17_grid_r1_s0, %att17_inrel_r1_s0, %att17_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx17_r1_s0, %att13_out_r1_s0)
      : (
        !nest.event<"att17_grid_r1_s0">, !nest.event<"att17_inrel_r1_s0">,
        !nest.event<"att17_out_r1_s0">)
    %pf_bidx18_r1_s0 = nest.dma.prefetch.async %426 into %block_idx_p0_4
      depends_on(%att16_inrel_r1_s0) : !nest.event<"pf_bidx18_r1_s0">
    %att18_grid_r1_s0, %att18_inrel_r1_s0, %att18_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx18_r1_s0, %att14_out_r1_s0)
      : (
        !nest.event<"att18_grid_r1_s0">, !nest.event<"att18_inrel_r1_s0">,
        !nest.event<"att18_out_r1_s0">)
    %pf_bidx19_r1_s0 = nest.dma.prefetch.async %427 into %block_idx_p1_4
      depends_on(%att17_inrel_r1_s0) : !nest.event<"pf_bidx19_r1_s0">
    %att19_grid_r1_s0, %att19_inrel_r1_s0, %att19_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx19_r1_s0, %att15_out_r1_s0)
      : (
        !nest.event<"att19_grid_r1_s0">, !nest.event<"att19_inrel_r1_s0">,
        !nest.event<"att19_out_r1_s0">)
    %pf_bidx20_r1_s0 = nest.dma.prefetch.async %428 into %block_idx_p0_4
      depends_on(%att18_inrel_r1_s0) : !nest.event<"pf_bidx20_r1_s0">
    %att20_grid_r1_s0, %att20_inrel_r1_s0, %att20_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx20_r1_s0, %att16_out_r1_s0)
      : (
        !nest.event<"att20_grid_r1_s0">, !nest.event<"att20_inrel_r1_s0">,
        !nest.event<"att20_out_r1_s0">)
    %pf_bidx21_r1_s0 = nest.dma.prefetch.async %429 into %block_idx_p1_4
      depends_on(%att19_inrel_r1_s0) : !nest.event<"pf_bidx21_r1_s0">
    %att21_grid_r1_s0, %att21_inrel_r1_s0, %att21_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx21_r1_s0, %att17_out_r1_s0)
      : (
        !nest.event<"att21_grid_r1_s0">, !nest.event<"att21_inrel_r1_s0">,
        !nest.event<"att21_out_r1_s0">)
    %pf_bidx22_r1_s0 = nest.dma.prefetch.async %430 into %block_idx_p0_4
      depends_on(%att20_inrel_r1_s0) : !nest.event<"pf_bidx22_r1_s0">
    %att22_grid_r1_s0, %att22_inrel_r1_s0, %att22_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx22_r1_s0, %att18_out_r1_s0)
      : (
        !nest.event<"att22_grid_r1_s0">, !nest.event<"att22_inrel_r1_s0">,
        !nest.event<"att22_out_r1_s0">)
    %pf_bidx23_r1_s0 = nest.dma.prefetch.async %431 into %block_idx_p1_4
      depends_on(%att21_inrel_r1_s0) : !nest.event<"pf_bidx23_r1_s0">
    %att23_grid_r1_s0, %att23_inrel_r1_s0, %att23_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx23_r1_s0, %att19_out_r1_s0)
      : (
        !nest.event<"att23_grid_r1_s0">, !nest.event<"att23_inrel_r1_s0">,
        !nest.event<"att23_out_r1_s0">)
    %pf_bidx24_r1_s0 = nest.dma.prefetch.async %432 into %block_idx_p0_4
      depends_on(%att22_inrel_r1_s0) : !nest.event<"pf_bidx24_r1_s0">
    %att24_grid_r1_s0, %att24_inrel_r1_s0, %att24_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx24_r1_s0, %att20_out_r1_s0)
      : (
        !nest.event<"att24_grid_r1_s0">, !nest.event<"att24_inrel_r1_s0">,
        !nest.event<"att24_out_r1_s0">)
    %pf_bidx25_r1_s0 = nest.dma.prefetch.async %433 into %block_idx_p1_4
      depends_on(%att23_inrel_r1_s0) : !nest.event<"pf_bidx25_r1_s0">
    %att25_grid_r1_s0, %att25_inrel_r1_s0, %att25_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx25_r1_s0, %att21_out_r1_s0)
      : (
        !nest.event<"att25_grid_r1_s0">, !nest.event<"att25_inrel_r1_s0">,
        !nest.event<"att25_out_r1_s0">)
    %pf_bidx26_r1_s0 = nest.dma.prefetch.async %434 into %block_idx_p0_4
      depends_on(%att24_inrel_r1_s0) : !nest.event<"pf_bidx26_r1_s0">
    %att26_grid_r1_s0, %att26_inrel_r1_s0, %att26_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx26_r1_s0, %att22_out_r1_s0)
      : (
        !nest.event<"att26_grid_r1_s0">, !nest.event<"att26_inrel_r1_s0">,
        !nest.event<"att26_out_r1_s0">)
    %pf_bidx27_r1_s0 = nest.dma.prefetch.async %435 into %block_idx_p1_4
      depends_on(%att25_inrel_r1_s0) : !nest.event<"pf_bidx27_r1_s0">
    %att27_grid_r1_s0, %att27_inrel_r1_s0, %att27_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_bidx27_r1_s0, %att23_out_r1_s0)
      : (
        !nest.event<"att27_grid_r1_s0">, !nest.event<"att27_inrel_r1_s0">,
        !nest.event<"att27_out_r1_s0">)
    %pf_bidx28_r1_s0 = nest.dma.prefetch.async %436 into %block_idx_p0_4
      depends_on(%att26_inrel_r1_s0) : !nest.event<"pf_bidx28_r1_s0">
    %att28_grid_r1_s0, %att28_inrel_r1_s0, %att28_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p0_5, %acc_p0_5) outs(%state_p0_5, %acc_p0_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p0_r1_s0, %pf_acc_p0_r1_s0, %pf_bidx28_r1_s0, %att24_out_r1_s0)
      : (
        !nest.event<"att28_grid_r1_s0">, !nest.event<"att28_inrel_r1_s0">,
        !nest.event<"att28_out_r1_s0">)
    %pf_bidx29_r1_s0 = nest.dma.prefetch.async %437 into %block_idx_p1_4
      depends_on(%att27_inrel_r1_s0) : !nest.event<"pf_bidx29_r1_s0">
    %att29_grid_r1_s0, %att29_inrel_r1_s0, %att29_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5)
      ins(%block_idx_p1_4, %q_l2_19, %state_p1_5, %acc_p1_5) outs(%state_p1_5, %acc_p1_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p1_r1_s0, %pf_acc_p1_r1_s0, %pf_bidx29_r1_s0, %att25_out_r1_s0)
      : (
        !nest.event<"att29_grid_r1_s0">, !nest.event<"att29_inrel_r1_s0">,
        !nest.event<"att29_out_r1_s0">)
    %pf_bidx30_r1_s0 = nest.dma.prefetch.async %438 into %block_idx_p0_4
      depends_on(%att28_inrel_r1_s0) : !nest.event<"pf_bidx30_r1_s0">
    %att30_grid_r1_s0, %att30_inrel_r1_s0, %att30_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5)
      ins(%block_idx_p0_4, %q_l2_19, %state_p2_5, %acc_p2_5) outs(%state_p2_5, %acc_p2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p2_r1_s0, %pf_acc_p2_r1_s0, %pf_bidx30_r1_s0, %att26_out_r1_s0)
      : (
        !nest.event<"att30_grid_r1_s0">, !nest.event<"att30_inrel_r1_s0">,
        !nest.event<"att30_out_r1_s0">)
    %att31_grid_r1_s0, %att31_inrel_r1_s0, %att31_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_final_r1 l1_mode = 1 tasks(%439) globals(%394)
      bindings(%append_idx_4, %q_l2_19, %state_p3_5, %acc_p3_5)
      ins(%append_idx_4, %q_l2_19, %state_p3_5, %acc_p3_5) outs(%state_p3_5, %acc_p3_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s0, %pf_state_p3_r1_s0, %pf_acc_p3_r1_s0, %pf_aidx_r1_s0, %append_grid_r1_s0,
        %att27_out_r1_s0)
      : (
        !nest.event<"att31_grid_r1_s0">, !nest.event<"att31_inrel_r1_s0">,
        !nest.event<"att31_out_r1_s0">)
    %merge_grid_r1_s0, %merge_inrel_r1_s0, %merge_out_r1_s0 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%439) globals()
      bindings(
        %state_p0_5, %state_p1_5, %state_p2_5, %state_p3_5, %acc_p0_5, %acc_p1_5, %acc_p2_5,
        %acc_p3_5, %out_l2_5)
      ins(
        %state_p0_5, %state_p1_5, %state_p2_5, %state_p3_5, %acc_p0_5, %acc_p1_5, %acc_p2_5,
        %acc_p3_5)
      outs(%out_l2_5)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att28_out_r1_s0, %att29_out_r1_s0, %att30_out_r1_s0, %att31_out_r1_s0) : (
        !nest.event<"merge_grid_r1_s0">, !nest.event<"merge_inrel_r1_s0">,
        !nest.event<"merge_out_r1_s0">)
    %out_store_r1_s0 = nest.dma.store.async %out_l2_5 into %407 depends_on(%merge_out_r1_s0)
      : !nest.event<"out_store_r1_s0">
    nest.await %append_grid_r1_s0, %att31_grid_r1_s0, %merge_grid_r1_s0, %out_store_r1_s0
    nest.release %block_idx_p0_4 depends_on(
      %pf_bidx0_r1_s0, %pf_bidx2_r1_s0, %pf_bidx4_r1_s0, %pf_bidx6_r1_s0, %pf_bidx8_r1_s0,
      %pf_bidx10_r1_s0, %pf_bidx12_r1_s0, %pf_bidx14_r1_s0, %pf_bidx16_r1_s0, %pf_bidx18_r1_s0,
      %pf_bidx20_r1_s0, %pf_bidx22_r1_s0, %pf_bidx24_r1_s0, %pf_bidx26_r1_s0, %pf_bidx28_r1_s0,
      %pf_bidx30_r1_s0, %att0_inrel_r1_s0, %att2_inrel_r1_s0, %att4_inrel_r1_s0,
      %att6_inrel_r1_s0, %att8_inrel_r1_s0, %att10_inrel_r1_s0, %att12_inrel_r1_s0,
      %att14_inrel_r1_s0, %att16_inrel_r1_s0, %att18_inrel_r1_s0, %att20_inrel_r1_s0,
      %att22_inrel_r1_s0, %att24_inrel_r1_s0, %att26_inrel_r1_s0, %att28_inrel_r1_s0,
      %att30_inrel_r1_s0)
    nest.release %block_idx_p1_4 depends_on(
      %pf_bidx1_r1_s0, %pf_bidx3_r1_s0, %pf_bidx5_r1_s0, %pf_bidx7_r1_s0, %pf_bidx9_r1_s0,
      %pf_bidx11_r1_s0, %pf_bidx13_r1_s0, %pf_bidx15_r1_s0, %pf_bidx17_r1_s0, %pf_bidx19_r1_s0,
      %pf_bidx21_r1_s0, %pf_bidx23_r1_s0, %pf_bidx25_r1_s0, %pf_bidx27_r1_s0, %pf_bidx29_r1_s0,
      %att1_inrel_r1_s0, %att3_inrel_r1_s0, %att5_inrel_r1_s0, %att7_inrel_r1_s0,
      %att9_inrel_r1_s0, %att11_inrel_r1_s0, %att13_inrel_r1_s0, %att15_inrel_r1_s0,
      %att17_inrel_r1_s0, %att19_inrel_r1_s0, %att21_inrel_r1_s0, %att23_inrel_r1_s0,
      %att25_inrel_r1_s0, %att27_inrel_r1_s0, %att29_inrel_r1_s0)
    nest.release %k_new_4 depends_on(%pf_k_r1_s0, %append_inrel_r1_s0)
    nest.release %v_new_4 depends_on(%pf_v_r1_s0, %append_inrel_r1_s0)
    nest.release %append_idx_4 depends_on(%pf_aidx_r1_s0, %append_inrel_r1_s0, %att31_inrel_r1_s0)
    nest.release %q_l2_19 depends_on(
      %pf_q_r1_s0, %att0_inrel_r1_s0, %att1_inrel_r1_s0, %att2_inrel_r1_s0, %att3_inrel_r1_s0,
      %att4_inrel_r1_s0, %att5_inrel_r1_s0, %att6_inrel_r1_s0, %att7_inrel_r1_s0,
      %att8_inrel_r1_s0, %att9_inrel_r1_s0, %att10_inrel_r1_s0, %att11_inrel_r1_s0,
      %att12_inrel_r1_s0, %att13_inrel_r1_s0, %att14_inrel_r1_s0, %att15_inrel_r1_s0,
      %att16_inrel_r1_s0, %att17_inrel_r1_s0, %att18_inrel_r1_s0, %att19_inrel_r1_s0,
      %att20_inrel_r1_s0, %att21_inrel_r1_s0, %att22_inrel_r1_s0, %att23_inrel_r1_s0,
      %att24_inrel_r1_s0, %att25_inrel_r1_s0, %att26_inrel_r1_s0, %att27_inrel_r1_s0,
      %att28_inrel_r1_s0, %att29_inrel_r1_s0, %att30_inrel_r1_s0, %att31_inrel_r1_s0)
    nest.release %state_p0_5 depends_on(
      %pf_state_p0_r1_s0, %att0_inrel_r1_s0, %att4_inrel_r1_s0, %att8_inrel_r1_s0,
      %att12_inrel_r1_s0, %att16_inrel_r1_s0, %att20_inrel_r1_s0, %att24_inrel_r1_s0,
      %att28_inrel_r1_s0, %att0_out_r1_s0, %att4_out_r1_s0, %att8_out_r1_s0, %att12_out_r1_s0,
      %att16_out_r1_s0, %att20_out_r1_s0, %att24_out_r1_s0, %att28_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %acc_p0_5 depends_on(
      %pf_acc_p0_r1_s0, %att0_inrel_r1_s0, %att4_inrel_r1_s0, %att8_inrel_r1_s0,
      %att12_inrel_r1_s0, %att16_inrel_r1_s0, %att20_inrel_r1_s0, %att24_inrel_r1_s0,
      %att28_inrel_r1_s0, %att0_out_r1_s0, %att4_out_r1_s0, %att8_out_r1_s0, %att12_out_r1_s0,
      %att16_out_r1_s0, %att20_out_r1_s0, %att24_out_r1_s0, %att28_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %state_p1_5 depends_on(
      %pf_state_p1_r1_s0, %att1_inrel_r1_s0, %att5_inrel_r1_s0, %att9_inrel_r1_s0,
      %att13_inrel_r1_s0, %att17_inrel_r1_s0, %att21_inrel_r1_s0, %att25_inrel_r1_s0,
      %att29_inrel_r1_s0, %att1_out_r1_s0, %att5_out_r1_s0, %att9_out_r1_s0, %att13_out_r1_s0,
      %att17_out_r1_s0, %att21_out_r1_s0, %att25_out_r1_s0, %att29_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %acc_p1_5 depends_on(
      %pf_acc_p1_r1_s0, %att1_inrel_r1_s0, %att5_inrel_r1_s0, %att9_inrel_r1_s0,
      %att13_inrel_r1_s0, %att17_inrel_r1_s0, %att21_inrel_r1_s0, %att25_inrel_r1_s0,
      %att29_inrel_r1_s0, %att1_out_r1_s0, %att5_out_r1_s0, %att9_out_r1_s0, %att13_out_r1_s0,
      %att17_out_r1_s0, %att21_out_r1_s0, %att25_out_r1_s0, %att29_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %state_p2_5 depends_on(
      %pf_state_p2_r1_s0, %att2_inrel_r1_s0, %att6_inrel_r1_s0, %att10_inrel_r1_s0,
      %att14_inrel_r1_s0, %att18_inrel_r1_s0, %att22_inrel_r1_s0, %att26_inrel_r1_s0,
      %att30_inrel_r1_s0, %att2_out_r1_s0, %att6_out_r1_s0, %att10_out_r1_s0, %att14_out_r1_s0,
      %att18_out_r1_s0, %att22_out_r1_s0, %att26_out_r1_s0, %att30_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %acc_p2_5 depends_on(
      %pf_acc_p2_r1_s0, %att2_inrel_r1_s0, %att6_inrel_r1_s0, %att10_inrel_r1_s0,
      %att14_inrel_r1_s0, %att18_inrel_r1_s0, %att22_inrel_r1_s0, %att26_inrel_r1_s0,
      %att30_inrel_r1_s0, %att2_out_r1_s0, %att6_out_r1_s0, %att10_out_r1_s0, %att14_out_r1_s0,
      %att18_out_r1_s0, %att22_out_r1_s0, %att26_out_r1_s0, %att30_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %state_p3_5 depends_on(
      %pf_state_p3_r1_s0, %att3_inrel_r1_s0, %att7_inrel_r1_s0, %att11_inrel_r1_s0,
      %att15_inrel_r1_s0, %att19_inrel_r1_s0, %att23_inrel_r1_s0, %att27_inrel_r1_s0,
      %att31_inrel_r1_s0, %att3_out_r1_s0, %att7_out_r1_s0, %att11_out_r1_s0, %att15_out_r1_s0,
      %att19_out_r1_s0, %att23_out_r1_s0, %att27_out_r1_s0, %att31_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %acc_p3_5 depends_on(
      %pf_acc_p3_r1_s0, %att3_inrel_r1_s0, %att7_inrel_r1_s0, %att11_inrel_r1_s0,
      %att15_inrel_r1_s0, %att19_inrel_r1_s0, %att23_inrel_r1_s0, %att27_inrel_r1_s0,
      %att31_inrel_r1_s0, %att3_out_r1_s0, %att7_out_r1_s0, %att11_out_r1_s0, %att15_out_r1_s0,
      %att19_out_r1_s0, %att23_out_r1_s0, %att27_out_r1_s0, %att31_out_r1_s0, %merge_inrel_r1_s0)
    nest.release %out_l2_5 depends_on(%merge_out_r1_s0, %out_store_r1_s0)
    nest.return
  }
  nest.context @step_r1_s1(
    %POOL_5: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_5: !nest.global_memref<147xi32>,
    %APPEND_IDS_5: !nest.global_memref<12xi32>, %Q_IN_5: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_5: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_5: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_5: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_5: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_5: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 140, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_5 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_5 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_5 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_20 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_6 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_6 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_6 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_6 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_6 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_6 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_6 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_6 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_5 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_5 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_6 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %441 = nest.subview %POOL_5 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %442 = nest.subview %K_NEW_5 offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %443 = nest.subview %V_NEW_5 offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %444 = nest.subview %APPEND_IDS_5 offsets = [5] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %445 = nest.subview %Q_IN_5 offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %446 = nest.subview %S_INIT_5 offsets = [1, 1, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %447 = nest.subview %S_INIT_5 offsets = [1, 1, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %448 = nest.subview %S_INIT_5 offsets = [1, 1, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %449 = nest.subview %S_INIT_5 offsets = [1, 1, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %450 = nest.subview %O_INIT_5 offsets = [1, 1, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %451 = nest.subview %O_INIT_5 offsets = [1, 1, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %452 = nest.subview %O_INIT_5 offsets = [1, 1, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %453 = nest.subview %O_INIT_5 offsets = [1, 1, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %454 = nest.subview %OUT_5 offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %455 = nest.subview %BLOCK_TABLE_5 offsets = [49] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %456 = nest.subview %BLOCK_TABLE_5 offsets = [50] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %457 = nest.subview %BLOCK_TABLE_5 offsets = [51] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %458 = nest.subview %BLOCK_TABLE_5 offsets = [52] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %459 = nest.subview %BLOCK_TABLE_5 offsets = [53] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %460 = nest.subview %BLOCK_TABLE_5 offsets = [54] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %461 = nest.subview %BLOCK_TABLE_5 offsets = [55] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %462 = nest.subview %BLOCK_TABLE_5 offsets = [56] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %463 = nest.subview %BLOCK_TABLE_5 offsets = [57] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %464 = nest.subview %BLOCK_TABLE_5 offsets = [58] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %465 = nest.subview %BLOCK_TABLE_5 offsets = [59] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %466 = nest.subview %BLOCK_TABLE_5 offsets = [60] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %467 = nest.subview %BLOCK_TABLE_5 offsets = [61] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %468 = nest.subview %BLOCK_TABLE_5 offsets = [62] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %469 = nest.subview %BLOCK_TABLE_5 offsets = [63] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %470 = nest.subview %BLOCK_TABLE_5 offsets = [64] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %471 = nest.subview %BLOCK_TABLE_5 offsets = [65] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %472 = nest.subview %BLOCK_TABLE_5 offsets = [66] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %473 = nest.subview %BLOCK_TABLE_5 offsets = [67] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %474 = nest.subview %BLOCK_TABLE_5 offsets = [68] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %475 = nest.subview %BLOCK_TABLE_5 offsets = [69] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %476 = nest.subview %BLOCK_TABLE_5 offsets = [70] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %477 = nest.subview %BLOCK_TABLE_5 offsets = [71] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %478 = nest.subview %BLOCK_TABLE_5 offsets = [72] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %479 = nest.subview %BLOCK_TABLE_5 offsets = [73] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %480 = nest.subview %BLOCK_TABLE_5 offsets = [74] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %481 = nest.subview %BLOCK_TABLE_5 offsets = [75] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %482 = nest.subview %BLOCK_TABLE_5 offsets = [76] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %483 = nest.subview %BLOCK_TABLE_5 offsets = [77] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %484 = nest.subview %BLOCK_TABLE_5 offsets = [78] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %485 = nest.subview %BLOCK_TABLE_5 offsets = [79] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %486 = nest.subview %BLOCK_TABLE_5 offsets = [80] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r1_s1 = nest.dma.prefetch.async %442 into %k_new_5 : !nest.event<"pf_k_r1_s1">
    %pf_v_r1_s1 = nest.dma.prefetch.async %443 into %v_new_5 : !nest.event<"pf_v_r1_s1">
    %pf_aidx_r1_s1 = nest.dma.prefetch.async %444 into %append_idx_5 : !nest.event<"pf_aidx_r1_s1">
    %pf_q_r1_s1 = nest.dma.prefetch.async %445 into %q_l2_20 : !nest.event<"pf_q_r1_s1">
    %pf_state_p0_r1_s1 = nest.dma.prefetch.async %446 into %state_p0_6
      : !nest.event<"pf_state_p0_r1_s1">
    %pf_state_p1_r1_s1 = nest.dma.prefetch.async %447 into %state_p1_6
      : !nest.event<"pf_state_p1_r1_s1">
    %pf_state_p2_r1_s1 = nest.dma.prefetch.async %448 into %state_p2_6
      : !nest.event<"pf_state_p2_r1_s1">
    %pf_state_p3_r1_s1 = nest.dma.prefetch.async %449 into %state_p3_6
      : !nest.event<"pf_state_p3_r1_s1">
    %pf_acc_p0_r1_s1 = nest.dma.prefetch.async %450 into %acc_p0_6 : !nest.event<"pf_acc_p0_r1_s1">
    %pf_acc_p1_r1_s1 = nest.dma.prefetch.async %451 into %acc_p1_6 : !nest.event<"pf_acc_p1_r1_s1">
    %pf_acc_p2_r1_s1 = nest.dma.prefetch.async %452 into %acc_p2_6 : !nest.event<"pf_acc_p2_r1_s1">
    %pf_acc_p3_r1_s1 = nest.dma.prefetch.async %453 into %acc_p3_6 : !nest.event<"pf_acc_p3_r1_s1">
    %487 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r1_s1, %append_inrel_r1_s1, %488 =
      nest.dispatch.tasks.async @paged_attention_append_r1_tip0 l1_mode = 1 tasks(%487)
      globals(%441) bindings(%k_new_5, %v_new_5, %append_idx_5)
      ins(%k_new_5, %v_new_5, %append_idx_5) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r1_s1, %pf_v_r1_s1, %pf_aidx_r1_s1)
      : (!nest.event<"append_grid_r1_s1">, !nest.event<"append_inrel_r1_s1">, !nest.event<"">)
    %pf_bidx0_r1_s1 = nest.dma.prefetch.async %455 into %block_idx_p0_5
      : !nest.event<"pf_bidx0_r1_s1">
    %att0_grid_r1_s1, %att0_inrel_r1_s1, %att0_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx0_r1_s1) : (
        !nest.event<"att0_grid_r1_s1">, !nest.event<"att0_inrel_r1_s1">,
        !nest.event<"att0_out_r1_s1">)
    %pf_bidx1_r1_s1 = nest.dma.prefetch.async %456 into %block_idx_p1_5
      : !nest.event<"pf_bidx1_r1_s1">
    %att1_grid_r1_s1, %att1_inrel_r1_s1, %att1_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx1_r1_s1) : (
        !nest.event<"att1_grid_r1_s1">, !nest.event<"att1_inrel_r1_s1">,
        !nest.event<"att1_out_r1_s1">)
    %pf_bidx2_r1_s1 = nest.dma.prefetch.async %457 into %block_idx_p0_5
      depends_on(%att0_inrel_r1_s1) : !nest.event<"pf_bidx2_r1_s1">
    %att2_grid_r1_s1, %att2_inrel_r1_s1, %att2_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx2_r1_s1) : (
        !nest.event<"att2_grid_r1_s1">, !nest.event<"att2_inrel_r1_s1">,
        !nest.event<"att2_out_r1_s1">)
    %pf_bidx3_r1_s1 = nest.dma.prefetch.async %458 into %block_idx_p1_5
      depends_on(%att1_inrel_r1_s1) : !nest.event<"pf_bidx3_r1_s1">
    %att3_grid_r1_s1, %att3_inrel_r1_s1, %att3_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx3_r1_s1) : (
        !nest.event<"att3_grid_r1_s1">, !nest.event<"att3_inrel_r1_s1">,
        !nest.event<"att3_out_r1_s1">)
    %pf_bidx4_r1_s1 = nest.dma.prefetch.async %459 into %block_idx_p0_5
      depends_on(%att2_inrel_r1_s1) : !nest.event<"pf_bidx4_r1_s1">
    %att4_grid_r1_s1, %att4_inrel_r1_s1, %att4_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx4_r1_s1, %att0_out_r1_s1)
      : (
        !nest.event<"att4_grid_r1_s1">, !nest.event<"att4_inrel_r1_s1">,
        !nest.event<"att4_out_r1_s1">)
    %pf_bidx5_r1_s1 = nest.dma.prefetch.async %460 into %block_idx_p1_5
      depends_on(%att3_inrel_r1_s1) : !nest.event<"pf_bidx5_r1_s1">
    %att5_grid_r1_s1, %att5_inrel_r1_s1, %att5_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx5_r1_s1, %att1_out_r1_s1)
      : (
        !nest.event<"att5_grid_r1_s1">, !nest.event<"att5_inrel_r1_s1">,
        !nest.event<"att5_out_r1_s1">)
    %pf_bidx6_r1_s1 = nest.dma.prefetch.async %461 into %block_idx_p0_5
      depends_on(%att4_inrel_r1_s1) : !nest.event<"pf_bidx6_r1_s1">
    %att6_grid_r1_s1, %att6_inrel_r1_s1, %att6_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx6_r1_s1, %att2_out_r1_s1)
      : (
        !nest.event<"att6_grid_r1_s1">, !nest.event<"att6_inrel_r1_s1">,
        !nest.event<"att6_out_r1_s1">)
    %pf_bidx7_r1_s1 = nest.dma.prefetch.async %462 into %block_idx_p1_5
      depends_on(%att5_inrel_r1_s1) : !nest.event<"pf_bidx7_r1_s1">
    %att7_grid_r1_s1, %att7_inrel_r1_s1, %att7_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx7_r1_s1, %att3_out_r1_s1)
      : (
        !nest.event<"att7_grid_r1_s1">, !nest.event<"att7_inrel_r1_s1">,
        !nest.event<"att7_out_r1_s1">)
    %pf_bidx8_r1_s1 = nest.dma.prefetch.async %463 into %block_idx_p0_5
      depends_on(%att6_inrel_r1_s1) : !nest.event<"pf_bidx8_r1_s1">
    %att8_grid_r1_s1, %att8_inrel_r1_s1, %att8_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx8_r1_s1, %att4_out_r1_s1)
      : (
        !nest.event<"att8_grid_r1_s1">, !nest.event<"att8_inrel_r1_s1">,
        !nest.event<"att8_out_r1_s1">)
    %pf_bidx9_r1_s1 = nest.dma.prefetch.async %464 into %block_idx_p1_5
      depends_on(%att7_inrel_r1_s1) : !nest.event<"pf_bidx9_r1_s1">
    %att9_grid_r1_s1, %att9_inrel_r1_s1, %att9_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx9_r1_s1, %att5_out_r1_s1)
      : (
        !nest.event<"att9_grid_r1_s1">, !nest.event<"att9_inrel_r1_s1">,
        !nest.event<"att9_out_r1_s1">)
    %pf_bidx10_r1_s1 = nest.dma.prefetch.async %465 into %block_idx_p0_5
      depends_on(%att8_inrel_r1_s1) : !nest.event<"pf_bidx10_r1_s1">
    %att10_grid_r1_s1, %att10_inrel_r1_s1, %att10_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx10_r1_s1, %att6_out_r1_s1)
      : (
        !nest.event<"att10_grid_r1_s1">, !nest.event<"att10_inrel_r1_s1">,
        !nest.event<"att10_out_r1_s1">)
    %pf_bidx11_r1_s1 = nest.dma.prefetch.async %466 into %block_idx_p1_5
      depends_on(%att9_inrel_r1_s1) : !nest.event<"pf_bidx11_r1_s1">
    %att11_grid_r1_s1, %att11_inrel_r1_s1, %att11_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx11_r1_s1, %att7_out_r1_s1)
      : (
        !nest.event<"att11_grid_r1_s1">, !nest.event<"att11_inrel_r1_s1">,
        !nest.event<"att11_out_r1_s1">)
    %pf_bidx12_r1_s1 = nest.dma.prefetch.async %467 into %block_idx_p0_5
      depends_on(%att10_inrel_r1_s1) : !nest.event<"pf_bidx12_r1_s1">
    %att12_grid_r1_s1, %att12_inrel_r1_s1, %att12_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx12_r1_s1, %att8_out_r1_s1)
      : (
        !nest.event<"att12_grid_r1_s1">, !nest.event<"att12_inrel_r1_s1">,
        !nest.event<"att12_out_r1_s1">)
    %pf_bidx13_r1_s1 = nest.dma.prefetch.async %468 into %block_idx_p1_5
      depends_on(%att11_inrel_r1_s1) : !nest.event<"pf_bidx13_r1_s1">
    %att13_grid_r1_s1, %att13_inrel_r1_s1, %att13_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx13_r1_s1, %att9_out_r1_s1)
      : (
        !nest.event<"att13_grid_r1_s1">, !nest.event<"att13_inrel_r1_s1">,
        !nest.event<"att13_out_r1_s1">)
    %pf_bidx14_r1_s1 = nest.dma.prefetch.async %469 into %block_idx_p0_5
      depends_on(%att12_inrel_r1_s1) : !nest.event<"pf_bidx14_r1_s1">
    %att14_grid_r1_s1, %att14_inrel_r1_s1, %att14_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx14_r1_s1, %att10_out_r1_s1)
      : (
        !nest.event<"att14_grid_r1_s1">, !nest.event<"att14_inrel_r1_s1">,
        !nest.event<"att14_out_r1_s1">)
    %pf_bidx15_r1_s1 = nest.dma.prefetch.async %470 into %block_idx_p1_5
      depends_on(%att13_inrel_r1_s1) : !nest.event<"pf_bidx15_r1_s1">
    %att15_grid_r1_s1, %att15_inrel_r1_s1, %att15_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx15_r1_s1, %att11_out_r1_s1)
      : (
        !nest.event<"att15_grid_r1_s1">, !nest.event<"att15_inrel_r1_s1">,
        !nest.event<"att15_out_r1_s1">)
    %pf_bidx16_r1_s1 = nest.dma.prefetch.async %471 into %block_idx_p0_5
      depends_on(%att14_inrel_r1_s1) : !nest.event<"pf_bidx16_r1_s1">
    %att16_grid_r1_s1, %att16_inrel_r1_s1, %att16_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx16_r1_s1, %att12_out_r1_s1)
      : (
        !nest.event<"att16_grid_r1_s1">, !nest.event<"att16_inrel_r1_s1">,
        !nest.event<"att16_out_r1_s1">)
    %pf_bidx17_r1_s1 = nest.dma.prefetch.async %472 into %block_idx_p1_5
      depends_on(%att15_inrel_r1_s1) : !nest.event<"pf_bidx17_r1_s1">
    %att17_grid_r1_s1, %att17_inrel_r1_s1, %att17_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx17_r1_s1, %att13_out_r1_s1)
      : (
        !nest.event<"att17_grid_r1_s1">, !nest.event<"att17_inrel_r1_s1">,
        !nest.event<"att17_out_r1_s1">)
    %pf_bidx18_r1_s1 = nest.dma.prefetch.async %473 into %block_idx_p0_5
      depends_on(%att16_inrel_r1_s1) : !nest.event<"pf_bidx18_r1_s1">
    %att18_grid_r1_s1, %att18_inrel_r1_s1, %att18_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx18_r1_s1, %att14_out_r1_s1)
      : (
        !nest.event<"att18_grid_r1_s1">, !nest.event<"att18_inrel_r1_s1">,
        !nest.event<"att18_out_r1_s1">)
    %pf_bidx19_r1_s1 = nest.dma.prefetch.async %474 into %block_idx_p1_5
      depends_on(%att17_inrel_r1_s1) : !nest.event<"pf_bidx19_r1_s1">
    %att19_grid_r1_s1, %att19_inrel_r1_s1, %att19_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx19_r1_s1, %att15_out_r1_s1)
      : (
        !nest.event<"att19_grid_r1_s1">, !nest.event<"att19_inrel_r1_s1">,
        !nest.event<"att19_out_r1_s1">)
    %pf_bidx20_r1_s1 = nest.dma.prefetch.async %475 into %block_idx_p0_5
      depends_on(%att18_inrel_r1_s1) : !nest.event<"pf_bidx20_r1_s1">
    %att20_grid_r1_s1, %att20_inrel_r1_s1, %att20_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx20_r1_s1, %att16_out_r1_s1)
      : (
        !nest.event<"att20_grid_r1_s1">, !nest.event<"att20_inrel_r1_s1">,
        !nest.event<"att20_out_r1_s1">)
    %pf_bidx21_r1_s1 = nest.dma.prefetch.async %476 into %block_idx_p1_5
      depends_on(%att19_inrel_r1_s1) : !nest.event<"pf_bidx21_r1_s1">
    %att21_grid_r1_s1, %att21_inrel_r1_s1, %att21_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx21_r1_s1, %att17_out_r1_s1)
      : (
        !nest.event<"att21_grid_r1_s1">, !nest.event<"att21_inrel_r1_s1">,
        !nest.event<"att21_out_r1_s1">)
    %pf_bidx22_r1_s1 = nest.dma.prefetch.async %477 into %block_idx_p0_5
      depends_on(%att20_inrel_r1_s1) : !nest.event<"pf_bidx22_r1_s1">
    %att22_grid_r1_s1, %att22_inrel_r1_s1, %att22_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx22_r1_s1, %att18_out_r1_s1)
      : (
        !nest.event<"att22_grid_r1_s1">, !nest.event<"att22_inrel_r1_s1">,
        !nest.event<"att22_out_r1_s1">)
    %pf_bidx23_r1_s1 = nest.dma.prefetch.async %478 into %block_idx_p1_5
      depends_on(%att21_inrel_r1_s1) : !nest.event<"pf_bidx23_r1_s1">
    %att23_grid_r1_s1, %att23_inrel_r1_s1, %att23_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx23_r1_s1, %att19_out_r1_s1)
      : (
        !nest.event<"att23_grid_r1_s1">, !nest.event<"att23_inrel_r1_s1">,
        !nest.event<"att23_out_r1_s1">)
    %pf_bidx24_r1_s1 = nest.dma.prefetch.async %479 into %block_idx_p0_5
      depends_on(%att22_inrel_r1_s1) : !nest.event<"pf_bidx24_r1_s1">
    %att24_grid_r1_s1, %att24_inrel_r1_s1, %att24_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx24_r1_s1, %att20_out_r1_s1)
      : (
        !nest.event<"att24_grid_r1_s1">, !nest.event<"att24_inrel_r1_s1">,
        !nest.event<"att24_out_r1_s1">)
    %pf_bidx25_r1_s1 = nest.dma.prefetch.async %480 into %block_idx_p1_5
      depends_on(%att23_inrel_r1_s1) : !nest.event<"pf_bidx25_r1_s1">
    %att25_grid_r1_s1, %att25_inrel_r1_s1, %att25_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx25_r1_s1, %att21_out_r1_s1)
      : (
        !nest.event<"att25_grid_r1_s1">, !nest.event<"att25_inrel_r1_s1">,
        !nest.event<"att25_out_r1_s1">)
    %pf_bidx26_r1_s1 = nest.dma.prefetch.async %481 into %block_idx_p0_5
      depends_on(%att24_inrel_r1_s1) : !nest.event<"pf_bidx26_r1_s1">
    %att26_grid_r1_s1, %att26_inrel_r1_s1, %att26_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx26_r1_s1, %att22_out_r1_s1)
      : (
        !nest.event<"att26_grid_r1_s1">, !nest.event<"att26_inrel_r1_s1">,
        !nest.event<"att26_out_r1_s1">)
    %pf_bidx27_r1_s1 = nest.dma.prefetch.async %482 into %block_idx_p1_5
      depends_on(%att25_inrel_r1_s1) : !nest.event<"pf_bidx27_r1_s1">
    %att27_grid_r1_s1, %att27_inrel_r1_s1, %att27_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx27_r1_s1, %att23_out_r1_s1)
      : (
        !nest.event<"att27_grid_r1_s1">, !nest.event<"att27_inrel_r1_s1">,
        !nest.event<"att27_out_r1_s1">)
    %pf_bidx28_r1_s1 = nest.dma.prefetch.async %483 into %block_idx_p0_5
      depends_on(%att26_inrel_r1_s1) : !nest.event<"pf_bidx28_r1_s1">
    %att28_grid_r1_s1, %att28_inrel_r1_s1, %att28_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_bidx28_r1_s1, %att24_out_r1_s1)
      : (
        !nest.event<"att28_grid_r1_s1">, !nest.event<"att28_inrel_r1_s1">,
        !nest.event<"att28_out_r1_s1">)
    %pf_bidx29_r1_s1 = nest.dma.prefetch.async %484 into %block_idx_p1_5
      depends_on(%att27_inrel_r1_s1) : !nest.event<"pf_bidx29_r1_s1">
    %att29_grid_r1_s1, %att29_inrel_r1_s1, %att29_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p1_6, %acc_p1_6) outs(%state_p1_6, %acc_p1_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p1_r1_s1, %pf_acc_p1_r1_s1, %pf_bidx29_r1_s1, %att25_out_r1_s1)
      : (
        !nest.event<"att29_grid_r1_s1">, !nest.event<"att29_inrel_r1_s1">,
        !nest.event<"att29_out_r1_s1">)
    %pf_bidx30_r1_s1 = nest.dma.prefetch.async %485 into %block_idx_p0_5
      depends_on(%att28_inrel_r1_s1) : !nest.event<"pf_bidx30_r1_s1">
    %att30_grid_r1_s1, %att30_inrel_r1_s1, %att30_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6)
      ins(%block_idx_p0_5, %q_l2_20, %state_p2_6, %acc_p2_6) outs(%state_p2_6, %acc_p2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p2_r1_s1, %pf_acc_p2_r1_s1, %pf_bidx30_r1_s1, %att26_out_r1_s1)
      : (
        !nest.event<"att30_grid_r1_s1">, !nest.event<"att30_inrel_r1_s1">,
        !nest.event<"att30_out_r1_s1">)
    %pf_bidx31_r1_s1 = nest.dma.prefetch.async %486 into %block_idx_p1_5
      depends_on(%att29_inrel_r1_s1) : !nest.event<"pf_bidx31_r1_s1">
    %att31_grid_r1_s1, %att31_inrel_r1_s1, %att31_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6)
      ins(%block_idx_p1_5, %q_l2_20, %state_p3_6, %acc_p3_6) outs(%state_p3_6, %acc_p3_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p3_r1_s1, %pf_acc_p3_r1_s1, %pf_bidx31_r1_s1, %att27_out_r1_s1)
      : (
        !nest.event<"att31_grid_r1_s1">, !nest.event<"att31_inrel_r1_s1">,
        !nest.event<"att31_out_r1_s1">)
    %att32_grid_r1_s1, %att32_inrel_r1_s1, %att32_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_t1_final_r1 l1_mode = 1 tasks(%487) globals(%441)
      bindings(%append_idx_5, %q_l2_20, %state_p0_6, %acc_p0_6)
      ins(%append_idx_5, %q_l2_20, %state_p0_6, %acc_p0_6) outs(%state_p0_6, %acc_p0_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s1, %pf_state_p0_r1_s1, %pf_acc_p0_r1_s1, %pf_aidx_r1_s1, %append_grid_r1_s1,
        %att28_out_r1_s1)
      : (
        !nest.event<"att32_grid_r1_s1">, !nest.event<"att32_inrel_r1_s1">,
        !nest.event<"att32_out_r1_s1">)
    %merge_grid_r1_s1, %merge_inrel_r1_s1, %merge_out_r1_s1 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%487) globals()
      bindings(
        %state_p0_6, %state_p1_6, %state_p2_6, %state_p3_6, %acc_p0_6, %acc_p1_6, %acc_p2_6,
        %acc_p3_6, %out_l2_6)
      ins(
        %state_p0_6, %state_p1_6, %state_p2_6, %state_p3_6, %acc_p0_6, %acc_p1_6, %acc_p2_6,
        %acc_p3_6)
      outs(%out_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att32_out_r1_s1, %att29_out_r1_s1, %att30_out_r1_s1, %att31_out_r1_s1) : (
        !nest.event<"merge_grid_r1_s1">, !nest.event<"merge_inrel_r1_s1">,
        !nest.event<"merge_out_r1_s1">)
    %out_store_r1_s1 = nest.dma.store.async %out_l2_6 into %454 depends_on(%merge_out_r1_s1)
      : !nest.event<"out_store_r1_s1">
    nest.await %append_grid_r1_s1, %att32_grid_r1_s1, %merge_grid_r1_s1, %out_store_r1_s1
    nest.release %block_idx_p0_5 depends_on(
      %pf_bidx0_r1_s1, %pf_bidx2_r1_s1, %pf_bidx4_r1_s1, %pf_bidx6_r1_s1, %pf_bidx8_r1_s1,
      %pf_bidx10_r1_s1, %pf_bidx12_r1_s1, %pf_bidx14_r1_s1, %pf_bidx16_r1_s1, %pf_bidx18_r1_s1,
      %pf_bidx20_r1_s1, %pf_bidx22_r1_s1, %pf_bidx24_r1_s1, %pf_bidx26_r1_s1, %pf_bidx28_r1_s1,
      %pf_bidx30_r1_s1, %att0_inrel_r1_s1, %att2_inrel_r1_s1, %att4_inrel_r1_s1,
      %att6_inrel_r1_s1, %att8_inrel_r1_s1, %att10_inrel_r1_s1, %att12_inrel_r1_s1,
      %att14_inrel_r1_s1, %att16_inrel_r1_s1, %att18_inrel_r1_s1, %att20_inrel_r1_s1,
      %att22_inrel_r1_s1, %att24_inrel_r1_s1, %att26_inrel_r1_s1, %att28_inrel_r1_s1,
      %att30_inrel_r1_s1)
    nest.release %block_idx_p1_5 depends_on(
      %pf_bidx1_r1_s1, %pf_bidx3_r1_s1, %pf_bidx5_r1_s1, %pf_bidx7_r1_s1, %pf_bidx9_r1_s1,
      %pf_bidx11_r1_s1, %pf_bidx13_r1_s1, %pf_bidx15_r1_s1, %pf_bidx17_r1_s1, %pf_bidx19_r1_s1,
      %pf_bidx21_r1_s1, %pf_bidx23_r1_s1, %pf_bidx25_r1_s1, %pf_bidx27_r1_s1, %pf_bidx29_r1_s1,
      %pf_bidx31_r1_s1, %att1_inrel_r1_s1, %att3_inrel_r1_s1, %att5_inrel_r1_s1,
      %att7_inrel_r1_s1, %att9_inrel_r1_s1, %att11_inrel_r1_s1, %att13_inrel_r1_s1,
      %att15_inrel_r1_s1, %att17_inrel_r1_s1, %att19_inrel_r1_s1, %att21_inrel_r1_s1,
      %att23_inrel_r1_s1, %att25_inrel_r1_s1, %att27_inrel_r1_s1, %att29_inrel_r1_s1,
      %att31_inrel_r1_s1)
    nest.release %k_new_5 depends_on(%pf_k_r1_s1, %append_inrel_r1_s1)
    nest.release %v_new_5 depends_on(%pf_v_r1_s1, %append_inrel_r1_s1)
    nest.release %append_idx_5 depends_on(%pf_aidx_r1_s1, %append_inrel_r1_s1, %att32_inrel_r1_s1)
    nest.release %q_l2_20 depends_on(
      %pf_q_r1_s1, %att0_inrel_r1_s1, %att1_inrel_r1_s1, %att2_inrel_r1_s1, %att3_inrel_r1_s1,
      %att4_inrel_r1_s1, %att5_inrel_r1_s1, %att6_inrel_r1_s1, %att7_inrel_r1_s1,
      %att8_inrel_r1_s1, %att9_inrel_r1_s1, %att10_inrel_r1_s1, %att11_inrel_r1_s1,
      %att12_inrel_r1_s1, %att13_inrel_r1_s1, %att14_inrel_r1_s1, %att15_inrel_r1_s1,
      %att16_inrel_r1_s1, %att17_inrel_r1_s1, %att18_inrel_r1_s1, %att19_inrel_r1_s1,
      %att20_inrel_r1_s1, %att21_inrel_r1_s1, %att22_inrel_r1_s1, %att23_inrel_r1_s1,
      %att24_inrel_r1_s1, %att25_inrel_r1_s1, %att26_inrel_r1_s1, %att27_inrel_r1_s1,
      %att28_inrel_r1_s1, %att29_inrel_r1_s1, %att30_inrel_r1_s1, %att31_inrel_r1_s1,
      %att32_inrel_r1_s1)
    nest.release %state_p0_6 depends_on(
      %pf_state_p0_r1_s1, %att0_inrel_r1_s1, %att4_inrel_r1_s1, %att8_inrel_r1_s1,
      %att12_inrel_r1_s1, %att16_inrel_r1_s1, %att20_inrel_r1_s1, %att24_inrel_r1_s1,
      %att28_inrel_r1_s1, %att32_inrel_r1_s1, %att0_out_r1_s1, %att4_out_r1_s1, %att8_out_r1_s1,
      %att12_out_r1_s1, %att16_out_r1_s1, %att20_out_r1_s1, %att24_out_r1_s1, %att28_out_r1_s1,
      %att32_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %acc_p0_6 depends_on(
      %pf_acc_p0_r1_s1, %att0_inrel_r1_s1, %att4_inrel_r1_s1, %att8_inrel_r1_s1,
      %att12_inrel_r1_s1, %att16_inrel_r1_s1, %att20_inrel_r1_s1, %att24_inrel_r1_s1,
      %att28_inrel_r1_s1, %att32_inrel_r1_s1, %att0_out_r1_s1, %att4_out_r1_s1, %att8_out_r1_s1,
      %att12_out_r1_s1, %att16_out_r1_s1, %att20_out_r1_s1, %att24_out_r1_s1, %att28_out_r1_s1,
      %att32_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %state_p1_6 depends_on(
      %pf_state_p1_r1_s1, %att1_inrel_r1_s1, %att5_inrel_r1_s1, %att9_inrel_r1_s1,
      %att13_inrel_r1_s1, %att17_inrel_r1_s1, %att21_inrel_r1_s1, %att25_inrel_r1_s1,
      %att29_inrel_r1_s1, %att1_out_r1_s1, %att5_out_r1_s1, %att9_out_r1_s1, %att13_out_r1_s1,
      %att17_out_r1_s1, %att21_out_r1_s1, %att25_out_r1_s1, %att29_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %acc_p1_6 depends_on(
      %pf_acc_p1_r1_s1, %att1_inrel_r1_s1, %att5_inrel_r1_s1, %att9_inrel_r1_s1,
      %att13_inrel_r1_s1, %att17_inrel_r1_s1, %att21_inrel_r1_s1, %att25_inrel_r1_s1,
      %att29_inrel_r1_s1, %att1_out_r1_s1, %att5_out_r1_s1, %att9_out_r1_s1, %att13_out_r1_s1,
      %att17_out_r1_s1, %att21_out_r1_s1, %att25_out_r1_s1, %att29_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %state_p2_6 depends_on(
      %pf_state_p2_r1_s1, %att2_inrel_r1_s1, %att6_inrel_r1_s1, %att10_inrel_r1_s1,
      %att14_inrel_r1_s1, %att18_inrel_r1_s1, %att22_inrel_r1_s1, %att26_inrel_r1_s1,
      %att30_inrel_r1_s1, %att2_out_r1_s1, %att6_out_r1_s1, %att10_out_r1_s1, %att14_out_r1_s1,
      %att18_out_r1_s1, %att22_out_r1_s1, %att26_out_r1_s1, %att30_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %acc_p2_6 depends_on(
      %pf_acc_p2_r1_s1, %att2_inrel_r1_s1, %att6_inrel_r1_s1, %att10_inrel_r1_s1,
      %att14_inrel_r1_s1, %att18_inrel_r1_s1, %att22_inrel_r1_s1, %att26_inrel_r1_s1,
      %att30_inrel_r1_s1, %att2_out_r1_s1, %att6_out_r1_s1, %att10_out_r1_s1, %att14_out_r1_s1,
      %att18_out_r1_s1, %att22_out_r1_s1, %att26_out_r1_s1, %att30_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %state_p3_6 depends_on(
      %pf_state_p3_r1_s1, %att3_inrel_r1_s1, %att7_inrel_r1_s1, %att11_inrel_r1_s1,
      %att15_inrel_r1_s1, %att19_inrel_r1_s1, %att23_inrel_r1_s1, %att27_inrel_r1_s1,
      %att31_inrel_r1_s1, %att3_out_r1_s1, %att7_out_r1_s1, %att11_out_r1_s1, %att15_out_r1_s1,
      %att19_out_r1_s1, %att23_out_r1_s1, %att27_out_r1_s1, %att31_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %acc_p3_6 depends_on(
      %pf_acc_p3_r1_s1, %att3_inrel_r1_s1, %att7_inrel_r1_s1, %att11_inrel_r1_s1,
      %att15_inrel_r1_s1, %att19_inrel_r1_s1, %att23_inrel_r1_s1, %att27_inrel_r1_s1,
      %att31_inrel_r1_s1, %att3_out_r1_s1, %att7_out_r1_s1, %att11_out_r1_s1, %att15_out_r1_s1,
      %att19_out_r1_s1, %att23_out_r1_s1, %att27_out_r1_s1, %att31_out_r1_s1, %merge_inrel_r1_s1)
    nest.release %out_l2_6 depends_on(%merge_out_r1_s1, %out_store_r1_s1)
    nest.return
  }
  nest.context @step_r1_s2(
    %POOL_6: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_6: !nest.global_memref<147xi32>,
    %APPEND_IDS_6: !nest.global_memref<12xi32>, %Q_IN_6: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_6: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_6: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_6: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_6: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_6: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 140, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_6 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_6 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_6 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_21 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_7 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_7 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_7 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_7 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_7 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_7 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_7 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_7 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_6 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_6 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_7 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %489 = nest.subview %POOL_6 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %490 = nest.subview %K_NEW_6 offsets = [1, 2, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %491 = nest.subview %V_NEW_6 offsets = [1, 2, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %492 = nest.subview %APPEND_IDS_6 offsets = [6] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %493 = nest.subview %Q_IN_6 offsets = [1, 2, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %494 = nest.subview %S_INIT_6 offsets = [1, 2, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %495 = nest.subview %S_INIT_6 offsets = [1, 2, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %496 = nest.subview %S_INIT_6 offsets = [1, 2, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %497 = nest.subview %S_INIT_6 offsets = [1, 2, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %498 = nest.subview %O_INIT_6 offsets = [1, 2, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %499 = nest.subview %O_INIT_6 offsets = [1, 2, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %500 = nest.subview %O_INIT_6 offsets = [1, 2, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %501 = nest.subview %O_INIT_6 offsets = [1, 2, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %502 = nest.subview %OUT_6 offsets = [1, 2, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %503 = nest.subview %BLOCK_TABLE_6 offsets = [49] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %504 = nest.subview %BLOCK_TABLE_6 offsets = [50] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %505 = nest.subview %BLOCK_TABLE_6 offsets = [51] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %506 = nest.subview %BLOCK_TABLE_6 offsets = [52] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %507 = nest.subview %BLOCK_TABLE_6 offsets = [53] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %508 = nest.subview %BLOCK_TABLE_6 offsets = [54] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %509 = nest.subview %BLOCK_TABLE_6 offsets = [55] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %510 = nest.subview %BLOCK_TABLE_6 offsets = [56] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %511 = nest.subview %BLOCK_TABLE_6 offsets = [57] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %512 = nest.subview %BLOCK_TABLE_6 offsets = [58] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %513 = nest.subview %BLOCK_TABLE_6 offsets = [59] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %514 = nest.subview %BLOCK_TABLE_6 offsets = [60] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %515 = nest.subview %BLOCK_TABLE_6 offsets = [61] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %516 = nest.subview %BLOCK_TABLE_6 offsets = [62] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %517 = nest.subview %BLOCK_TABLE_6 offsets = [63] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %518 = nest.subview %BLOCK_TABLE_6 offsets = [64] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %519 = nest.subview %BLOCK_TABLE_6 offsets = [65] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %520 = nest.subview %BLOCK_TABLE_6 offsets = [66] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %521 = nest.subview %BLOCK_TABLE_6 offsets = [67] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %522 = nest.subview %BLOCK_TABLE_6 offsets = [68] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %523 = nest.subview %BLOCK_TABLE_6 offsets = [69] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %524 = nest.subview %BLOCK_TABLE_6 offsets = [70] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %525 = nest.subview %BLOCK_TABLE_6 offsets = [71] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %526 = nest.subview %BLOCK_TABLE_6 offsets = [72] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %527 = nest.subview %BLOCK_TABLE_6 offsets = [73] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %528 = nest.subview %BLOCK_TABLE_6 offsets = [74] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %529 = nest.subview %BLOCK_TABLE_6 offsets = [75] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %530 = nest.subview %BLOCK_TABLE_6 offsets = [76] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %531 = nest.subview %BLOCK_TABLE_6 offsets = [77] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %532 = nest.subview %BLOCK_TABLE_6 offsets = [78] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %533 = nest.subview %BLOCK_TABLE_6 offsets = [79] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %534 = nest.subview %BLOCK_TABLE_6 offsets = [80] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r1_s2 = nest.dma.prefetch.async %490 into %k_new_6 : !nest.event<"pf_k_r1_s2">
    %pf_v_r1_s2 = nest.dma.prefetch.async %491 into %v_new_6 : !nest.event<"pf_v_r1_s2">
    %pf_aidx_r1_s2 = nest.dma.prefetch.async %492 into %append_idx_6 : !nest.event<"pf_aidx_r1_s2">
    %pf_q_r1_s2 = nest.dma.prefetch.async %493 into %q_l2_21 : !nest.event<"pf_q_r1_s2">
    %pf_state_p0_r1_s2 = nest.dma.prefetch.async %494 into %state_p0_7
      : !nest.event<"pf_state_p0_r1_s2">
    %pf_state_p1_r1_s2 = nest.dma.prefetch.async %495 into %state_p1_7
      : !nest.event<"pf_state_p1_r1_s2">
    %pf_state_p2_r1_s2 = nest.dma.prefetch.async %496 into %state_p2_7
      : !nest.event<"pf_state_p2_r1_s2">
    %pf_state_p3_r1_s2 = nest.dma.prefetch.async %497 into %state_p3_7
      : !nest.event<"pf_state_p3_r1_s2">
    %pf_acc_p0_r1_s2 = nest.dma.prefetch.async %498 into %acc_p0_7 : !nest.event<"pf_acc_p0_r1_s2">
    %pf_acc_p1_r1_s2 = nest.dma.prefetch.async %499 into %acc_p1_7 : !nest.event<"pf_acc_p1_r1_s2">
    %pf_acc_p2_r1_s2 = nest.dma.prefetch.async %500 into %acc_p2_7 : !nest.event<"pf_acc_p2_r1_s2">
    %pf_acc_p3_r1_s2 = nest.dma.prefetch.async %501 into %acc_p3_7 : !nest.event<"pf_acc_p3_r1_s2">
    %535 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r1_s2, %append_inrel_r1_s2, %536 =
      nest.dispatch.tasks.async @paged_attention_append_r1_tip1 l1_mode = 1 tasks(%535)
      globals(%489) bindings(%k_new_6, %v_new_6, %append_idx_6)
      ins(%k_new_6, %v_new_6, %append_idx_6) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r1_s2, %pf_v_r1_s2, %pf_aidx_r1_s2)
      : (!nest.event<"append_grid_r1_s2">, !nest.event<"append_inrel_r1_s2">, !nest.event<"">)
    %pf_bidx0_r1_s2 = nest.dma.prefetch.async %503 into %block_idx_p0_6
      : !nest.event<"pf_bidx0_r1_s2">
    %att0_grid_r1_s2, %att0_inrel_r1_s2, %att0_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx0_r1_s2) : (
        !nest.event<"att0_grid_r1_s2">, !nest.event<"att0_inrel_r1_s2">,
        !nest.event<"att0_out_r1_s2">)
    %pf_bidx1_r1_s2 = nest.dma.prefetch.async %504 into %block_idx_p1_6
      : !nest.event<"pf_bidx1_r1_s2">
    %att1_grid_r1_s2, %att1_inrel_r1_s2, %att1_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx1_r1_s2) : (
        !nest.event<"att1_grid_r1_s2">, !nest.event<"att1_inrel_r1_s2">,
        !nest.event<"att1_out_r1_s2">)
    %pf_bidx2_r1_s2 = nest.dma.prefetch.async %505 into %block_idx_p0_6
      depends_on(%att0_inrel_r1_s2) : !nest.event<"pf_bidx2_r1_s2">
    %att2_grid_r1_s2, %att2_inrel_r1_s2, %att2_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx2_r1_s2) : (
        !nest.event<"att2_grid_r1_s2">, !nest.event<"att2_inrel_r1_s2">,
        !nest.event<"att2_out_r1_s2">)
    %pf_bidx3_r1_s2 = nest.dma.prefetch.async %506 into %block_idx_p1_6
      depends_on(%att1_inrel_r1_s2) : !nest.event<"pf_bidx3_r1_s2">
    %att3_grid_r1_s2, %att3_inrel_r1_s2, %att3_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx3_r1_s2) : (
        !nest.event<"att3_grid_r1_s2">, !nest.event<"att3_inrel_r1_s2">,
        !nest.event<"att3_out_r1_s2">)
    %pf_bidx4_r1_s2 = nest.dma.prefetch.async %507 into %block_idx_p0_6
      depends_on(%att2_inrel_r1_s2) : !nest.event<"pf_bidx4_r1_s2">
    %att4_grid_r1_s2, %att4_inrel_r1_s2, %att4_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx4_r1_s2, %att0_out_r1_s2)
      : (
        !nest.event<"att4_grid_r1_s2">, !nest.event<"att4_inrel_r1_s2">,
        !nest.event<"att4_out_r1_s2">)
    %pf_bidx5_r1_s2 = nest.dma.prefetch.async %508 into %block_idx_p1_6
      depends_on(%att3_inrel_r1_s2) : !nest.event<"pf_bidx5_r1_s2">
    %att5_grid_r1_s2, %att5_inrel_r1_s2, %att5_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx5_r1_s2, %att1_out_r1_s2)
      : (
        !nest.event<"att5_grid_r1_s2">, !nest.event<"att5_inrel_r1_s2">,
        !nest.event<"att5_out_r1_s2">)
    %pf_bidx6_r1_s2 = nest.dma.prefetch.async %509 into %block_idx_p0_6
      depends_on(%att4_inrel_r1_s2) : !nest.event<"pf_bidx6_r1_s2">
    %att6_grid_r1_s2, %att6_inrel_r1_s2, %att6_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx6_r1_s2, %att2_out_r1_s2)
      : (
        !nest.event<"att6_grid_r1_s2">, !nest.event<"att6_inrel_r1_s2">,
        !nest.event<"att6_out_r1_s2">)
    %pf_bidx7_r1_s2 = nest.dma.prefetch.async %510 into %block_idx_p1_6
      depends_on(%att5_inrel_r1_s2) : !nest.event<"pf_bidx7_r1_s2">
    %att7_grid_r1_s2, %att7_inrel_r1_s2, %att7_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx7_r1_s2, %att3_out_r1_s2)
      : (
        !nest.event<"att7_grid_r1_s2">, !nest.event<"att7_inrel_r1_s2">,
        !nest.event<"att7_out_r1_s2">)
    %pf_bidx8_r1_s2 = nest.dma.prefetch.async %511 into %block_idx_p0_6
      depends_on(%att6_inrel_r1_s2) : !nest.event<"pf_bidx8_r1_s2">
    %att8_grid_r1_s2, %att8_inrel_r1_s2, %att8_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx8_r1_s2, %att4_out_r1_s2)
      : (
        !nest.event<"att8_grid_r1_s2">, !nest.event<"att8_inrel_r1_s2">,
        !nest.event<"att8_out_r1_s2">)
    %pf_bidx9_r1_s2 = nest.dma.prefetch.async %512 into %block_idx_p1_6
      depends_on(%att7_inrel_r1_s2) : !nest.event<"pf_bidx9_r1_s2">
    %att9_grid_r1_s2, %att9_inrel_r1_s2, %att9_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx9_r1_s2, %att5_out_r1_s2)
      : (
        !nest.event<"att9_grid_r1_s2">, !nest.event<"att9_inrel_r1_s2">,
        !nest.event<"att9_out_r1_s2">)
    %pf_bidx10_r1_s2 = nest.dma.prefetch.async %513 into %block_idx_p0_6
      depends_on(%att8_inrel_r1_s2) : !nest.event<"pf_bidx10_r1_s2">
    %att10_grid_r1_s2, %att10_inrel_r1_s2, %att10_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx10_r1_s2, %att6_out_r1_s2)
      : (
        !nest.event<"att10_grid_r1_s2">, !nest.event<"att10_inrel_r1_s2">,
        !nest.event<"att10_out_r1_s2">)
    %pf_bidx11_r1_s2 = nest.dma.prefetch.async %514 into %block_idx_p1_6
      depends_on(%att9_inrel_r1_s2) : !nest.event<"pf_bidx11_r1_s2">
    %att11_grid_r1_s2, %att11_inrel_r1_s2, %att11_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx11_r1_s2, %att7_out_r1_s2)
      : (
        !nest.event<"att11_grid_r1_s2">, !nest.event<"att11_inrel_r1_s2">,
        !nest.event<"att11_out_r1_s2">)
    %pf_bidx12_r1_s2 = nest.dma.prefetch.async %515 into %block_idx_p0_6
      depends_on(%att10_inrel_r1_s2) : !nest.event<"pf_bidx12_r1_s2">
    %att12_grid_r1_s2, %att12_inrel_r1_s2, %att12_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx12_r1_s2, %att8_out_r1_s2)
      : (
        !nest.event<"att12_grid_r1_s2">, !nest.event<"att12_inrel_r1_s2">,
        !nest.event<"att12_out_r1_s2">)
    %pf_bidx13_r1_s2 = nest.dma.prefetch.async %516 into %block_idx_p1_6
      depends_on(%att11_inrel_r1_s2) : !nest.event<"pf_bidx13_r1_s2">
    %att13_grid_r1_s2, %att13_inrel_r1_s2, %att13_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx13_r1_s2, %att9_out_r1_s2)
      : (
        !nest.event<"att13_grid_r1_s2">, !nest.event<"att13_inrel_r1_s2">,
        !nest.event<"att13_out_r1_s2">)
    %pf_bidx14_r1_s2 = nest.dma.prefetch.async %517 into %block_idx_p0_6
      depends_on(%att12_inrel_r1_s2) : !nest.event<"pf_bidx14_r1_s2">
    %att14_grid_r1_s2, %att14_inrel_r1_s2, %att14_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx14_r1_s2, %att10_out_r1_s2)
      : (
        !nest.event<"att14_grid_r1_s2">, !nest.event<"att14_inrel_r1_s2">,
        !nest.event<"att14_out_r1_s2">)
    %pf_bidx15_r1_s2 = nest.dma.prefetch.async %518 into %block_idx_p1_6
      depends_on(%att13_inrel_r1_s2) : !nest.event<"pf_bidx15_r1_s2">
    %att15_grid_r1_s2, %att15_inrel_r1_s2, %att15_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx15_r1_s2, %att11_out_r1_s2)
      : (
        !nest.event<"att15_grid_r1_s2">, !nest.event<"att15_inrel_r1_s2">,
        !nest.event<"att15_out_r1_s2">)
    %pf_bidx16_r1_s2 = nest.dma.prefetch.async %519 into %block_idx_p0_6
      depends_on(%att14_inrel_r1_s2) : !nest.event<"pf_bidx16_r1_s2">
    %att16_grid_r1_s2, %att16_inrel_r1_s2, %att16_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx16_r1_s2, %att12_out_r1_s2)
      : (
        !nest.event<"att16_grid_r1_s2">, !nest.event<"att16_inrel_r1_s2">,
        !nest.event<"att16_out_r1_s2">)
    %pf_bidx17_r1_s2 = nest.dma.prefetch.async %520 into %block_idx_p1_6
      depends_on(%att15_inrel_r1_s2) : !nest.event<"pf_bidx17_r1_s2">
    %att17_grid_r1_s2, %att17_inrel_r1_s2, %att17_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx17_r1_s2, %att13_out_r1_s2)
      : (
        !nest.event<"att17_grid_r1_s2">, !nest.event<"att17_inrel_r1_s2">,
        !nest.event<"att17_out_r1_s2">)
    %pf_bidx18_r1_s2 = nest.dma.prefetch.async %521 into %block_idx_p0_6
      depends_on(%att16_inrel_r1_s2) : !nest.event<"pf_bidx18_r1_s2">
    %att18_grid_r1_s2, %att18_inrel_r1_s2, %att18_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx18_r1_s2, %att14_out_r1_s2)
      : (
        !nest.event<"att18_grid_r1_s2">, !nest.event<"att18_inrel_r1_s2">,
        !nest.event<"att18_out_r1_s2">)
    %pf_bidx19_r1_s2 = nest.dma.prefetch.async %522 into %block_idx_p1_6
      depends_on(%att17_inrel_r1_s2) : !nest.event<"pf_bidx19_r1_s2">
    %att19_grid_r1_s2, %att19_inrel_r1_s2, %att19_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx19_r1_s2, %att15_out_r1_s2)
      : (
        !nest.event<"att19_grid_r1_s2">, !nest.event<"att19_inrel_r1_s2">,
        !nest.event<"att19_out_r1_s2">)
    %pf_bidx20_r1_s2 = nest.dma.prefetch.async %523 into %block_idx_p0_6
      depends_on(%att18_inrel_r1_s2) : !nest.event<"pf_bidx20_r1_s2">
    %att20_grid_r1_s2, %att20_inrel_r1_s2, %att20_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx20_r1_s2, %att16_out_r1_s2)
      : (
        !nest.event<"att20_grid_r1_s2">, !nest.event<"att20_inrel_r1_s2">,
        !nest.event<"att20_out_r1_s2">)
    %pf_bidx21_r1_s2 = nest.dma.prefetch.async %524 into %block_idx_p1_6
      depends_on(%att19_inrel_r1_s2) : !nest.event<"pf_bidx21_r1_s2">
    %att21_grid_r1_s2, %att21_inrel_r1_s2, %att21_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx21_r1_s2, %att17_out_r1_s2)
      : (
        !nest.event<"att21_grid_r1_s2">, !nest.event<"att21_inrel_r1_s2">,
        !nest.event<"att21_out_r1_s2">)
    %pf_bidx22_r1_s2 = nest.dma.prefetch.async %525 into %block_idx_p0_6
      depends_on(%att20_inrel_r1_s2) : !nest.event<"pf_bidx22_r1_s2">
    %att22_grid_r1_s2, %att22_inrel_r1_s2, %att22_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx22_r1_s2, %att18_out_r1_s2)
      : (
        !nest.event<"att22_grid_r1_s2">, !nest.event<"att22_inrel_r1_s2">,
        !nest.event<"att22_out_r1_s2">)
    %pf_bidx23_r1_s2 = nest.dma.prefetch.async %526 into %block_idx_p1_6
      depends_on(%att21_inrel_r1_s2) : !nest.event<"pf_bidx23_r1_s2">
    %att23_grid_r1_s2, %att23_inrel_r1_s2, %att23_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx23_r1_s2, %att19_out_r1_s2)
      : (
        !nest.event<"att23_grid_r1_s2">, !nest.event<"att23_inrel_r1_s2">,
        !nest.event<"att23_out_r1_s2">)
    %pf_bidx24_r1_s2 = nest.dma.prefetch.async %527 into %block_idx_p0_6
      depends_on(%att22_inrel_r1_s2) : !nest.event<"pf_bidx24_r1_s2">
    %att24_grid_r1_s2, %att24_inrel_r1_s2, %att24_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx24_r1_s2, %att20_out_r1_s2)
      : (
        !nest.event<"att24_grid_r1_s2">, !nest.event<"att24_inrel_r1_s2">,
        !nest.event<"att24_out_r1_s2">)
    %pf_bidx25_r1_s2 = nest.dma.prefetch.async %528 into %block_idx_p1_6
      depends_on(%att23_inrel_r1_s2) : !nest.event<"pf_bidx25_r1_s2">
    %att25_grid_r1_s2, %att25_inrel_r1_s2, %att25_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx25_r1_s2, %att21_out_r1_s2)
      : (
        !nest.event<"att25_grid_r1_s2">, !nest.event<"att25_inrel_r1_s2">,
        !nest.event<"att25_out_r1_s2">)
    %pf_bidx26_r1_s2 = nest.dma.prefetch.async %529 into %block_idx_p0_6
      depends_on(%att24_inrel_r1_s2) : !nest.event<"pf_bidx26_r1_s2">
    %att26_grid_r1_s2, %att26_inrel_r1_s2, %att26_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx26_r1_s2, %att22_out_r1_s2)
      : (
        !nest.event<"att26_grid_r1_s2">, !nest.event<"att26_inrel_r1_s2">,
        !nest.event<"att26_out_r1_s2">)
    %pf_bidx27_r1_s2 = nest.dma.prefetch.async %530 into %block_idx_p1_6
      depends_on(%att25_inrel_r1_s2) : !nest.event<"pf_bidx27_r1_s2">
    %att27_grid_r1_s2, %att27_inrel_r1_s2, %att27_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx27_r1_s2, %att23_out_r1_s2)
      : (
        !nest.event<"att27_grid_r1_s2">, !nest.event<"att27_inrel_r1_s2">,
        !nest.event<"att27_out_r1_s2">)
    %pf_bidx28_r1_s2 = nest.dma.prefetch.async %531 into %block_idx_p0_6
      depends_on(%att26_inrel_r1_s2) : !nest.event<"pf_bidx28_r1_s2">
    %att28_grid_r1_s2, %att28_inrel_r1_s2, %att28_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_bidx28_r1_s2, %att24_out_r1_s2)
      : (
        !nest.event<"att28_grid_r1_s2">, !nest.event<"att28_inrel_r1_s2">,
        !nest.event<"att28_out_r1_s2">)
    %pf_bidx29_r1_s2 = nest.dma.prefetch.async %532 into %block_idx_p1_6
      depends_on(%att27_inrel_r1_s2) : !nest.event<"pf_bidx29_r1_s2">
    %att29_grid_r1_s2, %att29_inrel_r1_s2, %att29_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p1_7, %acc_p1_7) outs(%state_p1_7, %acc_p1_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p1_r1_s2, %pf_acc_p1_r1_s2, %pf_bidx29_r1_s2, %att25_out_r1_s2)
      : (
        !nest.event<"att29_grid_r1_s2">, !nest.event<"att29_inrel_r1_s2">,
        !nest.event<"att29_out_r1_s2">)
    %pf_bidx30_r1_s2 = nest.dma.prefetch.async %533 into %block_idx_p0_6
      depends_on(%att28_inrel_r1_s2) : !nest.event<"pf_bidx30_r1_s2">
    %att30_grid_r1_s2, %att30_inrel_r1_s2, %att30_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7)
      ins(%block_idx_p0_6, %q_l2_21, %state_p2_7, %acc_p2_7) outs(%state_p2_7, %acc_p2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p2_r1_s2, %pf_acc_p2_r1_s2, %pf_bidx30_r1_s2, %att26_out_r1_s2)
      : (
        !nest.event<"att30_grid_r1_s2">, !nest.event<"att30_inrel_r1_s2">,
        !nest.event<"att30_out_r1_s2">)
    %pf_bidx31_r1_s2 = nest.dma.prefetch.async %534 into %block_idx_p1_6
      depends_on(%att29_inrel_r1_s2) : !nest.event<"pf_bidx31_r1_s2">
    %att31_grid_r1_s2, %att31_inrel_r1_s2, %att31_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7)
      ins(%block_idx_p1_6, %q_l2_21, %state_p3_7, %acc_p3_7) outs(%state_p3_7, %acc_p3_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p3_r1_s2, %pf_acc_p3_r1_s2, %pf_bidx31_r1_s2, %att27_out_r1_s2)
      : (
        !nest.event<"att31_grid_r1_s2">, !nest.event<"att31_inrel_r1_s2">,
        !nest.event<"att31_out_r1_s2">)
    %att32_grid_r1_s2, %att32_inrel_r1_s2, %att32_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_t2_final_r1 l1_mode = 1 tasks(%535) globals(%489)
      bindings(%append_idx_6, %q_l2_21, %state_p0_7, %acc_p0_7)
      ins(%append_idx_6, %q_l2_21, %state_p0_7, %acc_p0_7) outs(%state_p0_7, %acc_p0_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s2, %pf_state_p0_r1_s2, %pf_acc_p0_r1_s2, %pf_aidx_r1_s2, %append_grid_r1_s2,
        %att28_out_r1_s2)
      : (
        !nest.event<"att32_grid_r1_s2">, !nest.event<"att32_inrel_r1_s2">,
        !nest.event<"att32_out_r1_s2">)
    %merge_grid_r1_s2, %merge_inrel_r1_s2, %merge_out_r1_s2 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%535) globals()
      bindings(
        %state_p0_7, %state_p1_7, %state_p2_7, %state_p3_7, %acc_p0_7, %acc_p1_7, %acc_p2_7,
        %acc_p3_7, %out_l2_7)
      ins(
        %state_p0_7, %state_p1_7, %state_p2_7, %state_p3_7, %acc_p0_7, %acc_p1_7, %acc_p2_7,
        %acc_p3_7)
      outs(%out_l2_7)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att32_out_r1_s2, %att29_out_r1_s2, %att30_out_r1_s2, %att31_out_r1_s2) : (
        !nest.event<"merge_grid_r1_s2">, !nest.event<"merge_inrel_r1_s2">,
        !nest.event<"merge_out_r1_s2">)
    %out_store_r1_s2 = nest.dma.store.async %out_l2_7 into %502 depends_on(%merge_out_r1_s2)
      : !nest.event<"out_store_r1_s2">
    nest.await %append_grid_r1_s2, %att32_grid_r1_s2, %merge_grid_r1_s2, %out_store_r1_s2
    nest.release %block_idx_p0_6 depends_on(
      %pf_bidx0_r1_s2, %pf_bidx2_r1_s2, %pf_bidx4_r1_s2, %pf_bidx6_r1_s2, %pf_bidx8_r1_s2,
      %pf_bidx10_r1_s2, %pf_bidx12_r1_s2, %pf_bidx14_r1_s2, %pf_bidx16_r1_s2, %pf_bidx18_r1_s2,
      %pf_bidx20_r1_s2, %pf_bidx22_r1_s2, %pf_bidx24_r1_s2, %pf_bidx26_r1_s2, %pf_bidx28_r1_s2,
      %pf_bidx30_r1_s2, %att0_inrel_r1_s2, %att2_inrel_r1_s2, %att4_inrel_r1_s2,
      %att6_inrel_r1_s2, %att8_inrel_r1_s2, %att10_inrel_r1_s2, %att12_inrel_r1_s2,
      %att14_inrel_r1_s2, %att16_inrel_r1_s2, %att18_inrel_r1_s2, %att20_inrel_r1_s2,
      %att22_inrel_r1_s2, %att24_inrel_r1_s2, %att26_inrel_r1_s2, %att28_inrel_r1_s2,
      %att30_inrel_r1_s2)
    nest.release %block_idx_p1_6 depends_on(
      %pf_bidx1_r1_s2, %pf_bidx3_r1_s2, %pf_bidx5_r1_s2, %pf_bidx7_r1_s2, %pf_bidx9_r1_s2,
      %pf_bidx11_r1_s2, %pf_bidx13_r1_s2, %pf_bidx15_r1_s2, %pf_bidx17_r1_s2, %pf_bidx19_r1_s2,
      %pf_bidx21_r1_s2, %pf_bidx23_r1_s2, %pf_bidx25_r1_s2, %pf_bidx27_r1_s2, %pf_bidx29_r1_s2,
      %pf_bidx31_r1_s2, %att1_inrel_r1_s2, %att3_inrel_r1_s2, %att5_inrel_r1_s2,
      %att7_inrel_r1_s2, %att9_inrel_r1_s2, %att11_inrel_r1_s2, %att13_inrel_r1_s2,
      %att15_inrel_r1_s2, %att17_inrel_r1_s2, %att19_inrel_r1_s2, %att21_inrel_r1_s2,
      %att23_inrel_r1_s2, %att25_inrel_r1_s2, %att27_inrel_r1_s2, %att29_inrel_r1_s2,
      %att31_inrel_r1_s2)
    nest.release %k_new_6 depends_on(%pf_k_r1_s2, %append_inrel_r1_s2)
    nest.release %v_new_6 depends_on(%pf_v_r1_s2, %append_inrel_r1_s2)
    nest.release %append_idx_6 depends_on(%pf_aidx_r1_s2, %append_inrel_r1_s2, %att32_inrel_r1_s2)
    nest.release %q_l2_21 depends_on(
      %pf_q_r1_s2, %att0_inrel_r1_s2, %att1_inrel_r1_s2, %att2_inrel_r1_s2, %att3_inrel_r1_s2,
      %att4_inrel_r1_s2, %att5_inrel_r1_s2, %att6_inrel_r1_s2, %att7_inrel_r1_s2,
      %att8_inrel_r1_s2, %att9_inrel_r1_s2, %att10_inrel_r1_s2, %att11_inrel_r1_s2,
      %att12_inrel_r1_s2, %att13_inrel_r1_s2, %att14_inrel_r1_s2, %att15_inrel_r1_s2,
      %att16_inrel_r1_s2, %att17_inrel_r1_s2, %att18_inrel_r1_s2, %att19_inrel_r1_s2,
      %att20_inrel_r1_s2, %att21_inrel_r1_s2, %att22_inrel_r1_s2, %att23_inrel_r1_s2,
      %att24_inrel_r1_s2, %att25_inrel_r1_s2, %att26_inrel_r1_s2, %att27_inrel_r1_s2,
      %att28_inrel_r1_s2, %att29_inrel_r1_s2, %att30_inrel_r1_s2, %att31_inrel_r1_s2,
      %att32_inrel_r1_s2)
    nest.release %state_p0_7 depends_on(
      %pf_state_p0_r1_s2, %att0_inrel_r1_s2, %att4_inrel_r1_s2, %att8_inrel_r1_s2,
      %att12_inrel_r1_s2, %att16_inrel_r1_s2, %att20_inrel_r1_s2, %att24_inrel_r1_s2,
      %att28_inrel_r1_s2, %att32_inrel_r1_s2, %att0_out_r1_s2, %att4_out_r1_s2, %att8_out_r1_s2,
      %att12_out_r1_s2, %att16_out_r1_s2, %att20_out_r1_s2, %att24_out_r1_s2, %att28_out_r1_s2,
      %att32_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %acc_p0_7 depends_on(
      %pf_acc_p0_r1_s2, %att0_inrel_r1_s2, %att4_inrel_r1_s2, %att8_inrel_r1_s2,
      %att12_inrel_r1_s2, %att16_inrel_r1_s2, %att20_inrel_r1_s2, %att24_inrel_r1_s2,
      %att28_inrel_r1_s2, %att32_inrel_r1_s2, %att0_out_r1_s2, %att4_out_r1_s2, %att8_out_r1_s2,
      %att12_out_r1_s2, %att16_out_r1_s2, %att20_out_r1_s2, %att24_out_r1_s2, %att28_out_r1_s2,
      %att32_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %state_p1_7 depends_on(
      %pf_state_p1_r1_s2, %att1_inrel_r1_s2, %att5_inrel_r1_s2, %att9_inrel_r1_s2,
      %att13_inrel_r1_s2, %att17_inrel_r1_s2, %att21_inrel_r1_s2, %att25_inrel_r1_s2,
      %att29_inrel_r1_s2, %att1_out_r1_s2, %att5_out_r1_s2, %att9_out_r1_s2, %att13_out_r1_s2,
      %att17_out_r1_s2, %att21_out_r1_s2, %att25_out_r1_s2, %att29_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %acc_p1_7 depends_on(
      %pf_acc_p1_r1_s2, %att1_inrel_r1_s2, %att5_inrel_r1_s2, %att9_inrel_r1_s2,
      %att13_inrel_r1_s2, %att17_inrel_r1_s2, %att21_inrel_r1_s2, %att25_inrel_r1_s2,
      %att29_inrel_r1_s2, %att1_out_r1_s2, %att5_out_r1_s2, %att9_out_r1_s2, %att13_out_r1_s2,
      %att17_out_r1_s2, %att21_out_r1_s2, %att25_out_r1_s2, %att29_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %state_p2_7 depends_on(
      %pf_state_p2_r1_s2, %att2_inrel_r1_s2, %att6_inrel_r1_s2, %att10_inrel_r1_s2,
      %att14_inrel_r1_s2, %att18_inrel_r1_s2, %att22_inrel_r1_s2, %att26_inrel_r1_s2,
      %att30_inrel_r1_s2, %att2_out_r1_s2, %att6_out_r1_s2, %att10_out_r1_s2, %att14_out_r1_s2,
      %att18_out_r1_s2, %att22_out_r1_s2, %att26_out_r1_s2, %att30_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %acc_p2_7 depends_on(
      %pf_acc_p2_r1_s2, %att2_inrel_r1_s2, %att6_inrel_r1_s2, %att10_inrel_r1_s2,
      %att14_inrel_r1_s2, %att18_inrel_r1_s2, %att22_inrel_r1_s2, %att26_inrel_r1_s2,
      %att30_inrel_r1_s2, %att2_out_r1_s2, %att6_out_r1_s2, %att10_out_r1_s2, %att14_out_r1_s2,
      %att18_out_r1_s2, %att22_out_r1_s2, %att26_out_r1_s2, %att30_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %state_p3_7 depends_on(
      %pf_state_p3_r1_s2, %att3_inrel_r1_s2, %att7_inrel_r1_s2, %att11_inrel_r1_s2,
      %att15_inrel_r1_s2, %att19_inrel_r1_s2, %att23_inrel_r1_s2, %att27_inrel_r1_s2,
      %att31_inrel_r1_s2, %att3_out_r1_s2, %att7_out_r1_s2, %att11_out_r1_s2, %att15_out_r1_s2,
      %att19_out_r1_s2, %att23_out_r1_s2, %att27_out_r1_s2, %att31_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %acc_p3_7 depends_on(
      %pf_acc_p3_r1_s2, %att3_inrel_r1_s2, %att7_inrel_r1_s2, %att11_inrel_r1_s2,
      %att15_inrel_r1_s2, %att19_inrel_r1_s2, %att23_inrel_r1_s2, %att27_inrel_r1_s2,
      %att31_inrel_r1_s2, %att3_out_r1_s2, %att7_out_r1_s2, %att11_out_r1_s2, %att15_out_r1_s2,
      %att19_out_r1_s2, %att23_out_r1_s2, %att27_out_r1_s2, %att31_out_r1_s2, %merge_inrel_r1_s2)
    nest.release %out_l2_7 depends_on(%merge_out_r1_s2, %out_store_r1_s2)
    nest.return
  }
  nest.context @step_r1_s3(
    %POOL_7: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_7: !nest.global_memref<147xi32>,
    %APPEND_IDS_7: !nest.global_memref<12xi32>, %Q_IN_7: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_7: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_7: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_7: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_7: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_7: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 140, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_7 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_7 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_7 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_22 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_8 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_8 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_8 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_8 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_8 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_8 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_8 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_8 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_7 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_7 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_8 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %537 = nest.subview %POOL_7 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %538 = nest.subview %K_NEW_7 offsets = [1, 3, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %539 = nest.subview %V_NEW_7 offsets = [1, 3, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %540 = nest.subview %APPEND_IDS_7 offsets = [7] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %541 = nest.subview %Q_IN_7 offsets = [1, 3, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %542 = nest.subview %S_INIT_7 offsets = [1, 3, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %543 = nest.subview %S_INIT_7 offsets = [1, 3, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %544 = nest.subview %S_INIT_7 offsets = [1, 3, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %545 = nest.subview %S_INIT_7 offsets = [1, 3, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %546 = nest.subview %O_INIT_7 offsets = [1, 3, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %547 = nest.subview %O_INIT_7 offsets = [1, 3, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %548 = nest.subview %O_INIT_7 offsets = [1, 3, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %549 = nest.subview %O_INIT_7 offsets = [1, 3, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %550 = nest.subview %OUT_7 offsets = [1, 3, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %551 = nest.subview %BLOCK_TABLE_7 offsets = [49] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %552 = nest.subview %BLOCK_TABLE_7 offsets = [50] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %553 = nest.subview %BLOCK_TABLE_7 offsets = [51] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %554 = nest.subview %BLOCK_TABLE_7 offsets = [52] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %555 = nest.subview %BLOCK_TABLE_7 offsets = [53] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %556 = nest.subview %BLOCK_TABLE_7 offsets = [54] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %557 = nest.subview %BLOCK_TABLE_7 offsets = [55] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %558 = nest.subview %BLOCK_TABLE_7 offsets = [56] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %559 = nest.subview %BLOCK_TABLE_7 offsets = [57] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %560 = nest.subview %BLOCK_TABLE_7 offsets = [58] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %561 = nest.subview %BLOCK_TABLE_7 offsets = [59] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %562 = nest.subview %BLOCK_TABLE_7 offsets = [60] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %563 = nest.subview %BLOCK_TABLE_7 offsets = [61] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %564 = nest.subview %BLOCK_TABLE_7 offsets = [62] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %565 = nest.subview %BLOCK_TABLE_7 offsets = [63] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %566 = nest.subview %BLOCK_TABLE_7 offsets = [64] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %567 = nest.subview %BLOCK_TABLE_7 offsets = [65] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %568 = nest.subview %BLOCK_TABLE_7 offsets = [66] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %569 = nest.subview %BLOCK_TABLE_7 offsets = [67] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %570 = nest.subview %BLOCK_TABLE_7 offsets = [68] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %571 = nest.subview %BLOCK_TABLE_7 offsets = [69] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %572 = nest.subview %BLOCK_TABLE_7 offsets = [70] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %573 = nest.subview %BLOCK_TABLE_7 offsets = [71] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %574 = nest.subview %BLOCK_TABLE_7 offsets = [72] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %575 = nest.subview %BLOCK_TABLE_7 offsets = [73] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %576 = nest.subview %BLOCK_TABLE_7 offsets = [74] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %577 = nest.subview %BLOCK_TABLE_7 offsets = [75] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %578 = nest.subview %BLOCK_TABLE_7 offsets = [76] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %579 = nest.subview %BLOCK_TABLE_7 offsets = [77] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %580 = nest.subview %BLOCK_TABLE_7 offsets = [78] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %581 = nest.subview %BLOCK_TABLE_7 offsets = [79] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %582 = nest.subview %BLOCK_TABLE_7 offsets = [80] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r1_s3 = nest.dma.prefetch.async %538 into %k_new_7 : !nest.event<"pf_k_r1_s3">
    %pf_v_r1_s3 = nest.dma.prefetch.async %539 into %v_new_7 : !nest.event<"pf_v_r1_s3">
    %pf_aidx_r1_s3 = nest.dma.prefetch.async %540 into %append_idx_7 : !nest.event<"pf_aidx_r1_s3">
    %pf_q_r1_s3 = nest.dma.prefetch.async %541 into %q_l2_22 : !nest.event<"pf_q_r1_s3">
    %pf_state_p0_r1_s3 = nest.dma.prefetch.async %542 into %state_p0_8
      : !nest.event<"pf_state_p0_r1_s3">
    %pf_state_p1_r1_s3 = nest.dma.prefetch.async %543 into %state_p1_8
      : !nest.event<"pf_state_p1_r1_s3">
    %pf_state_p2_r1_s3 = nest.dma.prefetch.async %544 into %state_p2_8
      : !nest.event<"pf_state_p2_r1_s3">
    %pf_state_p3_r1_s3 = nest.dma.prefetch.async %545 into %state_p3_8
      : !nest.event<"pf_state_p3_r1_s3">
    %pf_acc_p0_r1_s3 = nest.dma.prefetch.async %546 into %acc_p0_8 : !nest.event<"pf_acc_p0_r1_s3">
    %pf_acc_p1_r1_s3 = nest.dma.prefetch.async %547 into %acc_p1_8 : !nest.event<"pf_acc_p1_r1_s3">
    %pf_acc_p2_r1_s3 = nest.dma.prefetch.async %548 into %acc_p2_8 : !nest.event<"pf_acc_p2_r1_s3">
    %pf_acc_p3_r1_s3 = nest.dma.prefetch.async %549 into %acc_p3_8 : !nest.event<"pf_acc_p3_r1_s3">
    %583 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r1_s3, %append_inrel_r1_s3, %584 =
      nest.dispatch.tasks.async @paged_attention_append_r1_tip2 l1_mode = 1 tasks(%583)
      globals(%537) bindings(%k_new_7, %v_new_7, %append_idx_7)
      ins(%k_new_7, %v_new_7, %append_idx_7) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r1_s3, %pf_v_r1_s3, %pf_aidx_r1_s3)
      : (!nest.event<"append_grid_r1_s3">, !nest.event<"append_inrel_r1_s3">, !nest.event<"">)
    %pf_bidx0_r1_s3 = nest.dma.prefetch.async %551 into %block_idx_p0_7
      : !nest.event<"pf_bidx0_r1_s3">
    %att0_grid_r1_s3, %att0_inrel_r1_s3, %att0_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx0_r1_s3) : (
        !nest.event<"att0_grid_r1_s3">, !nest.event<"att0_inrel_r1_s3">,
        !nest.event<"att0_out_r1_s3">)
    %pf_bidx1_r1_s3 = nest.dma.prefetch.async %552 into %block_idx_p1_7
      : !nest.event<"pf_bidx1_r1_s3">
    %att1_grid_r1_s3, %att1_inrel_r1_s3, %att1_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx1_r1_s3) : (
        !nest.event<"att1_grid_r1_s3">, !nest.event<"att1_inrel_r1_s3">,
        !nest.event<"att1_out_r1_s3">)
    %pf_bidx2_r1_s3 = nest.dma.prefetch.async %553 into %block_idx_p0_7
      depends_on(%att0_inrel_r1_s3) : !nest.event<"pf_bidx2_r1_s3">
    %att2_grid_r1_s3, %att2_inrel_r1_s3, %att2_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx2_r1_s3) : (
        !nest.event<"att2_grid_r1_s3">, !nest.event<"att2_inrel_r1_s3">,
        !nest.event<"att2_out_r1_s3">)
    %pf_bidx3_r1_s3 = nest.dma.prefetch.async %554 into %block_idx_p1_7
      depends_on(%att1_inrel_r1_s3) : !nest.event<"pf_bidx3_r1_s3">
    %att3_grid_r1_s3, %att3_inrel_r1_s3, %att3_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx3_r1_s3) : (
        !nest.event<"att3_grid_r1_s3">, !nest.event<"att3_inrel_r1_s3">,
        !nest.event<"att3_out_r1_s3">)
    %pf_bidx4_r1_s3 = nest.dma.prefetch.async %555 into %block_idx_p0_7
      depends_on(%att2_inrel_r1_s3) : !nest.event<"pf_bidx4_r1_s3">
    %att4_grid_r1_s3, %att4_inrel_r1_s3, %att4_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx4_r1_s3, %att0_out_r1_s3)
      : (
        !nest.event<"att4_grid_r1_s3">, !nest.event<"att4_inrel_r1_s3">,
        !nest.event<"att4_out_r1_s3">)
    %pf_bidx5_r1_s3 = nest.dma.prefetch.async %556 into %block_idx_p1_7
      depends_on(%att3_inrel_r1_s3) : !nest.event<"pf_bidx5_r1_s3">
    %att5_grid_r1_s3, %att5_inrel_r1_s3, %att5_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx5_r1_s3, %att1_out_r1_s3)
      : (
        !nest.event<"att5_grid_r1_s3">, !nest.event<"att5_inrel_r1_s3">,
        !nest.event<"att5_out_r1_s3">)
    %pf_bidx6_r1_s3 = nest.dma.prefetch.async %557 into %block_idx_p0_7
      depends_on(%att4_inrel_r1_s3) : !nest.event<"pf_bidx6_r1_s3">
    %att6_grid_r1_s3, %att6_inrel_r1_s3, %att6_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx6_r1_s3, %att2_out_r1_s3)
      : (
        !nest.event<"att6_grid_r1_s3">, !nest.event<"att6_inrel_r1_s3">,
        !nest.event<"att6_out_r1_s3">)
    %pf_bidx7_r1_s3 = nest.dma.prefetch.async %558 into %block_idx_p1_7
      depends_on(%att5_inrel_r1_s3) : !nest.event<"pf_bidx7_r1_s3">
    %att7_grid_r1_s3, %att7_inrel_r1_s3, %att7_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx7_r1_s3, %att3_out_r1_s3)
      : (
        !nest.event<"att7_grid_r1_s3">, !nest.event<"att7_inrel_r1_s3">,
        !nest.event<"att7_out_r1_s3">)
    %pf_bidx8_r1_s3 = nest.dma.prefetch.async %559 into %block_idx_p0_7
      depends_on(%att6_inrel_r1_s3) : !nest.event<"pf_bidx8_r1_s3">
    %att8_grid_r1_s3, %att8_inrel_r1_s3, %att8_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx8_r1_s3, %att4_out_r1_s3)
      : (
        !nest.event<"att8_grid_r1_s3">, !nest.event<"att8_inrel_r1_s3">,
        !nest.event<"att8_out_r1_s3">)
    %pf_bidx9_r1_s3 = nest.dma.prefetch.async %560 into %block_idx_p1_7
      depends_on(%att7_inrel_r1_s3) : !nest.event<"pf_bidx9_r1_s3">
    %att9_grid_r1_s3, %att9_inrel_r1_s3, %att9_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx9_r1_s3, %att5_out_r1_s3)
      : (
        !nest.event<"att9_grid_r1_s3">, !nest.event<"att9_inrel_r1_s3">,
        !nest.event<"att9_out_r1_s3">)
    %pf_bidx10_r1_s3 = nest.dma.prefetch.async %561 into %block_idx_p0_7
      depends_on(%att8_inrel_r1_s3) : !nest.event<"pf_bidx10_r1_s3">
    %att10_grid_r1_s3, %att10_inrel_r1_s3, %att10_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx10_r1_s3, %att6_out_r1_s3)
      : (
        !nest.event<"att10_grid_r1_s3">, !nest.event<"att10_inrel_r1_s3">,
        !nest.event<"att10_out_r1_s3">)
    %pf_bidx11_r1_s3 = nest.dma.prefetch.async %562 into %block_idx_p1_7
      depends_on(%att9_inrel_r1_s3) : !nest.event<"pf_bidx11_r1_s3">
    %att11_grid_r1_s3, %att11_inrel_r1_s3, %att11_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx11_r1_s3, %att7_out_r1_s3)
      : (
        !nest.event<"att11_grid_r1_s3">, !nest.event<"att11_inrel_r1_s3">,
        !nest.event<"att11_out_r1_s3">)
    %pf_bidx12_r1_s3 = nest.dma.prefetch.async %563 into %block_idx_p0_7
      depends_on(%att10_inrel_r1_s3) : !nest.event<"pf_bidx12_r1_s3">
    %att12_grid_r1_s3, %att12_inrel_r1_s3, %att12_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx12_r1_s3, %att8_out_r1_s3)
      : (
        !nest.event<"att12_grid_r1_s3">, !nest.event<"att12_inrel_r1_s3">,
        !nest.event<"att12_out_r1_s3">)
    %pf_bidx13_r1_s3 = nest.dma.prefetch.async %564 into %block_idx_p1_7
      depends_on(%att11_inrel_r1_s3) : !nest.event<"pf_bidx13_r1_s3">
    %att13_grid_r1_s3, %att13_inrel_r1_s3, %att13_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx13_r1_s3, %att9_out_r1_s3)
      : (
        !nest.event<"att13_grid_r1_s3">, !nest.event<"att13_inrel_r1_s3">,
        !nest.event<"att13_out_r1_s3">)
    %pf_bidx14_r1_s3 = nest.dma.prefetch.async %565 into %block_idx_p0_7
      depends_on(%att12_inrel_r1_s3) : !nest.event<"pf_bidx14_r1_s3">
    %att14_grid_r1_s3, %att14_inrel_r1_s3, %att14_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx14_r1_s3, %att10_out_r1_s3)
      : (
        !nest.event<"att14_grid_r1_s3">, !nest.event<"att14_inrel_r1_s3">,
        !nest.event<"att14_out_r1_s3">)
    %pf_bidx15_r1_s3 = nest.dma.prefetch.async %566 into %block_idx_p1_7
      depends_on(%att13_inrel_r1_s3) : !nest.event<"pf_bidx15_r1_s3">
    %att15_grid_r1_s3, %att15_inrel_r1_s3, %att15_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx15_r1_s3, %att11_out_r1_s3)
      : (
        !nest.event<"att15_grid_r1_s3">, !nest.event<"att15_inrel_r1_s3">,
        !nest.event<"att15_out_r1_s3">)
    %pf_bidx16_r1_s3 = nest.dma.prefetch.async %567 into %block_idx_p0_7
      depends_on(%att14_inrel_r1_s3) : !nest.event<"pf_bidx16_r1_s3">
    %att16_grid_r1_s3, %att16_inrel_r1_s3, %att16_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx16_r1_s3, %att12_out_r1_s3)
      : (
        !nest.event<"att16_grid_r1_s3">, !nest.event<"att16_inrel_r1_s3">,
        !nest.event<"att16_out_r1_s3">)
    %pf_bidx17_r1_s3 = nest.dma.prefetch.async %568 into %block_idx_p1_7
      depends_on(%att15_inrel_r1_s3) : !nest.event<"pf_bidx17_r1_s3">
    %att17_grid_r1_s3, %att17_inrel_r1_s3, %att17_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx17_r1_s3, %att13_out_r1_s3)
      : (
        !nest.event<"att17_grid_r1_s3">, !nest.event<"att17_inrel_r1_s3">,
        !nest.event<"att17_out_r1_s3">)
    %pf_bidx18_r1_s3 = nest.dma.prefetch.async %569 into %block_idx_p0_7
      depends_on(%att16_inrel_r1_s3) : !nest.event<"pf_bidx18_r1_s3">
    %att18_grid_r1_s3, %att18_inrel_r1_s3, %att18_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx18_r1_s3, %att14_out_r1_s3)
      : (
        !nest.event<"att18_grid_r1_s3">, !nest.event<"att18_inrel_r1_s3">,
        !nest.event<"att18_out_r1_s3">)
    %pf_bidx19_r1_s3 = nest.dma.prefetch.async %570 into %block_idx_p1_7
      depends_on(%att17_inrel_r1_s3) : !nest.event<"pf_bidx19_r1_s3">
    %att19_grid_r1_s3, %att19_inrel_r1_s3, %att19_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx19_r1_s3, %att15_out_r1_s3)
      : (
        !nest.event<"att19_grid_r1_s3">, !nest.event<"att19_inrel_r1_s3">,
        !nest.event<"att19_out_r1_s3">)
    %pf_bidx20_r1_s3 = nest.dma.prefetch.async %571 into %block_idx_p0_7
      depends_on(%att18_inrel_r1_s3) : !nest.event<"pf_bidx20_r1_s3">
    %att20_grid_r1_s3, %att20_inrel_r1_s3, %att20_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx20_r1_s3, %att16_out_r1_s3)
      : (
        !nest.event<"att20_grid_r1_s3">, !nest.event<"att20_inrel_r1_s3">,
        !nest.event<"att20_out_r1_s3">)
    %pf_bidx21_r1_s3 = nest.dma.prefetch.async %572 into %block_idx_p1_7
      depends_on(%att19_inrel_r1_s3) : !nest.event<"pf_bidx21_r1_s3">
    %att21_grid_r1_s3, %att21_inrel_r1_s3, %att21_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx21_r1_s3, %att17_out_r1_s3)
      : (
        !nest.event<"att21_grid_r1_s3">, !nest.event<"att21_inrel_r1_s3">,
        !nest.event<"att21_out_r1_s3">)
    %pf_bidx22_r1_s3 = nest.dma.prefetch.async %573 into %block_idx_p0_7
      depends_on(%att20_inrel_r1_s3) : !nest.event<"pf_bidx22_r1_s3">
    %att22_grid_r1_s3, %att22_inrel_r1_s3, %att22_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx22_r1_s3, %att18_out_r1_s3)
      : (
        !nest.event<"att22_grid_r1_s3">, !nest.event<"att22_inrel_r1_s3">,
        !nest.event<"att22_out_r1_s3">)
    %pf_bidx23_r1_s3 = nest.dma.prefetch.async %574 into %block_idx_p1_7
      depends_on(%att21_inrel_r1_s3) : !nest.event<"pf_bidx23_r1_s3">
    %att23_grid_r1_s3, %att23_inrel_r1_s3, %att23_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx23_r1_s3, %att19_out_r1_s3)
      : (
        !nest.event<"att23_grid_r1_s3">, !nest.event<"att23_inrel_r1_s3">,
        !nest.event<"att23_out_r1_s3">)
    %pf_bidx24_r1_s3 = nest.dma.prefetch.async %575 into %block_idx_p0_7
      depends_on(%att22_inrel_r1_s3) : !nest.event<"pf_bidx24_r1_s3">
    %att24_grid_r1_s3, %att24_inrel_r1_s3, %att24_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx24_r1_s3, %att20_out_r1_s3)
      : (
        !nest.event<"att24_grid_r1_s3">, !nest.event<"att24_inrel_r1_s3">,
        !nest.event<"att24_out_r1_s3">)
    %pf_bidx25_r1_s3 = nest.dma.prefetch.async %576 into %block_idx_p1_7
      depends_on(%att23_inrel_r1_s3) : !nest.event<"pf_bidx25_r1_s3">
    %att25_grid_r1_s3, %att25_inrel_r1_s3, %att25_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx25_r1_s3, %att21_out_r1_s3)
      : (
        !nest.event<"att25_grid_r1_s3">, !nest.event<"att25_inrel_r1_s3">,
        !nest.event<"att25_out_r1_s3">)
    %pf_bidx26_r1_s3 = nest.dma.prefetch.async %577 into %block_idx_p0_7
      depends_on(%att24_inrel_r1_s3) : !nest.event<"pf_bidx26_r1_s3">
    %att26_grid_r1_s3, %att26_inrel_r1_s3, %att26_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx26_r1_s3, %att22_out_r1_s3)
      : (
        !nest.event<"att26_grid_r1_s3">, !nest.event<"att26_inrel_r1_s3">,
        !nest.event<"att26_out_r1_s3">)
    %pf_bidx27_r1_s3 = nest.dma.prefetch.async %578 into %block_idx_p1_7
      depends_on(%att25_inrel_r1_s3) : !nest.event<"pf_bidx27_r1_s3">
    %att27_grid_r1_s3, %att27_inrel_r1_s3, %att27_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx27_r1_s3, %att23_out_r1_s3)
      : (
        !nest.event<"att27_grid_r1_s3">, !nest.event<"att27_inrel_r1_s3">,
        !nest.event<"att27_out_r1_s3">)
    %pf_bidx28_r1_s3 = nest.dma.prefetch.async %579 into %block_idx_p0_7
      depends_on(%att26_inrel_r1_s3) : !nest.event<"pf_bidx28_r1_s3">
    %att28_grid_r1_s3, %att28_inrel_r1_s3, %att28_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_bidx28_r1_s3, %att24_out_r1_s3)
      : (
        !nest.event<"att28_grid_r1_s3">, !nest.event<"att28_inrel_r1_s3">,
        !nest.event<"att28_out_r1_s3">)
    %pf_bidx29_r1_s3 = nest.dma.prefetch.async %580 into %block_idx_p1_7
      depends_on(%att27_inrel_r1_s3) : !nest.event<"pf_bidx29_r1_s3">
    %att29_grid_r1_s3, %att29_inrel_r1_s3, %att29_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p1_8, %acc_p1_8) outs(%state_p1_8, %acc_p1_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p1_r1_s3, %pf_acc_p1_r1_s3, %pf_bidx29_r1_s3, %att25_out_r1_s3)
      : (
        !nest.event<"att29_grid_r1_s3">, !nest.event<"att29_inrel_r1_s3">,
        !nest.event<"att29_out_r1_s3">)
    %pf_bidx30_r1_s3 = nest.dma.prefetch.async %581 into %block_idx_p0_7
      depends_on(%att28_inrel_r1_s3) : !nest.event<"pf_bidx30_r1_s3">
    %att30_grid_r1_s3, %att30_inrel_r1_s3, %att30_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8)
      ins(%block_idx_p0_7, %q_l2_22, %state_p2_8, %acc_p2_8) outs(%state_p2_8, %acc_p2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p2_r1_s3, %pf_acc_p2_r1_s3, %pf_bidx30_r1_s3, %att26_out_r1_s3)
      : (
        !nest.event<"att30_grid_r1_s3">, !nest.event<"att30_inrel_r1_s3">,
        !nest.event<"att30_out_r1_s3">)
    %pf_bidx31_r1_s3 = nest.dma.prefetch.async %582 into %block_idx_p1_7
      depends_on(%att29_inrel_r1_s3) : !nest.event<"pf_bidx31_r1_s3">
    %att31_grid_r1_s3, %att31_inrel_r1_s3, %att31_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8)
      ins(%block_idx_p1_7, %q_l2_22, %state_p3_8, %acc_p3_8) outs(%state_p3_8, %acc_p3_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p3_r1_s3, %pf_acc_p3_r1_s3, %pf_bidx31_r1_s3, %att27_out_r1_s3)
      : (
        !nest.event<"att31_grid_r1_s3">, !nest.event<"att31_inrel_r1_s3">,
        !nest.event<"att31_out_r1_s3">)
    %att32_grid_r1_s3, %att32_inrel_r1_s3, %att32_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_t3_final_r1 l1_mode = 1 tasks(%583) globals(%537)
      bindings(%append_idx_7, %q_l2_22, %state_p0_8, %acc_p0_8)
      ins(%append_idx_7, %q_l2_22, %state_p0_8, %acc_p0_8) outs(%state_p0_8, %acc_p0_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r1_s3, %pf_state_p0_r1_s3, %pf_acc_p0_r1_s3, %pf_aidx_r1_s3, %append_grid_r1_s3,
        %att28_out_r1_s3)
      : (
        !nest.event<"att32_grid_r1_s3">, !nest.event<"att32_inrel_r1_s3">,
        !nest.event<"att32_out_r1_s3">)
    %merge_grid_r1_s3, %merge_inrel_r1_s3, %merge_out_r1_s3 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%583) globals()
      bindings(
        %state_p0_8, %state_p1_8, %state_p2_8, %state_p3_8, %acc_p0_8, %acc_p1_8, %acc_p2_8,
        %acc_p3_8, %out_l2_8)
      ins(
        %state_p0_8, %state_p1_8, %state_p2_8, %state_p3_8, %acc_p0_8, %acc_p1_8, %acc_p2_8,
        %acc_p3_8)
      outs(%out_l2_8)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att32_out_r1_s3, %att29_out_r1_s3, %att30_out_r1_s3, %att31_out_r1_s3) : (
        !nest.event<"merge_grid_r1_s3">, !nest.event<"merge_inrel_r1_s3">,
        !nest.event<"merge_out_r1_s3">)
    %out_store_r1_s3 = nest.dma.store.async %out_l2_8 into %550 depends_on(%merge_out_r1_s3)
      : !nest.event<"out_store_r1_s3">
    nest.await %append_grid_r1_s3, %att32_grid_r1_s3, %merge_grid_r1_s3, %out_store_r1_s3
    nest.release %block_idx_p0_7 depends_on(
      %pf_bidx0_r1_s3, %pf_bidx2_r1_s3, %pf_bidx4_r1_s3, %pf_bidx6_r1_s3, %pf_bidx8_r1_s3,
      %pf_bidx10_r1_s3, %pf_bidx12_r1_s3, %pf_bidx14_r1_s3, %pf_bidx16_r1_s3, %pf_bidx18_r1_s3,
      %pf_bidx20_r1_s3, %pf_bidx22_r1_s3, %pf_bidx24_r1_s3, %pf_bidx26_r1_s3, %pf_bidx28_r1_s3,
      %pf_bidx30_r1_s3, %att0_inrel_r1_s3, %att2_inrel_r1_s3, %att4_inrel_r1_s3,
      %att6_inrel_r1_s3, %att8_inrel_r1_s3, %att10_inrel_r1_s3, %att12_inrel_r1_s3,
      %att14_inrel_r1_s3, %att16_inrel_r1_s3, %att18_inrel_r1_s3, %att20_inrel_r1_s3,
      %att22_inrel_r1_s3, %att24_inrel_r1_s3, %att26_inrel_r1_s3, %att28_inrel_r1_s3,
      %att30_inrel_r1_s3)
    nest.release %block_idx_p1_7 depends_on(
      %pf_bidx1_r1_s3, %pf_bidx3_r1_s3, %pf_bidx5_r1_s3, %pf_bidx7_r1_s3, %pf_bidx9_r1_s3,
      %pf_bidx11_r1_s3, %pf_bidx13_r1_s3, %pf_bidx15_r1_s3, %pf_bidx17_r1_s3, %pf_bidx19_r1_s3,
      %pf_bidx21_r1_s3, %pf_bidx23_r1_s3, %pf_bidx25_r1_s3, %pf_bidx27_r1_s3, %pf_bidx29_r1_s3,
      %pf_bidx31_r1_s3, %att1_inrel_r1_s3, %att3_inrel_r1_s3, %att5_inrel_r1_s3,
      %att7_inrel_r1_s3, %att9_inrel_r1_s3, %att11_inrel_r1_s3, %att13_inrel_r1_s3,
      %att15_inrel_r1_s3, %att17_inrel_r1_s3, %att19_inrel_r1_s3, %att21_inrel_r1_s3,
      %att23_inrel_r1_s3, %att25_inrel_r1_s3, %att27_inrel_r1_s3, %att29_inrel_r1_s3,
      %att31_inrel_r1_s3)
    nest.release %k_new_7 depends_on(%pf_k_r1_s3, %append_inrel_r1_s3)
    nest.release %v_new_7 depends_on(%pf_v_r1_s3, %append_inrel_r1_s3)
    nest.release %append_idx_7 depends_on(%pf_aidx_r1_s3, %append_inrel_r1_s3, %att32_inrel_r1_s3)
    nest.release %q_l2_22 depends_on(
      %pf_q_r1_s3, %att0_inrel_r1_s3, %att1_inrel_r1_s3, %att2_inrel_r1_s3, %att3_inrel_r1_s3,
      %att4_inrel_r1_s3, %att5_inrel_r1_s3, %att6_inrel_r1_s3, %att7_inrel_r1_s3,
      %att8_inrel_r1_s3, %att9_inrel_r1_s3, %att10_inrel_r1_s3, %att11_inrel_r1_s3,
      %att12_inrel_r1_s3, %att13_inrel_r1_s3, %att14_inrel_r1_s3, %att15_inrel_r1_s3,
      %att16_inrel_r1_s3, %att17_inrel_r1_s3, %att18_inrel_r1_s3, %att19_inrel_r1_s3,
      %att20_inrel_r1_s3, %att21_inrel_r1_s3, %att22_inrel_r1_s3, %att23_inrel_r1_s3,
      %att24_inrel_r1_s3, %att25_inrel_r1_s3, %att26_inrel_r1_s3, %att27_inrel_r1_s3,
      %att28_inrel_r1_s3, %att29_inrel_r1_s3, %att30_inrel_r1_s3, %att31_inrel_r1_s3,
      %att32_inrel_r1_s3)
    nest.release %state_p0_8 depends_on(
      %pf_state_p0_r1_s3, %att0_inrel_r1_s3, %att4_inrel_r1_s3, %att8_inrel_r1_s3,
      %att12_inrel_r1_s3, %att16_inrel_r1_s3, %att20_inrel_r1_s3, %att24_inrel_r1_s3,
      %att28_inrel_r1_s3, %att32_inrel_r1_s3, %att0_out_r1_s3, %att4_out_r1_s3, %att8_out_r1_s3,
      %att12_out_r1_s3, %att16_out_r1_s3, %att20_out_r1_s3, %att24_out_r1_s3, %att28_out_r1_s3,
      %att32_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %acc_p0_8 depends_on(
      %pf_acc_p0_r1_s3, %att0_inrel_r1_s3, %att4_inrel_r1_s3, %att8_inrel_r1_s3,
      %att12_inrel_r1_s3, %att16_inrel_r1_s3, %att20_inrel_r1_s3, %att24_inrel_r1_s3,
      %att28_inrel_r1_s3, %att32_inrel_r1_s3, %att0_out_r1_s3, %att4_out_r1_s3, %att8_out_r1_s3,
      %att12_out_r1_s3, %att16_out_r1_s3, %att20_out_r1_s3, %att24_out_r1_s3, %att28_out_r1_s3,
      %att32_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %state_p1_8 depends_on(
      %pf_state_p1_r1_s3, %att1_inrel_r1_s3, %att5_inrel_r1_s3, %att9_inrel_r1_s3,
      %att13_inrel_r1_s3, %att17_inrel_r1_s3, %att21_inrel_r1_s3, %att25_inrel_r1_s3,
      %att29_inrel_r1_s3, %att1_out_r1_s3, %att5_out_r1_s3, %att9_out_r1_s3, %att13_out_r1_s3,
      %att17_out_r1_s3, %att21_out_r1_s3, %att25_out_r1_s3, %att29_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %acc_p1_8 depends_on(
      %pf_acc_p1_r1_s3, %att1_inrel_r1_s3, %att5_inrel_r1_s3, %att9_inrel_r1_s3,
      %att13_inrel_r1_s3, %att17_inrel_r1_s3, %att21_inrel_r1_s3, %att25_inrel_r1_s3,
      %att29_inrel_r1_s3, %att1_out_r1_s3, %att5_out_r1_s3, %att9_out_r1_s3, %att13_out_r1_s3,
      %att17_out_r1_s3, %att21_out_r1_s3, %att25_out_r1_s3, %att29_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %state_p2_8 depends_on(
      %pf_state_p2_r1_s3, %att2_inrel_r1_s3, %att6_inrel_r1_s3, %att10_inrel_r1_s3,
      %att14_inrel_r1_s3, %att18_inrel_r1_s3, %att22_inrel_r1_s3, %att26_inrel_r1_s3,
      %att30_inrel_r1_s3, %att2_out_r1_s3, %att6_out_r1_s3, %att10_out_r1_s3, %att14_out_r1_s3,
      %att18_out_r1_s3, %att22_out_r1_s3, %att26_out_r1_s3, %att30_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %acc_p2_8 depends_on(
      %pf_acc_p2_r1_s3, %att2_inrel_r1_s3, %att6_inrel_r1_s3, %att10_inrel_r1_s3,
      %att14_inrel_r1_s3, %att18_inrel_r1_s3, %att22_inrel_r1_s3, %att26_inrel_r1_s3,
      %att30_inrel_r1_s3, %att2_out_r1_s3, %att6_out_r1_s3, %att10_out_r1_s3, %att14_out_r1_s3,
      %att18_out_r1_s3, %att22_out_r1_s3, %att26_out_r1_s3, %att30_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %state_p3_8 depends_on(
      %pf_state_p3_r1_s3, %att3_inrel_r1_s3, %att7_inrel_r1_s3, %att11_inrel_r1_s3,
      %att15_inrel_r1_s3, %att19_inrel_r1_s3, %att23_inrel_r1_s3, %att27_inrel_r1_s3,
      %att31_inrel_r1_s3, %att3_out_r1_s3, %att7_out_r1_s3, %att11_out_r1_s3, %att15_out_r1_s3,
      %att19_out_r1_s3, %att23_out_r1_s3, %att27_out_r1_s3, %att31_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %acc_p3_8 depends_on(
      %pf_acc_p3_r1_s3, %att3_inrel_r1_s3, %att7_inrel_r1_s3, %att11_inrel_r1_s3,
      %att15_inrel_r1_s3, %att19_inrel_r1_s3, %att23_inrel_r1_s3, %att27_inrel_r1_s3,
      %att31_inrel_r1_s3, %att3_out_r1_s3, %att7_out_r1_s3, %att11_out_r1_s3, %att15_out_r1_s3,
      %att19_out_r1_s3, %att23_out_r1_s3, %att27_out_r1_s3, %att31_out_r1_s3, %merge_inrel_r1_s3)
    nest.release %out_l2_8 depends_on(%merge_out_r1_s3, %out_store_r1_s3)
    nest.return
  }
  nest.context @step_r2_s0(
    %POOL_8: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_8: !nest.global_memref<147xi32>,
    %APPEND_IDS_8: !nest.global_memref<12xi32>, %Q_IN_8: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_8: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_8: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_8: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_8: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_8: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 200, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_8 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_8 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_8 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_23 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_9 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_9 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_9 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_9 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_9 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_9 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_9 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_9 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_8 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_8 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_9 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %585 = nest.subview %POOL_8 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %586 = nest.subview %K_NEW_8 offsets = [2, 0, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %587 = nest.subview %V_NEW_8 offsets = [2, 0, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %588 = nest.subview %APPEND_IDS_8 offsets = [8] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %589 = nest.subview %Q_IN_8 offsets = [2, 0, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %590 = nest.subview %S_INIT_8 offsets = [2, 0, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %591 = nest.subview %S_INIT_8 offsets = [2, 0, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %592 = nest.subview %S_INIT_8 offsets = [2, 0, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %593 = nest.subview %S_INIT_8 offsets = [2, 0, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %594 = nest.subview %O_INIT_8 offsets = [2, 0, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %595 = nest.subview %O_INIT_8 offsets = [2, 0, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %596 = nest.subview %O_INIT_8 offsets = [2, 0, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %597 = nest.subview %O_INIT_8 offsets = [2, 0, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %598 = nest.subview %OUT_8 offsets = [2, 0, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %599 = nest.subview %BLOCK_TABLE_8 offsets = [98] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %600 = nest.subview %BLOCK_TABLE_8 offsets = [99] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %601 = nest.subview %BLOCK_TABLE_8 offsets = [100] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %602 = nest.subview %BLOCK_TABLE_8 offsets = [101] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %603 = nest.subview %BLOCK_TABLE_8 offsets = [102] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %604 = nest.subview %BLOCK_TABLE_8 offsets = [103] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %605 = nest.subview %BLOCK_TABLE_8 offsets = [104] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %606 = nest.subview %BLOCK_TABLE_8 offsets = [105] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %607 = nest.subview %BLOCK_TABLE_8 offsets = [106] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %608 = nest.subview %BLOCK_TABLE_8 offsets = [107] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %609 = nest.subview %BLOCK_TABLE_8 offsets = [108] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %610 = nest.subview %BLOCK_TABLE_8 offsets = [109] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %611 = nest.subview %BLOCK_TABLE_8 offsets = [110] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %612 = nest.subview %BLOCK_TABLE_8 offsets = [111] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %613 = nest.subview %BLOCK_TABLE_8 offsets = [112] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %614 = nest.subview %BLOCK_TABLE_8 offsets = [113] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %615 = nest.subview %BLOCK_TABLE_8 offsets = [114] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %616 = nest.subview %BLOCK_TABLE_8 offsets = [115] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %617 = nest.subview %BLOCK_TABLE_8 offsets = [116] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %618 = nest.subview %BLOCK_TABLE_8 offsets = [117] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %619 = nest.subview %BLOCK_TABLE_8 offsets = [118] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %620 = nest.subview %BLOCK_TABLE_8 offsets = [119] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %621 = nest.subview %BLOCK_TABLE_8 offsets = [120] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %622 = nest.subview %BLOCK_TABLE_8 offsets = [121] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %623 = nest.subview %BLOCK_TABLE_8 offsets = [122] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %624 = nest.subview %BLOCK_TABLE_8 offsets = [123] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %625 = nest.subview %BLOCK_TABLE_8 offsets = [124] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %626 = nest.subview %BLOCK_TABLE_8 offsets = [125] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %627 = nest.subview %BLOCK_TABLE_8 offsets = [126] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %628 = nest.subview %BLOCK_TABLE_8 offsets = [127] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %629 = nest.subview %BLOCK_TABLE_8 offsets = [128] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %630 = nest.subview %BLOCK_TABLE_8 offsets = [129] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %631 = nest.subview %BLOCK_TABLE_8 offsets = [130] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %632 = nest.subview %BLOCK_TABLE_8 offsets = [131] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %633 = nest.subview %BLOCK_TABLE_8 offsets = [132] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %634 = nest.subview %BLOCK_TABLE_8 offsets = [133] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %635 = nest.subview %BLOCK_TABLE_8 offsets = [134] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %636 = nest.subview %BLOCK_TABLE_8 offsets = [135] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %637 = nest.subview %BLOCK_TABLE_8 offsets = [136] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %638 = nest.subview %BLOCK_TABLE_8 offsets = [137] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %639 = nest.subview %BLOCK_TABLE_8 offsets = [138] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %640 = nest.subview %BLOCK_TABLE_8 offsets = [139] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %641 = nest.subview %BLOCK_TABLE_8 offsets = [140] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %642 = nest.subview %BLOCK_TABLE_8 offsets = [141] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %643 = nest.subview %BLOCK_TABLE_8 offsets = [142] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %644 = nest.subview %BLOCK_TABLE_8 offsets = [143] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %645 = nest.subview %BLOCK_TABLE_8 offsets = [144] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r2_s0 = nest.dma.prefetch.async %586 into %k_new_8 : !nest.event<"pf_k_r2_s0">
    %pf_v_r2_s0 = nest.dma.prefetch.async %587 into %v_new_8 : !nest.event<"pf_v_r2_s0">
    %pf_aidx_r2_s0 = nest.dma.prefetch.async %588 into %append_idx_8 : !nest.event<"pf_aidx_r2_s0">
    %pf_q_r2_s0 = nest.dma.prefetch.async %589 into %q_l2_23 : !nest.event<"pf_q_r2_s0">
    %pf_state_p0_r2_s0 = nest.dma.prefetch.async %590 into %state_p0_9
      : !nest.event<"pf_state_p0_r2_s0">
    %pf_state_p1_r2_s0 = nest.dma.prefetch.async %591 into %state_p1_9
      : !nest.event<"pf_state_p1_r2_s0">
    %pf_state_p2_r2_s0 = nest.dma.prefetch.async %592 into %state_p2_9
      : !nest.event<"pf_state_p2_r2_s0">
    %pf_state_p3_r2_s0 = nest.dma.prefetch.async %593 into %state_p3_9
      : !nest.event<"pf_state_p3_r2_s0">
    %pf_acc_p0_r2_s0 = nest.dma.prefetch.async %594 into %acc_p0_9 : !nest.event<"pf_acc_p0_r2_s0">
    %pf_acc_p1_r2_s0 = nest.dma.prefetch.async %595 into %acc_p1_9 : !nest.event<"pf_acc_p1_r2_s0">
    %pf_acc_p2_r2_s0 = nest.dma.prefetch.async %596 into %acc_p2_9 : !nest.event<"pf_acc_p2_r2_s0">
    %pf_acc_p3_r2_s0 = nest.dma.prefetch.async %597 into %acc_p3_9 : !nest.event<"pf_acc_p3_r2_s0">
    %646 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r2_s0, %append_inrel_r2_s0, %647 =
      nest.dispatch.tasks.async @paged_attention_append_r2_tip15 l1_mode = 1 tasks(%646)
      globals(%585) bindings(%k_new_8, %v_new_8, %append_idx_8)
      ins(%k_new_8, %v_new_8, %append_idx_8) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r2_s0, %pf_v_r2_s0, %pf_aidx_r2_s0)
      : (!nest.event<"append_grid_r2_s0">, !nest.event<"append_inrel_r2_s0">, !nest.event<"">)
    %pf_bidx0_r2_s0 = nest.dma.prefetch.async %599 into %block_idx_p0_8
      : !nest.event<"pf_bidx0_r2_s0">
    %att0_grid_r2_s0, %att0_inrel_r2_s0, %att0_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx0_r2_s0) : (
        !nest.event<"att0_grid_r2_s0">, !nest.event<"att0_inrel_r2_s0">,
        !nest.event<"att0_out_r2_s0">)
    %pf_bidx1_r2_s0 = nest.dma.prefetch.async %600 into %block_idx_p1_8
      : !nest.event<"pf_bidx1_r2_s0">
    %att1_grid_r2_s0, %att1_inrel_r2_s0, %att1_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx1_r2_s0) : (
        !nest.event<"att1_grid_r2_s0">, !nest.event<"att1_inrel_r2_s0">,
        !nest.event<"att1_out_r2_s0">)
    %pf_bidx2_r2_s0 = nest.dma.prefetch.async %601 into %block_idx_p0_8
      depends_on(%att0_inrel_r2_s0) : !nest.event<"pf_bidx2_r2_s0">
    %att2_grid_r2_s0, %att2_inrel_r2_s0, %att2_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx2_r2_s0) : (
        !nest.event<"att2_grid_r2_s0">, !nest.event<"att2_inrel_r2_s0">,
        !nest.event<"att2_out_r2_s0">)
    %pf_bidx3_r2_s0 = nest.dma.prefetch.async %602 into %block_idx_p1_8
      depends_on(%att1_inrel_r2_s0) : !nest.event<"pf_bidx3_r2_s0">
    %att3_grid_r2_s0, %att3_inrel_r2_s0, %att3_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx3_r2_s0) : (
        !nest.event<"att3_grid_r2_s0">, !nest.event<"att3_inrel_r2_s0">,
        !nest.event<"att3_out_r2_s0">)
    %pf_bidx4_r2_s0 = nest.dma.prefetch.async %603 into %block_idx_p0_8
      depends_on(%att2_inrel_r2_s0) : !nest.event<"pf_bidx4_r2_s0">
    %att4_grid_r2_s0, %att4_inrel_r2_s0, %att4_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx4_r2_s0, %att0_out_r2_s0)
      : (
        !nest.event<"att4_grid_r2_s0">, !nest.event<"att4_inrel_r2_s0">,
        !nest.event<"att4_out_r2_s0">)
    %pf_bidx5_r2_s0 = nest.dma.prefetch.async %604 into %block_idx_p1_8
      depends_on(%att3_inrel_r2_s0) : !nest.event<"pf_bidx5_r2_s0">
    %att5_grid_r2_s0, %att5_inrel_r2_s0, %att5_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx5_r2_s0, %att1_out_r2_s0)
      : (
        !nest.event<"att5_grid_r2_s0">, !nest.event<"att5_inrel_r2_s0">,
        !nest.event<"att5_out_r2_s0">)
    %pf_bidx6_r2_s0 = nest.dma.prefetch.async %605 into %block_idx_p0_8
      depends_on(%att4_inrel_r2_s0) : !nest.event<"pf_bidx6_r2_s0">
    %att6_grid_r2_s0, %att6_inrel_r2_s0, %att6_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx6_r2_s0, %att2_out_r2_s0)
      : (
        !nest.event<"att6_grid_r2_s0">, !nest.event<"att6_inrel_r2_s0">,
        !nest.event<"att6_out_r2_s0">)
    %pf_bidx7_r2_s0 = nest.dma.prefetch.async %606 into %block_idx_p1_8
      depends_on(%att5_inrel_r2_s0) : !nest.event<"pf_bidx7_r2_s0">
    %att7_grid_r2_s0, %att7_inrel_r2_s0, %att7_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx7_r2_s0, %att3_out_r2_s0)
      : (
        !nest.event<"att7_grid_r2_s0">, !nest.event<"att7_inrel_r2_s0">,
        !nest.event<"att7_out_r2_s0">)
    %pf_bidx8_r2_s0 = nest.dma.prefetch.async %607 into %block_idx_p0_8
      depends_on(%att6_inrel_r2_s0) : !nest.event<"pf_bidx8_r2_s0">
    %att8_grid_r2_s0, %att8_inrel_r2_s0, %att8_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx8_r2_s0, %att4_out_r2_s0)
      : (
        !nest.event<"att8_grid_r2_s0">, !nest.event<"att8_inrel_r2_s0">,
        !nest.event<"att8_out_r2_s0">)
    %pf_bidx9_r2_s0 = nest.dma.prefetch.async %608 into %block_idx_p1_8
      depends_on(%att7_inrel_r2_s0) : !nest.event<"pf_bidx9_r2_s0">
    %att9_grid_r2_s0, %att9_inrel_r2_s0, %att9_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx9_r2_s0, %att5_out_r2_s0)
      : (
        !nest.event<"att9_grid_r2_s0">, !nest.event<"att9_inrel_r2_s0">,
        !nest.event<"att9_out_r2_s0">)
    %pf_bidx10_r2_s0 = nest.dma.prefetch.async %609 into %block_idx_p0_8
      depends_on(%att8_inrel_r2_s0) : !nest.event<"pf_bidx10_r2_s0">
    %att10_grid_r2_s0, %att10_inrel_r2_s0, %att10_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx10_r2_s0, %att6_out_r2_s0)
      : (
        !nest.event<"att10_grid_r2_s0">, !nest.event<"att10_inrel_r2_s0">,
        !nest.event<"att10_out_r2_s0">)
    %pf_bidx11_r2_s0 = nest.dma.prefetch.async %610 into %block_idx_p1_8
      depends_on(%att9_inrel_r2_s0) : !nest.event<"pf_bidx11_r2_s0">
    %att11_grid_r2_s0, %att11_inrel_r2_s0, %att11_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx11_r2_s0, %att7_out_r2_s0)
      : (
        !nest.event<"att11_grid_r2_s0">, !nest.event<"att11_inrel_r2_s0">,
        !nest.event<"att11_out_r2_s0">)
    %pf_bidx12_r2_s0 = nest.dma.prefetch.async %611 into %block_idx_p0_8
      depends_on(%att10_inrel_r2_s0) : !nest.event<"pf_bidx12_r2_s0">
    %att12_grid_r2_s0, %att12_inrel_r2_s0, %att12_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx12_r2_s0, %att8_out_r2_s0)
      : (
        !nest.event<"att12_grid_r2_s0">, !nest.event<"att12_inrel_r2_s0">,
        !nest.event<"att12_out_r2_s0">)
    %pf_bidx13_r2_s0 = nest.dma.prefetch.async %612 into %block_idx_p1_8
      depends_on(%att11_inrel_r2_s0) : !nest.event<"pf_bidx13_r2_s0">
    %att13_grid_r2_s0, %att13_inrel_r2_s0, %att13_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx13_r2_s0, %att9_out_r2_s0)
      : (
        !nest.event<"att13_grid_r2_s0">, !nest.event<"att13_inrel_r2_s0">,
        !nest.event<"att13_out_r2_s0">)
    %pf_bidx14_r2_s0 = nest.dma.prefetch.async %613 into %block_idx_p0_8
      depends_on(%att12_inrel_r2_s0) : !nest.event<"pf_bidx14_r2_s0">
    %att14_grid_r2_s0, %att14_inrel_r2_s0, %att14_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx14_r2_s0, %att10_out_r2_s0)
      : (
        !nest.event<"att14_grid_r2_s0">, !nest.event<"att14_inrel_r2_s0">,
        !nest.event<"att14_out_r2_s0">)
    %pf_bidx15_r2_s0 = nest.dma.prefetch.async %614 into %block_idx_p1_8
      depends_on(%att13_inrel_r2_s0) : !nest.event<"pf_bidx15_r2_s0">
    %att15_grid_r2_s0, %att15_inrel_r2_s0, %att15_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx15_r2_s0, %att11_out_r2_s0)
      : (
        !nest.event<"att15_grid_r2_s0">, !nest.event<"att15_inrel_r2_s0">,
        !nest.event<"att15_out_r2_s0">)
    %pf_bidx16_r2_s0 = nest.dma.prefetch.async %615 into %block_idx_p0_8
      depends_on(%att14_inrel_r2_s0) : !nest.event<"pf_bidx16_r2_s0">
    %att16_grid_r2_s0, %att16_inrel_r2_s0, %att16_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx16_r2_s0, %att12_out_r2_s0)
      : (
        !nest.event<"att16_grid_r2_s0">, !nest.event<"att16_inrel_r2_s0">,
        !nest.event<"att16_out_r2_s0">)
    %pf_bidx17_r2_s0 = nest.dma.prefetch.async %616 into %block_idx_p1_8
      depends_on(%att15_inrel_r2_s0) : !nest.event<"pf_bidx17_r2_s0">
    %att17_grid_r2_s0, %att17_inrel_r2_s0, %att17_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx17_r2_s0, %att13_out_r2_s0)
      : (
        !nest.event<"att17_grid_r2_s0">, !nest.event<"att17_inrel_r2_s0">,
        !nest.event<"att17_out_r2_s0">)
    %pf_bidx18_r2_s0 = nest.dma.prefetch.async %617 into %block_idx_p0_8
      depends_on(%att16_inrel_r2_s0) : !nest.event<"pf_bidx18_r2_s0">
    %att18_grid_r2_s0, %att18_inrel_r2_s0, %att18_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx18_r2_s0, %att14_out_r2_s0)
      : (
        !nest.event<"att18_grid_r2_s0">, !nest.event<"att18_inrel_r2_s0">,
        !nest.event<"att18_out_r2_s0">)
    %pf_bidx19_r2_s0 = nest.dma.prefetch.async %618 into %block_idx_p1_8
      depends_on(%att17_inrel_r2_s0) : !nest.event<"pf_bidx19_r2_s0">
    %att19_grid_r2_s0, %att19_inrel_r2_s0, %att19_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx19_r2_s0, %att15_out_r2_s0)
      : (
        !nest.event<"att19_grid_r2_s0">, !nest.event<"att19_inrel_r2_s0">,
        !nest.event<"att19_out_r2_s0">)
    %pf_bidx20_r2_s0 = nest.dma.prefetch.async %619 into %block_idx_p0_8
      depends_on(%att18_inrel_r2_s0) : !nest.event<"pf_bidx20_r2_s0">
    %att20_grid_r2_s0, %att20_inrel_r2_s0, %att20_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx20_r2_s0, %att16_out_r2_s0)
      : (
        !nest.event<"att20_grid_r2_s0">, !nest.event<"att20_inrel_r2_s0">,
        !nest.event<"att20_out_r2_s0">)
    %pf_bidx21_r2_s0 = nest.dma.prefetch.async %620 into %block_idx_p1_8
      depends_on(%att19_inrel_r2_s0) : !nest.event<"pf_bidx21_r2_s0">
    %att21_grid_r2_s0, %att21_inrel_r2_s0, %att21_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx21_r2_s0, %att17_out_r2_s0)
      : (
        !nest.event<"att21_grid_r2_s0">, !nest.event<"att21_inrel_r2_s0">,
        !nest.event<"att21_out_r2_s0">)
    %pf_bidx22_r2_s0 = nest.dma.prefetch.async %621 into %block_idx_p0_8
      depends_on(%att20_inrel_r2_s0) : !nest.event<"pf_bidx22_r2_s0">
    %att22_grid_r2_s0, %att22_inrel_r2_s0, %att22_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx22_r2_s0, %att18_out_r2_s0)
      : (
        !nest.event<"att22_grid_r2_s0">, !nest.event<"att22_inrel_r2_s0">,
        !nest.event<"att22_out_r2_s0">)
    %pf_bidx23_r2_s0 = nest.dma.prefetch.async %622 into %block_idx_p1_8
      depends_on(%att21_inrel_r2_s0) : !nest.event<"pf_bidx23_r2_s0">
    %att23_grid_r2_s0, %att23_inrel_r2_s0, %att23_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx23_r2_s0, %att19_out_r2_s0)
      : (
        !nest.event<"att23_grid_r2_s0">, !nest.event<"att23_inrel_r2_s0">,
        !nest.event<"att23_out_r2_s0">)
    %pf_bidx24_r2_s0 = nest.dma.prefetch.async %623 into %block_idx_p0_8
      depends_on(%att22_inrel_r2_s0) : !nest.event<"pf_bidx24_r2_s0">
    %att24_grid_r2_s0, %att24_inrel_r2_s0, %att24_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx24_r2_s0, %att20_out_r2_s0)
      : (
        !nest.event<"att24_grid_r2_s0">, !nest.event<"att24_inrel_r2_s0">,
        !nest.event<"att24_out_r2_s0">)
    %pf_bidx25_r2_s0 = nest.dma.prefetch.async %624 into %block_idx_p1_8
      depends_on(%att23_inrel_r2_s0) : !nest.event<"pf_bidx25_r2_s0">
    %att25_grid_r2_s0, %att25_inrel_r2_s0, %att25_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx25_r2_s0, %att21_out_r2_s0)
      : (
        !nest.event<"att25_grid_r2_s0">, !nest.event<"att25_inrel_r2_s0">,
        !nest.event<"att25_out_r2_s0">)
    %pf_bidx26_r2_s0 = nest.dma.prefetch.async %625 into %block_idx_p0_8
      depends_on(%att24_inrel_r2_s0) : !nest.event<"pf_bidx26_r2_s0">
    %att26_grid_r2_s0, %att26_inrel_r2_s0, %att26_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx26_r2_s0, %att22_out_r2_s0)
      : (
        !nest.event<"att26_grid_r2_s0">, !nest.event<"att26_inrel_r2_s0">,
        !nest.event<"att26_out_r2_s0">)
    %pf_bidx27_r2_s0 = nest.dma.prefetch.async %626 into %block_idx_p1_8
      depends_on(%att25_inrel_r2_s0) : !nest.event<"pf_bidx27_r2_s0">
    %att27_grid_r2_s0, %att27_inrel_r2_s0, %att27_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx27_r2_s0, %att23_out_r2_s0)
      : (
        !nest.event<"att27_grid_r2_s0">, !nest.event<"att27_inrel_r2_s0">,
        !nest.event<"att27_out_r2_s0">)
    %pf_bidx28_r2_s0 = nest.dma.prefetch.async %627 into %block_idx_p0_8
      depends_on(%att26_inrel_r2_s0) : !nest.event<"pf_bidx28_r2_s0">
    %att28_grid_r2_s0, %att28_inrel_r2_s0, %att28_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx28_r2_s0, %att24_out_r2_s0)
      : (
        !nest.event<"att28_grid_r2_s0">, !nest.event<"att28_inrel_r2_s0">,
        !nest.event<"att28_out_r2_s0">)
    %pf_bidx29_r2_s0 = nest.dma.prefetch.async %628 into %block_idx_p1_8
      depends_on(%att27_inrel_r2_s0) : !nest.event<"pf_bidx29_r2_s0">
    %att29_grid_r2_s0, %att29_inrel_r2_s0, %att29_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx29_r2_s0, %att25_out_r2_s0)
      : (
        !nest.event<"att29_grid_r2_s0">, !nest.event<"att29_inrel_r2_s0">,
        !nest.event<"att29_out_r2_s0">)
    %pf_bidx30_r2_s0 = nest.dma.prefetch.async %629 into %block_idx_p0_8
      depends_on(%att28_inrel_r2_s0) : !nest.event<"pf_bidx30_r2_s0">
    %att30_grid_r2_s0, %att30_inrel_r2_s0, %att30_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx30_r2_s0, %att26_out_r2_s0)
      : (
        !nest.event<"att30_grid_r2_s0">, !nest.event<"att30_inrel_r2_s0">,
        !nest.event<"att30_out_r2_s0">)
    %pf_bidx31_r2_s0 = nest.dma.prefetch.async %630 into %block_idx_p1_8
      depends_on(%att29_inrel_r2_s0) : !nest.event<"pf_bidx31_r2_s0">
    %att31_grid_r2_s0, %att31_inrel_r2_s0, %att31_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx31_r2_s0, %att27_out_r2_s0)
      : (
        !nest.event<"att31_grid_r2_s0">, !nest.event<"att31_inrel_r2_s0">,
        !nest.event<"att31_out_r2_s0">)
    %pf_bidx32_r2_s0 = nest.dma.prefetch.async %631 into %block_idx_p0_8
      depends_on(%att30_inrel_r2_s0) : !nest.event<"pf_bidx32_r2_s0">
    %att32_grid_r2_s0, %att32_inrel_r2_s0, %att32_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx32_r2_s0, %att28_out_r2_s0)
      : (
        !nest.event<"att32_grid_r2_s0">, !nest.event<"att32_inrel_r2_s0">,
        !nest.event<"att32_out_r2_s0">)
    %pf_bidx33_r2_s0 = nest.dma.prefetch.async %632 into %block_idx_p1_8
      depends_on(%att31_inrel_r2_s0) : !nest.event<"pf_bidx33_r2_s0">
    %att33_grid_r2_s0, %att33_inrel_r2_s0, %att33_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx33_r2_s0, %att29_out_r2_s0)
      : (
        !nest.event<"att33_grid_r2_s0">, !nest.event<"att33_inrel_r2_s0">,
        !nest.event<"att33_out_r2_s0">)
    %pf_bidx34_r2_s0 = nest.dma.prefetch.async %633 into %block_idx_p0_8
      depends_on(%att32_inrel_r2_s0) : !nest.event<"pf_bidx34_r2_s0">
    %att34_grid_r2_s0, %att34_inrel_r2_s0, %att34_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx34_r2_s0, %att30_out_r2_s0)
      : (
        !nest.event<"att34_grid_r2_s0">, !nest.event<"att34_inrel_r2_s0">,
        !nest.event<"att34_out_r2_s0">)
    %pf_bidx35_r2_s0 = nest.dma.prefetch.async %634 into %block_idx_p1_8
      depends_on(%att33_inrel_r2_s0) : !nest.event<"pf_bidx35_r2_s0">
    %att35_grid_r2_s0, %att35_inrel_r2_s0, %att35_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx35_r2_s0, %att31_out_r2_s0)
      : (
        !nest.event<"att35_grid_r2_s0">, !nest.event<"att35_inrel_r2_s0">,
        !nest.event<"att35_out_r2_s0">)
    %pf_bidx36_r2_s0 = nest.dma.prefetch.async %635 into %block_idx_p0_8
      depends_on(%att34_inrel_r2_s0) : !nest.event<"pf_bidx36_r2_s0">
    %att36_grid_r2_s0, %att36_inrel_r2_s0, %att36_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx36_r2_s0, %att32_out_r2_s0)
      : (
        !nest.event<"att36_grid_r2_s0">, !nest.event<"att36_inrel_r2_s0">,
        !nest.event<"att36_out_r2_s0">)
    %pf_bidx37_r2_s0 = nest.dma.prefetch.async %636 into %block_idx_p1_8
      depends_on(%att35_inrel_r2_s0) : !nest.event<"pf_bidx37_r2_s0">
    %att37_grid_r2_s0, %att37_inrel_r2_s0, %att37_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx37_r2_s0, %att33_out_r2_s0)
      : (
        !nest.event<"att37_grid_r2_s0">, !nest.event<"att37_inrel_r2_s0">,
        !nest.event<"att37_out_r2_s0">)
    %pf_bidx38_r2_s0 = nest.dma.prefetch.async %637 into %block_idx_p0_8
      depends_on(%att36_inrel_r2_s0) : !nest.event<"pf_bidx38_r2_s0">
    %att38_grid_r2_s0, %att38_inrel_r2_s0, %att38_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx38_r2_s0, %att34_out_r2_s0)
      : (
        !nest.event<"att38_grid_r2_s0">, !nest.event<"att38_inrel_r2_s0">,
        !nest.event<"att38_out_r2_s0">)
    %pf_bidx39_r2_s0 = nest.dma.prefetch.async %638 into %block_idx_p1_8
      depends_on(%att37_inrel_r2_s0) : !nest.event<"pf_bidx39_r2_s0">
    %att39_grid_r2_s0, %att39_inrel_r2_s0, %att39_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx39_r2_s0, %att35_out_r2_s0)
      : (
        !nest.event<"att39_grid_r2_s0">, !nest.event<"att39_inrel_r2_s0">,
        !nest.event<"att39_out_r2_s0">)
    %pf_bidx40_r2_s0 = nest.dma.prefetch.async %639 into %block_idx_p0_8
      depends_on(%att38_inrel_r2_s0) : !nest.event<"pf_bidx40_r2_s0">
    %att40_grid_r2_s0, %att40_inrel_r2_s0, %att40_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx40_r2_s0, %att36_out_r2_s0)
      : (
        !nest.event<"att40_grid_r2_s0">, !nest.event<"att40_inrel_r2_s0">,
        !nest.event<"att40_out_r2_s0">)
    %pf_bidx41_r2_s0 = nest.dma.prefetch.async %640 into %block_idx_p1_8
      depends_on(%att39_inrel_r2_s0) : !nest.event<"pf_bidx41_r2_s0">
    %att41_grid_r2_s0, %att41_inrel_r2_s0, %att41_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx41_r2_s0, %att37_out_r2_s0)
      : (
        !nest.event<"att41_grid_r2_s0">, !nest.event<"att41_inrel_r2_s0">,
        !nest.event<"att41_out_r2_s0">)
    %pf_bidx42_r2_s0 = nest.dma.prefetch.async %641 into %block_idx_p0_8
      depends_on(%att40_inrel_r2_s0) : !nest.event<"pf_bidx42_r2_s0">
    %att42_grid_r2_s0, %att42_inrel_r2_s0, %att42_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx42_r2_s0, %att38_out_r2_s0)
      : (
        !nest.event<"att42_grid_r2_s0">, !nest.event<"att42_inrel_r2_s0">,
        !nest.event<"att42_out_r2_s0">)
    %pf_bidx43_r2_s0 = nest.dma.prefetch.async %642 into %block_idx_p1_8
      depends_on(%att41_inrel_r2_s0) : !nest.event<"pf_bidx43_r2_s0">
    %att43_grid_r2_s0, %att43_inrel_r2_s0, %att43_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_bidx43_r2_s0, %att39_out_r2_s0)
      : (
        !nest.event<"att43_grid_r2_s0">, !nest.event<"att43_inrel_r2_s0">,
        !nest.event<"att43_out_r2_s0">)
    %pf_bidx44_r2_s0 = nest.dma.prefetch.async %643 into %block_idx_p0_8
      depends_on(%att42_inrel_r2_s0) : !nest.event<"pf_bidx44_r2_s0">
    %att44_grid_r2_s0, %att44_inrel_r2_s0, %att44_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p0_9, %acc_p0_9) outs(%state_p0_9, %acc_p0_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p0_r2_s0, %pf_acc_p0_r2_s0, %pf_bidx44_r2_s0, %att40_out_r2_s0)
      : (
        !nest.event<"att44_grid_r2_s0">, !nest.event<"att44_inrel_r2_s0">,
        !nest.event<"att44_out_r2_s0">)
    %pf_bidx45_r2_s0 = nest.dma.prefetch.async %644 into %block_idx_p1_8
      depends_on(%att43_inrel_r2_s0) : !nest.event<"pf_bidx45_r2_s0">
    %att45_grid_r2_s0, %att45_inrel_r2_s0, %att45_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9)
      ins(%block_idx_p1_8, %q_l2_23, %state_p1_9, %acc_p1_9) outs(%state_p1_9, %acc_p1_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p1_r2_s0, %pf_acc_p1_r2_s0, %pf_bidx45_r2_s0, %att41_out_r2_s0)
      : (
        !nest.event<"att45_grid_r2_s0">, !nest.event<"att45_inrel_r2_s0">,
        !nest.event<"att45_out_r2_s0">)
    %pf_bidx46_r2_s0 = nest.dma.prefetch.async %645 into %block_idx_p0_8
      depends_on(%att44_inrel_r2_s0) : !nest.event<"pf_bidx46_r2_s0">
    %att46_grid_r2_s0, %att46_inrel_r2_s0, %att46_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9)
      ins(%block_idx_p0_8, %q_l2_23, %state_p2_9, %acc_p2_9) outs(%state_p2_9, %acc_p2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p2_r2_s0, %pf_acc_p2_r2_s0, %pf_bidx46_r2_s0, %att42_out_r2_s0)
      : (
        !nest.event<"att46_grid_r2_s0">, !nest.event<"att46_inrel_r2_s0">,
        !nest.event<"att46_out_r2_s0">)
    %att47_grid_r2_s0, %att47_inrel_r2_s0, %att47_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_t16_final_r2 l1_mode = 1 tasks(%646) globals(%585)
      bindings(%append_idx_8, %q_l2_23, %state_p3_9, %acc_p3_9)
      ins(%append_idx_8, %q_l2_23, %state_p3_9, %acc_p3_9) outs(%state_p3_9, %acc_p3_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s0, %pf_state_p3_r2_s0, %pf_acc_p3_r2_s0, %pf_aidx_r2_s0, %append_grid_r2_s0,
        %att43_out_r2_s0)
      : (
        !nest.event<"att47_grid_r2_s0">, !nest.event<"att47_inrel_r2_s0">,
        !nest.event<"att47_out_r2_s0">)
    %merge_grid_r2_s0, %merge_inrel_r2_s0, %merge_out_r2_s0 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%646) globals()
      bindings(
        %state_p0_9, %state_p1_9, %state_p2_9, %state_p3_9, %acc_p0_9, %acc_p1_9, %acc_p2_9,
        %acc_p3_9, %out_l2_9)
      ins(
        %state_p0_9, %state_p1_9, %state_p2_9, %state_p3_9, %acc_p0_9, %acc_p1_9, %acc_p2_9,
        %acc_p3_9)
      outs(%out_l2_9)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att44_out_r2_s0, %att45_out_r2_s0, %att46_out_r2_s0, %att47_out_r2_s0) : (
        !nest.event<"merge_grid_r2_s0">, !nest.event<"merge_inrel_r2_s0">,
        !nest.event<"merge_out_r2_s0">)
    %out_store_r2_s0 = nest.dma.store.async %out_l2_9 into %598 depends_on(%merge_out_r2_s0)
      : !nest.event<"out_store_r2_s0">
    nest.await %append_grid_r2_s0, %att47_grid_r2_s0, %merge_grid_r2_s0, %out_store_r2_s0
    nest.release %block_idx_p0_8 depends_on(
      %pf_bidx0_r2_s0, %pf_bidx2_r2_s0, %pf_bidx4_r2_s0, %pf_bidx6_r2_s0, %pf_bidx8_r2_s0,
      %pf_bidx10_r2_s0, %pf_bidx12_r2_s0, %pf_bidx14_r2_s0, %pf_bidx16_r2_s0, %pf_bidx18_r2_s0,
      %pf_bidx20_r2_s0, %pf_bidx22_r2_s0, %pf_bidx24_r2_s0, %pf_bidx26_r2_s0, %pf_bidx28_r2_s0,
      %pf_bidx30_r2_s0, %pf_bidx32_r2_s0, %pf_bidx34_r2_s0, %pf_bidx36_r2_s0, %pf_bidx38_r2_s0,
      %pf_bidx40_r2_s0, %pf_bidx42_r2_s0, %pf_bidx44_r2_s0, %pf_bidx46_r2_s0, %att0_inrel_r2_s0,
      %att2_inrel_r2_s0, %att4_inrel_r2_s0, %att6_inrel_r2_s0, %att8_inrel_r2_s0,
      %att10_inrel_r2_s0, %att12_inrel_r2_s0, %att14_inrel_r2_s0, %att16_inrel_r2_s0,
      %att18_inrel_r2_s0, %att20_inrel_r2_s0, %att22_inrel_r2_s0, %att24_inrel_r2_s0,
      %att26_inrel_r2_s0, %att28_inrel_r2_s0, %att30_inrel_r2_s0, %att32_inrel_r2_s0,
      %att34_inrel_r2_s0, %att36_inrel_r2_s0, %att38_inrel_r2_s0, %att40_inrel_r2_s0,
      %att42_inrel_r2_s0, %att44_inrel_r2_s0, %att46_inrel_r2_s0)
    nest.release %block_idx_p1_8 depends_on(
      %pf_bidx1_r2_s0, %pf_bidx3_r2_s0, %pf_bidx5_r2_s0, %pf_bidx7_r2_s0, %pf_bidx9_r2_s0,
      %pf_bidx11_r2_s0, %pf_bidx13_r2_s0, %pf_bidx15_r2_s0, %pf_bidx17_r2_s0, %pf_bidx19_r2_s0,
      %pf_bidx21_r2_s0, %pf_bidx23_r2_s0, %pf_bidx25_r2_s0, %pf_bidx27_r2_s0, %pf_bidx29_r2_s0,
      %pf_bidx31_r2_s0, %pf_bidx33_r2_s0, %pf_bidx35_r2_s0, %pf_bidx37_r2_s0, %pf_bidx39_r2_s0,
      %pf_bidx41_r2_s0, %pf_bidx43_r2_s0, %pf_bidx45_r2_s0, %att1_inrel_r2_s0, %att3_inrel_r2_s0,
      %att5_inrel_r2_s0, %att7_inrel_r2_s0, %att9_inrel_r2_s0, %att11_inrel_r2_s0,
      %att13_inrel_r2_s0, %att15_inrel_r2_s0, %att17_inrel_r2_s0, %att19_inrel_r2_s0,
      %att21_inrel_r2_s0, %att23_inrel_r2_s0, %att25_inrel_r2_s0, %att27_inrel_r2_s0,
      %att29_inrel_r2_s0, %att31_inrel_r2_s0, %att33_inrel_r2_s0, %att35_inrel_r2_s0,
      %att37_inrel_r2_s0, %att39_inrel_r2_s0, %att41_inrel_r2_s0, %att43_inrel_r2_s0,
      %att45_inrel_r2_s0)
    nest.release %k_new_8 depends_on(%pf_k_r2_s0, %append_inrel_r2_s0)
    nest.release %v_new_8 depends_on(%pf_v_r2_s0, %append_inrel_r2_s0)
    nest.release %append_idx_8 depends_on(%pf_aidx_r2_s0, %append_inrel_r2_s0, %att47_inrel_r2_s0)
    nest.release %q_l2_23 depends_on(
      %pf_q_r2_s0, %att0_inrel_r2_s0, %att1_inrel_r2_s0, %att2_inrel_r2_s0, %att3_inrel_r2_s0,
      %att4_inrel_r2_s0, %att5_inrel_r2_s0, %att6_inrel_r2_s0, %att7_inrel_r2_s0,
      %att8_inrel_r2_s0, %att9_inrel_r2_s0, %att10_inrel_r2_s0, %att11_inrel_r2_s0,
      %att12_inrel_r2_s0, %att13_inrel_r2_s0, %att14_inrel_r2_s0, %att15_inrel_r2_s0,
      %att16_inrel_r2_s0, %att17_inrel_r2_s0, %att18_inrel_r2_s0, %att19_inrel_r2_s0,
      %att20_inrel_r2_s0, %att21_inrel_r2_s0, %att22_inrel_r2_s0, %att23_inrel_r2_s0,
      %att24_inrel_r2_s0, %att25_inrel_r2_s0, %att26_inrel_r2_s0, %att27_inrel_r2_s0,
      %att28_inrel_r2_s0, %att29_inrel_r2_s0, %att30_inrel_r2_s0, %att31_inrel_r2_s0,
      %att32_inrel_r2_s0, %att33_inrel_r2_s0, %att34_inrel_r2_s0, %att35_inrel_r2_s0,
      %att36_inrel_r2_s0, %att37_inrel_r2_s0, %att38_inrel_r2_s0, %att39_inrel_r2_s0,
      %att40_inrel_r2_s0, %att41_inrel_r2_s0, %att42_inrel_r2_s0, %att43_inrel_r2_s0,
      %att44_inrel_r2_s0, %att45_inrel_r2_s0, %att46_inrel_r2_s0, %att47_inrel_r2_s0)
    nest.release %state_p0_9 depends_on(
      %pf_state_p0_r2_s0, %att0_inrel_r2_s0, %att4_inrel_r2_s0, %att8_inrel_r2_s0,
      %att12_inrel_r2_s0, %att16_inrel_r2_s0, %att20_inrel_r2_s0, %att24_inrel_r2_s0,
      %att28_inrel_r2_s0, %att32_inrel_r2_s0, %att36_inrel_r2_s0, %att40_inrel_r2_s0,
      %att44_inrel_r2_s0, %att0_out_r2_s0, %att4_out_r2_s0, %att8_out_r2_s0, %att12_out_r2_s0,
      %att16_out_r2_s0, %att20_out_r2_s0, %att24_out_r2_s0, %att28_out_r2_s0, %att32_out_r2_s0,
      %att36_out_r2_s0, %att40_out_r2_s0, %att44_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %acc_p0_9 depends_on(
      %pf_acc_p0_r2_s0, %att0_inrel_r2_s0, %att4_inrel_r2_s0, %att8_inrel_r2_s0,
      %att12_inrel_r2_s0, %att16_inrel_r2_s0, %att20_inrel_r2_s0, %att24_inrel_r2_s0,
      %att28_inrel_r2_s0, %att32_inrel_r2_s0, %att36_inrel_r2_s0, %att40_inrel_r2_s0,
      %att44_inrel_r2_s0, %att0_out_r2_s0, %att4_out_r2_s0, %att8_out_r2_s0, %att12_out_r2_s0,
      %att16_out_r2_s0, %att20_out_r2_s0, %att24_out_r2_s0, %att28_out_r2_s0, %att32_out_r2_s0,
      %att36_out_r2_s0, %att40_out_r2_s0, %att44_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %state_p1_9 depends_on(
      %pf_state_p1_r2_s0, %att1_inrel_r2_s0, %att5_inrel_r2_s0, %att9_inrel_r2_s0,
      %att13_inrel_r2_s0, %att17_inrel_r2_s0, %att21_inrel_r2_s0, %att25_inrel_r2_s0,
      %att29_inrel_r2_s0, %att33_inrel_r2_s0, %att37_inrel_r2_s0, %att41_inrel_r2_s0,
      %att45_inrel_r2_s0, %att1_out_r2_s0, %att5_out_r2_s0, %att9_out_r2_s0, %att13_out_r2_s0,
      %att17_out_r2_s0, %att21_out_r2_s0, %att25_out_r2_s0, %att29_out_r2_s0, %att33_out_r2_s0,
      %att37_out_r2_s0, %att41_out_r2_s0, %att45_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %acc_p1_9 depends_on(
      %pf_acc_p1_r2_s0, %att1_inrel_r2_s0, %att5_inrel_r2_s0, %att9_inrel_r2_s0,
      %att13_inrel_r2_s0, %att17_inrel_r2_s0, %att21_inrel_r2_s0, %att25_inrel_r2_s0,
      %att29_inrel_r2_s0, %att33_inrel_r2_s0, %att37_inrel_r2_s0, %att41_inrel_r2_s0,
      %att45_inrel_r2_s0, %att1_out_r2_s0, %att5_out_r2_s0, %att9_out_r2_s0, %att13_out_r2_s0,
      %att17_out_r2_s0, %att21_out_r2_s0, %att25_out_r2_s0, %att29_out_r2_s0, %att33_out_r2_s0,
      %att37_out_r2_s0, %att41_out_r2_s0, %att45_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %state_p2_9 depends_on(
      %pf_state_p2_r2_s0, %att2_inrel_r2_s0, %att6_inrel_r2_s0, %att10_inrel_r2_s0,
      %att14_inrel_r2_s0, %att18_inrel_r2_s0, %att22_inrel_r2_s0, %att26_inrel_r2_s0,
      %att30_inrel_r2_s0, %att34_inrel_r2_s0, %att38_inrel_r2_s0, %att42_inrel_r2_s0,
      %att46_inrel_r2_s0, %att2_out_r2_s0, %att6_out_r2_s0, %att10_out_r2_s0, %att14_out_r2_s0,
      %att18_out_r2_s0, %att22_out_r2_s0, %att26_out_r2_s0, %att30_out_r2_s0, %att34_out_r2_s0,
      %att38_out_r2_s0, %att42_out_r2_s0, %att46_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %acc_p2_9 depends_on(
      %pf_acc_p2_r2_s0, %att2_inrel_r2_s0, %att6_inrel_r2_s0, %att10_inrel_r2_s0,
      %att14_inrel_r2_s0, %att18_inrel_r2_s0, %att22_inrel_r2_s0, %att26_inrel_r2_s0,
      %att30_inrel_r2_s0, %att34_inrel_r2_s0, %att38_inrel_r2_s0, %att42_inrel_r2_s0,
      %att46_inrel_r2_s0, %att2_out_r2_s0, %att6_out_r2_s0, %att10_out_r2_s0, %att14_out_r2_s0,
      %att18_out_r2_s0, %att22_out_r2_s0, %att26_out_r2_s0, %att30_out_r2_s0, %att34_out_r2_s0,
      %att38_out_r2_s0, %att42_out_r2_s0, %att46_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %state_p3_9 depends_on(
      %pf_state_p3_r2_s0, %att3_inrel_r2_s0, %att7_inrel_r2_s0, %att11_inrel_r2_s0,
      %att15_inrel_r2_s0, %att19_inrel_r2_s0, %att23_inrel_r2_s0, %att27_inrel_r2_s0,
      %att31_inrel_r2_s0, %att35_inrel_r2_s0, %att39_inrel_r2_s0, %att43_inrel_r2_s0,
      %att47_inrel_r2_s0, %att3_out_r2_s0, %att7_out_r2_s0, %att11_out_r2_s0, %att15_out_r2_s0,
      %att19_out_r2_s0, %att23_out_r2_s0, %att27_out_r2_s0, %att31_out_r2_s0, %att35_out_r2_s0,
      %att39_out_r2_s0, %att43_out_r2_s0, %att47_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %acc_p3_9 depends_on(
      %pf_acc_p3_r2_s0, %att3_inrel_r2_s0, %att7_inrel_r2_s0, %att11_inrel_r2_s0,
      %att15_inrel_r2_s0, %att19_inrel_r2_s0, %att23_inrel_r2_s0, %att27_inrel_r2_s0,
      %att31_inrel_r2_s0, %att35_inrel_r2_s0, %att39_inrel_r2_s0, %att43_inrel_r2_s0,
      %att47_inrel_r2_s0, %att3_out_r2_s0, %att7_out_r2_s0, %att11_out_r2_s0, %att15_out_r2_s0,
      %att19_out_r2_s0, %att23_out_r2_s0, %att27_out_r2_s0, %att31_out_r2_s0, %att35_out_r2_s0,
      %att39_out_r2_s0, %att43_out_r2_s0, %att47_out_r2_s0, %merge_inrel_r2_s0)
    nest.release %out_l2_9 depends_on(%merge_out_r2_s0, %out_store_r2_s0)
    nest.return
  }
  nest.context @step_r2_s1(
    %POOL_9: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_9: !nest.global_memref<147xi32>,
    %APPEND_IDS_9: !nest.global_memref<12xi32>, %Q_IN_9: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_9: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_9: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_9: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_9: !nest.global_memref<3x4x4x4x4x64xf32>, %OUT_9: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 204, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_9 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_9 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_9 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_24 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_10 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_10 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_10 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_10 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_10 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_10 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_10 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_10 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_9 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_9 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_10 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %648 = nest.subview %POOL_9 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %649 = nest.subview %K_NEW_9 offsets = [2, 1, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %650 = nest.subview %V_NEW_9 offsets = [2, 1, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %651 = nest.subview %APPEND_IDS_9 offsets = [9] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %652 = nest.subview %Q_IN_9 offsets = [2, 1, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %653 = nest.subview %S_INIT_9 offsets = [2, 1, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %654 = nest.subview %S_INIT_9 offsets = [2, 1, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %655 = nest.subview %S_INIT_9 offsets = [2, 1, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %656 = nest.subview %S_INIT_9 offsets = [2, 1, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %657 = nest.subview %O_INIT_9 offsets = [2, 1, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %658 = nest.subview %O_INIT_9 offsets = [2, 1, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %659 = nest.subview %O_INIT_9 offsets = [2, 1, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %660 = nest.subview %O_INIT_9 offsets = [2, 1, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %661 = nest.subview %OUT_9 offsets = [2, 1, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %662 = nest.subview %BLOCK_TABLE_9 offsets = [98] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %663 = nest.subview %BLOCK_TABLE_9 offsets = [99] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %664 = nest.subview %BLOCK_TABLE_9 offsets = [100] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %665 = nest.subview %BLOCK_TABLE_9 offsets = [101] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %666 = nest.subview %BLOCK_TABLE_9 offsets = [102] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %667 = nest.subview %BLOCK_TABLE_9 offsets = [103] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %668 = nest.subview %BLOCK_TABLE_9 offsets = [104] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %669 = nest.subview %BLOCK_TABLE_9 offsets = [105] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %670 = nest.subview %BLOCK_TABLE_9 offsets = [106] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %671 = nest.subview %BLOCK_TABLE_9 offsets = [107] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %672 = nest.subview %BLOCK_TABLE_9 offsets = [108] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %673 = nest.subview %BLOCK_TABLE_9 offsets = [109] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %674 = nest.subview %BLOCK_TABLE_9 offsets = [110] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %675 = nest.subview %BLOCK_TABLE_9 offsets = [111] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %676 = nest.subview %BLOCK_TABLE_9 offsets = [112] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %677 = nest.subview %BLOCK_TABLE_9 offsets = [113] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %678 = nest.subview %BLOCK_TABLE_9 offsets = [114] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %679 = nest.subview %BLOCK_TABLE_9 offsets = [115] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %680 = nest.subview %BLOCK_TABLE_9 offsets = [116] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %681 = nest.subview %BLOCK_TABLE_9 offsets = [117] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %682 = nest.subview %BLOCK_TABLE_9 offsets = [118] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %683 = nest.subview %BLOCK_TABLE_9 offsets = [119] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %684 = nest.subview %BLOCK_TABLE_9 offsets = [120] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %685 = nest.subview %BLOCK_TABLE_9 offsets = [121] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %686 = nest.subview %BLOCK_TABLE_9 offsets = [122] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %687 = nest.subview %BLOCK_TABLE_9 offsets = [123] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %688 = nest.subview %BLOCK_TABLE_9 offsets = [124] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %689 = nest.subview %BLOCK_TABLE_9 offsets = [125] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %690 = nest.subview %BLOCK_TABLE_9 offsets = [126] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %691 = nest.subview %BLOCK_TABLE_9 offsets = [127] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %692 = nest.subview %BLOCK_TABLE_9 offsets = [128] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %693 = nest.subview %BLOCK_TABLE_9 offsets = [129] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %694 = nest.subview %BLOCK_TABLE_9 offsets = [130] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %695 = nest.subview %BLOCK_TABLE_9 offsets = [131] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %696 = nest.subview %BLOCK_TABLE_9 offsets = [132] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %697 = nest.subview %BLOCK_TABLE_9 offsets = [133] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %698 = nest.subview %BLOCK_TABLE_9 offsets = [134] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %699 = nest.subview %BLOCK_TABLE_9 offsets = [135] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %700 = nest.subview %BLOCK_TABLE_9 offsets = [136] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %701 = nest.subview %BLOCK_TABLE_9 offsets = [137] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %702 = nest.subview %BLOCK_TABLE_9 offsets = [138] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %703 = nest.subview %BLOCK_TABLE_9 offsets = [139] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %704 = nest.subview %BLOCK_TABLE_9 offsets = [140] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %705 = nest.subview %BLOCK_TABLE_9 offsets = [141] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %706 = nest.subview %BLOCK_TABLE_9 offsets = [142] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %707 = nest.subview %BLOCK_TABLE_9 offsets = [143] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %708 = nest.subview %BLOCK_TABLE_9 offsets = [144] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %709 = nest.subview %BLOCK_TABLE_9 offsets = [145] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r2_s1 = nest.dma.prefetch.async %649 into %k_new_9 : !nest.event<"pf_k_r2_s1">
    %pf_v_r2_s1 = nest.dma.prefetch.async %650 into %v_new_9 : !nest.event<"pf_v_r2_s1">
    %pf_aidx_r2_s1 = nest.dma.prefetch.async %651 into %append_idx_9 : !nest.event<"pf_aidx_r2_s1">
    %pf_q_r2_s1 = nest.dma.prefetch.async %652 into %q_l2_24 : !nest.event<"pf_q_r2_s1">
    %pf_state_p0_r2_s1 = nest.dma.prefetch.async %653 into %state_p0_10
      : !nest.event<"pf_state_p0_r2_s1">
    %pf_state_p1_r2_s1 = nest.dma.prefetch.async %654 into %state_p1_10
      : !nest.event<"pf_state_p1_r2_s1">
    %pf_state_p2_r2_s1 = nest.dma.prefetch.async %655 into %state_p2_10
      : !nest.event<"pf_state_p2_r2_s1">
    %pf_state_p3_r2_s1 = nest.dma.prefetch.async %656 into %state_p3_10
      : !nest.event<"pf_state_p3_r2_s1">
    %pf_acc_p0_r2_s1 = nest.dma.prefetch.async %657 into %acc_p0_10 : !nest.event<"pf_acc_p0_r2_s1">
    %pf_acc_p1_r2_s1 = nest.dma.prefetch.async %658 into %acc_p1_10 : !nest.event<"pf_acc_p1_r2_s1">
    %pf_acc_p2_r2_s1 = nest.dma.prefetch.async %659 into %acc_p2_10 : !nest.event<"pf_acc_p2_r2_s1">
    %pf_acc_p3_r2_s1 = nest.dma.prefetch.async %660 into %acc_p3_10 : !nest.event<"pf_acc_p3_r2_s1">
    %710 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r2_s1, %append_inrel_r2_s1, %711 =
      nest.dispatch.tasks.async @paged_attention_append_r2_tip0 l1_mode = 1 tasks(%710)
      globals(%648) bindings(%k_new_9, %v_new_9, %append_idx_9)
      ins(%k_new_9, %v_new_9, %append_idx_9) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r2_s1, %pf_v_r2_s1, %pf_aidx_r2_s1)
      : (!nest.event<"append_grid_r2_s1">, !nest.event<"append_inrel_r2_s1">, !nest.event<"">)
    %pf_bidx0_r2_s1 = nest.dma.prefetch.async %662 into %block_idx_p0_9
      : !nest.event<"pf_bidx0_r2_s1">
    %att0_grid_r2_s1, %att0_inrel_r2_s1, %att0_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx0_r2_s1) : (
        !nest.event<"att0_grid_r2_s1">, !nest.event<"att0_inrel_r2_s1">,
        !nest.event<"att0_out_r2_s1">)
    %pf_bidx1_r2_s1 = nest.dma.prefetch.async %663 into %block_idx_p1_9
      : !nest.event<"pf_bidx1_r2_s1">
    %att1_grid_r2_s1, %att1_inrel_r2_s1, %att1_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx1_r2_s1) : (
        !nest.event<"att1_grid_r2_s1">, !nest.event<"att1_inrel_r2_s1">,
        !nest.event<"att1_out_r2_s1">)
    %pf_bidx2_r2_s1 = nest.dma.prefetch.async %664 into %block_idx_p0_9
      depends_on(%att0_inrel_r2_s1) : !nest.event<"pf_bidx2_r2_s1">
    %att2_grid_r2_s1, %att2_inrel_r2_s1, %att2_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx2_r2_s1) : (
        !nest.event<"att2_grid_r2_s1">, !nest.event<"att2_inrel_r2_s1">,
        !nest.event<"att2_out_r2_s1">)
    %pf_bidx3_r2_s1 = nest.dma.prefetch.async %665 into %block_idx_p1_9
      depends_on(%att1_inrel_r2_s1) : !nest.event<"pf_bidx3_r2_s1">
    %att3_grid_r2_s1, %att3_inrel_r2_s1, %att3_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx3_r2_s1) : (
        !nest.event<"att3_grid_r2_s1">, !nest.event<"att3_inrel_r2_s1">,
        !nest.event<"att3_out_r2_s1">)
    %pf_bidx4_r2_s1 = nest.dma.prefetch.async %666 into %block_idx_p0_9
      depends_on(%att2_inrel_r2_s1) : !nest.event<"pf_bidx4_r2_s1">
    %att4_grid_r2_s1, %att4_inrel_r2_s1, %att4_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx4_r2_s1, %att0_out_r2_s1)
      : (
        !nest.event<"att4_grid_r2_s1">, !nest.event<"att4_inrel_r2_s1">,
        !nest.event<"att4_out_r2_s1">)
    %pf_bidx5_r2_s1 = nest.dma.prefetch.async %667 into %block_idx_p1_9
      depends_on(%att3_inrel_r2_s1) : !nest.event<"pf_bidx5_r2_s1">
    %att5_grid_r2_s1, %att5_inrel_r2_s1, %att5_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx5_r2_s1, %att1_out_r2_s1)
      : (
        !nest.event<"att5_grid_r2_s1">, !nest.event<"att5_inrel_r2_s1">,
        !nest.event<"att5_out_r2_s1">)
    %pf_bidx6_r2_s1 = nest.dma.prefetch.async %668 into %block_idx_p0_9
      depends_on(%att4_inrel_r2_s1) : !nest.event<"pf_bidx6_r2_s1">
    %att6_grid_r2_s1, %att6_inrel_r2_s1, %att6_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx6_r2_s1, %att2_out_r2_s1)
      : (
        !nest.event<"att6_grid_r2_s1">, !nest.event<"att6_inrel_r2_s1">,
        !nest.event<"att6_out_r2_s1">)
    %pf_bidx7_r2_s1 = nest.dma.prefetch.async %669 into %block_idx_p1_9
      depends_on(%att5_inrel_r2_s1) : !nest.event<"pf_bidx7_r2_s1">
    %att7_grid_r2_s1, %att7_inrel_r2_s1, %att7_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx7_r2_s1, %att3_out_r2_s1)
      : (
        !nest.event<"att7_grid_r2_s1">, !nest.event<"att7_inrel_r2_s1">,
        !nest.event<"att7_out_r2_s1">)
    %pf_bidx8_r2_s1 = nest.dma.prefetch.async %670 into %block_idx_p0_9
      depends_on(%att6_inrel_r2_s1) : !nest.event<"pf_bidx8_r2_s1">
    %att8_grid_r2_s1, %att8_inrel_r2_s1, %att8_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx8_r2_s1, %att4_out_r2_s1)
      : (
        !nest.event<"att8_grid_r2_s1">, !nest.event<"att8_inrel_r2_s1">,
        !nest.event<"att8_out_r2_s1">)
    %pf_bidx9_r2_s1 = nest.dma.prefetch.async %671 into %block_idx_p1_9
      depends_on(%att7_inrel_r2_s1) : !nest.event<"pf_bidx9_r2_s1">
    %att9_grid_r2_s1, %att9_inrel_r2_s1, %att9_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx9_r2_s1, %att5_out_r2_s1)
      : (
        !nest.event<"att9_grid_r2_s1">, !nest.event<"att9_inrel_r2_s1">,
        !nest.event<"att9_out_r2_s1">)
    %pf_bidx10_r2_s1 = nest.dma.prefetch.async %672 into %block_idx_p0_9
      depends_on(%att8_inrel_r2_s1) : !nest.event<"pf_bidx10_r2_s1">
    %att10_grid_r2_s1, %att10_inrel_r2_s1, %att10_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx10_r2_s1, %att6_out_r2_s1)
      : (
        !nest.event<"att10_grid_r2_s1">, !nest.event<"att10_inrel_r2_s1">,
        !nest.event<"att10_out_r2_s1">)
    %pf_bidx11_r2_s1 = nest.dma.prefetch.async %673 into %block_idx_p1_9
      depends_on(%att9_inrel_r2_s1) : !nest.event<"pf_bidx11_r2_s1">
    %att11_grid_r2_s1, %att11_inrel_r2_s1, %att11_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx11_r2_s1, %att7_out_r2_s1)
      : (
        !nest.event<"att11_grid_r2_s1">, !nest.event<"att11_inrel_r2_s1">,
        !nest.event<"att11_out_r2_s1">)
    %pf_bidx12_r2_s1 = nest.dma.prefetch.async %674 into %block_idx_p0_9
      depends_on(%att10_inrel_r2_s1) : !nest.event<"pf_bidx12_r2_s1">
    %att12_grid_r2_s1, %att12_inrel_r2_s1, %att12_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx12_r2_s1, %att8_out_r2_s1)
      : (
        !nest.event<"att12_grid_r2_s1">, !nest.event<"att12_inrel_r2_s1">,
        !nest.event<"att12_out_r2_s1">)
    %pf_bidx13_r2_s1 = nest.dma.prefetch.async %675 into %block_idx_p1_9
      depends_on(%att11_inrel_r2_s1) : !nest.event<"pf_bidx13_r2_s1">
    %att13_grid_r2_s1, %att13_inrel_r2_s1, %att13_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx13_r2_s1, %att9_out_r2_s1)
      : (
        !nest.event<"att13_grid_r2_s1">, !nest.event<"att13_inrel_r2_s1">,
        !nest.event<"att13_out_r2_s1">)
    %pf_bidx14_r2_s1 = nest.dma.prefetch.async %676 into %block_idx_p0_9
      depends_on(%att12_inrel_r2_s1) : !nest.event<"pf_bidx14_r2_s1">
    %att14_grid_r2_s1, %att14_inrel_r2_s1, %att14_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx14_r2_s1, %att10_out_r2_s1)
      : (
        !nest.event<"att14_grid_r2_s1">, !nest.event<"att14_inrel_r2_s1">,
        !nest.event<"att14_out_r2_s1">)
    %pf_bidx15_r2_s1 = nest.dma.prefetch.async %677 into %block_idx_p1_9
      depends_on(%att13_inrel_r2_s1) : !nest.event<"pf_bidx15_r2_s1">
    %att15_grid_r2_s1, %att15_inrel_r2_s1, %att15_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx15_r2_s1, %att11_out_r2_s1)
      : (
        !nest.event<"att15_grid_r2_s1">, !nest.event<"att15_inrel_r2_s1">,
        !nest.event<"att15_out_r2_s1">)
    %pf_bidx16_r2_s1 = nest.dma.prefetch.async %678 into %block_idx_p0_9
      depends_on(%att14_inrel_r2_s1) : !nest.event<"pf_bidx16_r2_s1">
    %att16_grid_r2_s1, %att16_inrel_r2_s1, %att16_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx16_r2_s1, %att12_out_r2_s1)
      : (
        !nest.event<"att16_grid_r2_s1">, !nest.event<"att16_inrel_r2_s1">,
        !nest.event<"att16_out_r2_s1">)
    %pf_bidx17_r2_s1 = nest.dma.prefetch.async %679 into %block_idx_p1_9
      depends_on(%att15_inrel_r2_s1) : !nest.event<"pf_bidx17_r2_s1">
    %att17_grid_r2_s1, %att17_inrel_r2_s1, %att17_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx17_r2_s1, %att13_out_r2_s1)
      : (
        !nest.event<"att17_grid_r2_s1">, !nest.event<"att17_inrel_r2_s1">,
        !nest.event<"att17_out_r2_s1">)
    %pf_bidx18_r2_s1 = nest.dma.prefetch.async %680 into %block_idx_p0_9
      depends_on(%att16_inrel_r2_s1) : !nest.event<"pf_bidx18_r2_s1">
    %att18_grid_r2_s1, %att18_inrel_r2_s1, %att18_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx18_r2_s1, %att14_out_r2_s1)
      : (
        !nest.event<"att18_grid_r2_s1">, !nest.event<"att18_inrel_r2_s1">,
        !nest.event<"att18_out_r2_s1">)
    %pf_bidx19_r2_s1 = nest.dma.prefetch.async %681 into %block_idx_p1_9
      depends_on(%att17_inrel_r2_s1) : !nest.event<"pf_bidx19_r2_s1">
    %att19_grid_r2_s1, %att19_inrel_r2_s1, %att19_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx19_r2_s1, %att15_out_r2_s1)
      : (
        !nest.event<"att19_grid_r2_s1">, !nest.event<"att19_inrel_r2_s1">,
        !nest.event<"att19_out_r2_s1">)
    %pf_bidx20_r2_s1 = nest.dma.prefetch.async %682 into %block_idx_p0_9
      depends_on(%att18_inrel_r2_s1) : !nest.event<"pf_bidx20_r2_s1">
    %att20_grid_r2_s1, %att20_inrel_r2_s1, %att20_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx20_r2_s1, %att16_out_r2_s1)
      : (
        !nest.event<"att20_grid_r2_s1">, !nest.event<"att20_inrel_r2_s1">,
        !nest.event<"att20_out_r2_s1">)
    %pf_bidx21_r2_s1 = nest.dma.prefetch.async %683 into %block_idx_p1_9
      depends_on(%att19_inrel_r2_s1) : !nest.event<"pf_bidx21_r2_s1">
    %att21_grid_r2_s1, %att21_inrel_r2_s1, %att21_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx21_r2_s1, %att17_out_r2_s1)
      : (
        !nest.event<"att21_grid_r2_s1">, !nest.event<"att21_inrel_r2_s1">,
        !nest.event<"att21_out_r2_s1">)
    %pf_bidx22_r2_s1 = nest.dma.prefetch.async %684 into %block_idx_p0_9
      depends_on(%att20_inrel_r2_s1) : !nest.event<"pf_bidx22_r2_s1">
    %att22_grid_r2_s1, %att22_inrel_r2_s1, %att22_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx22_r2_s1, %att18_out_r2_s1)
      : (
        !nest.event<"att22_grid_r2_s1">, !nest.event<"att22_inrel_r2_s1">,
        !nest.event<"att22_out_r2_s1">)
    %pf_bidx23_r2_s1 = nest.dma.prefetch.async %685 into %block_idx_p1_9
      depends_on(%att21_inrel_r2_s1) : !nest.event<"pf_bidx23_r2_s1">
    %att23_grid_r2_s1, %att23_inrel_r2_s1, %att23_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx23_r2_s1, %att19_out_r2_s1)
      : (
        !nest.event<"att23_grid_r2_s1">, !nest.event<"att23_inrel_r2_s1">,
        !nest.event<"att23_out_r2_s1">)
    %pf_bidx24_r2_s1 = nest.dma.prefetch.async %686 into %block_idx_p0_9
      depends_on(%att22_inrel_r2_s1) : !nest.event<"pf_bidx24_r2_s1">
    %att24_grid_r2_s1, %att24_inrel_r2_s1, %att24_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx24_r2_s1, %att20_out_r2_s1)
      : (
        !nest.event<"att24_grid_r2_s1">, !nest.event<"att24_inrel_r2_s1">,
        !nest.event<"att24_out_r2_s1">)
    %pf_bidx25_r2_s1 = nest.dma.prefetch.async %687 into %block_idx_p1_9
      depends_on(%att23_inrel_r2_s1) : !nest.event<"pf_bidx25_r2_s1">
    %att25_grid_r2_s1, %att25_inrel_r2_s1, %att25_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx25_r2_s1, %att21_out_r2_s1)
      : (
        !nest.event<"att25_grid_r2_s1">, !nest.event<"att25_inrel_r2_s1">,
        !nest.event<"att25_out_r2_s1">)
    %pf_bidx26_r2_s1 = nest.dma.prefetch.async %688 into %block_idx_p0_9
      depends_on(%att24_inrel_r2_s1) : !nest.event<"pf_bidx26_r2_s1">
    %att26_grid_r2_s1, %att26_inrel_r2_s1, %att26_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx26_r2_s1, %att22_out_r2_s1)
      : (
        !nest.event<"att26_grid_r2_s1">, !nest.event<"att26_inrel_r2_s1">,
        !nest.event<"att26_out_r2_s1">)
    %pf_bidx27_r2_s1 = nest.dma.prefetch.async %689 into %block_idx_p1_9
      depends_on(%att25_inrel_r2_s1) : !nest.event<"pf_bidx27_r2_s1">
    %att27_grid_r2_s1, %att27_inrel_r2_s1, %att27_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx27_r2_s1, %att23_out_r2_s1)
      : (
        !nest.event<"att27_grid_r2_s1">, !nest.event<"att27_inrel_r2_s1">,
        !nest.event<"att27_out_r2_s1">)
    %pf_bidx28_r2_s1 = nest.dma.prefetch.async %690 into %block_idx_p0_9
      depends_on(%att26_inrel_r2_s1) : !nest.event<"pf_bidx28_r2_s1">
    %att28_grid_r2_s1, %att28_inrel_r2_s1, %att28_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx28_r2_s1, %att24_out_r2_s1)
      : (
        !nest.event<"att28_grid_r2_s1">, !nest.event<"att28_inrel_r2_s1">,
        !nest.event<"att28_out_r2_s1">)
    %pf_bidx29_r2_s1 = nest.dma.prefetch.async %691 into %block_idx_p1_9
      depends_on(%att27_inrel_r2_s1) : !nest.event<"pf_bidx29_r2_s1">
    %att29_grid_r2_s1, %att29_inrel_r2_s1, %att29_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx29_r2_s1, %att25_out_r2_s1)
      : (
        !nest.event<"att29_grid_r2_s1">, !nest.event<"att29_inrel_r2_s1">,
        !nest.event<"att29_out_r2_s1">)
    %pf_bidx30_r2_s1 = nest.dma.prefetch.async %692 into %block_idx_p0_9
      depends_on(%att28_inrel_r2_s1) : !nest.event<"pf_bidx30_r2_s1">
    %att30_grid_r2_s1, %att30_inrel_r2_s1, %att30_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx30_r2_s1, %att26_out_r2_s1)
      : (
        !nest.event<"att30_grid_r2_s1">, !nest.event<"att30_inrel_r2_s1">,
        !nest.event<"att30_out_r2_s1">)
    %pf_bidx31_r2_s1 = nest.dma.prefetch.async %693 into %block_idx_p1_9
      depends_on(%att29_inrel_r2_s1) : !nest.event<"pf_bidx31_r2_s1">
    %att31_grid_r2_s1, %att31_inrel_r2_s1, %att31_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx31_r2_s1, %att27_out_r2_s1)
      : (
        !nest.event<"att31_grid_r2_s1">, !nest.event<"att31_inrel_r2_s1">,
        !nest.event<"att31_out_r2_s1">)
    %pf_bidx32_r2_s1 = nest.dma.prefetch.async %694 into %block_idx_p0_9
      depends_on(%att30_inrel_r2_s1) : !nest.event<"pf_bidx32_r2_s1">
    %att32_grid_r2_s1, %att32_inrel_r2_s1, %att32_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx32_r2_s1, %att28_out_r2_s1)
      : (
        !nest.event<"att32_grid_r2_s1">, !nest.event<"att32_inrel_r2_s1">,
        !nest.event<"att32_out_r2_s1">)
    %pf_bidx33_r2_s1 = nest.dma.prefetch.async %695 into %block_idx_p1_9
      depends_on(%att31_inrel_r2_s1) : !nest.event<"pf_bidx33_r2_s1">
    %att33_grid_r2_s1, %att33_inrel_r2_s1, %att33_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx33_r2_s1, %att29_out_r2_s1)
      : (
        !nest.event<"att33_grid_r2_s1">, !nest.event<"att33_inrel_r2_s1">,
        !nest.event<"att33_out_r2_s1">)
    %pf_bidx34_r2_s1 = nest.dma.prefetch.async %696 into %block_idx_p0_9
      depends_on(%att32_inrel_r2_s1) : !nest.event<"pf_bidx34_r2_s1">
    %att34_grid_r2_s1, %att34_inrel_r2_s1, %att34_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx34_r2_s1, %att30_out_r2_s1)
      : (
        !nest.event<"att34_grid_r2_s1">, !nest.event<"att34_inrel_r2_s1">,
        !nest.event<"att34_out_r2_s1">)
    %pf_bidx35_r2_s1 = nest.dma.prefetch.async %697 into %block_idx_p1_9
      depends_on(%att33_inrel_r2_s1) : !nest.event<"pf_bidx35_r2_s1">
    %att35_grid_r2_s1, %att35_inrel_r2_s1, %att35_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx35_r2_s1, %att31_out_r2_s1)
      : (
        !nest.event<"att35_grid_r2_s1">, !nest.event<"att35_inrel_r2_s1">,
        !nest.event<"att35_out_r2_s1">)
    %pf_bidx36_r2_s1 = nest.dma.prefetch.async %698 into %block_idx_p0_9
      depends_on(%att34_inrel_r2_s1) : !nest.event<"pf_bidx36_r2_s1">
    %att36_grid_r2_s1, %att36_inrel_r2_s1, %att36_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx36_r2_s1, %att32_out_r2_s1)
      : (
        !nest.event<"att36_grid_r2_s1">, !nest.event<"att36_inrel_r2_s1">,
        !nest.event<"att36_out_r2_s1">)
    %pf_bidx37_r2_s1 = nest.dma.prefetch.async %699 into %block_idx_p1_9
      depends_on(%att35_inrel_r2_s1) : !nest.event<"pf_bidx37_r2_s1">
    %att37_grid_r2_s1, %att37_inrel_r2_s1, %att37_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx37_r2_s1, %att33_out_r2_s1)
      : (
        !nest.event<"att37_grid_r2_s1">, !nest.event<"att37_inrel_r2_s1">,
        !nest.event<"att37_out_r2_s1">)
    %pf_bidx38_r2_s1 = nest.dma.prefetch.async %700 into %block_idx_p0_9
      depends_on(%att36_inrel_r2_s1) : !nest.event<"pf_bidx38_r2_s1">
    %att38_grid_r2_s1, %att38_inrel_r2_s1, %att38_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx38_r2_s1, %att34_out_r2_s1)
      : (
        !nest.event<"att38_grid_r2_s1">, !nest.event<"att38_inrel_r2_s1">,
        !nest.event<"att38_out_r2_s1">)
    %pf_bidx39_r2_s1 = nest.dma.prefetch.async %701 into %block_idx_p1_9
      depends_on(%att37_inrel_r2_s1) : !nest.event<"pf_bidx39_r2_s1">
    %att39_grid_r2_s1, %att39_inrel_r2_s1, %att39_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx39_r2_s1, %att35_out_r2_s1)
      : (
        !nest.event<"att39_grid_r2_s1">, !nest.event<"att39_inrel_r2_s1">,
        !nest.event<"att39_out_r2_s1">)
    %pf_bidx40_r2_s1 = nest.dma.prefetch.async %702 into %block_idx_p0_9
      depends_on(%att38_inrel_r2_s1) : !nest.event<"pf_bidx40_r2_s1">
    %att40_grid_r2_s1, %att40_inrel_r2_s1, %att40_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx40_r2_s1, %att36_out_r2_s1)
      : (
        !nest.event<"att40_grid_r2_s1">, !nest.event<"att40_inrel_r2_s1">,
        !nest.event<"att40_out_r2_s1">)
    %pf_bidx41_r2_s1 = nest.dma.prefetch.async %703 into %block_idx_p1_9
      depends_on(%att39_inrel_r2_s1) : !nest.event<"pf_bidx41_r2_s1">
    %att41_grid_r2_s1, %att41_inrel_r2_s1, %att41_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx41_r2_s1, %att37_out_r2_s1)
      : (
        !nest.event<"att41_grid_r2_s1">, !nest.event<"att41_inrel_r2_s1">,
        !nest.event<"att41_out_r2_s1">)
    %pf_bidx42_r2_s1 = nest.dma.prefetch.async %704 into %block_idx_p0_9
      depends_on(%att40_inrel_r2_s1) : !nest.event<"pf_bidx42_r2_s1">
    %att42_grid_r2_s1, %att42_inrel_r2_s1, %att42_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx42_r2_s1, %att38_out_r2_s1)
      : (
        !nest.event<"att42_grid_r2_s1">, !nest.event<"att42_inrel_r2_s1">,
        !nest.event<"att42_out_r2_s1">)
    %pf_bidx43_r2_s1 = nest.dma.prefetch.async %705 into %block_idx_p1_9
      depends_on(%att41_inrel_r2_s1) : !nest.event<"pf_bidx43_r2_s1">
    %att43_grid_r2_s1, %att43_inrel_r2_s1, %att43_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx43_r2_s1, %att39_out_r2_s1)
      : (
        !nest.event<"att43_grid_r2_s1">, !nest.event<"att43_inrel_r2_s1">,
        !nest.event<"att43_out_r2_s1">)
    %pf_bidx44_r2_s1 = nest.dma.prefetch.async %706 into %block_idx_p0_9
      depends_on(%att42_inrel_r2_s1) : !nest.event<"pf_bidx44_r2_s1">
    %att44_grid_r2_s1, %att44_inrel_r2_s1, %att44_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_bidx44_r2_s1, %att40_out_r2_s1)
      : (
        !nest.event<"att44_grid_r2_s1">, !nest.event<"att44_inrel_r2_s1">,
        !nest.event<"att44_out_r2_s1">)
    %pf_bidx45_r2_s1 = nest.dma.prefetch.async %707 into %block_idx_p1_9
      depends_on(%att43_inrel_r2_s1) : !nest.event<"pf_bidx45_r2_s1">
    %att45_grid_r2_s1, %att45_inrel_r2_s1, %att45_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p1_10, %acc_p1_10) outs(%state_p1_10, %acc_p1_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p1_r2_s1, %pf_acc_p1_r2_s1, %pf_bidx45_r2_s1, %att41_out_r2_s1)
      : (
        !nest.event<"att45_grid_r2_s1">, !nest.event<"att45_inrel_r2_s1">,
        !nest.event<"att45_out_r2_s1">)
    %pf_bidx46_r2_s1 = nest.dma.prefetch.async %708 into %block_idx_p0_9
      depends_on(%att44_inrel_r2_s1) : !nest.event<"pf_bidx46_r2_s1">
    %att46_grid_r2_s1, %att46_inrel_r2_s1, %att46_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10)
      ins(%block_idx_p0_9, %q_l2_24, %state_p2_10, %acc_p2_10) outs(%state_p2_10, %acc_p2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p2_r2_s1, %pf_acc_p2_r2_s1, %pf_bidx46_r2_s1, %att42_out_r2_s1)
      : (
        !nest.event<"att46_grid_r2_s1">, !nest.event<"att46_inrel_r2_s1">,
        !nest.event<"att46_out_r2_s1">)
    %pf_bidx47_r2_s1 = nest.dma.prefetch.async %709 into %block_idx_p1_9
      depends_on(%att45_inrel_r2_s1) : !nest.event<"pf_bidx47_r2_s1">
    %att47_grid_r2_s1, %att47_inrel_r2_s1, %att47_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10)
      ins(%block_idx_p1_9, %q_l2_24, %state_p3_10, %acc_p3_10) outs(%state_p3_10, %acc_p3_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p3_r2_s1, %pf_acc_p3_r2_s1, %pf_bidx47_r2_s1, %att43_out_r2_s1)
      : (
        !nest.event<"att47_grid_r2_s1">, !nest.event<"att47_inrel_r2_s1">,
        !nest.event<"att47_out_r2_s1">)
    %att48_grid_r2_s1, %att48_inrel_r2_s1, %att48_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_t1_final_r2 l1_mode = 1 tasks(%710) globals(%648)
      bindings(%append_idx_9, %q_l2_24, %state_p0_10, %acc_p0_10)
      ins(%append_idx_9, %q_l2_24, %state_p0_10, %acc_p0_10) outs(%state_p0_10, %acc_p0_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s1, %pf_state_p0_r2_s1, %pf_acc_p0_r2_s1, %pf_aidx_r2_s1, %append_grid_r2_s1,
        %att44_out_r2_s1)
      : (
        !nest.event<"att48_grid_r2_s1">, !nest.event<"att48_inrel_r2_s1">,
        !nest.event<"att48_out_r2_s1">)
    %merge_grid_r2_s1, %merge_inrel_r2_s1, %merge_out_r2_s1 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%710) globals()
      bindings(
        %state_p0_10, %state_p1_10, %state_p2_10, %state_p3_10, %acc_p0_10, %acc_p1_10,
        %acc_p2_10, %acc_p3_10, %out_l2_10)
      ins(
        %state_p0_10, %state_p1_10, %state_p2_10, %state_p3_10, %acc_p0_10, %acc_p1_10,
        %acc_p2_10, %acc_p3_10)
      outs(%out_l2_10)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att48_out_r2_s1, %att45_out_r2_s1, %att46_out_r2_s1, %att47_out_r2_s1) : (
        !nest.event<"merge_grid_r2_s1">, !nest.event<"merge_inrel_r2_s1">,
        !nest.event<"merge_out_r2_s1">)
    %out_store_r2_s1 = nest.dma.store.async %out_l2_10 into %661 depends_on(%merge_out_r2_s1)
      : !nest.event<"out_store_r2_s1">
    nest.await %append_grid_r2_s1, %att48_grid_r2_s1, %merge_grid_r2_s1, %out_store_r2_s1
    nest.release %block_idx_p0_9 depends_on(
      %pf_bidx0_r2_s1, %pf_bidx2_r2_s1, %pf_bidx4_r2_s1, %pf_bidx6_r2_s1, %pf_bidx8_r2_s1,
      %pf_bidx10_r2_s1, %pf_bidx12_r2_s1, %pf_bidx14_r2_s1, %pf_bidx16_r2_s1, %pf_bidx18_r2_s1,
      %pf_bidx20_r2_s1, %pf_bidx22_r2_s1, %pf_bidx24_r2_s1, %pf_bidx26_r2_s1, %pf_bidx28_r2_s1,
      %pf_bidx30_r2_s1, %pf_bidx32_r2_s1, %pf_bidx34_r2_s1, %pf_bidx36_r2_s1, %pf_bidx38_r2_s1,
      %pf_bidx40_r2_s1, %pf_bidx42_r2_s1, %pf_bidx44_r2_s1, %pf_bidx46_r2_s1, %att0_inrel_r2_s1,
      %att2_inrel_r2_s1, %att4_inrel_r2_s1, %att6_inrel_r2_s1, %att8_inrel_r2_s1,
      %att10_inrel_r2_s1, %att12_inrel_r2_s1, %att14_inrel_r2_s1, %att16_inrel_r2_s1,
      %att18_inrel_r2_s1, %att20_inrel_r2_s1, %att22_inrel_r2_s1, %att24_inrel_r2_s1,
      %att26_inrel_r2_s1, %att28_inrel_r2_s1, %att30_inrel_r2_s1, %att32_inrel_r2_s1,
      %att34_inrel_r2_s1, %att36_inrel_r2_s1, %att38_inrel_r2_s1, %att40_inrel_r2_s1,
      %att42_inrel_r2_s1, %att44_inrel_r2_s1, %att46_inrel_r2_s1)
    nest.release %block_idx_p1_9 depends_on(
      %pf_bidx1_r2_s1, %pf_bidx3_r2_s1, %pf_bidx5_r2_s1, %pf_bidx7_r2_s1, %pf_bidx9_r2_s1,
      %pf_bidx11_r2_s1, %pf_bidx13_r2_s1, %pf_bidx15_r2_s1, %pf_bidx17_r2_s1, %pf_bidx19_r2_s1,
      %pf_bidx21_r2_s1, %pf_bidx23_r2_s1, %pf_bidx25_r2_s1, %pf_bidx27_r2_s1, %pf_bidx29_r2_s1,
      %pf_bidx31_r2_s1, %pf_bidx33_r2_s1, %pf_bidx35_r2_s1, %pf_bidx37_r2_s1, %pf_bidx39_r2_s1,
      %pf_bidx41_r2_s1, %pf_bidx43_r2_s1, %pf_bidx45_r2_s1, %pf_bidx47_r2_s1, %att1_inrel_r2_s1,
      %att3_inrel_r2_s1, %att5_inrel_r2_s1, %att7_inrel_r2_s1, %att9_inrel_r2_s1,
      %att11_inrel_r2_s1, %att13_inrel_r2_s1, %att15_inrel_r2_s1, %att17_inrel_r2_s1,
      %att19_inrel_r2_s1, %att21_inrel_r2_s1, %att23_inrel_r2_s1, %att25_inrel_r2_s1,
      %att27_inrel_r2_s1, %att29_inrel_r2_s1, %att31_inrel_r2_s1, %att33_inrel_r2_s1,
      %att35_inrel_r2_s1, %att37_inrel_r2_s1, %att39_inrel_r2_s1, %att41_inrel_r2_s1,
      %att43_inrel_r2_s1, %att45_inrel_r2_s1, %att47_inrel_r2_s1)
    nest.release %k_new_9 depends_on(%pf_k_r2_s1, %append_inrel_r2_s1)
    nest.release %v_new_9 depends_on(%pf_v_r2_s1, %append_inrel_r2_s1)
    nest.release %append_idx_9 depends_on(%pf_aidx_r2_s1, %append_inrel_r2_s1, %att48_inrel_r2_s1)
    nest.release %q_l2_24 depends_on(
      %pf_q_r2_s1, %att0_inrel_r2_s1, %att1_inrel_r2_s1, %att2_inrel_r2_s1, %att3_inrel_r2_s1,
      %att4_inrel_r2_s1, %att5_inrel_r2_s1, %att6_inrel_r2_s1, %att7_inrel_r2_s1,
      %att8_inrel_r2_s1, %att9_inrel_r2_s1, %att10_inrel_r2_s1, %att11_inrel_r2_s1,
      %att12_inrel_r2_s1, %att13_inrel_r2_s1, %att14_inrel_r2_s1, %att15_inrel_r2_s1,
      %att16_inrel_r2_s1, %att17_inrel_r2_s1, %att18_inrel_r2_s1, %att19_inrel_r2_s1,
      %att20_inrel_r2_s1, %att21_inrel_r2_s1, %att22_inrel_r2_s1, %att23_inrel_r2_s1,
      %att24_inrel_r2_s1, %att25_inrel_r2_s1, %att26_inrel_r2_s1, %att27_inrel_r2_s1,
      %att28_inrel_r2_s1, %att29_inrel_r2_s1, %att30_inrel_r2_s1, %att31_inrel_r2_s1,
      %att32_inrel_r2_s1, %att33_inrel_r2_s1, %att34_inrel_r2_s1, %att35_inrel_r2_s1,
      %att36_inrel_r2_s1, %att37_inrel_r2_s1, %att38_inrel_r2_s1, %att39_inrel_r2_s1,
      %att40_inrel_r2_s1, %att41_inrel_r2_s1, %att42_inrel_r2_s1, %att43_inrel_r2_s1,
      %att44_inrel_r2_s1, %att45_inrel_r2_s1, %att46_inrel_r2_s1, %att47_inrel_r2_s1,
      %att48_inrel_r2_s1)
    nest.release %state_p0_10 depends_on(
      %pf_state_p0_r2_s1, %att0_inrel_r2_s1, %att4_inrel_r2_s1, %att8_inrel_r2_s1,
      %att12_inrel_r2_s1, %att16_inrel_r2_s1, %att20_inrel_r2_s1, %att24_inrel_r2_s1,
      %att28_inrel_r2_s1, %att32_inrel_r2_s1, %att36_inrel_r2_s1, %att40_inrel_r2_s1,
      %att44_inrel_r2_s1, %att48_inrel_r2_s1, %att0_out_r2_s1, %att4_out_r2_s1, %att8_out_r2_s1,
      %att12_out_r2_s1, %att16_out_r2_s1, %att20_out_r2_s1, %att24_out_r2_s1, %att28_out_r2_s1,
      %att32_out_r2_s1, %att36_out_r2_s1, %att40_out_r2_s1, %att44_out_r2_s1, %att48_out_r2_s1,
      %merge_inrel_r2_s1)
    nest.release %acc_p0_10 depends_on(
      %pf_acc_p0_r2_s1, %att0_inrel_r2_s1, %att4_inrel_r2_s1, %att8_inrel_r2_s1,
      %att12_inrel_r2_s1, %att16_inrel_r2_s1, %att20_inrel_r2_s1, %att24_inrel_r2_s1,
      %att28_inrel_r2_s1, %att32_inrel_r2_s1, %att36_inrel_r2_s1, %att40_inrel_r2_s1,
      %att44_inrel_r2_s1, %att48_inrel_r2_s1, %att0_out_r2_s1, %att4_out_r2_s1, %att8_out_r2_s1,
      %att12_out_r2_s1, %att16_out_r2_s1, %att20_out_r2_s1, %att24_out_r2_s1, %att28_out_r2_s1,
      %att32_out_r2_s1, %att36_out_r2_s1, %att40_out_r2_s1, %att44_out_r2_s1, %att48_out_r2_s1,
      %merge_inrel_r2_s1)
    nest.release %state_p1_10 depends_on(
      %pf_state_p1_r2_s1, %att1_inrel_r2_s1, %att5_inrel_r2_s1, %att9_inrel_r2_s1,
      %att13_inrel_r2_s1, %att17_inrel_r2_s1, %att21_inrel_r2_s1, %att25_inrel_r2_s1,
      %att29_inrel_r2_s1, %att33_inrel_r2_s1, %att37_inrel_r2_s1, %att41_inrel_r2_s1,
      %att45_inrel_r2_s1, %att1_out_r2_s1, %att5_out_r2_s1, %att9_out_r2_s1, %att13_out_r2_s1,
      %att17_out_r2_s1, %att21_out_r2_s1, %att25_out_r2_s1, %att29_out_r2_s1, %att33_out_r2_s1,
      %att37_out_r2_s1, %att41_out_r2_s1, %att45_out_r2_s1, %merge_inrel_r2_s1)
    nest.release %acc_p1_10 depends_on(
      %pf_acc_p1_r2_s1, %att1_inrel_r2_s1, %att5_inrel_r2_s1, %att9_inrel_r2_s1,
      %att13_inrel_r2_s1, %att17_inrel_r2_s1, %att21_inrel_r2_s1, %att25_inrel_r2_s1,
      %att29_inrel_r2_s1, %att33_inrel_r2_s1, %att37_inrel_r2_s1, %att41_inrel_r2_s1,
      %att45_inrel_r2_s1, %att1_out_r2_s1, %att5_out_r2_s1, %att9_out_r2_s1, %att13_out_r2_s1,
      %att17_out_r2_s1, %att21_out_r2_s1, %att25_out_r2_s1, %att29_out_r2_s1, %att33_out_r2_s1,
      %att37_out_r2_s1, %att41_out_r2_s1, %att45_out_r2_s1, %merge_inrel_r2_s1)
    nest.release %state_p2_10 depends_on(
      %pf_state_p2_r2_s1, %att2_inrel_r2_s1, %att6_inrel_r2_s1, %att10_inrel_r2_s1,
      %att14_inrel_r2_s1, %att18_inrel_r2_s1, %att22_inrel_r2_s1, %att26_inrel_r2_s1,
      %att30_inrel_r2_s1, %att34_inrel_r2_s1, %att38_inrel_r2_s1, %att42_inrel_r2_s1,
      %att46_inrel_r2_s1, %att2_out_r2_s1, %att6_out_r2_s1, %att10_out_r2_s1, %att14_out_r2_s1,
      %att18_out_r2_s1, %att22_out_r2_s1, %att26_out_r2_s1, %att30_out_r2_s1, %att34_out_r2_s1,
      %att38_out_r2_s1, %att42_out_r2_s1, %att46_out_r2_s1, %merge_inrel_r2_s1)
    nest.release %acc_p2_10 depends_on(
      %pf_acc_p2_r2_s1, %att2_inrel_r2_s1, %att6_inrel_r2_s1, %att10_inrel_r2_s1,
      %att14_inrel_r2_s1, %att18_inrel_r2_s1, %att22_inrel_r2_s1, %att26_inrel_r2_s1,
      %att30_inrel_r2_s1, %att34_inrel_r2_s1, %att38_inrel_r2_s1, %att42_inrel_r2_s1,
      %att46_inrel_r2_s1, %att2_out_r2_s1, %att6_out_r2_s1, %att10_out_r2_s1, %att14_out_r2_s1,
      %att18_out_r2_s1, %att22_out_r2_s1, %att26_out_r2_s1, %att30_out_r2_s1, %att34_out_r2_s1,
      %att38_out_r2_s1, %att42_out_r2_s1, %att46_out_r2_s1, %merge_inrel_r2_s1)
    nest.release %state_p3_10 depends_on(
      %pf_state_p3_r2_s1, %att3_inrel_r2_s1, %att7_inrel_r2_s1, %att11_inrel_r2_s1,
      %att15_inrel_r2_s1, %att19_inrel_r2_s1, %att23_inrel_r2_s1, %att27_inrel_r2_s1,
      %att31_inrel_r2_s1, %att35_inrel_r2_s1, %att39_inrel_r2_s1, %att43_inrel_r2_s1,
      %att47_inrel_r2_s1, %att3_out_r2_s1, %att7_out_r2_s1, %att11_out_r2_s1, %att15_out_r2_s1,
      %att19_out_r2_s1, %att23_out_r2_s1, %att27_out_r2_s1, %att31_out_r2_s1, %att35_out_r2_s1,
      %att39_out_r2_s1, %att43_out_r2_s1, %att47_out_r2_s1, %merge_inrel_r2_s1)
    nest.release %acc_p3_10 depends_on(
      %pf_acc_p3_r2_s1, %att3_inrel_r2_s1, %att7_inrel_r2_s1, %att11_inrel_r2_s1,
      %att15_inrel_r2_s1, %att19_inrel_r2_s1, %att23_inrel_r2_s1, %att27_inrel_r2_s1,
      %att31_inrel_r2_s1, %att35_inrel_r2_s1, %att39_inrel_r2_s1, %att43_inrel_r2_s1,
      %att47_inrel_r2_s1, %att3_out_r2_s1, %att7_out_r2_s1, %att11_out_r2_s1, %att15_out_r2_s1,
      %att19_out_r2_s1, %att23_out_r2_s1, %att27_out_r2_s1, %att31_out_r2_s1, %att35_out_r2_s1,
      %att39_out_r2_s1, %att43_out_r2_s1, %att47_out_r2_s1, %merge_inrel_r2_s1)
    nest.release %out_l2_10 depends_on(%merge_out_r2_s1, %out_store_r2_s1)
    nest.return
  }
  nest.context @step_r2_s2(
    %POOL_10: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_10: !nest.global_memref<147xi32>,
    %APPEND_IDS_10: !nest.global_memref<12xi32>, %Q_IN_10: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_10: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_10: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_10: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_10: !nest.global_memref<3x4x4x4x4x64xf32>,
    %OUT_10: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 204, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_10 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_10 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_10 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_25 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_11 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_11 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_11 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_11 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_11 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_11 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_11 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_11 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_10 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_10 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_11 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %712 = nest.subview %POOL_10 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %713 = nest.subview %K_NEW_10 offsets = [2, 2, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %714 = nest.subview %V_NEW_10 offsets = [2, 2, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %715 = nest.subview %APPEND_IDS_10 offsets = [10] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %716 = nest.subview %Q_IN_10 offsets = [2, 2, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %717 = nest.subview %S_INIT_10 offsets = [2, 2, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %718 = nest.subview %S_INIT_10 offsets = [2, 2, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %719 = nest.subview %S_INIT_10 offsets = [2, 2, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %720 = nest.subview %S_INIT_10 offsets = [2, 2, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %721 = nest.subview %O_INIT_10 offsets = [2, 2, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %722 = nest.subview %O_INIT_10 offsets = [2, 2, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %723 = nest.subview %O_INIT_10 offsets = [2, 2, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %724 = nest.subview %O_INIT_10 offsets = [2, 2, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %725 = nest.subview %OUT_10 offsets = [2, 2, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %726 = nest.subview %BLOCK_TABLE_10 offsets = [98] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %727 = nest.subview %BLOCK_TABLE_10 offsets = [99] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %728 = nest.subview %BLOCK_TABLE_10 offsets = [100] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %729 = nest.subview %BLOCK_TABLE_10 offsets = [101] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %730 = nest.subview %BLOCK_TABLE_10 offsets = [102] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %731 = nest.subview %BLOCK_TABLE_10 offsets = [103] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %732 = nest.subview %BLOCK_TABLE_10 offsets = [104] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %733 = nest.subview %BLOCK_TABLE_10 offsets = [105] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %734 = nest.subview %BLOCK_TABLE_10 offsets = [106] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %735 = nest.subview %BLOCK_TABLE_10 offsets = [107] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %736 = nest.subview %BLOCK_TABLE_10 offsets = [108] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %737 = nest.subview %BLOCK_TABLE_10 offsets = [109] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %738 = nest.subview %BLOCK_TABLE_10 offsets = [110] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %739 = nest.subview %BLOCK_TABLE_10 offsets = [111] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %740 = nest.subview %BLOCK_TABLE_10 offsets = [112] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %741 = nest.subview %BLOCK_TABLE_10 offsets = [113] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %742 = nest.subview %BLOCK_TABLE_10 offsets = [114] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %743 = nest.subview %BLOCK_TABLE_10 offsets = [115] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %744 = nest.subview %BLOCK_TABLE_10 offsets = [116] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %745 = nest.subview %BLOCK_TABLE_10 offsets = [117] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %746 = nest.subview %BLOCK_TABLE_10 offsets = [118] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %747 = nest.subview %BLOCK_TABLE_10 offsets = [119] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %748 = nest.subview %BLOCK_TABLE_10 offsets = [120] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %749 = nest.subview %BLOCK_TABLE_10 offsets = [121] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %750 = nest.subview %BLOCK_TABLE_10 offsets = [122] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %751 = nest.subview %BLOCK_TABLE_10 offsets = [123] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %752 = nest.subview %BLOCK_TABLE_10 offsets = [124] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %753 = nest.subview %BLOCK_TABLE_10 offsets = [125] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %754 = nest.subview %BLOCK_TABLE_10 offsets = [126] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %755 = nest.subview %BLOCK_TABLE_10 offsets = [127] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %756 = nest.subview %BLOCK_TABLE_10 offsets = [128] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %757 = nest.subview %BLOCK_TABLE_10 offsets = [129] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %758 = nest.subview %BLOCK_TABLE_10 offsets = [130] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %759 = nest.subview %BLOCK_TABLE_10 offsets = [131] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %760 = nest.subview %BLOCK_TABLE_10 offsets = [132] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %761 = nest.subview %BLOCK_TABLE_10 offsets = [133] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %762 = nest.subview %BLOCK_TABLE_10 offsets = [134] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %763 = nest.subview %BLOCK_TABLE_10 offsets = [135] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %764 = nest.subview %BLOCK_TABLE_10 offsets = [136] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %765 = nest.subview %BLOCK_TABLE_10 offsets = [137] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %766 = nest.subview %BLOCK_TABLE_10 offsets = [138] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %767 = nest.subview %BLOCK_TABLE_10 offsets = [139] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %768 = nest.subview %BLOCK_TABLE_10 offsets = [140] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %769 = nest.subview %BLOCK_TABLE_10 offsets = [141] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %770 = nest.subview %BLOCK_TABLE_10 offsets = [142] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %771 = nest.subview %BLOCK_TABLE_10 offsets = [143] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %772 = nest.subview %BLOCK_TABLE_10 offsets = [144] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %773 = nest.subview %BLOCK_TABLE_10 offsets = [145] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r2_s2 = nest.dma.prefetch.async %713 into %k_new_10 : !nest.event<"pf_k_r2_s2">
    %pf_v_r2_s2 = nest.dma.prefetch.async %714 into %v_new_10 : !nest.event<"pf_v_r2_s2">
    %pf_aidx_r2_s2 = nest.dma.prefetch.async %715 into %append_idx_10 : !nest.event<"pf_aidx_r2_s2">
    %pf_q_r2_s2 = nest.dma.prefetch.async %716 into %q_l2_25 : !nest.event<"pf_q_r2_s2">
    %pf_state_p0_r2_s2 = nest.dma.prefetch.async %717 into %state_p0_11
      : !nest.event<"pf_state_p0_r2_s2">
    %pf_state_p1_r2_s2 = nest.dma.prefetch.async %718 into %state_p1_11
      : !nest.event<"pf_state_p1_r2_s2">
    %pf_state_p2_r2_s2 = nest.dma.prefetch.async %719 into %state_p2_11
      : !nest.event<"pf_state_p2_r2_s2">
    %pf_state_p3_r2_s2 = nest.dma.prefetch.async %720 into %state_p3_11
      : !nest.event<"pf_state_p3_r2_s2">
    %pf_acc_p0_r2_s2 = nest.dma.prefetch.async %721 into %acc_p0_11 : !nest.event<"pf_acc_p0_r2_s2">
    %pf_acc_p1_r2_s2 = nest.dma.prefetch.async %722 into %acc_p1_11 : !nest.event<"pf_acc_p1_r2_s2">
    %pf_acc_p2_r2_s2 = nest.dma.prefetch.async %723 into %acc_p2_11 : !nest.event<"pf_acc_p2_r2_s2">
    %pf_acc_p3_r2_s2 = nest.dma.prefetch.async %724 into %acc_p3_11 : !nest.event<"pf_acc_p3_r2_s2">
    %774 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r2_s2, %append_inrel_r2_s2, %775 =
      nest.dispatch.tasks.async @paged_attention_append_r2_tip1 l1_mode = 1 tasks(%774)
      globals(%712) bindings(%k_new_10, %v_new_10, %append_idx_10)
      ins(%k_new_10, %v_new_10, %append_idx_10) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r2_s2, %pf_v_r2_s2, %pf_aidx_r2_s2)
      : (!nest.event<"append_grid_r2_s2">, !nest.event<"append_inrel_r2_s2">, !nest.event<"">)
    %pf_bidx0_r2_s2 = nest.dma.prefetch.async %726 into %block_idx_p0_10
      : !nest.event<"pf_bidx0_r2_s2">
    %att0_grid_r2_s2, %att0_inrel_r2_s2, %att0_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx0_r2_s2) : (
        !nest.event<"att0_grid_r2_s2">, !nest.event<"att0_inrel_r2_s2">,
        !nest.event<"att0_out_r2_s2">)
    %pf_bidx1_r2_s2 = nest.dma.prefetch.async %727 into %block_idx_p1_10
      : !nest.event<"pf_bidx1_r2_s2">
    %att1_grid_r2_s2, %att1_inrel_r2_s2, %att1_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx1_r2_s2) : (
        !nest.event<"att1_grid_r2_s2">, !nest.event<"att1_inrel_r2_s2">,
        !nest.event<"att1_out_r2_s2">)
    %pf_bidx2_r2_s2 = nest.dma.prefetch.async %728 into %block_idx_p0_10
      depends_on(%att0_inrel_r2_s2) : !nest.event<"pf_bidx2_r2_s2">
    %att2_grid_r2_s2, %att2_inrel_r2_s2, %att2_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx2_r2_s2) : (
        !nest.event<"att2_grid_r2_s2">, !nest.event<"att2_inrel_r2_s2">,
        !nest.event<"att2_out_r2_s2">)
    %pf_bidx3_r2_s2 = nest.dma.prefetch.async %729 into %block_idx_p1_10
      depends_on(%att1_inrel_r2_s2) : !nest.event<"pf_bidx3_r2_s2">
    %att3_grid_r2_s2, %att3_inrel_r2_s2, %att3_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx3_r2_s2) : (
        !nest.event<"att3_grid_r2_s2">, !nest.event<"att3_inrel_r2_s2">,
        !nest.event<"att3_out_r2_s2">)
    %pf_bidx4_r2_s2 = nest.dma.prefetch.async %730 into %block_idx_p0_10
      depends_on(%att2_inrel_r2_s2) : !nest.event<"pf_bidx4_r2_s2">
    %att4_grid_r2_s2, %att4_inrel_r2_s2, %att4_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx4_r2_s2, %att0_out_r2_s2)
      : (
        !nest.event<"att4_grid_r2_s2">, !nest.event<"att4_inrel_r2_s2">,
        !nest.event<"att4_out_r2_s2">)
    %pf_bidx5_r2_s2 = nest.dma.prefetch.async %731 into %block_idx_p1_10
      depends_on(%att3_inrel_r2_s2) : !nest.event<"pf_bidx5_r2_s2">
    %att5_grid_r2_s2, %att5_inrel_r2_s2, %att5_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx5_r2_s2, %att1_out_r2_s2)
      : (
        !nest.event<"att5_grid_r2_s2">, !nest.event<"att5_inrel_r2_s2">,
        !nest.event<"att5_out_r2_s2">)
    %pf_bidx6_r2_s2 = nest.dma.prefetch.async %732 into %block_idx_p0_10
      depends_on(%att4_inrel_r2_s2) : !nest.event<"pf_bidx6_r2_s2">
    %att6_grid_r2_s2, %att6_inrel_r2_s2, %att6_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx6_r2_s2, %att2_out_r2_s2)
      : (
        !nest.event<"att6_grid_r2_s2">, !nest.event<"att6_inrel_r2_s2">,
        !nest.event<"att6_out_r2_s2">)
    %pf_bidx7_r2_s2 = nest.dma.prefetch.async %733 into %block_idx_p1_10
      depends_on(%att5_inrel_r2_s2) : !nest.event<"pf_bidx7_r2_s2">
    %att7_grid_r2_s2, %att7_inrel_r2_s2, %att7_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx7_r2_s2, %att3_out_r2_s2)
      : (
        !nest.event<"att7_grid_r2_s2">, !nest.event<"att7_inrel_r2_s2">,
        !nest.event<"att7_out_r2_s2">)
    %pf_bidx8_r2_s2 = nest.dma.prefetch.async %734 into %block_idx_p0_10
      depends_on(%att6_inrel_r2_s2) : !nest.event<"pf_bidx8_r2_s2">
    %att8_grid_r2_s2, %att8_inrel_r2_s2, %att8_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx8_r2_s2, %att4_out_r2_s2)
      : (
        !nest.event<"att8_grid_r2_s2">, !nest.event<"att8_inrel_r2_s2">,
        !nest.event<"att8_out_r2_s2">)
    %pf_bidx9_r2_s2 = nest.dma.prefetch.async %735 into %block_idx_p1_10
      depends_on(%att7_inrel_r2_s2) : !nest.event<"pf_bidx9_r2_s2">
    %att9_grid_r2_s2, %att9_inrel_r2_s2, %att9_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx9_r2_s2, %att5_out_r2_s2)
      : (
        !nest.event<"att9_grid_r2_s2">, !nest.event<"att9_inrel_r2_s2">,
        !nest.event<"att9_out_r2_s2">)
    %pf_bidx10_r2_s2 = nest.dma.prefetch.async %736 into %block_idx_p0_10
      depends_on(%att8_inrel_r2_s2) : !nest.event<"pf_bidx10_r2_s2">
    %att10_grid_r2_s2, %att10_inrel_r2_s2, %att10_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx10_r2_s2, %att6_out_r2_s2)
      : (
        !nest.event<"att10_grid_r2_s2">, !nest.event<"att10_inrel_r2_s2">,
        !nest.event<"att10_out_r2_s2">)
    %pf_bidx11_r2_s2 = nest.dma.prefetch.async %737 into %block_idx_p1_10
      depends_on(%att9_inrel_r2_s2) : !nest.event<"pf_bidx11_r2_s2">
    %att11_grid_r2_s2, %att11_inrel_r2_s2, %att11_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx11_r2_s2, %att7_out_r2_s2)
      : (
        !nest.event<"att11_grid_r2_s2">, !nest.event<"att11_inrel_r2_s2">,
        !nest.event<"att11_out_r2_s2">)
    %pf_bidx12_r2_s2 = nest.dma.prefetch.async %738 into %block_idx_p0_10
      depends_on(%att10_inrel_r2_s2) : !nest.event<"pf_bidx12_r2_s2">
    %att12_grid_r2_s2, %att12_inrel_r2_s2, %att12_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx12_r2_s2, %att8_out_r2_s2)
      : (
        !nest.event<"att12_grid_r2_s2">, !nest.event<"att12_inrel_r2_s2">,
        !nest.event<"att12_out_r2_s2">)
    %pf_bidx13_r2_s2 = nest.dma.prefetch.async %739 into %block_idx_p1_10
      depends_on(%att11_inrel_r2_s2) : !nest.event<"pf_bidx13_r2_s2">
    %att13_grid_r2_s2, %att13_inrel_r2_s2, %att13_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx13_r2_s2, %att9_out_r2_s2)
      : (
        !nest.event<"att13_grid_r2_s2">, !nest.event<"att13_inrel_r2_s2">,
        !nest.event<"att13_out_r2_s2">)
    %pf_bidx14_r2_s2 = nest.dma.prefetch.async %740 into %block_idx_p0_10
      depends_on(%att12_inrel_r2_s2) : !nest.event<"pf_bidx14_r2_s2">
    %att14_grid_r2_s2, %att14_inrel_r2_s2, %att14_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx14_r2_s2, %att10_out_r2_s2)
      : (
        !nest.event<"att14_grid_r2_s2">, !nest.event<"att14_inrel_r2_s2">,
        !nest.event<"att14_out_r2_s2">)
    %pf_bidx15_r2_s2 = nest.dma.prefetch.async %741 into %block_idx_p1_10
      depends_on(%att13_inrel_r2_s2) : !nest.event<"pf_bidx15_r2_s2">
    %att15_grid_r2_s2, %att15_inrel_r2_s2, %att15_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx15_r2_s2, %att11_out_r2_s2)
      : (
        !nest.event<"att15_grid_r2_s2">, !nest.event<"att15_inrel_r2_s2">,
        !nest.event<"att15_out_r2_s2">)
    %pf_bidx16_r2_s2 = nest.dma.prefetch.async %742 into %block_idx_p0_10
      depends_on(%att14_inrel_r2_s2) : !nest.event<"pf_bidx16_r2_s2">
    %att16_grid_r2_s2, %att16_inrel_r2_s2, %att16_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx16_r2_s2, %att12_out_r2_s2)
      : (
        !nest.event<"att16_grid_r2_s2">, !nest.event<"att16_inrel_r2_s2">,
        !nest.event<"att16_out_r2_s2">)
    %pf_bidx17_r2_s2 = nest.dma.prefetch.async %743 into %block_idx_p1_10
      depends_on(%att15_inrel_r2_s2) : !nest.event<"pf_bidx17_r2_s2">
    %att17_grid_r2_s2, %att17_inrel_r2_s2, %att17_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx17_r2_s2, %att13_out_r2_s2)
      : (
        !nest.event<"att17_grid_r2_s2">, !nest.event<"att17_inrel_r2_s2">,
        !nest.event<"att17_out_r2_s2">)
    %pf_bidx18_r2_s2 = nest.dma.prefetch.async %744 into %block_idx_p0_10
      depends_on(%att16_inrel_r2_s2) : !nest.event<"pf_bidx18_r2_s2">
    %att18_grid_r2_s2, %att18_inrel_r2_s2, %att18_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx18_r2_s2, %att14_out_r2_s2)
      : (
        !nest.event<"att18_grid_r2_s2">, !nest.event<"att18_inrel_r2_s2">,
        !nest.event<"att18_out_r2_s2">)
    %pf_bidx19_r2_s2 = nest.dma.prefetch.async %745 into %block_idx_p1_10
      depends_on(%att17_inrel_r2_s2) : !nest.event<"pf_bidx19_r2_s2">
    %att19_grid_r2_s2, %att19_inrel_r2_s2, %att19_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx19_r2_s2, %att15_out_r2_s2)
      : (
        !nest.event<"att19_grid_r2_s2">, !nest.event<"att19_inrel_r2_s2">,
        !nest.event<"att19_out_r2_s2">)
    %pf_bidx20_r2_s2 = nest.dma.prefetch.async %746 into %block_idx_p0_10
      depends_on(%att18_inrel_r2_s2) : !nest.event<"pf_bidx20_r2_s2">
    %att20_grid_r2_s2, %att20_inrel_r2_s2, %att20_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx20_r2_s2, %att16_out_r2_s2)
      : (
        !nest.event<"att20_grid_r2_s2">, !nest.event<"att20_inrel_r2_s2">,
        !nest.event<"att20_out_r2_s2">)
    %pf_bidx21_r2_s2 = nest.dma.prefetch.async %747 into %block_idx_p1_10
      depends_on(%att19_inrel_r2_s2) : !nest.event<"pf_bidx21_r2_s2">
    %att21_grid_r2_s2, %att21_inrel_r2_s2, %att21_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx21_r2_s2, %att17_out_r2_s2)
      : (
        !nest.event<"att21_grid_r2_s2">, !nest.event<"att21_inrel_r2_s2">,
        !nest.event<"att21_out_r2_s2">)
    %pf_bidx22_r2_s2 = nest.dma.prefetch.async %748 into %block_idx_p0_10
      depends_on(%att20_inrel_r2_s2) : !nest.event<"pf_bidx22_r2_s2">
    %att22_grid_r2_s2, %att22_inrel_r2_s2, %att22_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx22_r2_s2, %att18_out_r2_s2)
      : (
        !nest.event<"att22_grid_r2_s2">, !nest.event<"att22_inrel_r2_s2">,
        !nest.event<"att22_out_r2_s2">)
    %pf_bidx23_r2_s2 = nest.dma.prefetch.async %749 into %block_idx_p1_10
      depends_on(%att21_inrel_r2_s2) : !nest.event<"pf_bidx23_r2_s2">
    %att23_grid_r2_s2, %att23_inrel_r2_s2, %att23_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx23_r2_s2, %att19_out_r2_s2)
      : (
        !nest.event<"att23_grid_r2_s2">, !nest.event<"att23_inrel_r2_s2">,
        !nest.event<"att23_out_r2_s2">)
    %pf_bidx24_r2_s2 = nest.dma.prefetch.async %750 into %block_idx_p0_10
      depends_on(%att22_inrel_r2_s2) : !nest.event<"pf_bidx24_r2_s2">
    %att24_grid_r2_s2, %att24_inrel_r2_s2, %att24_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx24_r2_s2, %att20_out_r2_s2)
      : (
        !nest.event<"att24_grid_r2_s2">, !nest.event<"att24_inrel_r2_s2">,
        !nest.event<"att24_out_r2_s2">)
    %pf_bidx25_r2_s2 = nest.dma.prefetch.async %751 into %block_idx_p1_10
      depends_on(%att23_inrel_r2_s2) : !nest.event<"pf_bidx25_r2_s2">
    %att25_grid_r2_s2, %att25_inrel_r2_s2, %att25_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx25_r2_s2, %att21_out_r2_s2)
      : (
        !nest.event<"att25_grid_r2_s2">, !nest.event<"att25_inrel_r2_s2">,
        !nest.event<"att25_out_r2_s2">)
    %pf_bidx26_r2_s2 = nest.dma.prefetch.async %752 into %block_idx_p0_10
      depends_on(%att24_inrel_r2_s2) : !nest.event<"pf_bidx26_r2_s2">
    %att26_grid_r2_s2, %att26_inrel_r2_s2, %att26_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx26_r2_s2, %att22_out_r2_s2)
      : (
        !nest.event<"att26_grid_r2_s2">, !nest.event<"att26_inrel_r2_s2">,
        !nest.event<"att26_out_r2_s2">)
    %pf_bidx27_r2_s2 = nest.dma.prefetch.async %753 into %block_idx_p1_10
      depends_on(%att25_inrel_r2_s2) : !nest.event<"pf_bidx27_r2_s2">
    %att27_grid_r2_s2, %att27_inrel_r2_s2, %att27_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx27_r2_s2, %att23_out_r2_s2)
      : (
        !nest.event<"att27_grid_r2_s2">, !nest.event<"att27_inrel_r2_s2">,
        !nest.event<"att27_out_r2_s2">)
    %pf_bidx28_r2_s2 = nest.dma.prefetch.async %754 into %block_idx_p0_10
      depends_on(%att26_inrel_r2_s2) : !nest.event<"pf_bidx28_r2_s2">
    %att28_grid_r2_s2, %att28_inrel_r2_s2, %att28_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx28_r2_s2, %att24_out_r2_s2)
      : (
        !nest.event<"att28_grid_r2_s2">, !nest.event<"att28_inrel_r2_s2">,
        !nest.event<"att28_out_r2_s2">)
    %pf_bidx29_r2_s2 = nest.dma.prefetch.async %755 into %block_idx_p1_10
      depends_on(%att27_inrel_r2_s2) : !nest.event<"pf_bidx29_r2_s2">
    %att29_grid_r2_s2, %att29_inrel_r2_s2, %att29_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx29_r2_s2, %att25_out_r2_s2)
      : (
        !nest.event<"att29_grid_r2_s2">, !nest.event<"att29_inrel_r2_s2">,
        !nest.event<"att29_out_r2_s2">)
    %pf_bidx30_r2_s2 = nest.dma.prefetch.async %756 into %block_idx_p0_10
      depends_on(%att28_inrel_r2_s2) : !nest.event<"pf_bidx30_r2_s2">
    %att30_grid_r2_s2, %att30_inrel_r2_s2, %att30_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx30_r2_s2, %att26_out_r2_s2)
      : (
        !nest.event<"att30_grid_r2_s2">, !nest.event<"att30_inrel_r2_s2">,
        !nest.event<"att30_out_r2_s2">)
    %pf_bidx31_r2_s2 = nest.dma.prefetch.async %757 into %block_idx_p1_10
      depends_on(%att29_inrel_r2_s2) : !nest.event<"pf_bidx31_r2_s2">
    %att31_grid_r2_s2, %att31_inrel_r2_s2, %att31_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx31_r2_s2, %att27_out_r2_s2)
      : (
        !nest.event<"att31_grid_r2_s2">, !nest.event<"att31_inrel_r2_s2">,
        !nest.event<"att31_out_r2_s2">)
    %pf_bidx32_r2_s2 = nest.dma.prefetch.async %758 into %block_idx_p0_10
      depends_on(%att30_inrel_r2_s2) : !nest.event<"pf_bidx32_r2_s2">
    %att32_grid_r2_s2, %att32_inrel_r2_s2, %att32_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx32_r2_s2, %att28_out_r2_s2)
      : (
        !nest.event<"att32_grid_r2_s2">, !nest.event<"att32_inrel_r2_s2">,
        !nest.event<"att32_out_r2_s2">)
    %pf_bidx33_r2_s2 = nest.dma.prefetch.async %759 into %block_idx_p1_10
      depends_on(%att31_inrel_r2_s2) : !nest.event<"pf_bidx33_r2_s2">
    %att33_grid_r2_s2, %att33_inrel_r2_s2, %att33_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx33_r2_s2, %att29_out_r2_s2)
      : (
        !nest.event<"att33_grid_r2_s2">, !nest.event<"att33_inrel_r2_s2">,
        !nest.event<"att33_out_r2_s2">)
    %pf_bidx34_r2_s2 = nest.dma.prefetch.async %760 into %block_idx_p0_10
      depends_on(%att32_inrel_r2_s2) : !nest.event<"pf_bidx34_r2_s2">
    %att34_grid_r2_s2, %att34_inrel_r2_s2, %att34_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx34_r2_s2, %att30_out_r2_s2)
      : (
        !nest.event<"att34_grid_r2_s2">, !nest.event<"att34_inrel_r2_s2">,
        !nest.event<"att34_out_r2_s2">)
    %pf_bidx35_r2_s2 = nest.dma.prefetch.async %761 into %block_idx_p1_10
      depends_on(%att33_inrel_r2_s2) : !nest.event<"pf_bidx35_r2_s2">
    %att35_grid_r2_s2, %att35_inrel_r2_s2, %att35_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx35_r2_s2, %att31_out_r2_s2)
      : (
        !nest.event<"att35_grid_r2_s2">, !nest.event<"att35_inrel_r2_s2">,
        !nest.event<"att35_out_r2_s2">)
    %pf_bidx36_r2_s2 = nest.dma.prefetch.async %762 into %block_idx_p0_10
      depends_on(%att34_inrel_r2_s2) : !nest.event<"pf_bidx36_r2_s2">
    %att36_grid_r2_s2, %att36_inrel_r2_s2, %att36_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx36_r2_s2, %att32_out_r2_s2)
      : (
        !nest.event<"att36_grid_r2_s2">, !nest.event<"att36_inrel_r2_s2">,
        !nest.event<"att36_out_r2_s2">)
    %pf_bidx37_r2_s2 = nest.dma.prefetch.async %763 into %block_idx_p1_10
      depends_on(%att35_inrel_r2_s2) : !nest.event<"pf_bidx37_r2_s2">
    %att37_grid_r2_s2, %att37_inrel_r2_s2, %att37_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx37_r2_s2, %att33_out_r2_s2)
      : (
        !nest.event<"att37_grid_r2_s2">, !nest.event<"att37_inrel_r2_s2">,
        !nest.event<"att37_out_r2_s2">)
    %pf_bidx38_r2_s2 = nest.dma.prefetch.async %764 into %block_idx_p0_10
      depends_on(%att36_inrel_r2_s2) : !nest.event<"pf_bidx38_r2_s2">
    %att38_grid_r2_s2, %att38_inrel_r2_s2, %att38_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx38_r2_s2, %att34_out_r2_s2)
      : (
        !nest.event<"att38_grid_r2_s2">, !nest.event<"att38_inrel_r2_s2">,
        !nest.event<"att38_out_r2_s2">)
    %pf_bidx39_r2_s2 = nest.dma.prefetch.async %765 into %block_idx_p1_10
      depends_on(%att37_inrel_r2_s2) : !nest.event<"pf_bidx39_r2_s2">
    %att39_grid_r2_s2, %att39_inrel_r2_s2, %att39_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx39_r2_s2, %att35_out_r2_s2)
      : (
        !nest.event<"att39_grid_r2_s2">, !nest.event<"att39_inrel_r2_s2">,
        !nest.event<"att39_out_r2_s2">)
    %pf_bidx40_r2_s2 = nest.dma.prefetch.async %766 into %block_idx_p0_10
      depends_on(%att38_inrel_r2_s2) : !nest.event<"pf_bidx40_r2_s2">
    %att40_grid_r2_s2, %att40_inrel_r2_s2, %att40_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx40_r2_s2, %att36_out_r2_s2)
      : (
        !nest.event<"att40_grid_r2_s2">, !nest.event<"att40_inrel_r2_s2">,
        !nest.event<"att40_out_r2_s2">)
    %pf_bidx41_r2_s2 = nest.dma.prefetch.async %767 into %block_idx_p1_10
      depends_on(%att39_inrel_r2_s2) : !nest.event<"pf_bidx41_r2_s2">
    %att41_grid_r2_s2, %att41_inrel_r2_s2, %att41_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx41_r2_s2, %att37_out_r2_s2)
      : (
        !nest.event<"att41_grid_r2_s2">, !nest.event<"att41_inrel_r2_s2">,
        !nest.event<"att41_out_r2_s2">)
    %pf_bidx42_r2_s2 = nest.dma.prefetch.async %768 into %block_idx_p0_10
      depends_on(%att40_inrel_r2_s2) : !nest.event<"pf_bidx42_r2_s2">
    %att42_grid_r2_s2, %att42_inrel_r2_s2, %att42_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx42_r2_s2, %att38_out_r2_s2)
      : (
        !nest.event<"att42_grid_r2_s2">, !nest.event<"att42_inrel_r2_s2">,
        !nest.event<"att42_out_r2_s2">)
    %pf_bidx43_r2_s2 = nest.dma.prefetch.async %769 into %block_idx_p1_10
      depends_on(%att41_inrel_r2_s2) : !nest.event<"pf_bidx43_r2_s2">
    %att43_grid_r2_s2, %att43_inrel_r2_s2, %att43_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx43_r2_s2, %att39_out_r2_s2)
      : (
        !nest.event<"att43_grid_r2_s2">, !nest.event<"att43_inrel_r2_s2">,
        !nest.event<"att43_out_r2_s2">)
    %pf_bidx44_r2_s2 = nest.dma.prefetch.async %770 into %block_idx_p0_10
      depends_on(%att42_inrel_r2_s2) : !nest.event<"pf_bidx44_r2_s2">
    %att44_grid_r2_s2, %att44_inrel_r2_s2, %att44_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_bidx44_r2_s2, %att40_out_r2_s2)
      : (
        !nest.event<"att44_grid_r2_s2">, !nest.event<"att44_inrel_r2_s2">,
        !nest.event<"att44_out_r2_s2">)
    %pf_bidx45_r2_s2 = nest.dma.prefetch.async %771 into %block_idx_p1_10
      depends_on(%att43_inrel_r2_s2) : !nest.event<"pf_bidx45_r2_s2">
    %att45_grid_r2_s2, %att45_inrel_r2_s2, %att45_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p1_11, %acc_p1_11) outs(%state_p1_11, %acc_p1_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p1_r2_s2, %pf_acc_p1_r2_s2, %pf_bidx45_r2_s2, %att41_out_r2_s2)
      : (
        !nest.event<"att45_grid_r2_s2">, !nest.event<"att45_inrel_r2_s2">,
        !nest.event<"att45_out_r2_s2">)
    %pf_bidx46_r2_s2 = nest.dma.prefetch.async %772 into %block_idx_p0_10
      depends_on(%att44_inrel_r2_s2) : !nest.event<"pf_bidx46_r2_s2">
    %att46_grid_r2_s2, %att46_inrel_r2_s2, %att46_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11)
      ins(%block_idx_p0_10, %q_l2_25, %state_p2_11, %acc_p2_11) outs(%state_p2_11, %acc_p2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p2_r2_s2, %pf_acc_p2_r2_s2, %pf_bidx46_r2_s2, %att42_out_r2_s2)
      : (
        !nest.event<"att46_grid_r2_s2">, !nest.event<"att46_inrel_r2_s2">,
        !nest.event<"att46_out_r2_s2">)
    %pf_bidx47_r2_s2 = nest.dma.prefetch.async %773 into %block_idx_p1_10
      depends_on(%att45_inrel_r2_s2) : !nest.event<"pf_bidx47_r2_s2">
    %att47_grid_r2_s2, %att47_inrel_r2_s2, %att47_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11)
      ins(%block_idx_p1_10, %q_l2_25, %state_p3_11, %acc_p3_11) outs(%state_p3_11, %acc_p3_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p3_r2_s2, %pf_acc_p3_r2_s2, %pf_bidx47_r2_s2, %att43_out_r2_s2)
      : (
        !nest.event<"att47_grid_r2_s2">, !nest.event<"att47_inrel_r2_s2">,
        !nest.event<"att47_out_r2_s2">)
    %att48_grid_r2_s2, %att48_inrel_r2_s2, %att48_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_t2_final_r2 l1_mode = 1 tasks(%774) globals(%712)
      bindings(%append_idx_10, %q_l2_25, %state_p0_11, %acc_p0_11)
      ins(%append_idx_10, %q_l2_25, %state_p0_11, %acc_p0_11) outs(%state_p0_11, %acc_p0_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s2, %pf_state_p0_r2_s2, %pf_acc_p0_r2_s2, %pf_aidx_r2_s2, %append_grid_r2_s2,
        %att44_out_r2_s2)
      : (
        !nest.event<"att48_grid_r2_s2">, !nest.event<"att48_inrel_r2_s2">,
        !nest.event<"att48_out_r2_s2">)
    %merge_grid_r2_s2, %merge_inrel_r2_s2, %merge_out_r2_s2 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%774) globals()
      bindings(
        %state_p0_11, %state_p1_11, %state_p2_11, %state_p3_11, %acc_p0_11, %acc_p1_11,
        %acc_p2_11, %acc_p3_11, %out_l2_11)
      ins(
        %state_p0_11, %state_p1_11, %state_p2_11, %state_p3_11, %acc_p0_11, %acc_p1_11,
        %acc_p2_11, %acc_p3_11)
      outs(%out_l2_11)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att48_out_r2_s2, %att45_out_r2_s2, %att46_out_r2_s2, %att47_out_r2_s2) : (
        !nest.event<"merge_grid_r2_s2">, !nest.event<"merge_inrel_r2_s2">,
        !nest.event<"merge_out_r2_s2">)
    %out_store_r2_s2 = nest.dma.store.async %out_l2_11 into %725 depends_on(%merge_out_r2_s2)
      : !nest.event<"out_store_r2_s2">
    nest.await %append_grid_r2_s2, %att48_grid_r2_s2, %merge_grid_r2_s2, %out_store_r2_s2
    nest.release %block_idx_p0_10 depends_on(
      %pf_bidx0_r2_s2, %pf_bidx2_r2_s2, %pf_bidx4_r2_s2, %pf_bidx6_r2_s2, %pf_bidx8_r2_s2,
      %pf_bidx10_r2_s2, %pf_bidx12_r2_s2, %pf_bidx14_r2_s2, %pf_bidx16_r2_s2, %pf_bidx18_r2_s2,
      %pf_bidx20_r2_s2, %pf_bidx22_r2_s2, %pf_bidx24_r2_s2, %pf_bidx26_r2_s2, %pf_bidx28_r2_s2,
      %pf_bidx30_r2_s2, %pf_bidx32_r2_s2, %pf_bidx34_r2_s2, %pf_bidx36_r2_s2, %pf_bidx38_r2_s2,
      %pf_bidx40_r2_s2, %pf_bidx42_r2_s2, %pf_bidx44_r2_s2, %pf_bidx46_r2_s2, %att0_inrel_r2_s2,
      %att2_inrel_r2_s2, %att4_inrel_r2_s2, %att6_inrel_r2_s2, %att8_inrel_r2_s2,
      %att10_inrel_r2_s2, %att12_inrel_r2_s2, %att14_inrel_r2_s2, %att16_inrel_r2_s2,
      %att18_inrel_r2_s2, %att20_inrel_r2_s2, %att22_inrel_r2_s2, %att24_inrel_r2_s2,
      %att26_inrel_r2_s2, %att28_inrel_r2_s2, %att30_inrel_r2_s2, %att32_inrel_r2_s2,
      %att34_inrel_r2_s2, %att36_inrel_r2_s2, %att38_inrel_r2_s2, %att40_inrel_r2_s2,
      %att42_inrel_r2_s2, %att44_inrel_r2_s2, %att46_inrel_r2_s2)
    nest.release %block_idx_p1_10 depends_on(
      %pf_bidx1_r2_s2, %pf_bidx3_r2_s2, %pf_bidx5_r2_s2, %pf_bidx7_r2_s2, %pf_bidx9_r2_s2,
      %pf_bidx11_r2_s2, %pf_bidx13_r2_s2, %pf_bidx15_r2_s2, %pf_bidx17_r2_s2, %pf_bidx19_r2_s2,
      %pf_bidx21_r2_s2, %pf_bidx23_r2_s2, %pf_bidx25_r2_s2, %pf_bidx27_r2_s2, %pf_bidx29_r2_s2,
      %pf_bidx31_r2_s2, %pf_bidx33_r2_s2, %pf_bidx35_r2_s2, %pf_bidx37_r2_s2, %pf_bidx39_r2_s2,
      %pf_bidx41_r2_s2, %pf_bidx43_r2_s2, %pf_bidx45_r2_s2, %pf_bidx47_r2_s2, %att1_inrel_r2_s2,
      %att3_inrel_r2_s2, %att5_inrel_r2_s2, %att7_inrel_r2_s2, %att9_inrel_r2_s2,
      %att11_inrel_r2_s2, %att13_inrel_r2_s2, %att15_inrel_r2_s2, %att17_inrel_r2_s2,
      %att19_inrel_r2_s2, %att21_inrel_r2_s2, %att23_inrel_r2_s2, %att25_inrel_r2_s2,
      %att27_inrel_r2_s2, %att29_inrel_r2_s2, %att31_inrel_r2_s2, %att33_inrel_r2_s2,
      %att35_inrel_r2_s2, %att37_inrel_r2_s2, %att39_inrel_r2_s2, %att41_inrel_r2_s2,
      %att43_inrel_r2_s2, %att45_inrel_r2_s2, %att47_inrel_r2_s2)
    nest.release %k_new_10 depends_on(%pf_k_r2_s2, %append_inrel_r2_s2)
    nest.release %v_new_10 depends_on(%pf_v_r2_s2, %append_inrel_r2_s2)
    nest.release %append_idx_10 depends_on(%pf_aidx_r2_s2, %append_inrel_r2_s2, %att48_inrel_r2_s2)
    nest.release %q_l2_25 depends_on(
      %pf_q_r2_s2, %att0_inrel_r2_s2, %att1_inrel_r2_s2, %att2_inrel_r2_s2, %att3_inrel_r2_s2,
      %att4_inrel_r2_s2, %att5_inrel_r2_s2, %att6_inrel_r2_s2, %att7_inrel_r2_s2,
      %att8_inrel_r2_s2, %att9_inrel_r2_s2, %att10_inrel_r2_s2, %att11_inrel_r2_s2,
      %att12_inrel_r2_s2, %att13_inrel_r2_s2, %att14_inrel_r2_s2, %att15_inrel_r2_s2,
      %att16_inrel_r2_s2, %att17_inrel_r2_s2, %att18_inrel_r2_s2, %att19_inrel_r2_s2,
      %att20_inrel_r2_s2, %att21_inrel_r2_s2, %att22_inrel_r2_s2, %att23_inrel_r2_s2,
      %att24_inrel_r2_s2, %att25_inrel_r2_s2, %att26_inrel_r2_s2, %att27_inrel_r2_s2,
      %att28_inrel_r2_s2, %att29_inrel_r2_s2, %att30_inrel_r2_s2, %att31_inrel_r2_s2,
      %att32_inrel_r2_s2, %att33_inrel_r2_s2, %att34_inrel_r2_s2, %att35_inrel_r2_s2,
      %att36_inrel_r2_s2, %att37_inrel_r2_s2, %att38_inrel_r2_s2, %att39_inrel_r2_s2,
      %att40_inrel_r2_s2, %att41_inrel_r2_s2, %att42_inrel_r2_s2, %att43_inrel_r2_s2,
      %att44_inrel_r2_s2, %att45_inrel_r2_s2, %att46_inrel_r2_s2, %att47_inrel_r2_s2,
      %att48_inrel_r2_s2)
    nest.release %state_p0_11 depends_on(
      %pf_state_p0_r2_s2, %att0_inrel_r2_s2, %att4_inrel_r2_s2, %att8_inrel_r2_s2,
      %att12_inrel_r2_s2, %att16_inrel_r2_s2, %att20_inrel_r2_s2, %att24_inrel_r2_s2,
      %att28_inrel_r2_s2, %att32_inrel_r2_s2, %att36_inrel_r2_s2, %att40_inrel_r2_s2,
      %att44_inrel_r2_s2, %att48_inrel_r2_s2, %att0_out_r2_s2, %att4_out_r2_s2, %att8_out_r2_s2,
      %att12_out_r2_s2, %att16_out_r2_s2, %att20_out_r2_s2, %att24_out_r2_s2, %att28_out_r2_s2,
      %att32_out_r2_s2, %att36_out_r2_s2, %att40_out_r2_s2, %att44_out_r2_s2, %att48_out_r2_s2,
      %merge_inrel_r2_s2)
    nest.release %acc_p0_11 depends_on(
      %pf_acc_p0_r2_s2, %att0_inrel_r2_s2, %att4_inrel_r2_s2, %att8_inrel_r2_s2,
      %att12_inrel_r2_s2, %att16_inrel_r2_s2, %att20_inrel_r2_s2, %att24_inrel_r2_s2,
      %att28_inrel_r2_s2, %att32_inrel_r2_s2, %att36_inrel_r2_s2, %att40_inrel_r2_s2,
      %att44_inrel_r2_s2, %att48_inrel_r2_s2, %att0_out_r2_s2, %att4_out_r2_s2, %att8_out_r2_s2,
      %att12_out_r2_s2, %att16_out_r2_s2, %att20_out_r2_s2, %att24_out_r2_s2, %att28_out_r2_s2,
      %att32_out_r2_s2, %att36_out_r2_s2, %att40_out_r2_s2, %att44_out_r2_s2, %att48_out_r2_s2,
      %merge_inrel_r2_s2)
    nest.release %state_p1_11 depends_on(
      %pf_state_p1_r2_s2, %att1_inrel_r2_s2, %att5_inrel_r2_s2, %att9_inrel_r2_s2,
      %att13_inrel_r2_s2, %att17_inrel_r2_s2, %att21_inrel_r2_s2, %att25_inrel_r2_s2,
      %att29_inrel_r2_s2, %att33_inrel_r2_s2, %att37_inrel_r2_s2, %att41_inrel_r2_s2,
      %att45_inrel_r2_s2, %att1_out_r2_s2, %att5_out_r2_s2, %att9_out_r2_s2, %att13_out_r2_s2,
      %att17_out_r2_s2, %att21_out_r2_s2, %att25_out_r2_s2, %att29_out_r2_s2, %att33_out_r2_s2,
      %att37_out_r2_s2, %att41_out_r2_s2, %att45_out_r2_s2, %merge_inrel_r2_s2)
    nest.release %acc_p1_11 depends_on(
      %pf_acc_p1_r2_s2, %att1_inrel_r2_s2, %att5_inrel_r2_s2, %att9_inrel_r2_s2,
      %att13_inrel_r2_s2, %att17_inrel_r2_s2, %att21_inrel_r2_s2, %att25_inrel_r2_s2,
      %att29_inrel_r2_s2, %att33_inrel_r2_s2, %att37_inrel_r2_s2, %att41_inrel_r2_s2,
      %att45_inrel_r2_s2, %att1_out_r2_s2, %att5_out_r2_s2, %att9_out_r2_s2, %att13_out_r2_s2,
      %att17_out_r2_s2, %att21_out_r2_s2, %att25_out_r2_s2, %att29_out_r2_s2, %att33_out_r2_s2,
      %att37_out_r2_s2, %att41_out_r2_s2, %att45_out_r2_s2, %merge_inrel_r2_s2)
    nest.release %state_p2_11 depends_on(
      %pf_state_p2_r2_s2, %att2_inrel_r2_s2, %att6_inrel_r2_s2, %att10_inrel_r2_s2,
      %att14_inrel_r2_s2, %att18_inrel_r2_s2, %att22_inrel_r2_s2, %att26_inrel_r2_s2,
      %att30_inrel_r2_s2, %att34_inrel_r2_s2, %att38_inrel_r2_s2, %att42_inrel_r2_s2,
      %att46_inrel_r2_s2, %att2_out_r2_s2, %att6_out_r2_s2, %att10_out_r2_s2, %att14_out_r2_s2,
      %att18_out_r2_s2, %att22_out_r2_s2, %att26_out_r2_s2, %att30_out_r2_s2, %att34_out_r2_s2,
      %att38_out_r2_s2, %att42_out_r2_s2, %att46_out_r2_s2, %merge_inrel_r2_s2)
    nest.release %acc_p2_11 depends_on(
      %pf_acc_p2_r2_s2, %att2_inrel_r2_s2, %att6_inrel_r2_s2, %att10_inrel_r2_s2,
      %att14_inrel_r2_s2, %att18_inrel_r2_s2, %att22_inrel_r2_s2, %att26_inrel_r2_s2,
      %att30_inrel_r2_s2, %att34_inrel_r2_s2, %att38_inrel_r2_s2, %att42_inrel_r2_s2,
      %att46_inrel_r2_s2, %att2_out_r2_s2, %att6_out_r2_s2, %att10_out_r2_s2, %att14_out_r2_s2,
      %att18_out_r2_s2, %att22_out_r2_s2, %att26_out_r2_s2, %att30_out_r2_s2, %att34_out_r2_s2,
      %att38_out_r2_s2, %att42_out_r2_s2, %att46_out_r2_s2, %merge_inrel_r2_s2)
    nest.release %state_p3_11 depends_on(
      %pf_state_p3_r2_s2, %att3_inrel_r2_s2, %att7_inrel_r2_s2, %att11_inrel_r2_s2,
      %att15_inrel_r2_s2, %att19_inrel_r2_s2, %att23_inrel_r2_s2, %att27_inrel_r2_s2,
      %att31_inrel_r2_s2, %att35_inrel_r2_s2, %att39_inrel_r2_s2, %att43_inrel_r2_s2,
      %att47_inrel_r2_s2, %att3_out_r2_s2, %att7_out_r2_s2, %att11_out_r2_s2, %att15_out_r2_s2,
      %att19_out_r2_s2, %att23_out_r2_s2, %att27_out_r2_s2, %att31_out_r2_s2, %att35_out_r2_s2,
      %att39_out_r2_s2, %att43_out_r2_s2, %att47_out_r2_s2, %merge_inrel_r2_s2)
    nest.release %acc_p3_11 depends_on(
      %pf_acc_p3_r2_s2, %att3_inrel_r2_s2, %att7_inrel_r2_s2, %att11_inrel_r2_s2,
      %att15_inrel_r2_s2, %att19_inrel_r2_s2, %att23_inrel_r2_s2, %att27_inrel_r2_s2,
      %att31_inrel_r2_s2, %att35_inrel_r2_s2, %att39_inrel_r2_s2, %att43_inrel_r2_s2,
      %att47_inrel_r2_s2, %att3_out_r2_s2, %att7_out_r2_s2, %att11_out_r2_s2, %att15_out_r2_s2,
      %att19_out_r2_s2, %att23_out_r2_s2, %att27_out_r2_s2, %att31_out_r2_s2, %att35_out_r2_s2,
      %att39_out_r2_s2, %att43_out_r2_s2, %att47_out_r2_s2, %merge_inrel_r2_s2)
    nest.release %out_l2_11 depends_on(%merge_out_r2_s2, %out_store_r2_s2)
    nest.return
  }
  nest.context @step_r2_s3(
    %POOL_11: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_11: !nest.global_memref<147xi32>,
    %APPEND_IDS_11: !nest.global_memref<12xi32>, %Q_IN_11: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_11: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_11: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_11: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_11: !nest.global_memref<3x4x4x4x4x64xf32>,
    %OUT_11: !nest.global_memref<3x4x4x4x64xf32>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [0, 1, 2],
      logical_tasks = 204, l2_spm_bytes = 61440, requested_contexts_per_tile = 4,
      l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536}> {
    %k_new_11 = nest.alloc slot = "k_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %v_new_11 = nest.alloc slot = "v_new" role = "in" shape = [4, 1, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1x64xbf16>
    %append_idx_11 = nest.alloc slot = "append_idx" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %q_l2_26 = nest.alloc slot = "q_l2" role = "in" shape = [4, 4, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x4x64xbf16>
    %state_p0_12 = nest.alloc slot = "state_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p1_12 = nest.alloc slot = "state_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p2_12 = nest.alloc slot = "state_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %state_p3_12 = nest.alloc slot = "state_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 2] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x2xf32>
    %acc_p0_12 = nest.alloc slot = "acc_p0" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p1_12 = nest.alloc slot = "acc_p1" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p2_12 = nest.alloc slot = "acc_p2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %acc_p3_12 = nest.alloc slot = "acc_p3" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %block_idx_p0_11 = nest.alloc slot = "block_idx_p0" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %block_idx_p1_11 = nest.alloc slot = "block_idx_p1" role = "in" shape = [1] dtype = "i32"
      alignment = 64 : !nest.l2_buffer<1xi32>
    %out_l2_12 = nest.alloc slot = "out_l2" role = "inout" sharing = "context-local"
      shape = [4, 4, 64] dtype = "f32" alignment = 64 : !nest.l2_buffer<4x4x64xf32>
    %776 = nest.subview %POOL_11 offsets = [0, 0] sizes = [128, 8224] strides = [1, 1]
      : !nest.global_view<128x8224xbf16>
    %777 = nest.subview %K_NEW_11 offsets = [2, 3, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %778 = nest.subview %V_NEW_11 offsets = [2, 3, 0, 0, 0] sizes = [1, 1, 4, 1, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x1x64xbf16>
    %779 = nest.subview %APPEND_IDS_11 offsets = [11] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %780 = nest.subview %Q_IN_11 offsets = [2, 3, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xbf16>
    %781 = nest.subview %S_INIT_11 offsets = [2, 3, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %782 = nest.subview %S_INIT_11 offsets = [2, 3, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %783 = nest.subview %S_INIT_11 offsets = [2, 3, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %784 = nest.subview %S_INIT_11 offsets = [2, 3, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 2]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x2xf32>
    %785 = nest.subview %O_INIT_11 offsets = [2, 3, 0, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %786 = nest.subview %O_INIT_11 offsets = [2, 3, 1, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %787 = nest.subview %O_INIT_11 offsets = [2, 3, 2, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %788 = nest.subview %O_INIT_11 offsets = [2, 3, 3, 0, 0, 0] sizes = [1, 1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1, 1] : !nest.global_view<1x1x1x4x4x64xf32>
    %789 = nest.subview %OUT_11 offsets = [2, 3, 0, 0, 0] sizes = [1, 1, 4, 4, 64]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x4x64xf32>
    %790 = nest.subview %BLOCK_TABLE_11 offsets = [98] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %791 = nest.subview %BLOCK_TABLE_11 offsets = [99] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %792 = nest.subview %BLOCK_TABLE_11 offsets = [100] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %793 = nest.subview %BLOCK_TABLE_11 offsets = [101] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %794 = nest.subview %BLOCK_TABLE_11 offsets = [102] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %795 = nest.subview %BLOCK_TABLE_11 offsets = [103] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %796 = nest.subview %BLOCK_TABLE_11 offsets = [104] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %797 = nest.subview %BLOCK_TABLE_11 offsets = [105] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %798 = nest.subview %BLOCK_TABLE_11 offsets = [106] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %799 = nest.subview %BLOCK_TABLE_11 offsets = [107] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %800 = nest.subview %BLOCK_TABLE_11 offsets = [108] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %801 = nest.subview %BLOCK_TABLE_11 offsets = [109] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %802 = nest.subview %BLOCK_TABLE_11 offsets = [110] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %803 = nest.subview %BLOCK_TABLE_11 offsets = [111] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %804 = nest.subview %BLOCK_TABLE_11 offsets = [112] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %805 = nest.subview %BLOCK_TABLE_11 offsets = [113] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %806 = nest.subview %BLOCK_TABLE_11 offsets = [114] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %807 = nest.subview %BLOCK_TABLE_11 offsets = [115] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %808 = nest.subview %BLOCK_TABLE_11 offsets = [116] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %809 = nest.subview %BLOCK_TABLE_11 offsets = [117] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %810 = nest.subview %BLOCK_TABLE_11 offsets = [118] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %811 = nest.subview %BLOCK_TABLE_11 offsets = [119] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %812 = nest.subview %BLOCK_TABLE_11 offsets = [120] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %813 = nest.subview %BLOCK_TABLE_11 offsets = [121] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %814 = nest.subview %BLOCK_TABLE_11 offsets = [122] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %815 = nest.subview %BLOCK_TABLE_11 offsets = [123] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %816 = nest.subview %BLOCK_TABLE_11 offsets = [124] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %817 = nest.subview %BLOCK_TABLE_11 offsets = [125] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %818 = nest.subview %BLOCK_TABLE_11 offsets = [126] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %819 = nest.subview %BLOCK_TABLE_11 offsets = [127] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %820 = nest.subview %BLOCK_TABLE_11 offsets = [128] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %821 = nest.subview %BLOCK_TABLE_11 offsets = [129] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %822 = nest.subview %BLOCK_TABLE_11 offsets = [130] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %823 = nest.subview %BLOCK_TABLE_11 offsets = [131] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %824 = nest.subview %BLOCK_TABLE_11 offsets = [132] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %825 = nest.subview %BLOCK_TABLE_11 offsets = [133] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %826 = nest.subview %BLOCK_TABLE_11 offsets = [134] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %827 = nest.subview %BLOCK_TABLE_11 offsets = [135] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %828 = nest.subview %BLOCK_TABLE_11 offsets = [136] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %829 = nest.subview %BLOCK_TABLE_11 offsets = [137] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %830 = nest.subview %BLOCK_TABLE_11 offsets = [138] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %831 = nest.subview %BLOCK_TABLE_11 offsets = [139] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %832 = nest.subview %BLOCK_TABLE_11 offsets = [140] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %833 = nest.subview %BLOCK_TABLE_11 offsets = [141] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %834 = nest.subview %BLOCK_TABLE_11 offsets = [142] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %835 = nest.subview %BLOCK_TABLE_11 offsets = [143] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %836 = nest.subview %BLOCK_TABLE_11 offsets = [144] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %837 = nest.subview %BLOCK_TABLE_11 offsets = [145] sizes = [1] strides = [1]
      : !nest.global_view<1xi32>
    %pf_k_r2_s3 = nest.dma.prefetch.async %777 into %k_new_11 : !nest.event<"pf_k_r2_s3">
    %pf_v_r2_s3 = nest.dma.prefetch.async %778 into %v_new_11 : !nest.event<"pf_v_r2_s3">
    %pf_aidx_r2_s3 = nest.dma.prefetch.async %779 into %append_idx_11 : !nest.event<"pf_aidx_r2_s3">
    %pf_q_r2_s3 = nest.dma.prefetch.async %780 into %q_l2_26 : !nest.event<"pf_q_r2_s3">
    %pf_state_p0_r2_s3 = nest.dma.prefetch.async %781 into %state_p0_12
      : !nest.event<"pf_state_p0_r2_s3">
    %pf_state_p1_r2_s3 = nest.dma.prefetch.async %782 into %state_p1_12
      : !nest.event<"pf_state_p1_r2_s3">
    %pf_state_p2_r2_s3 = nest.dma.prefetch.async %783 into %state_p2_12
      : !nest.event<"pf_state_p2_r2_s3">
    %pf_state_p3_r2_s3 = nest.dma.prefetch.async %784 into %state_p3_12
      : !nest.event<"pf_state_p3_r2_s3">
    %pf_acc_p0_r2_s3 = nest.dma.prefetch.async %785 into %acc_p0_12 : !nest.event<"pf_acc_p0_r2_s3">
    %pf_acc_p1_r2_s3 = nest.dma.prefetch.async %786 into %acc_p1_12 : !nest.event<"pf_acc_p1_r2_s3">
    %pf_acc_p2_r2_s3 = nest.dma.prefetch.async %787 into %acc_p2_12 : !nest.event<"pf_acc_p2_r2_s3">
    %pf_acc_p3_r2_s3 = nest.dma.prefetch.async %788 into %acc_p3_12 : !nest.event<"pf_acc_p3_r2_s3">
    %838 = nest.task.range from = 0 to = 4 : !nest.task_range
    %append_grid_r2_s3, %append_inrel_r2_s3, %839 =
      nest.dispatch.tasks.async @paged_attention_append_r2_tip2 l1_mode = 1 tasks(%838)
      globals(%776) bindings(%k_new_11, %v_new_11, %append_idx_11)
      ins(%k_new_11, %v_new_11, %append_idx_11) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pf_k_r2_s3, %pf_v_r2_s3, %pf_aidx_r2_s3)
      : (!nest.event<"append_grid_r2_s3">, !nest.event<"append_inrel_r2_s3">, !nest.event<"">)
    %pf_bidx0_r2_s3 = nest.dma.prefetch.async %790 into %block_idx_p0_11
      : !nest.event<"pf_bidx0_r2_s3">
    %att0_grid_r2_s3, %att0_inrel_r2_s3, %att0_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx0_r2_s3) : (
        !nest.event<"att0_grid_r2_s3">, !nest.event<"att0_inrel_r2_s3">,
        !nest.event<"att0_out_r2_s3">)
    %pf_bidx1_r2_s3 = nest.dma.prefetch.async %791 into %block_idx_p1_11
      : !nest.event<"pf_bidx1_r2_s3">
    %att1_grid_r2_s3, %att1_inrel_r2_s3, %att1_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx1_r2_s3) : (
        !nest.event<"att1_grid_r2_s3">, !nest.event<"att1_inrel_r2_s3">,
        !nest.event<"att1_out_r2_s3">)
    %pf_bidx2_r2_s3 = nest.dma.prefetch.async %792 into %block_idx_p0_11
      depends_on(%att0_inrel_r2_s3) : !nest.event<"pf_bidx2_r2_s3">
    %att2_grid_r2_s3, %att2_inrel_r2_s3, %att2_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx2_r2_s3) : (
        !nest.event<"att2_grid_r2_s3">, !nest.event<"att2_inrel_r2_s3">,
        !nest.event<"att2_out_r2_s3">)
    %pf_bidx3_r2_s3 = nest.dma.prefetch.async %793 into %block_idx_p1_11
      depends_on(%att1_inrel_r2_s3) : !nest.event<"pf_bidx3_r2_s3">
    %att3_grid_r2_s3, %att3_inrel_r2_s3, %att3_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx3_r2_s3) : (
        !nest.event<"att3_grid_r2_s3">, !nest.event<"att3_inrel_r2_s3">,
        !nest.event<"att3_out_r2_s3">)
    %pf_bidx4_r2_s3 = nest.dma.prefetch.async %794 into %block_idx_p0_11
      depends_on(%att2_inrel_r2_s3) : !nest.event<"pf_bidx4_r2_s3">
    %att4_grid_r2_s3, %att4_inrel_r2_s3, %att4_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx4_r2_s3, %att0_out_r2_s3)
      : (
        !nest.event<"att4_grid_r2_s3">, !nest.event<"att4_inrel_r2_s3">,
        !nest.event<"att4_out_r2_s3">)
    %pf_bidx5_r2_s3 = nest.dma.prefetch.async %795 into %block_idx_p1_11
      depends_on(%att3_inrel_r2_s3) : !nest.event<"pf_bidx5_r2_s3">
    %att5_grid_r2_s3, %att5_inrel_r2_s3, %att5_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx5_r2_s3, %att1_out_r2_s3)
      : (
        !nest.event<"att5_grid_r2_s3">, !nest.event<"att5_inrel_r2_s3">,
        !nest.event<"att5_out_r2_s3">)
    %pf_bidx6_r2_s3 = nest.dma.prefetch.async %796 into %block_idx_p0_11
      depends_on(%att4_inrel_r2_s3) : !nest.event<"pf_bidx6_r2_s3">
    %att6_grid_r2_s3, %att6_inrel_r2_s3, %att6_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx6_r2_s3, %att2_out_r2_s3)
      : (
        !nest.event<"att6_grid_r2_s3">, !nest.event<"att6_inrel_r2_s3">,
        !nest.event<"att6_out_r2_s3">)
    %pf_bidx7_r2_s3 = nest.dma.prefetch.async %797 into %block_idx_p1_11
      depends_on(%att5_inrel_r2_s3) : !nest.event<"pf_bidx7_r2_s3">
    %att7_grid_r2_s3, %att7_inrel_r2_s3, %att7_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx7_r2_s3, %att3_out_r2_s3)
      : (
        !nest.event<"att7_grid_r2_s3">, !nest.event<"att7_inrel_r2_s3">,
        !nest.event<"att7_out_r2_s3">)
    %pf_bidx8_r2_s3 = nest.dma.prefetch.async %798 into %block_idx_p0_11
      depends_on(%att6_inrel_r2_s3) : !nest.event<"pf_bidx8_r2_s3">
    %att8_grid_r2_s3, %att8_inrel_r2_s3, %att8_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx8_r2_s3, %att4_out_r2_s3)
      : (
        !nest.event<"att8_grid_r2_s3">, !nest.event<"att8_inrel_r2_s3">,
        !nest.event<"att8_out_r2_s3">)
    %pf_bidx9_r2_s3 = nest.dma.prefetch.async %799 into %block_idx_p1_11
      depends_on(%att7_inrel_r2_s3) : !nest.event<"pf_bidx9_r2_s3">
    %att9_grid_r2_s3, %att9_inrel_r2_s3, %att9_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx9_r2_s3, %att5_out_r2_s3)
      : (
        !nest.event<"att9_grid_r2_s3">, !nest.event<"att9_inrel_r2_s3">,
        !nest.event<"att9_out_r2_s3">)
    %pf_bidx10_r2_s3 = nest.dma.prefetch.async %800 into %block_idx_p0_11
      depends_on(%att8_inrel_r2_s3) : !nest.event<"pf_bidx10_r2_s3">
    %att10_grid_r2_s3, %att10_inrel_r2_s3, %att10_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx10_r2_s3, %att6_out_r2_s3)
      : (
        !nest.event<"att10_grid_r2_s3">, !nest.event<"att10_inrel_r2_s3">,
        !nest.event<"att10_out_r2_s3">)
    %pf_bidx11_r2_s3 = nest.dma.prefetch.async %801 into %block_idx_p1_11
      depends_on(%att9_inrel_r2_s3) : !nest.event<"pf_bidx11_r2_s3">
    %att11_grid_r2_s3, %att11_inrel_r2_s3, %att11_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx11_r2_s3, %att7_out_r2_s3)
      : (
        !nest.event<"att11_grid_r2_s3">, !nest.event<"att11_inrel_r2_s3">,
        !nest.event<"att11_out_r2_s3">)
    %pf_bidx12_r2_s3 = nest.dma.prefetch.async %802 into %block_idx_p0_11
      depends_on(%att10_inrel_r2_s3) : !nest.event<"pf_bidx12_r2_s3">
    %att12_grid_r2_s3, %att12_inrel_r2_s3, %att12_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx12_r2_s3, %att8_out_r2_s3)
      : (
        !nest.event<"att12_grid_r2_s3">, !nest.event<"att12_inrel_r2_s3">,
        !nest.event<"att12_out_r2_s3">)
    %pf_bidx13_r2_s3 = nest.dma.prefetch.async %803 into %block_idx_p1_11
      depends_on(%att11_inrel_r2_s3) : !nest.event<"pf_bidx13_r2_s3">
    %att13_grid_r2_s3, %att13_inrel_r2_s3, %att13_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx13_r2_s3, %att9_out_r2_s3)
      : (
        !nest.event<"att13_grid_r2_s3">, !nest.event<"att13_inrel_r2_s3">,
        !nest.event<"att13_out_r2_s3">)
    %pf_bidx14_r2_s3 = nest.dma.prefetch.async %804 into %block_idx_p0_11
      depends_on(%att12_inrel_r2_s3) : !nest.event<"pf_bidx14_r2_s3">
    %att14_grid_r2_s3, %att14_inrel_r2_s3, %att14_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx14_r2_s3, %att10_out_r2_s3)
      : (
        !nest.event<"att14_grid_r2_s3">, !nest.event<"att14_inrel_r2_s3">,
        !nest.event<"att14_out_r2_s3">)
    %pf_bidx15_r2_s3 = nest.dma.prefetch.async %805 into %block_idx_p1_11
      depends_on(%att13_inrel_r2_s3) : !nest.event<"pf_bidx15_r2_s3">
    %att15_grid_r2_s3, %att15_inrel_r2_s3, %att15_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx15_r2_s3, %att11_out_r2_s3)
      : (
        !nest.event<"att15_grid_r2_s3">, !nest.event<"att15_inrel_r2_s3">,
        !nest.event<"att15_out_r2_s3">)
    %pf_bidx16_r2_s3 = nest.dma.prefetch.async %806 into %block_idx_p0_11
      depends_on(%att14_inrel_r2_s3) : !nest.event<"pf_bidx16_r2_s3">
    %att16_grid_r2_s3, %att16_inrel_r2_s3, %att16_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx16_r2_s3, %att12_out_r2_s3)
      : (
        !nest.event<"att16_grid_r2_s3">, !nest.event<"att16_inrel_r2_s3">,
        !nest.event<"att16_out_r2_s3">)
    %pf_bidx17_r2_s3 = nest.dma.prefetch.async %807 into %block_idx_p1_11
      depends_on(%att15_inrel_r2_s3) : !nest.event<"pf_bidx17_r2_s3">
    %att17_grid_r2_s3, %att17_inrel_r2_s3, %att17_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx17_r2_s3, %att13_out_r2_s3)
      : (
        !nest.event<"att17_grid_r2_s3">, !nest.event<"att17_inrel_r2_s3">,
        !nest.event<"att17_out_r2_s3">)
    %pf_bidx18_r2_s3 = nest.dma.prefetch.async %808 into %block_idx_p0_11
      depends_on(%att16_inrel_r2_s3) : !nest.event<"pf_bidx18_r2_s3">
    %att18_grid_r2_s3, %att18_inrel_r2_s3, %att18_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx18_r2_s3, %att14_out_r2_s3)
      : (
        !nest.event<"att18_grid_r2_s3">, !nest.event<"att18_inrel_r2_s3">,
        !nest.event<"att18_out_r2_s3">)
    %pf_bidx19_r2_s3 = nest.dma.prefetch.async %809 into %block_idx_p1_11
      depends_on(%att17_inrel_r2_s3) : !nest.event<"pf_bidx19_r2_s3">
    %att19_grid_r2_s3, %att19_inrel_r2_s3, %att19_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx19_r2_s3, %att15_out_r2_s3)
      : (
        !nest.event<"att19_grid_r2_s3">, !nest.event<"att19_inrel_r2_s3">,
        !nest.event<"att19_out_r2_s3">)
    %pf_bidx20_r2_s3 = nest.dma.prefetch.async %810 into %block_idx_p0_11
      depends_on(%att18_inrel_r2_s3) : !nest.event<"pf_bidx20_r2_s3">
    %att20_grid_r2_s3, %att20_inrel_r2_s3, %att20_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx20_r2_s3, %att16_out_r2_s3)
      : (
        !nest.event<"att20_grid_r2_s3">, !nest.event<"att20_inrel_r2_s3">,
        !nest.event<"att20_out_r2_s3">)
    %pf_bidx21_r2_s3 = nest.dma.prefetch.async %811 into %block_idx_p1_11
      depends_on(%att19_inrel_r2_s3) : !nest.event<"pf_bidx21_r2_s3">
    %att21_grid_r2_s3, %att21_inrel_r2_s3, %att21_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx21_r2_s3, %att17_out_r2_s3)
      : (
        !nest.event<"att21_grid_r2_s3">, !nest.event<"att21_inrel_r2_s3">,
        !nest.event<"att21_out_r2_s3">)
    %pf_bidx22_r2_s3 = nest.dma.prefetch.async %812 into %block_idx_p0_11
      depends_on(%att20_inrel_r2_s3) : !nest.event<"pf_bidx22_r2_s3">
    %att22_grid_r2_s3, %att22_inrel_r2_s3, %att22_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx22_r2_s3, %att18_out_r2_s3)
      : (
        !nest.event<"att22_grid_r2_s3">, !nest.event<"att22_inrel_r2_s3">,
        !nest.event<"att22_out_r2_s3">)
    %pf_bidx23_r2_s3 = nest.dma.prefetch.async %813 into %block_idx_p1_11
      depends_on(%att21_inrel_r2_s3) : !nest.event<"pf_bidx23_r2_s3">
    %att23_grid_r2_s3, %att23_inrel_r2_s3, %att23_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx23_r2_s3, %att19_out_r2_s3)
      : (
        !nest.event<"att23_grid_r2_s3">, !nest.event<"att23_inrel_r2_s3">,
        !nest.event<"att23_out_r2_s3">)
    %pf_bidx24_r2_s3 = nest.dma.prefetch.async %814 into %block_idx_p0_11
      depends_on(%att22_inrel_r2_s3) : !nest.event<"pf_bidx24_r2_s3">
    %att24_grid_r2_s3, %att24_inrel_r2_s3, %att24_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx24_r2_s3, %att20_out_r2_s3)
      : (
        !nest.event<"att24_grid_r2_s3">, !nest.event<"att24_inrel_r2_s3">,
        !nest.event<"att24_out_r2_s3">)
    %pf_bidx25_r2_s3 = nest.dma.prefetch.async %815 into %block_idx_p1_11
      depends_on(%att23_inrel_r2_s3) : !nest.event<"pf_bidx25_r2_s3">
    %att25_grid_r2_s3, %att25_inrel_r2_s3, %att25_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx25_r2_s3, %att21_out_r2_s3)
      : (
        !nest.event<"att25_grid_r2_s3">, !nest.event<"att25_inrel_r2_s3">,
        !nest.event<"att25_out_r2_s3">)
    %pf_bidx26_r2_s3 = nest.dma.prefetch.async %816 into %block_idx_p0_11
      depends_on(%att24_inrel_r2_s3) : !nest.event<"pf_bidx26_r2_s3">
    %att26_grid_r2_s3, %att26_inrel_r2_s3, %att26_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx26_r2_s3, %att22_out_r2_s3)
      : (
        !nest.event<"att26_grid_r2_s3">, !nest.event<"att26_inrel_r2_s3">,
        !nest.event<"att26_out_r2_s3">)
    %pf_bidx27_r2_s3 = nest.dma.prefetch.async %817 into %block_idx_p1_11
      depends_on(%att25_inrel_r2_s3) : !nest.event<"pf_bidx27_r2_s3">
    %att27_grid_r2_s3, %att27_inrel_r2_s3, %att27_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx27_r2_s3, %att23_out_r2_s3)
      : (
        !nest.event<"att27_grid_r2_s3">, !nest.event<"att27_inrel_r2_s3">,
        !nest.event<"att27_out_r2_s3">)
    %pf_bidx28_r2_s3 = nest.dma.prefetch.async %818 into %block_idx_p0_11
      depends_on(%att26_inrel_r2_s3) : !nest.event<"pf_bidx28_r2_s3">
    %att28_grid_r2_s3, %att28_inrel_r2_s3, %att28_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx28_r2_s3, %att24_out_r2_s3)
      : (
        !nest.event<"att28_grid_r2_s3">, !nest.event<"att28_inrel_r2_s3">,
        !nest.event<"att28_out_r2_s3">)
    %pf_bidx29_r2_s3 = nest.dma.prefetch.async %819 into %block_idx_p1_11
      depends_on(%att27_inrel_r2_s3) : !nest.event<"pf_bidx29_r2_s3">
    %att29_grid_r2_s3, %att29_inrel_r2_s3, %att29_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx29_r2_s3, %att25_out_r2_s3)
      : (
        !nest.event<"att29_grid_r2_s3">, !nest.event<"att29_inrel_r2_s3">,
        !nest.event<"att29_out_r2_s3">)
    %pf_bidx30_r2_s3 = nest.dma.prefetch.async %820 into %block_idx_p0_11
      depends_on(%att28_inrel_r2_s3) : !nest.event<"pf_bidx30_r2_s3">
    %att30_grid_r2_s3, %att30_inrel_r2_s3, %att30_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx30_r2_s3, %att26_out_r2_s3)
      : (
        !nest.event<"att30_grid_r2_s3">, !nest.event<"att30_inrel_r2_s3">,
        !nest.event<"att30_out_r2_s3">)
    %pf_bidx31_r2_s3 = nest.dma.prefetch.async %821 into %block_idx_p1_11
      depends_on(%att29_inrel_r2_s3) : !nest.event<"pf_bidx31_r2_s3">
    %att31_grid_r2_s3, %att31_inrel_r2_s3, %att31_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx31_r2_s3, %att27_out_r2_s3)
      : (
        !nest.event<"att31_grid_r2_s3">, !nest.event<"att31_inrel_r2_s3">,
        !nest.event<"att31_out_r2_s3">)
    %pf_bidx32_r2_s3 = nest.dma.prefetch.async %822 into %block_idx_p0_11
      depends_on(%att30_inrel_r2_s3) : !nest.event<"pf_bidx32_r2_s3">
    %att32_grid_r2_s3, %att32_inrel_r2_s3, %att32_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx32_r2_s3, %att28_out_r2_s3)
      : (
        !nest.event<"att32_grid_r2_s3">, !nest.event<"att32_inrel_r2_s3">,
        !nest.event<"att32_out_r2_s3">)
    %pf_bidx33_r2_s3 = nest.dma.prefetch.async %823 into %block_idx_p1_11
      depends_on(%att31_inrel_r2_s3) : !nest.event<"pf_bidx33_r2_s3">
    %att33_grid_r2_s3, %att33_inrel_r2_s3, %att33_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx33_r2_s3, %att29_out_r2_s3)
      : (
        !nest.event<"att33_grid_r2_s3">, !nest.event<"att33_inrel_r2_s3">,
        !nest.event<"att33_out_r2_s3">)
    %pf_bidx34_r2_s3 = nest.dma.prefetch.async %824 into %block_idx_p0_11
      depends_on(%att32_inrel_r2_s3) : !nest.event<"pf_bidx34_r2_s3">
    %att34_grid_r2_s3, %att34_inrel_r2_s3, %att34_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx34_r2_s3, %att30_out_r2_s3)
      : (
        !nest.event<"att34_grid_r2_s3">, !nest.event<"att34_inrel_r2_s3">,
        !nest.event<"att34_out_r2_s3">)
    %pf_bidx35_r2_s3 = nest.dma.prefetch.async %825 into %block_idx_p1_11
      depends_on(%att33_inrel_r2_s3) : !nest.event<"pf_bidx35_r2_s3">
    %att35_grid_r2_s3, %att35_inrel_r2_s3, %att35_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx35_r2_s3, %att31_out_r2_s3)
      : (
        !nest.event<"att35_grid_r2_s3">, !nest.event<"att35_inrel_r2_s3">,
        !nest.event<"att35_out_r2_s3">)
    %pf_bidx36_r2_s3 = nest.dma.prefetch.async %826 into %block_idx_p0_11
      depends_on(%att34_inrel_r2_s3) : !nest.event<"pf_bidx36_r2_s3">
    %att36_grid_r2_s3, %att36_inrel_r2_s3, %att36_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx36_r2_s3, %att32_out_r2_s3)
      : (
        !nest.event<"att36_grid_r2_s3">, !nest.event<"att36_inrel_r2_s3">,
        !nest.event<"att36_out_r2_s3">)
    %pf_bidx37_r2_s3 = nest.dma.prefetch.async %827 into %block_idx_p1_11
      depends_on(%att35_inrel_r2_s3) : !nest.event<"pf_bidx37_r2_s3">
    %att37_grid_r2_s3, %att37_inrel_r2_s3, %att37_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx37_r2_s3, %att33_out_r2_s3)
      : (
        !nest.event<"att37_grid_r2_s3">, !nest.event<"att37_inrel_r2_s3">,
        !nest.event<"att37_out_r2_s3">)
    %pf_bidx38_r2_s3 = nest.dma.prefetch.async %828 into %block_idx_p0_11
      depends_on(%att36_inrel_r2_s3) : !nest.event<"pf_bidx38_r2_s3">
    %att38_grid_r2_s3, %att38_inrel_r2_s3, %att38_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx38_r2_s3, %att34_out_r2_s3)
      : (
        !nest.event<"att38_grid_r2_s3">, !nest.event<"att38_inrel_r2_s3">,
        !nest.event<"att38_out_r2_s3">)
    %pf_bidx39_r2_s3 = nest.dma.prefetch.async %829 into %block_idx_p1_11
      depends_on(%att37_inrel_r2_s3) : !nest.event<"pf_bidx39_r2_s3">
    %att39_grid_r2_s3, %att39_inrel_r2_s3, %att39_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx39_r2_s3, %att35_out_r2_s3)
      : (
        !nest.event<"att39_grid_r2_s3">, !nest.event<"att39_inrel_r2_s3">,
        !nest.event<"att39_out_r2_s3">)
    %pf_bidx40_r2_s3 = nest.dma.prefetch.async %830 into %block_idx_p0_11
      depends_on(%att38_inrel_r2_s3) : !nest.event<"pf_bidx40_r2_s3">
    %att40_grid_r2_s3, %att40_inrel_r2_s3, %att40_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx40_r2_s3, %att36_out_r2_s3)
      : (
        !nest.event<"att40_grid_r2_s3">, !nest.event<"att40_inrel_r2_s3">,
        !nest.event<"att40_out_r2_s3">)
    %pf_bidx41_r2_s3 = nest.dma.prefetch.async %831 into %block_idx_p1_11
      depends_on(%att39_inrel_r2_s3) : !nest.event<"pf_bidx41_r2_s3">
    %att41_grid_r2_s3, %att41_inrel_r2_s3, %att41_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx41_r2_s3, %att37_out_r2_s3)
      : (
        !nest.event<"att41_grid_r2_s3">, !nest.event<"att41_inrel_r2_s3">,
        !nest.event<"att41_out_r2_s3">)
    %pf_bidx42_r2_s3 = nest.dma.prefetch.async %832 into %block_idx_p0_11
      depends_on(%att40_inrel_r2_s3) : !nest.event<"pf_bidx42_r2_s3">
    %att42_grid_r2_s3, %att42_inrel_r2_s3, %att42_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx42_r2_s3, %att38_out_r2_s3)
      : (
        !nest.event<"att42_grid_r2_s3">, !nest.event<"att42_inrel_r2_s3">,
        !nest.event<"att42_out_r2_s3">)
    %pf_bidx43_r2_s3 = nest.dma.prefetch.async %833 into %block_idx_p1_11
      depends_on(%att41_inrel_r2_s3) : !nest.event<"pf_bidx43_r2_s3">
    %att43_grid_r2_s3, %att43_inrel_r2_s3, %att43_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx43_r2_s3, %att39_out_r2_s3)
      : (
        !nest.event<"att43_grid_r2_s3">, !nest.event<"att43_inrel_r2_s3">,
        !nest.event<"att43_out_r2_s3">)
    %pf_bidx44_r2_s3 = nest.dma.prefetch.async %834 into %block_idx_p0_11
      depends_on(%att42_inrel_r2_s3) : !nest.event<"pf_bidx44_r2_s3">
    %att44_grid_r2_s3, %att44_inrel_r2_s3, %att44_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_bidx44_r2_s3, %att40_out_r2_s3)
      : (
        !nest.event<"att44_grid_r2_s3">, !nest.event<"att44_inrel_r2_s3">,
        !nest.event<"att44_out_r2_s3">)
    %pf_bidx45_r2_s3 = nest.dma.prefetch.async %835 into %block_idx_p1_11
      depends_on(%att43_inrel_r2_s3) : !nest.event<"pf_bidx45_r2_s3">
    %att45_grid_r2_s3, %att45_inrel_r2_s3, %att45_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p1_12, %acc_p1_12) outs(%state_p1_12, %acc_p1_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p1_r2_s3, %pf_acc_p1_r2_s3, %pf_bidx45_r2_s3, %att41_out_r2_s3)
      : (
        !nest.event<"att45_grid_r2_s3">, !nest.event<"att45_inrel_r2_s3">,
        !nest.event<"att45_out_r2_s3">)
    %pf_bidx46_r2_s3 = nest.dma.prefetch.async %836 into %block_idx_p0_11
      depends_on(%att44_inrel_r2_s3) : !nest.event<"pf_bidx46_r2_s3">
    %att46_grid_r2_s3, %att46_inrel_r2_s3, %att46_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12)
      ins(%block_idx_p0_11, %q_l2_26, %state_p2_12, %acc_p2_12) outs(%state_p2_12, %acc_p2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p2_r2_s3, %pf_acc_p2_r2_s3, %pf_bidx46_r2_s3, %att42_out_r2_s3)
      : (
        !nest.event<"att46_grid_r2_s3">, !nest.event<"att46_inrel_r2_s3">,
        !nest.event<"att46_out_r2_s3">)
    %pf_bidx47_r2_s3 = nest.dma.prefetch.async %837 into %block_idx_p1_11
      depends_on(%att45_inrel_r2_s3) : !nest.event<"pf_bidx47_r2_s3">
    %att47_grid_r2_s3, %att47_inrel_r2_s3, %att47_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t16_step_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12)
      ins(%block_idx_p1_11, %q_l2_26, %state_p3_12, %acc_p3_12) outs(%state_p3_12, %acc_p3_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p3_r2_s3, %pf_acc_p3_r2_s3, %pf_bidx47_r2_s3, %att43_out_r2_s3)
      : (
        !nest.event<"att47_grid_r2_s3">, !nest.event<"att47_inrel_r2_s3">,
        !nest.event<"att47_out_r2_s3">)
    %att48_grid_r2_s3, %att48_inrel_r2_s3, %att48_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_t3_final_r2 l1_mode = 1 tasks(%838) globals(%776)
      bindings(%append_idx_11, %q_l2_26, %state_p0_12, %acc_p0_12)
      ins(%append_idx_11, %q_l2_26, %state_p0_12, %acc_p0_12) outs(%state_p0_12, %acc_p0_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(
        %pf_q_r2_s3, %pf_state_p0_r2_s3, %pf_acc_p0_r2_s3, %pf_aidx_r2_s3, %append_grid_r2_s3,
        %att44_out_r2_s3)
      : (
        !nest.event<"att48_grid_r2_s3">, !nest.event<"att48_inrel_r2_s3">,
        !nest.event<"att48_out_r2_s3">)
    %merge_grid_r2_s3, %merge_inrel_r2_s3, %merge_out_r2_s3 =
      nest.dispatch.tasks.async @paged_attention_merge_p4 l1_mode = 1 tasks(%838) globals()
      bindings(
        %state_p0_12, %state_p1_12, %state_p2_12, %state_p3_12, %acc_p0_12, %acc_p1_12,
        %acc_p2_12, %acc_p3_12, %out_l2_12)
      ins(
        %state_p0_12, %state_p1_12, %state_p2_12, %state_p3_12, %acc_p0_12, %acc_p1_12,
        %acc_p2_12, %acc_p3_12)
      outs(%out_l2_12)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att48_out_r2_s3, %att45_out_r2_s3, %att46_out_r2_s3, %att47_out_r2_s3) : (
        !nest.event<"merge_grid_r2_s3">, !nest.event<"merge_inrel_r2_s3">,
        !nest.event<"merge_out_r2_s3">)
    %out_store_r2_s3 = nest.dma.store.async %out_l2_12 into %789 depends_on(%merge_out_r2_s3)
      : !nest.event<"out_store_r2_s3">
    nest.await %append_grid_r2_s3, %att48_grid_r2_s3, %merge_grid_r2_s3, %out_store_r2_s3
    nest.release %block_idx_p0_11 depends_on(
      %pf_bidx0_r2_s3, %pf_bidx2_r2_s3, %pf_bidx4_r2_s3, %pf_bidx6_r2_s3, %pf_bidx8_r2_s3,
      %pf_bidx10_r2_s3, %pf_bidx12_r2_s3, %pf_bidx14_r2_s3, %pf_bidx16_r2_s3, %pf_bidx18_r2_s3,
      %pf_bidx20_r2_s3, %pf_bidx22_r2_s3, %pf_bidx24_r2_s3, %pf_bidx26_r2_s3, %pf_bidx28_r2_s3,
      %pf_bidx30_r2_s3, %pf_bidx32_r2_s3, %pf_bidx34_r2_s3, %pf_bidx36_r2_s3, %pf_bidx38_r2_s3,
      %pf_bidx40_r2_s3, %pf_bidx42_r2_s3, %pf_bidx44_r2_s3, %pf_bidx46_r2_s3, %att0_inrel_r2_s3,
      %att2_inrel_r2_s3, %att4_inrel_r2_s3, %att6_inrel_r2_s3, %att8_inrel_r2_s3,
      %att10_inrel_r2_s3, %att12_inrel_r2_s3, %att14_inrel_r2_s3, %att16_inrel_r2_s3,
      %att18_inrel_r2_s3, %att20_inrel_r2_s3, %att22_inrel_r2_s3, %att24_inrel_r2_s3,
      %att26_inrel_r2_s3, %att28_inrel_r2_s3, %att30_inrel_r2_s3, %att32_inrel_r2_s3,
      %att34_inrel_r2_s3, %att36_inrel_r2_s3, %att38_inrel_r2_s3, %att40_inrel_r2_s3,
      %att42_inrel_r2_s3, %att44_inrel_r2_s3, %att46_inrel_r2_s3)
    nest.release %block_idx_p1_11 depends_on(
      %pf_bidx1_r2_s3, %pf_bidx3_r2_s3, %pf_bidx5_r2_s3, %pf_bidx7_r2_s3, %pf_bidx9_r2_s3,
      %pf_bidx11_r2_s3, %pf_bidx13_r2_s3, %pf_bidx15_r2_s3, %pf_bidx17_r2_s3, %pf_bidx19_r2_s3,
      %pf_bidx21_r2_s3, %pf_bidx23_r2_s3, %pf_bidx25_r2_s3, %pf_bidx27_r2_s3, %pf_bidx29_r2_s3,
      %pf_bidx31_r2_s3, %pf_bidx33_r2_s3, %pf_bidx35_r2_s3, %pf_bidx37_r2_s3, %pf_bidx39_r2_s3,
      %pf_bidx41_r2_s3, %pf_bidx43_r2_s3, %pf_bidx45_r2_s3, %pf_bidx47_r2_s3, %att1_inrel_r2_s3,
      %att3_inrel_r2_s3, %att5_inrel_r2_s3, %att7_inrel_r2_s3, %att9_inrel_r2_s3,
      %att11_inrel_r2_s3, %att13_inrel_r2_s3, %att15_inrel_r2_s3, %att17_inrel_r2_s3,
      %att19_inrel_r2_s3, %att21_inrel_r2_s3, %att23_inrel_r2_s3, %att25_inrel_r2_s3,
      %att27_inrel_r2_s3, %att29_inrel_r2_s3, %att31_inrel_r2_s3, %att33_inrel_r2_s3,
      %att35_inrel_r2_s3, %att37_inrel_r2_s3, %att39_inrel_r2_s3, %att41_inrel_r2_s3,
      %att43_inrel_r2_s3, %att45_inrel_r2_s3, %att47_inrel_r2_s3)
    nest.release %k_new_11 depends_on(%pf_k_r2_s3, %append_inrel_r2_s3)
    nest.release %v_new_11 depends_on(%pf_v_r2_s3, %append_inrel_r2_s3)
    nest.release %append_idx_11 depends_on(%pf_aidx_r2_s3, %append_inrel_r2_s3, %att48_inrel_r2_s3)
    nest.release %q_l2_26 depends_on(
      %pf_q_r2_s3, %att0_inrel_r2_s3, %att1_inrel_r2_s3, %att2_inrel_r2_s3, %att3_inrel_r2_s3,
      %att4_inrel_r2_s3, %att5_inrel_r2_s3, %att6_inrel_r2_s3, %att7_inrel_r2_s3,
      %att8_inrel_r2_s3, %att9_inrel_r2_s3, %att10_inrel_r2_s3, %att11_inrel_r2_s3,
      %att12_inrel_r2_s3, %att13_inrel_r2_s3, %att14_inrel_r2_s3, %att15_inrel_r2_s3,
      %att16_inrel_r2_s3, %att17_inrel_r2_s3, %att18_inrel_r2_s3, %att19_inrel_r2_s3,
      %att20_inrel_r2_s3, %att21_inrel_r2_s3, %att22_inrel_r2_s3, %att23_inrel_r2_s3,
      %att24_inrel_r2_s3, %att25_inrel_r2_s3, %att26_inrel_r2_s3, %att27_inrel_r2_s3,
      %att28_inrel_r2_s3, %att29_inrel_r2_s3, %att30_inrel_r2_s3, %att31_inrel_r2_s3,
      %att32_inrel_r2_s3, %att33_inrel_r2_s3, %att34_inrel_r2_s3, %att35_inrel_r2_s3,
      %att36_inrel_r2_s3, %att37_inrel_r2_s3, %att38_inrel_r2_s3, %att39_inrel_r2_s3,
      %att40_inrel_r2_s3, %att41_inrel_r2_s3, %att42_inrel_r2_s3, %att43_inrel_r2_s3,
      %att44_inrel_r2_s3, %att45_inrel_r2_s3, %att46_inrel_r2_s3, %att47_inrel_r2_s3,
      %att48_inrel_r2_s3)
    nest.release %state_p0_12 depends_on(
      %pf_state_p0_r2_s3, %att0_inrel_r2_s3, %att4_inrel_r2_s3, %att8_inrel_r2_s3,
      %att12_inrel_r2_s3, %att16_inrel_r2_s3, %att20_inrel_r2_s3, %att24_inrel_r2_s3,
      %att28_inrel_r2_s3, %att32_inrel_r2_s3, %att36_inrel_r2_s3, %att40_inrel_r2_s3,
      %att44_inrel_r2_s3, %att48_inrel_r2_s3, %att0_out_r2_s3, %att4_out_r2_s3, %att8_out_r2_s3,
      %att12_out_r2_s3, %att16_out_r2_s3, %att20_out_r2_s3, %att24_out_r2_s3, %att28_out_r2_s3,
      %att32_out_r2_s3, %att36_out_r2_s3, %att40_out_r2_s3, %att44_out_r2_s3, %att48_out_r2_s3,
      %merge_inrel_r2_s3)
    nest.release %acc_p0_12 depends_on(
      %pf_acc_p0_r2_s3, %att0_inrel_r2_s3, %att4_inrel_r2_s3, %att8_inrel_r2_s3,
      %att12_inrel_r2_s3, %att16_inrel_r2_s3, %att20_inrel_r2_s3, %att24_inrel_r2_s3,
      %att28_inrel_r2_s3, %att32_inrel_r2_s3, %att36_inrel_r2_s3, %att40_inrel_r2_s3,
      %att44_inrel_r2_s3, %att48_inrel_r2_s3, %att0_out_r2_s3, %att4_out_r2_s3, %att8_out_r2_s3,
      %att12_out_r2_s3, %att16_out_r2_s3, %att20_out_r2_s3, %att24_out_r2_s3, %att28_out_r2_s3,
      %att32_out_r2_s3, %att36_out_r2_s3, %att40_out_r2_s3, %att44_out_r2_s3, %att48_out_r2_s3,
      %merge_inrel_r2_s3)
    nest.release %state_p1_12 depends_on(
      %pf_state_p1_r2_s3, %att1_inrel_r2_s3, %att5_inrel_r2_s3, %att9_inrel_r2_s3,
      %att13_inrel_r2_s3, %att17_inrel_r2_s3, %att21_inrel_r2_s3, %att25_inrel_r2_s3,
      %att29_inrel_r2_s3, %att33_inrel_r2_s3, %att37_inrel_r2_s3, %att41_inrel_r2_s3,
      %att45_inrel_r2_s3, %att1_out_r2_s3, %att5_out_r2_s3, %att9_out_r2_s3, %att13_out_r2_s3,
      %att17_out_r2_s3, %att21_out_r2_s3, %att25_out_r2_s3, %att29_out_r2_s3, %att33_out_r2_s3,
      %att37_out_r2_s3, %att41_out_r2_s3, %att45_out_r2_s3, %merge_inrel_r2_s3)
    nest.release %acc_p1_12 depends_on(
      %pf_acc_p1_r2_s3, %att1_inrel_r2_s3, %att5_inrel_r2_s3, %att9_inrel_r2_s3,
      %att13_inrel_r2_s3, %att17_inrel_r2_s3, %att21_inrel_r2_s3, %att25_inrel_r2_s3,
      %att29_inrel_r2_s3, %att33_inrel_r2_s3, %att37_inrel_r2_s3, %att41_inrel_r2_s3,
      %att45_inrel_r2_s3, %att1_out_r2_s3, %att5_out_r2_s3, %att9_out_r2_s3, %att13_out_r2_s3,
      %att17_out_r2_s3, %att21_out_r2_s3, %att25_out_r2_s3, %att29_out_r2_s3, %att33_out_r2_s3,
      %att37_out_r2_s3, %att41_out_r2_s3, %att45_out_r2_s3, %merge_inrel_r2_s3)
    nest.release %state_p2_12 depends_on(
      %pf_state_p2_r2_s3, %att2_inrel_r2_s3, %att6_inrel_r2_s3, %att10_inrel_r2_s3,
      %att14_inrel_r2_s3, %att18_inrel_r2_s3, %att22_inrel_r2_s3, %att26_inrel_r2_s3,
      %att30_inrel_r2_s3, %att34_inrel_r2_s3, %att38_inrel_r2_s3, %att42_inrel_r2_s3,
      %att46_inrel_r2_s3, %att2_out_r2_s3, %att6_out_r2_s3, %att10_out_r2_s3, %att14_out_r2_s3,
      %att18_out_r2_s3, %att22_out_r2_s3, %att26_out_r2_s3, %att30_out_r2_s3, %att34_out_r2_s3,
      %att38_out_r2_s3, %att42_out_r2_s3, %att46_out_r2_s3, %merge_inrel_r2_s3)
    nest.release %acc_p2_12 depends_on(
      %pf_acc_p2_r2_s3, %att2_inrel_r2_s3, %att6_inrel_r2_s3, %att10_inrel_r2_s3,
      %att14_inrel_r2_s3, %att18_inrel_r2_s3, %att22_inrel_r2_s3, %att26_inrel_r2_s3,
      %att30_inrel_r2_s3, %att34_inrel_r2_s3, %att38_inrel_r2_s3, %att42_inrel_r2_s3,
      %att46_inrel_r2_s3, %att2_out_r2_s3, %att6_out_r2_s3, %att10_out_r2_s3, %att14_out_r2_s3,
      %att18_out_r2_s3, %att22_out_r2_s3, %att26_out_r2_s3, %att30_out_r2_s3, %att34_out_r2_s3,
      %att38_out_r2_s3, %att42_out_r2_s3, %att46_out_r2_s3, %merge_inrel_r2_s3)
    nest.release %state_p3_12 depends_on(
      %pf_state_p3_r2_s3, %att3_inrel_r2_s3, %att7_inrel_r2_s3, %att11_inrel_r2_s3,
      %att15_inrel_r2_s3, %att19_inrel_r2_s3, %att23_inrel_r2_s3, %att27_inrel_r2_s3,
      %att31_inrel_r2_s3, %att35_inrel_r2_s3, %att39_inrel_r2_s3, %att43_inrel_r2_s3,
      %att47_inrel_r2_s3, %att3_out_r2_s3, %att7_out_r2_s3, %att11_out_r2_s3, %att15_out_r2_s3,
      %att19_out_r2_s3, %att23_out_r2_s3, %att27_out_r2_s3, %att31_out_r2_s3, %att35_out_r2_s3,
      %att39_out_r2_s3, %att43_out_r2_s3, %att47_out_r2_s3, %merge_inrel_r2_s3)
    nest.release %acc_p3_12 depends_on(
      %pf_acc_p3_r2_s3, %att3_inrel_r2_s3, %att7_inrel_r2_s3, %att11_inrel_r2_s3,
      %att15_inrel_r2_s3, %att19_inrel_r2_s3, %att23_inrel_r2_s3, %att27_inrel_r2_s3,
      %att31_inrel_r2_s3, %att35_inrel_r2_s3, %att39_inrel_r2_s3, %att43_inrel_r2_s3,
      %att47_inrel_r2_s3, %att3_out_r2_s3, %att7_out_r2_s3, %att11_out_r2_s3, %att15_out_r2_s3,
      %att19_out_r2_s3, %att23_out_r2_s3, %att27_out_r2_s3, %att31_out_r2_s3, %att35_out_r2_s3,
      %att39_out_r2_s3, %att43_out_r2_s3, %att47_out_r2_s3, %merge_inrel_r2_s3)
    nest.release %out_l2_12 depends_on(%merge_out_r2_s3, %out_store_r2_s3)
    nest.return
  }
  nexus.program @paged_attention_decode_pipeline(
    %POOL_12: !nest.global_memref<128x8224xbf16>, %BLOCK_TABLE_12: !nest.global_memref<147xi32>,
    %LENGTHS: !nest.global_memref<3xi32>, %APPEND_IDS_12: !nest.global_memref<12xi32>,
    %Q_IN_12: !nest.global_memref<3x4x4x4x64xbf16>,
    %K_NEW_12: !nest.global_memref<3x4x4x1x64xbf16>,
    %V_NEW_12: !nest.global_memref<3x4x4x1x64xbf16>,
    %S_INIT_12: !nest.global_memref<3x4x4x4x4x2xf32>,
    %O_INIT_12: !nest.global_memref<3x4x4x4x4x64xf32>,
    %OUT_12: !nest.global_memref<3x4x4x4x64xf32>) {
    %prepared_r0_s0 = nexus.host.call.async "prepare_r0_s0" bindings(%APPEND_IDS_12)
      accesses = [{offset = 0, bytes = 4, mode = "write"}] scopes = ["owner_0"]
      : !nexus.event<"prepared_r0_s0">
    %step_done_r0_s0 =
      nexus.submit_context.async @step_r0_s0(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r0_s0) : !nexus.event<"step_done_r0_s0">
    %committed_r0_s0 = nexus.host.call.async "commit_r0_s0" bindings(%LENGTHS)
      accesses = [{offset = 0, bytes = 4, mode = "write"}] depends_on(%step_done_r0_s0)
      : !nexus.event<"committed_r0_s0">
    %prepared_r0_s1 = nexus.host.call.async "prepare_r0_s1" bindings(%APPEND_IDS_12)
      accesses = [{offset = 4, bytes = 4, mode = "write"}] scopes = ["owner_0"]
      depends_on(%committed_r0_s0) : !nexus.event<"prepared_r0_s1">
    %step_done_r0_s1 =
      nexus.submit_context.async @step_r0_s1(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r0_s1) : !nexus.event<"step_done_r0_s1">
    %committed_r0_s1 = nexus.host.call.async "commit_r0_s1" bindings(%BLOCK_TABLE_12, %LENGTHS)
      accesses = [{offset = 64, bytes = 4, mode = "write"}, {offset = 0, bytes = 4, mode = "write"}]
      depends_on(%step_done_r0_s1) : !nexus.event<"committed_r0_s1">
    %prepared_r0_s2 = nexus.host.call.async "prepare_r0_s2" bindings(%APPEND_IDS_12)
      accesses = [{offset = 8, bytes = 4, mode = "write"}] scopes = ["owner_0"]
      depends_on(%committed_r0_s1) : !nexus.event<"prepared_r0_s2">
    %step_done_r0_s2 =
      nexus.submit_context.async @step_r0_s2(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r0_s2) : !nexus.event<"step_done_r0_s2">
    %committed_r0_s2 = nexus.host.call.async "commit_r0_s2" bindings(%LENGTHS)
      accesses = [{offset = 0, bytes = 4, mode = "write"}] depends_on(%step_done_r0_s2)
      : !nexus.event<"committed_r0_s2">
    %prepared_r0_s3 = nexus.host.call.async "prepare_r0_s3" bindings(%APPEND_IDS_12)
      accesses = [{offset = 12, bytes = 4, mode = "write"}] scopes = ["owner_0"]
      depends_on(%committed_r0_s2) : !nexus.event<"prepared_r0_s3">
    %step_done_r0_s3 =
      nexus.submit_context.async @step_r0_s3(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r0_s3) : !nexus.event<"step_done_r0_s3">
    %committed_r0_s3 = nexus.host.call.async "commit_r0_s3" bindings(%LENGTHS)
      accesses = [{offset = 0, bytes = 4, mode = "write"}] depends_on(%step_done_r0_s3)
      : !nexus.event<"committed_r0_s3">
    %released_r0 = nexus.host.call.async "release_r0" bindings(%BLOCK_TABLE_12, %LENGTHS)
      accesses = [{offset = 0, bytes = 68, mode = "write"}, {offset = 0, bytes = 4, mode = "write"}]
      depends_on(%committed_r0_s3) : !nexus.event<"released_r0">
    %prepared_r1_s0 = nexus.host.call.async "prepare_r1_s0" bindings(%APPEND_IDS_12)
      accesses = [{offset = 16, bytes = 4, mode = "write"}] scopes = ["owner_1"]
      : !nexus.event<"prepared_r1_s0">
    %step_done_r1_s0 =
      nexus.submit_context.async @step_r1_s0(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r1_s0) : !nexus.event<"step_done_r1_s0">
    %committed_r1_s0 = nexus.host.call.async "commit_r1_s0" bindings(%LENGTHS)
      accesses = [{offset = 4, bytes = 4, mode = "write"}] depends_on(%step_done_r1_s0)
      : !nexus.event<"committed_r1_s0">
    %prepared_r1_s1 = nexus.host.call.async "prepare_r1_s1" bindings(%APPEND_IDS_12)
      accesses = [{offset = 20, bytes = 4, mode = "write"}] scopes = ["owner_1"]
      depends_on(%committed_r1_s0) : !nexus.event<"prepared_r1_s1">
    %step_done_r1_s1 =
      nexus.submit_context.async @step_r1_s1(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r1_s1) : !nexus.event<"step_done_r1_s1">
    %committed_r1_s1 = nexus.host.call.async "commit_r1_s1" bindings(%BLOCK_TABLE_12, %LENGTHS)
      accesses = [
        {offset = 128, bytes = 4, mode = "write"}, {offset = 4, bytes = 4, mode = "write"}]
      depends_on(%step_done_r1_s1) : !nexus.event<"committed_r1_s1">
    %prepared_r1_s2 = nexus.host.call.async "prepare_r1_s2" bindings(%APPEND_IDS_12)
      accesses = [{offset = 24, bytes = 4, mode = "write"}] scopes = ["owner_1"]
      depends_on(%committed_r1_s1) : !nexus.event<"prepared_r1_s2">
    %step_done_r1_s2 =
      nexus.submit_context.async @step_r1_s2(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r1_s2) : !nexus.event<"step_done_r1_s2">
    %committed_r1_s2 = nexus.host.call.async "commit_r1_s2" bindings(%LENGTHS)
      accesses = [{offset = 4, bytes = 4, mode = "write"}] depends_on(%step_done_r1_s2)
      : !nexus.event<"committed_r1_s2">
    %prepared_r1_s3 = nexus.host.call.async "prepare_r1_s3" bindings(%APPEND_IDS_12)
      accesses = [{offset = 28, bytes = 4, mode = "write"}] scopes = ["owner_1"]
      depends_on(%committed_r1_s2) : !nexus.event<"prepared_r1_s3">
    %step_done_r1_s3 =
      nexus.submit_context.async @step_r1_s3(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r1_s3) : !nexus.event<"step_done_r1_s3">
    %committed_r1_s3 = nexus.host.call.async "commit_r1_s3" bindings(%LENGTHS)
      accesses = [{offset = 4, bytes = 4, mode = "write"}] depends_on(%step_done_r1_s3)
      : !nexus.event<"committed_r1_s3">
    %released_r1 = nexus.host.call.async "release_r1" bindings(%BLOCK_TABLE_12, %LENGTHS)
      accesses = [
        {offset = 0, bytes = 132, mode = "write"}, {offset = 4, bytes = 4, mode = "write"}]
      depends_on(%committed_r1_s3) : !nexus.event<"released_r1">
    %prepared_r2_s0 = nexus.host.call.async "prepare_r2_s0" bindings(%APPEND_IDS_12)
      accesses = [{offset = 32, bytes = 4, mode = "write"}] scopes = ["owner_2"]
      : !nexus.event<"prepared_r2_s0">
    %step_done_r2_s0 =
      nexus.submit_context.async @step_r2_s0(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r2_s0) : !nexus.event<"step_done_r2_s0">
    %committed_r2_s0 = nexus.host.call.async "commit_r2_s0" bindings(%LENGTHS)
      accesses = [{offset = 8, bytes = 4, mode = "write"}] depends_on(%step_done_r2_s0)
      : !nexus.event<"committed_r2_s0">
    %prepared_r2_s1 = nexus.host.call.async "prepare_r2_s1" bindings(%APPEND_IDS_12)
      accesses = [{offset = 36, bytes = 4, mode = "write"}] scopes = ["owner_2"]
      depends_on(%committed_r2_s0) : !nexus.event<"prepared_r2_s1">
    %step_done_r2_s1 =
      nexus.submit_context.async @step_r2_s1(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r2_s1) : !nexus.event<"step_done_r2_s1">
    %committed_r2_s1 = nexus.host.call.async "commit_r2_s1" bindings(%BLOCK_TABLE_12, %LENGTHS)
      accesses = [
        {offset = 192, bytes = 4, mode = "write"}, {offset = 8, bytes = 4, mode = "write"}]
      depends_on(%step_done_r2_s1) : !nexus.event<"committed_r2_s1">
    %prepared_r2_s2 = nexus.host.call.async "prepare_r2_s2" bindings(%APPEND_IDS_12)
      accesses = [{offset = 40, bytes = 4, mode = "write"}] scopes = ["owner_2"]
      depends_on(%committed_r2_s1) : !nexus.event<"prepared_r2_s2">
    %step_done_r2_s2 =
      nexus.submit_context.async @step_r2_s2(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r2_s2) : !nexus.event<"step_done_r2_s2">
    %committed_r2_s2 = nexus.host.call.async "commit_r2_s2" bindings(%LENGTHS)
      accesses = [{offset = 8, bytes = 4, mode = "write"}] depends_on(%step_done_r2_s2)
      : !nexus.event<"committed_r2_s2">
    %prepared_r2_s3 = nexus.host.call.async "prepare_r2_s3" bindings(%APPEND_IDS_12)
      accesses = [{offset = 44, bytes = 4, mode = "write"}] scopes = ["owner_2"]
      depends_on(%committed_r2_s2) : !nexus.event<"prepared_r2_s3">
    %step_done_r2_s3 =
      nexus.submit_context.async @step_r2_s3(
        %POOL_12, %BLOCK_TABLE_12, %APPEND_IDS_12, %Q_IN_12, %K_NEW_12, %V_NEW_12, %S_INIT_12,
        %O_INIT_12, %OUT_12)
      depends_on(%prepared_r2_s3) : !nexus.event<"step_done_r2_s3">
    %committed_r2_s3 = nexus.host.call.async "commit_r2_s3" bindings(%LENGTHS)
      accesses = [{offset = 8, bytes = 4, mode = "write"}] depends_on(%step_done_r2_s3)
      : !nexus.event<"committed_r2_s3">
    %released_r2 = nexus.host.call.async "release_r2" bindings(%BLOCK_TABLE_12, %LENGTHS)
      accesses = [
        {offset = 0, bytes = 196, mode = "write"}, {offset = 8, bytes = 4, mode = "write"}]
      depends_on(%committed_r2_s3) : !nexus.event<"released_r2">
    nexus.await %step_done_r0_s0, %step_done_r0_s1, %step_done_r0_s2, %step_done_r0_s3, %released_r0
      , %step_done_r1_s0, %step_done_r1_s1, %step_done_r1_s2, %step_done_r1_s3, %released_r1,
      %step_done_r2_s0, %step_done_r2_s1, %step_done_r2_s2, %step_done_r2_s3, %released_r2
    nexus.return
  }
}
