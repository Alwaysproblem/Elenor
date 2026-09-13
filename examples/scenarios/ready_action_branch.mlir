// Same context, two independent branches. S0/S1 differ only in Group selection.
// Timing and lifecycle model, not tensor numerical verification.
// Use --context-mode 2 --input-binding arena=0x100000:8192:rw.
builtin.module {
  tile.program @slow(
    %task: !nest.task, %input: !nest.l2_buffer<1x32x32xbf16>,
    %output: !nest.l2_buffer<1x32x32xbf16>) {
    %iv = tile.subview %input task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 32, 32]
      strides = [1, 1, 1] : !nest.l2_view<1x32x32xbf16>
    %ov = tile.subview %output task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 32, 32]
      strides = [1, 1, 1] : !nest.l2_view<1x32x32xbf16>
    %local = tile.alloc shape = [32, 32] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<32x32xbf16>
    %load = tile.load.async %iv into %local : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
    %compute = tile.evu.async "slow" ops = 1048576 : !tile.event<"compute">
    tile.await %compute
    %store = tile.store.async %local into %ov : !tile.event<"store">
    tile.await %store
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @fast(
    %task: !nest.task, %input: !nest.l2_buffer<1x32x32xbf16>,
    %output: !nest.l2_buffer<1x32x32xbf16>) {
    %iv = tile.subview %input task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 32, 32]
      strides = [1, 1, 1] : !nest.l2_view<1x32x32xbf16>
    %ov = tile.subview %output task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 32, 32]
      strides = [1, 1, 1] : !nest.l2_view<1x32x32xbf16>
    %local = tile.alloc shape = [32, 32] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<32x32xbf16>
    %load = tile.load.async %iv into %local : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
    %compute = tile.boa.async "matmul" m = 32 n = 32 k = 32 ops = 65536 : !tile.event<"compute">
    tile.await %compute
    %store = tile.store.async %local into %ov : !tile.event<"store">
    tile.await %store
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @branches (%arena: !nest.global_memref<4096xbf16>) placement = 1 {
    %ai = nest.alloc slot = "ai" role = "in" shape = [1, 32, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32x32xbf16>
    %ao = nest.alloc slot = "ao" role = "out" shape = [1, 32, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32x32xbf16>
    %bi = nest.alloc slot = "bi" role = "in" shape = [1, 32, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32x32xbf16>
    %bo = nest.alloc slot = "bo" role = "out" shape = [1, 32, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32x32xbf16>
    %ha = nest.subview %arena offsets = [0] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %hoa = nest.subview %arena offsets = [1024] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %hb = nest.subview %arena offsets = [2048] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %hob = nest.subview %arena offsets = [3072] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %pa = nest.dma.prefetch.async %ha into %ai : !nest.event<"pa">
    %ga, %ira, %ora = nest.dispatch.tasks.async @slow tasks(%tasks) globals() bindings(%ai, %ao)
      ins(%ai) outs(%ao)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pa) : (!nest.event<"ga">, !nest.event<"ira">, !nest.event<"ora">)
    %sa = nest.dma.store.async %ao into %hoa depends_on(%ora) : !nest.event<"sa">
    nest.release %ai depends_on(%pa, %ira)
    nest.release %ao depends_on(%sa)
    %pb = nest.dma.prefetch.async %hb into %bi : !nest.event<"pb">
    %gb, %irb, %orb = nest.dispatch.tasks.async @fast tasks(%tasks) globals() bindings(%bi, %bo)
      ins(%bi) outs(%bo)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pb) : (!nest.event<"gb">, !nest.event<"irb">, !nest.event<"orb">)
    %sb = nest.dma.store.async %bo into %hob depends_on(%orb) : !nest.event<"sb">
    nest.release %bi depends_on(%pb, %irb)
    nest.release %bo depends_on(%sb)
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4096xbf16>) {
    %done = nexus.submit_context.async @branches(%arena) : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
