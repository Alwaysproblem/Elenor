builtin.module {
  // The validator models transfer, event and view lifetimes. The linked scalar ELF owns the +7 arithmetic.
  tile.program @scalar_ctl_v1(%task : !nest.task, %src : !nest.l2_buffer<8xi32>, %dst : !nest.l2_buffer<8xi32>)
      resource_contract = #tile.resources<allowed_profiles = [0], tile_l1_spm_bytes_per_context = 2048> {
    %from = tile.subview %src offsets = [0] sizes = [8] strides = [1] : !nest.l2_view<8xi32>
    %to = tile.subview %dst offsets = [0] sizes = [8] strides = [1] : !nest.l2_view<8xi32>
    %input = tile.alloc shape = [8] dtype = "i32" alignment = 32 : !tile.l1_buffer<8xi32>
    %output = tile.alloc shape = [8] dtype = "i32" alignment = 32 : !tile.l1_buffer<8xi32>
    %in = tile.load.async %from into %input : !tile.event<"load">
    tile.await %in
    %out = tile.store.async %output into %to : !tile.event<"store">
    tile.await %out
    tile.signal input_released(%task)
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @scalar_ctl(%X : !nest.global_memref<8xi32>, %Y : !nest.global_memref<8xi32>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0], logical_tasks = 1,
        l2_spm_bytes = 8192, requested_contexts_per_tile = 1> {
    %x = nest.subview %X offsets = [0] sizes = [8] strides = [1] : !nest.global_view<8xi32>
    %y = nest.subview %Y offsets = [0] sizes = [8] strides = [1] : !nest.global_view<8xi32>
    %src = nest.alloc slot = "input" role = "in" shape = [8] dtype = "i32" alignment = 256 : !nest.l2_buffer<8xi32>
    %dst = nest.alloc slot = "output" role = "out" shape = [8] dtype = "i32" alignment = 256 : !nest.l2_buffer<8xi32>
    %prefetch = nest.dma.prefetch.async %x into %src : !nest.event<"prefetch">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %done, %released, %ready = nest.dispatch.tasks.async @scalar_ctl_v1 l1_mode = 0 tasks(%tasks) globals()
      bindings(%src, %dst) ins(%src) outs(%dst) signal_policy {
        input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks>
      } depends_on(%prefetch) : (!nest.event<"done">, !nest.event<"released">, !nest.event<"ready">)
    nest.release %src depends_on(%prefetch, %released)
    %stored = nest.dma.store.async %dst into %y depends_on(%ready) : !nest.event<"stored">
    nest.release %dst depends_on(%stored)
    nest.await %stored, %done
    nest.return
  }
}
