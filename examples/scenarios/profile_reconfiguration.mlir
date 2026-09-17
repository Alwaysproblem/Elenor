// Profile reconfiguration review example (design/proposal/04 §4.7).
//
// Two Contexts demonstrate layered L1/L2 profile switching:
//   - ctx_tasks: L2 mode0, three Tasks with L1 0→2→2 (task_a → task_b → task_a).
//     The compiler inserts an ordinary nest.await before the L1 reconfig,
//     preserving the parent L2 Arena.
//   - ctx_l2_2: L2 mode2. The Device awaits ctx_tasks retirement before
//     generating the L2 reconfig command.
//
// Each Arena is 16 Banks × 256 B = 4096 B. R=1 for both contexts.
// No tile program reads or writes L2 data; the dispatches only exercise
// the L1 profile path and resource lease.

builtin.module {
  tile.program @task_a (%task : !nest.task)
    resource_contract = #tile.resources<
      allowed_profiles = [0, 2],
      tile_l1_spm_bytes_per_context = 4096>
  {
    %tmp = tile.alloc shape = [4096] dtype = "i8" alignment = 256
      : !tile.l1_buffer<4096xi8>
    tile.free %tmp
    tile.return
  }

  tile.program @task_b (%task : !nest.task)
    resource_contract = #tile.resources<
      allowed_profiles = [2],
      tile_l1_spm_bytes_per_context = 4096>
  {
    %tmp = tile.alloc shape = [4096] dtype = "i8" alignment = 256
      : !tile.l1_buffer<4096xi8>
    tile.free %tmp
    tile.return
  }

  nest.context @ctx_tasks placement = 1
    resource_contract = #nest.context_resources<
      l2_mode = 0,
      allowed_profiles = [0, 2],
      logical_tasks = 3,
      l2_spm_bytes = 4096,
      requested_contexts_per_tile = 1>
  {
    %keep = nest.alloc slot = "keep" role = "in"
      shape = [4096] dtype = "i8" alignment = 256
      : !nest.l2_buffer<4096xi8>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range

    %g0, %r0, %w0 = nest.dispatch.tasks.async @task_a
      l1_mode = 0
      tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
      : (!nest.event<"g0">, !nest.event<"">, !nest.event<"">)

    // The compiler inserts an ordinary nest.await here before the L1
    // reconfiguration to mode2, preserving the parent L2 Arena.
    nest.await %g0

    %g1, %r1, %w1 = nest.dispatch.tasks.async @task_b
      l1_mode = 2
      tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
      : (!nest.event<"g1">, !nest.event<"">, !nest.event<"">)

    // task_a requests baseline 0 but is allowed to run compatibly at
    // current L1 mode2 without switching back.
    %g2, %r2, %w2 = nest.dispatch.tasks.async @task_a
      l1_mode = 0
      tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
      : (!nest.event<"g2">, !nest.event<"">, !nest.event<"">)

    nest.await %g1, %g2
    nest.release %keep
    nest.return
  }

  nest.context @ctx_l2_2 placement = 1
    resource_contract = #nest.context_resources<
      l2_mode = 2,
      allowed_profiles = [2],
      logical_tasks = 0,
      l2_spm_bytes = 4096,
      requested_contexts_per_tile = 1>
  {
    %tmp = nest.alloc slot = "tmp" role = "in"
      shape = [4096] dtype = "i8" alignment = 256
      : !nest.l2_buffer<4096xi8>
    nest.release %tmp
    nest.return
  }

  nexus.program @review {
    %c0 = nexus.submit_context.async @ctx_tasks : !nexus.event<"c0">

    // Device-level await of the entire root before L2 reconfiguration.
    nexus.await %c0

    // The compiler generates the L2 configuration command here;
    // L1 stays at mode2.
    %c1 = nexus.submit_context.async @ctx_l2_2 : !nexus.event<"c1">
    nexus.await %c1
    nexus.return
  }
}
