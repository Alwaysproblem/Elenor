module {
  nexus.program @run_pow(
      %Y : !nest.global_memref<4x128x128xbf16>)
  {
      // ================================================================
      // 1. Obtain device and Context submission queue.
      //
      // This is the queue consumed by the L2 Context Scheduler.
      // ================================================================

      %device =
          nexus.device.get
              #nest.device<0>
          : !nexus.device

      %context_queue =
          nexus.context_queue.get
              %device
          {
              queue_id = 0
          }
          : !nexus.context_queue

      // ================================================================
      // 2. Submit one logical NEST Context.
      //
      // CPU does NOT specify:
      //
      //   - L2 buffer addresses
      //   - logical Tile task IDs
      //   - physical Tile IDs
      //   - Tile Hardware Context IDs
      //
      // Those are handled on device.
      // ================================================================

      %context_done =
          nexus.submit_context.async
              @tiled_pow_task(%Y)
              to %context_queue
          {
              placement =
                  #nest.tile_group<mask = 0xF>,

              priority =
                  #nest.priority<normal>,

              qos =
                  #nest.qos<throughput>,

              memory_effects = [
                  #nest.memory_effect<
                      %Y,
                      read_write>
              ]
          }

          : !nexus.event<context_done>

      // ================================================================
      // 3. CPU synchronization.
      //
      // context_done corresponds to nest.context.return, NOT grid_done.
      //
      // Therefore %Y is globally visible when this event fires.
      // ================================================================

      nexus.await %context_done

      nexus.return
  }

  // ==================================================================
  // Device-level NEST Context
  //
  //   HBM Y
  //     |
  //     | prefetch
  //     v
  //   L2 pow buffer
  //     |
  //     | task dispatch
  //     v
  //   Tile-local multi-context
  //     |
  //     | in-place pow
  //     v
  //   L2 pow buffer
  //     |
  //     | store
  //     v
  //   HBM Y
  // ==================================================================

  nest.context @tiled_pow_task(
      %Y : !nest.global_memref<4x128x128xbf16>)
  attributes {
      placement =
          #nest.tile_group<mask = 0xF>,

      execution_model =
          #nest.execution_model<tile_local_multicontext>,

      epoch_model =
          #nest.epoch_model<spmd>,

      // Compiler-generated resource summary.
      //
      // Used by L2 Context admission.
      resource_contract =
          #nest.context_resources<
              logical_tasks                  = 4,
              l2_scratchpad_bytes            = 131072,
              tile_l1_bytes_per_context      = 32768,
              requested_contexts_per_tile    = 4>,

      // Compatibility metadata only.
      legacy.role_map = [
          #nest.role<
              id      = 0,
              program = @pow_4k_tile>
      ]
  }
  {

      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index

      // ==============================================================
      // 1. Context-owned L2 buffer.
      //
      // 4 * 128 * 128 * sizeof(bf16)
      // = 131072 bytes.
      //
      // This buffer is owned by this NEST Context.
      // ==============================================================

      %l2_pow =
          nest.alloc {
              memory_space = #nest.memory_space<l2>,
              scope        = #nest.scope<context>,
              role         = #nest.buffer_role<inout>,
              shape        = [4, 128, 128],
              element_type = bf16,
              alignment    = 256
          }
          : !nest.l2_buffer<4x128x128xbf16>

      // ==============================================================
      // 2. HBM -> L2 prefetch.
      // ==============================================================

      %pow_input_ready =
          nest.prefetch.async
              %Y[
                  0 : 4,
                  0 : 128,
                  0 : 128]
              into %l2_pow
          : !nest.event<input_ready>

      // ==============================================================
      // 3. Logical task domain.
      //
      // task 0 -> Y[0,:,:]
      // task 1 -> Y[1,:,:]
      // task 2 -> Y[2,:,:]
      // task 3 -> Y[3,:,:]
      //
      // These task IDs are logical IDs.
      // They are not physical Tile IDs or Hardware Context IDs.
      // ==============================================================

      %tasks =
          nest.task.range
              %c0 to %c4
          : !nest.task_range

      // ==============================================================
      // 4. Dispatch logical tasks.
      //
      // L2 publishes:
      //
      //   program = @pow_4k_tile
      //   task range = [0, 4)
      //   shared L2 buffer = %l2_pow
      //
      // Tile-local scheduler decides which physical Tile Context
      // executes each logical task.
      // ==============================================================

      %grid_done,
      %input_released,
      %output_ready =
          nest.dispatch.tasks.async
              @pow_4k_tile

              tasks(%tasks)

              ins(%l2_pow)
              outs(%l2_pow)

              // Same physical L2 object is read and then overwritten.
              aliasing {
                  %l2_pow =
                      #nest.inplace_alias<
                          task_disjoint,
                          read_before_write>
              }

              execution {
                  distribution =
                      #nest.task_distribution<dynamic>,

                  tile_scheduler =
                      #nest.tile_scheduler<multi_context>,

                  // Used only when lowering to legacy ELENOR hardware.
                  legacy_fallback =
                      #nest.task_distribution<static_wave>
              }

              signal_policy {
                  input_released =
                      #nest.aggregate<all_tasks>,

                  output_ready =
                      #nest.aggregate<all_tasks>
              }

              depends_on(%pow_input_ready)

          attributes {
              legacy.role_id = 0
          }

          : (!nest.event<grid_done>,
             !nest.event<input_released>,
             !nest.event<output_ready>)

      // ==============================================================
      // Note:
      //
      // %input_released is intentionally NOT used to free %l2_pow,
      // because %l2_pow is also the output buffer.
      //
      // Its semantic value is:
      //
      //   all Tile Programs have completed their L2 read phase.
      //
      // This is useful for verifier / alias analysis even though the
      // physical buffer remains alive.
      // ==============================================================

      // ==============================================================
      // 5. L2 -> HBM final store.
      //
      // output_ready means all four Tile Programs have completed their
      // L1 -> L2 stores.
      // ==============================================================

      %pow_store_done =
          nest.store.async
              %l2_pow
              to %Y[
                  0 : 4,
                  0 : 128,
                  0 : 128]
              depends_on(%output_ready)
          : !nest.event<store_done>

      // ==============================================================
      // 6. Context-owned L2 buffer can only be reclaimed after GDMA
      // has finished reading it.
      // ==============================================================

      nest.release %l2_pow
          depends_on(%pow_store_done)

      // ==============================================================
      // 7. Context completion.
      //
      // grid_done:
      //   all logical Tile Tasks have returned.
      //
      // pow_store_done:
      //   final externally visible result has reached HBM.
      //
      // CPU-visible Context completion must cover both.
      // ==============================================================

      nest.await
          %grid_done,
          %pow_store_done

      nest.context.return
  }


  // ==================================================================
  // Tile Program
  // ==================================================================

  nest.tile.program @pow_4k_tile(
      %task   : !nest.task,

      %l2_buf : !nest.l2_buffer<
                    4x128x128xbf16>)
  attributes {
      execution_model =
          #tile.execution_model<hardware_multicontext>,

      resources =
          #tile.resources<
              l1_private_bytes_per_context = 32768,
              mfe_event_slots              = 2,
              evu_event_slots              = 1,
              active_contexts              = admission_controlled>,

      legacy.role_id = 0
  }
  {
      // ==============================================================
      // 1. Logical task identity.
      //
      // Not physical Tile ID.
      // Not hardware Context ID.
      // ==============================================================

      %task_id =
          tile.task.id %task
          : index

      // ==============================================================
      // 2. Context-private L1 storage.
      //
      // Every simultaneously active Hardware Context gets its own
      // 32 KiB operand/result buffer.
      // ==============================================================

      %l1_value =
          tile.alloc {
              memory_space     = #tile.memory_space<l1>,
              context_private  = true,

              shape            = [128, 128],
              element_type     = bf16,
              alignment        = 256
          }
          : !tile.l1_buffer<128x128xbf16>

      // ==============================================================
      // 3. Tile Program calculates its own L2 subview.
      // ==============================================================

      %l2_tile =
          tile.subview %l2_buf[
              %task_id,
              0 : 128,
              0 : 128]
          : !nest.l2_view<128x128xbf16>

      // ==============================================================
      // 4. L2 -> L1.
      //
      // This request is issued by the current Tile Hardware Context.
      // ==============================================================

      %input_loaded =
          tile.load.async
              %l2_tile
              into %l1_value
          : !tile.event<mfe_done>

      // Only this Hardware Context is suspended.
      tile.await %input_loaded

      // This task will no longer read its L2 input subview.
      tile.signal input_released(%task)

      // ==============================================================
      // 5. EVU Pow.
      //
      // If compiler is allowed to canonicalize pow(x, 2) -> x*x,
      // this should be controlled by math semantics rather than
      // "exact_operation = true".
      // ==============================================================

      %pow_done =
          tile.pow.async
              %l1_value
              exponent(2.0)
              into %l1_value
          {
              inplace   = true,
              math_mode = #tile.math_mode<relaxed>
          }
          : !tile.event<evu_done>

      tile.await %pow_done

      // ==============================================================
      // 6. L1 -> L2.
      // ==============================================================

      %output_stored =
          tile.store.async
              %l1_value
              into %l2_tile
          : !tile.event<mfe_done>

      tile.await %output_stored

      // Output of this logical task is now visible in L2.
      tile.signal output_ready(%task)

      // tile.return contributes to grid_done.
      tile.return
  }
}
