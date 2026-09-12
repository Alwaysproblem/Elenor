// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// participant count=0：省略工作，而不是放行非法 zero-range。
// case: {
//       "id": "N08",
//       "name": "n08_p0",
//       "nodes": [],
//       "edges": {},
//       "context_partition": [
//         []
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device_contexts": 4,
//         "fidelity": "full_memory",
//         "num_dma_channels": 2,
//         "arena_binding": "arena=0x1000000:8388608:rw",
//         "placement": 1
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "count0 使用合法空 Context；没有 task.range、dispatch、phase waiter 或伪 participant",
//         "数值": "未建模；engine descriptor 与未初始化的合成输出只用于时序",
//         "Liveness": "所有已接纳 Context 完成",
//         "Scheduling Quality": "仅记录实际 service/等待，不声称最优调度"
//       },
//       "forbidden_dependencies": "禁止用 task.range 0..0 冒充空工作，也禁止等待不存在的 aggregate",
//       "tensors": {},
//       "phase": {},
//       "pins": {},
//       "node_programs": {}
//     }
builtin.module {
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 1 {
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
