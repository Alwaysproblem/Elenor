// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "t03",
//       "name": "t03_node_zero",
//       "nodes": [],
//       "edges": {},
//       "context_partition": [
//         []
//       ],
//       "resource_config": {
//         "uce_context_mode": 4,
//         "device_context_mode": 4,
//         "placement": 15,
//         "fidelity": "full_memory",
//         "arena_binding": "arena=0x1000000:8388608:rw"
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查真实 HBM/L2 数据边、input_released/output_ready、最终 Store 与释放顺序",
//         "数值": "未建模；仅使用真实搬运和合成 engine service，不作为 tensor 数值证明",
//         "Liveness": "应完成；以同次运行 trace 和退出状态确认",
//         "Scheduling Quality": "0 次静态展开；活动 grid 数应为 0，不得解释为运行时 early-exit"
//       },
//       "forbidden_dependencies": "只允许表列真数据边；UCE 共享 pin 争用不是数据依赖；跨 Context 仅使用当前 IR 可见的 context_done/HBM 可见性",
//       "tensors": {
//         "State0": {
//           "formal": "arena",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         }
//       },
//       "node_programs": {},
//       "consumer_ready_granularity": {},
//       "notes": {
//         "iteration_count": 0,
//         "event_tags": [],
//         "zero_case": "0 步仍提交合法空 Context；没有 task.range、dispatch 或 phase waiter",
//         "stable_tensor_indices": {
//           "State0": 0,
//           "Input0": 1,
//           "Input1": 2,
//           "Input2": 3,
//           "Input3": 4,
//           "State1": 5,
//           "State2": 6,
//           "State3": 7,
//           "State4": 8
//         }
//       }
//     }
builtin.module {
  nest.context @ctx_empty (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_empty = nexus.submit_context.async @ctx_empty(%arena) : !nexus.event<"done_empty">
    nexus.await %done_empty
    nexus.return
  }
}
