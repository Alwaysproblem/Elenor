// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "s11",
//       "name": "s11_node_p0",
//       "nodes": [],
//       "edges": {},
//       "context_partition": [
//         []
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device": 4,
//         "placement": 15,
//         "fidelity": "full_memory"
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成",
//         "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描",
//         "Liveness": "预期完成",
//         "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"
//       },
//       "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性",
//       "tensors": {},
//       "node_programs": {},
//       "consumer_ready": {},
//       "说明": "participant count=0：一个合法空 Context，仅 nest.return；无 task.range、dispatch、phase waiter、reducer 或 consumer。"
//     }
builtin.module {
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
