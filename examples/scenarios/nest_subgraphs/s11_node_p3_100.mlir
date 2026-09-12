// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s11", "name": "s11_node_p3_100", "nodes": ["P0", "P1", "P2", "P3", "R0", "R1", "R", "N0", "N1", "N2", "N3"], "edges": {"R0": ["P0", "P1"], "R1": ["P2", "P3"], "R": ["R0", "R1"], "N0": ["R"], "N1": ["R"], "N2": ["R"], "N3": ["R"]}, "context_partition": [["P0"], ["P1"], ["P2"], ["P3"], ["R0"], ["R1"], ["R"], ["N0"], ["N1"], ["N2"], ["N3"]], "resource_config": {"uce": 4, "device": 4, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_P0": {"index": 0, "offset_elements": 0, "bytes": 32768}, "W_P0": {"index": 1, "offset_elements": 131072, "bytes": 32768}, "input_P1": {"index": 2, "offset_elements": 262144, "bytes": 32768}, "W_P1": {"index": 3, "offset_elements": 393216, "bytes": 32768}, "input_P2": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "W_P2": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "input_P3": {"index": 6, "offset_elements": 786432, "bytes": 32768}, "W_P3": {"index": 7, "offset_elements": 917504, "bytes": 32768}, "P0": {"index": 8, "offset_elements": 1048576, "bytes": 32768}, "P1": {"index": 9, "offset_elements": 1179648, "bytes": 32768}, "P2": {"index": 10, "offset_elements": 1310720, "bytes": 32768}, "P3": {"index": 11, "offset_elements": 1441792, "bytes": 32768}, "R0": {"index": 12, "offset_elements": 1572864, "bytes": 32768}, "R1": {"index": 13, "offset_elements": 1703936, "bytes": 32768}, "R": {"index": 14, "offset_elements": 1835008, "bytes": 32768}, "N0": {"index": 15, "offset_elements": 1966080, "bytes": 32768}, "N1": {"index": 16, "offset_elements": 2097152, "bytes": 32768}, "N2": {"index": 17, "offset_elements": 2228224, "bytes": 32768}, "N3": {"index": 18, "offset_elements": 2359296, "bytes": 32768}}, "node_programs": {"P0": {"program": "prog_P0", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "P1": {"program": "prog_P1", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "P2": {"program": "prog_P2", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "P3": {"program": "prog_P3", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 100, "loads": 1}, "R0": {"program": "prog_R0", "pin": "随 device slot", "engine": "evu", "op": "reduce", "repeat": 1, "loads": 1}, "R1": {"program": "prog_R1", "pin": "随 device slot", "engine": "evu", "op": "reduce", "repeat": 1, "loads": 1}, "R": {"program": "prog_R", "pin": "随 device slot", "engine": "evu", "op": "reduce", "repeat": 1, "loads": 1}, "N0": {"program": "prog_N0", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "N1": {"program": "prog_N1", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "N2": {"program": "prog_N2", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "N3": {"program": "prog_N3", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}}, "consumer_ready": {"P0": ["input_P0.prefetch", "W_P0.prefetch"], "P1": ["input_P1.prefetch", "W_P1.prefetch"], "P2": ["input_P2.prefetch", "W_P2.prefetch"], "P3": ["input_P3.prefetch", "W_P3.prefetch"], "R0": ["P0.context_done 后从同一 HBM 区间 prefetch", "P1.context_done 后从同一 HBM 区间 prefetch"], "R1": ["P2.context_done 后从同一 HBM 区间 prefetch", "P3.context_done 后从同一 HBM 区间 prefetch"], "R": ["R0.context_done 后从同一 HBM 区间 prefetch", "R1.context_done 后从同一 HBM 区间 prefetch"], "N0": ["R.context_done 后从同一 HBM 区间 prefetch"], "N1": ["R.context_done 后从同一 HBM 区间 prefetch"], "N2": ["R.context_done 后从同一 HBM 区间 prefetch"], "N3": ["R.context_done 后从同一 HBM 区间 prefetch"]}, "说明": "P3 BOA timing descriptor 静态重复一百次；reducer 仍只等待真实参与者。"}
builtin.module {
  tile.program @prog_P0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_P1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_P2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_P3 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"compute_99">
    tile.await %compute_99
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_R0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "reduce" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_R1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "reduce" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_R (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "reduce" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N3 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P0 = nest.alloc slot = "input_P0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P0 = nest.alloc slot = "W_P0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P0 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P0 = nest.alloc slot = "P0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P0 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P0 = nest.dma.prefetch.async %h_input_P0 into %b_input_P0 : !nest.event<"pref_input_P0">
    %pref_W_P0 = nest.dma.prefetch.async %h_W_P0 into %b_W_P0 : !nest.event<"pref_W_P0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P0, %read_P0, %ready_P0 = nest.dispatch.tasks.async @prog_P0 tasks(%tasks) globals() bindings(%b_input_P0, %b_W_P0, %b_P0) ins(%b_input_P0, %b_W_P0) outs(%b_P0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P0, %pref_W_P0) : (!nest.event<"grid_P0">, !nest.event<"read_P0">, !nest.event<"ready_P0">)
    nest.release %b_input_P0 depends_on(%read_P0, %pref_input_P0)
    nest.release %b_W_P0 depends_on(%read_P0, %pref_W_P0)
    %store_P0_0 = nest.dma.store.async %b_P0 into %h_P0 depends_on(%ready_P0) : !nest.event<"store_P0_0">
    nest.release %b_P0 depends_on(%store_P0_0)
    nest.await %grid_P0, %store_P0_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P1 = nest.alloc slot = "input_P1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P1 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P1 = nest.alloc slot = "W_P1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P1 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P1 = nest.alloc slot = "P1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P1 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P1 = nest.dma.prefetch.async %h_input_P1 into %b_input_P1 : !nest.event<"pref_input_P1">
    %pref_W_P1 = nest.dma.prefetch.async %h_W_P1 into %b_W_P1 : !nest.event<"pref_W_P1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P1, %read_P1, %ready_P1 = nest.dispatch.tasks.async @prog_P1 tasks(%tasks) globals() bindings(%b_input_P1, %b_W_P1, %b_P1) ins(%b_input_P1, %b_W_P1) outs(%b_P1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P1, %pref_W_P1) : (!nest.event<"grid_P1">, !nest.event<"read_P1">, !nest.event<"ready_P1">)
    nest.release %b_input_P1 depends_on(%read_P1, %pref_input_P1)
    nest.release %b_W_P1 depends_on(%read_P1, %pref_W_P1)
    %store_P1_0 = nest.dma.store.async %b_P1 into %h_P1 depends_on(%ready_P1) : !nest.event<"store_P1_0">
    nest.release %b_P1 depends_on(%store_P1_0)
    nest.await %grid_P1, %store_P1_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P2 = nest.alloc slot = "input_P2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P2 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P2 = nest.alloc slot = "W_P2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P2 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P2 = nest.alloc slot = "P2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P2 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P2 = nest.dma.prefetch.async %h_input_P2 into %b_input_P2 : !nest.event<"pref_input_P2">
    %pref_W_P2 = nest.dma.prefetch.async %h_W_P2 into %b_W_P2 : !nest.event<"pref_W_P2">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P2, %read_P2, %ready_P2 = nest.dispatch.tasks.async @prog_P2 tasks(%tasks) globals() bindings(%b_input_P2, %b_W_P2, %b_P2) ins(%b_input_P2, %b_W_P2) outs(%b_P2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P2, %pref_W_P2) : (!nest.event<"grid_P2">, !nest.event<"read_P2">, !nest.event<"ready_P2">)
    nest.release %b_input_P2 depends_on(%read_P2, %pref_input_P2)
    nest.release %b_W_P2 depends_on(%read_P2, %pref_W_P2)
    %store_P2_0 = nest.dma.store.async %b_P2 into %h_P2 depends_on(%ready_P2) : !nest.event<"store_P2_0">
    nest.release %b_P2 depends_on(%store_P2_0)
    nest.await %grid_P2, %store_P2_0
    nest.return
  }
  nest.context @ctx_3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P3 = nest.alloc slot = "input_P3" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P3 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P3 = nest.alloc slot = "W_P3" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P3 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P3 = nest.alloc slot = "P3" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P3 = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P3 = nest.dma.prefetch.async %h_input_P3 into %b_input_P3 : !nest.event<"pref_input_P3">
    %pref_W_P3 = nest.dma.prefetch.async %h_W_P3 into %b_W_P3 : !nest.event<"pref_W_P3">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P3, %read_P3, %ready_P3 = nest.dispatch.tasks.async @prog_P3 tasks(%tasks) globals() bindings(%b_input_P3, %b_W_P3, %b_P3) ins(%b_input_P3, %b_W_P3) outs(%b_P3) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P3, %pref_W_P3) : (!nest.event<"grid_P3">, !nest.event<"read_P3">, !nest.event<"ready_P3">)
    nest.release %b_input_P3 depends_on(%read_P3, %pref_input_P3)
    nest.release %b_W_P3 depends_on(%read_P3, %pref_W_P3)
    %store_P3_0 = nest.dma.store.async %b_P3 into %h_P3 depends_on(%ready_P3) : !nest.event<"store_P3_0">
    nest.release %b_P3 depends_on(%store_P3_0)
    nest.await %grid_P3, %store_P3_0
    nest.return
  }
  nest.context @ctx_4 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_P0 = nest.alloc slot = "P0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P0 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P1 = nest.alloc slot = "P1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P1 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R0 = nest.alloc slot = "R0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0 = nest.subview %arena offsets = [1572864] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_P0 = nest.dma.prefetch.async %h_P0 into %b_P0 : !nest.event<"pref_P0">
    %pref_P1 = nest.dma.prefetch.async %h_P1 into %b_P1 : !nest.event<"pref_P1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_R0, %read_R0, %ready_R0 = nest.dispatch.tasks.async @prog_R0 tasks(%tasks) globals() bindings(%b_P0, %b_P1, %b_R0) ins(%b_P0, %b_P1) outs(%b_R0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_P0, %pref_P1) : (!nest.event<"grid_R0">, !nest.event<"read_R0">, !nest.event<"ready_R0">)
    nest.release %b_P0 depends_on(%read_R0, %pref_P0)
    nest.release %b_P1 depends_on(%read_R0, %pref_P1)
    %store_R0_0 = nest.dma.store.async %b_R0 into %h_R0 depends_on(%ready_R0) : !nest.event<"store_R0_0">
    nest.release %b_R0 depends_on(%store_R0_0)
    nest.await %grid_R0, %store_R0_0
    nest.return
  }
  nest.context @ctx_5 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_P2 = nest.alloc slot = "P2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P2 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P3 = nest.alloc slot = "P3" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P3 = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R1 = nest.alloc slot = "R1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R1 = nest.subview %arena offsets = [1703936] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_P2 = nest.dma.prefetch.async %h_P2 into %b_P2 : !nest.event<"pref_P2">
    %pref_P3 = nest.dma.prefetch.async %h_P3 into %b_P3 : !nest.event<"pref_P3">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_R1, %read_R1, %ready_R1 = nest.dispatch.tasks.async @prog_R1 tasks(%tasks) globals() bindings(%b_P2, %b_P3, %b_R1) ins(%b_P2, %b_P3) outs(%b_R1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_P2, %pref_P3) : (!nest.event<"grid_R1">, !nest.event<"read_R1">, !nest.event<"ready_R1">)
    nest.release %b_P2 depends_on(%read_R1, %pref_P2)
    nest.release %b_P3 depends_on(%read_R1, %pref_P3)
    %store_R1_0 = nest.dma.store.async %b_R1 into %h_R1 depends_on(%ready_R1) : !nest.event<"store_R1_0">
    nest.release %b_R1 depends_on(%store_R1_0)
    nest.await %grid_R1, %store_R1_0
    nest.return
  }
  nest.context @ctx_6 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R0 = nest.alloc slot = "R0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0 = nest.subview %arena offsets = [1572864] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R1 = nest.alloc slot = "R1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R1 = nest.subview %arena offsets = [1703936] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R = nest.alloc slot = "R" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R0 = nest.dma.prefetch.async %h_R0 into %b_R0 : !nest.event<"pref_R0">
    %pref_R1 = nest.dma.prefetch.async %h_R1 into %b_R1 : !nest.event<"pref_R1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_R, %read_R, %ready_R = nest.dispatch.tasks.async @prog_R tasks(%tasks) globals() bindings(%b_R0, %b_R1, %b_R) ins(%b_R0, %b_R1) outs(%b_R) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R0, %pref_R1) : (!nest.event<"grid_R">, !nest.event<"read_R">, !nest.event<"ready_R">)
    nest.release %b_R0 depends_on(%read_R, %pref_R0)
    nest.release %b_R1 depends_on(%read_R, %pref_R1)
    %store_R_0 = nest.dma.store.async %b_R into %h_R depends_on(%ready_R) : !nest.event<"store_R_0">
    nest.release %b_R depends_on(%store_R_0)
    nest.await %grid_R, %store_R_0
    nest.return
  }
  nest.context @ctx_7 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N0 = nest.alloc slot = "N0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N0 = nest.subview %arena offsets = [1966080] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N0, %read_N0, %ready_N0 = nest.dispatch.tasks.async @prog_N0 tasks(%tasks) globals() bindings(%b_R, %b_N0) ins(%b_R) outs(%b_N0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N0">, !nest.event<"read_N0">, !nest.event<"ready_N0">)
    nest.release %b_R depends_on(%read_N0, %pref_R)
    %store_N0_0 = nest.dma.store.async %b_N0 into %h_N0 depends_on(%ready_N0) : !nest.event<"store_N0_0">
    nest.release %b_N0 depends_on(%store_N0_0)
    nest.await %grid_N0, %store_N0_0
    nest.return
  }
  nest.context @ctx_8 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N1 = nest.alloc slot = "N1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N1 = nest.subview %arena offsets = [2097152] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N1, %read_N1, %ready_N1 = nest.dispatch.tasks.async @prog_N1 tasks(%tasks) globals() bindings(%b_R, %b_N1) ins(%b_R) outs(%b_N1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N1">, !nest.event<"read_N1">, !nest.event<"ready_N1">)
    nest.release %b_R depends_on(%read_N1, %pref_R)
    %store_N1_0 = nest.dma.store.async %b_N1 into %h_N1 depends_on(%ready_N1) : !nest.event<"store_N1_0">
    nest.release %b_N1 depends_on(%store_N1_0)
    nest.await %grid_N1, %store_N1_0
    nest.return
  }
  nest.context @ctx_9 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N2 = nest.alloc slot = "N2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N2 = nest.subview %arena offsets = [2228224] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N2, %read_N2, %ready_N2 = nest.dispatch.tasks.async @prog_N2 tasks(%tasks) globals() bindings(%b_R, %b_N2) ins(%b_R) outs(%b_N2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N2">, !nest.event<"read_N2">, !nest.event<"ready_N2">)
    nest.release %b_R depends_on(%read_N2, %pref_R)
    %store_N2_0 = nest.dma.store.async %b_N2 into %h_N2 depends_on(%ready_N2) : !nest.event<"store_N2_0">
    nest.release %b_N2 depends_on(%store_N2_0)
    nest.await %grid_N2, %store_N2_0
    nest.return
  }
  nest.context @ctx_10 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N3 = nest.alloc slot = "N3" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N3 = nest.subview %arena offsets = [2359296] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N3, %read_N3, %ready_N3 = nest.dispatch.tasks.async @prog_N3 tasks(%tasks) globals() bindings(%b_R, %b_N3) ins(%b_R) outs(%b_N3) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N3">, !nest.event<"read_N3">, !nest.event<"ready_N3">)
    nest.release %b_R depends_on(%read_N3, %pref_R)
    %store_N3_0 = nest.dma.store.async %b_N3 into %h_N3 depends_on(%ready_N3) : !nest.event<"store_N3_0">
    nest.release %b_N3 depends_on(%store_N3_0)
    nest.await %grid_N3, %store_N3_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    %done_3 = nexus.submit_context.async @ctx_3(%arena) : !nexus.event<"done_3">
    nexus.await %done_0, %done_1
    %done_4 = nexus.submit_context.async @ctx_4(%arena) : !nexus.event<"done_4">
    nexus.await %done_2, %done_3
    %done_5 = nexus.submit_context.async @ctx_5(%arena) : !nexus.event<"done_5">
    nexus.await %done_4, %done_5
    %done_6 = nexus.submit_context.async @ctx_6(%arena) : !nexus.event<"done_6">
    nexus.await %done_6
    %done_7 = nexus.submit_context.async @ctx_7(%arena) : !nexus.event<"done_7">
    nexus.await %done_6
    %done_8 = nexus.submit_context.async @ctx_8(%arena) : !nexus.event<"done_8">
    nexus.await %done_6
    %done_9 = nexus.submit_context.async @ctx_9(%arena) : !nexus.event<"done_9">
    nexus.await %done_6
    %done_10 = nexus.submit_context.async @ctx_10(%arena) : !nexus.event<"done_10">
    nexus.await %done_0, %done_1, %done_2, %done_3, %done_4, %done_5, %done_6, %done_7, %done_8, %done_9, %done_10
    nexus.return
  }
}
