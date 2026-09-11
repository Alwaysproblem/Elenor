// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s12", "name": "s12_node_sm100", "nodes": ["X", "Q", "K", "V", "QK", "SM", "PV", "O"], "edges": {"Q": ["X"], "K": ["X"], "V": ["X"], "QK": ["Q", "K"], "SM": ["QK"], "PV": ["SM", "V"], "O": ["PV"]}, "context_partition": [["X"], ["Q"], ["K"], ["V"], ["QK"], ["SM"], ["PV"], ["O"]], "resource_config": {"uce": 4, "device": 4, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_X": {"index": 0, "offset_elements": 0, "bytes": 32768}, "W_Q": {"index": 1, "offset_elements": 131072, "bytes": 32768}, "W_K": {"index": 2, "offset_elements": 262144, "bytes": 32768}, "W_V": {"index": 3, "offset_elements": 393216, "bytes": 32768}, "X": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "Q": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "K": {"index": 6, "offset_elements": 786432, "bytes": 32768}, "V": {"index": 7, "offset_elements": 917504, "bytes": 32768}, "QK": {"index": 8, "offset_elements": 1048576, "bytes": 32768}, "SM": {"index": 9, "offset_elements": 1179648, "bytes": 32768}, "PV": {"index": 10, "offset_elements": 1310720, "bytes": 32768}, "O": {"index": 11, "offset_elements": 1441792, "bytes": 32768}}, "node_programs": {"X": {"program": "prog_X", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "Q": {"program": "prog_Q", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "K": {"program": "prog_K", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "V": {"program": "prog_V", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "QK": {"program": "prog_QK", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "SM": {"program": "prog_SM", "pin": "随 device slot", "engine": "evu", "op": "softmax", "repeat": 100, "loads": 1}, "PV": {"program": "prog_PV", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "O": {"program": "prog_O", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}}, "consumer_ready": {"X": ["input_X.prefetch"], "Q": ["X.context_done 后从同一 HBM 区间 prefetch", "W_Q.prefetch"], "K": ["X.context_done 后从同一 HBM 区间 prefetch", "W_K.prefetch"], "V": ["X.context_done 后从同一 HBM 区间 prefetch", "W_V.prefetch"], "QK": ["Q.context_done 后从同一 HBM 区间 prefetch", "K.context_done 后从同一 HBM 区间 prefetch"], "SM": ["QK.context_done 后从同一 HBM 区间 prefetch"], "PV": ["SM.context_done 后从同一 HBM 区间 prefetch", "V.context_done 后从同一 HBM 区间 prefetch"], "O": ["PV.context_done 后从同一 HBM 区间 prefetch"]}, "说明": "仅指定节点静态重复一百次；QK 只等待 Q/K，PV 只等待 SM/V。普通 V 仍是 BOA，资源争用不算数据依赖。"}
builtin.module {
  tile.program @prog_X (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_Q (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_K (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_V (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_QK (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_SM (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_99">
    tile.await %compute_99
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_PV (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_O (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %b_input_X = nest.alloc slot = "input_X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_X = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_X = nest.alloc slot = "X" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_X = nest.dma.prefetch.async %h_input_X into %b_input_X : !nest.event<"pref_input_X">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_X, %read_X, %ready_X = nest.dispatch.tasks.async @prog_X tasks(%tasks) globals() ins(%b_input_X, %b_X) outs(%b_input_X, %b_X) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_X) : (!nest.event<"grid_X">, !nest.event<"read_X">, !nest.event<"ready_X">)
    nest.release %b_input_X depends_on(%read_X)
    %store_X_0 = nest.dma.store.async %b_X into %h_X depends_on(%ready_X) : !nest.event<"store_X_0">
    nest.release %b_X depends_on(%store_X_0)
    nest.await %grid_X, %store_X_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_X = nest.alloc slot = "X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_Q = nest.alloc slot = "W_Q" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_Q = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_Q = nest.alloc slot = "Q" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Q = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_X = nest.dma.prefetch.async %h_X into %b_X : !nest.event<"pref_X">
    %pref_W_Q = nest.dma.prefetch.async %h_W_Q into %b_W_Q : !nest.event<"pref_W_Q">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Q, %read_Q, %ready_Q = nest.dispatch.tasks.async @prog_Q tasks(%tasks) globals() ins(%b_X, %b_W_Q, %b_Q) outs(%b_X, %b_W_Q, %b_Q) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X, %pref_W_Q) : (!nest.event<"grid_Q">, !nest.event<"read_Q">, !nest.event<"ready_Q">)
    nest.release %b_X depends_on(%read_Q)
    nest.release %b_W_Q depends_on(%read_Q)
    %store_Q_0 = nest.dma.store.async %b_Q into %h_Q depends_on(%ready_Q) : !nest.event<"store_Q_0">
    nest.release %b_Q depends_on(%store_Q_0)
    nest.await %grid_Q, %store_Q_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_X = nest.alloc slot = "X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_K = nest.alloc slot = "W_K" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_K = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_K = nest.alloc slot = "K" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_K = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_X = nest.dma.prefetch.async %h_X into %b_X : !nest.event<"pref_X">
    %pref_W_K = nest.dma.prefetch.async %h_W_K into %b_W_K : !nest.event<"pref_W_K">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_K, %read_K, %ready_K = nest.dispatch.tasks.async @prog_K tasks(%tasks) globals() ins(%b_X, %b_W_K, %b_K) outs(%b_X, %b_W_K, %b_K) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X, %pref_W_K) : (!nest.event<"grid_K">, !nest.event<"read_K">, !nest.event<"ready_K">)
    nest.release %b_X depends_on(%read_K)
    nest.release %b_W_K depends_on(%read_K)
    %store_K_0 = nest.dma.store.async %b_K into %h_K depends_on(%ready_K) : !nest.event<"store_K_0">
    nest.release %b_K depends_on(%store_K_0)
    nest.await %grid_K, %store_K_0
    nest.return
  }
  nest.context @ctx_3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_X = nest.alloc slot = "X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_V = nest.alloc slot = "W_V" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_V = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V = nest.alloc slot = "V" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_X = nest.dma.prefetch.async %h_X into %b_X : !nest.event<"pref_X">
    %pref_W_V = nest.dma.prefetch.async %h_W_V into %b_W_V : !nest.event<"pref_W_V">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V, %read_V, %ready_V = nest.dispatch.tasks.async @prog_V tasks(%tasks) globals() ins(%b_X, %b_W_V, %b_V) outs(%b_X, %b_W_V, %b_V) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X, %pref_W_V) : (!nest.event<"grid_V">, !nest.event<"read_V">, !nest.event<"ready_V">)
    nest.release %b_X depends_on(%read_V)
    nest.release %b_W_V depends_on(%read_V)
    %store_V_0 = nest.dma.store.async %b_V into %h_V depends_on(%ready_V) : !nest.event<"store_V_0">
    nest.release %b_V depends_on(%store_V_0)
    nest.await %grid_V, %store_V_0
    nest.return
  }
  nest.context @ctx_4 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_Q = nest.alloc slot = "Q" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Q = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_K = nest.alloc slot = "K" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_K = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_QK = nest.alloc slot = "QK" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_QK = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_Q = nest.dma.prefetch.async %h_Q into %b_Q : !nest.event<"pref_Q">
    %pref_K = nest.dma.prefetch.async %h_K into %b_K : !nest.event<"pref_K">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_QK, %read_QK, %ready_QK = nest.dispatch.tasks.async @prog_QK tasks(%tasks) globals() ins(%b_Q, %b_K, %b_QK) outs(%b_Q, %b_K, %b_QK) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_Q, %pref_K) : (!nest.event<"grid_QK">, !nest.event<"read_QK">, !nest.event<"ready_QK">)
    nest.release %b_Q depends_on(%read_QK)
    nest.release %b_K depends_on(%read_QK)
    %store_QK_0 = nest.dma.store.async %b_QK into %h_QK depends_on(%ready_QK) : !nest.event<"store_QK_0">
    nest.release %b_QK depends_on(%store_QK_0)
    nest.await %grid_QK, %store_QK_0
    nest.return
  }
  nest.context @ctx_5 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_QK = nest.alloc slot = "QK" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_QK = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_SM = nest.alloc slot = "SM" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_SM = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_QK = nest.dma.prefetch.async %h_QK into %b_QK : !nest.event<"pref_QK">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_SM, %read_SM, %ready_SM = nest.dispatch.tasks.async @prog_SM tasks(%tasks) globals() ins(%b_QK, %b_SM) outs(%b_QK, %b_SM) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_QK) : (!nest.event<"grid_SM">, !nest.event<"read_SM">, !nest.event<"ready_SM">)
    nest.release %b_QK depends_on(%read_SM)
    %store_SM_0 = nest.dma.store.async %b_SM into %h_SM depends_on(%ready_SM) : !nest.event<"store_SM_0">
    nest.release %b_SM depends_on(%store_SM_0)
    nest.await %grid_SM, %store_SM_0
    nest.return
  }
  nest.context @ctx_6 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_SM = nest.alloc slot = "SM" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_SM = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V = nest.alloc slot = "V" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_PV = nest.alloc slot = "PV" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_PV = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_SM = nest.dma.prefetch.async %h_SM into %b_SM : !nest.event<"pref_SM">
    %pref_V = nest.dma.prefetch.async %h_V into %b_V : !nest.event<"pref_V">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_PV, %read_PV, %ready_PV = nest.dispatch.tasks.async @prog_PV tasks(%tasks) globals() ins(%b_SM, %b_V, %b_PV) outs(%b_SM, %b_V, %b_PV) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_SM, %pref_V) : (!nest.event<"grid_PV">, !nest.event<"read_PV">, !nest.event<"ready_PV">)
    nest.release %b_SM depends_on(%read_PV)
    nest.release %b_V depends_on(%read_PV)
    %store_PV_0 = nest.dma.store.async %b_PV into %h_PV depends_on(%ready_PV) : !nest.event<"store_PV_0">
    nest.release %b_PV depends_on(%store_PV_0)
    nest.await %grid_PV, %store_PV_0
    nest.return
  }
  nest.context @ctx_7 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_PV = nest.alloc slot = "PV" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_PV = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_O = nest.alloc slot = "O" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_O = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_PV = nest.dma.prefetch.async %h_PV into %b_PV : !nest.event<"pref_PV">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_O, %read_O, %ready_O = nest.dispatch.tasks.async @prog_O tasks(%tasks) globals() ins(%b_PV, %b_O) outs(%b_PV, %b_O) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_PV) : (!nest.event<"grid_O">, !nest.event<"read_O">, !nest.event<"ready_O">)
    nest.release %b_PV depends_on(%read_O)
    %store_O_0 = nest.dma.store.async %b_O into %h_O depends_on(%ready_O) : !nest.event<"store_O_0">
    nest.release %b_O depends_on(%store_O_0)
    nest.await %grid_O, %store_O_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    nexus.await %done_0
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_0
    %done_3 = nexus.submit_context.async @ctx_3(%arena) : !nexus.event<"done_3">
    nexus.await %done_1, %done_2
    %done_4 = nexus.submit_context.async @ctx_4(%arena) : !nexus.event<"done_4">
    nexus.await %done_4
    %done_5 = nexus.submit_context.async @ctx_5(%arena) : !nexus.event<"done_5">
    nexus.await %done_5, %done_3
    %done_6 = nexus.submit_context.async @ctx_6(%arena) : !nexus.event<"done_6">
    nexus.await %done_6
    %done_7 = nexus.submit_context.async @ctx_7(%arena) : !nexus.event<"done_7">
    nexus.await %done_0, %done_1, %done_2, %done_3, %done_4, %done_5, %done_6, %done_7
    nexus.return
  }
}
