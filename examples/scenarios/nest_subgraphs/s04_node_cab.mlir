// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s04", "name": "s04_node_cab", "nodes": ["A", "B", "C", "D"], "edges": {"D": ["A", "B", "C"]}, "context_partition": [["A"], ["B"], ["C"], ["D"]], "resource_config": {"uce": 4, "device": 4, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_A": {"index": 0, "offset_elements": 0, "bytes": 32768}, "W_A": {"index": 1, "offset_elements": 131072, "bytes": 32768}, "input_B": {"index": 2, "offset_elements": 262144, "bytes": 32768}, "input_C": {"index": 3, "offset_elements": 393216, "bytes": 32768}, "A": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "B": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "C": {"index": 6, "offset_elements": 786432, "bytes": 32768}, "D": {"index": 7, "offset_elements": 917504, "bytes": 32768}}, "node_programs": {"A": {"program": "prog_A", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 10, "loads": 1}, "B": {"program": "prog_B", "pin": "随 device slot", "engine": "evu", "op": "pow", "repeat": 100, "loads": 1}, "C": {"program": "prog_C", "pin": "随 device slot", "engine": "evu", "op": "pow", "repeat": 1, "loads": 1}, "D": {"program": "prog_D", "pin": "随 device slot", "engine": "evu", "op": "add", "repeat": 1, "loads": 1}}, "consumer_ready": {"A": ["input_A.prefetch", "W_A.prefetch"], "B": ["input_B.prefetch"], "C": ["input_C.prefetch"], "D": ["A.context_done 后从同一 HBM 区间 prefetch", "B.context_done 后从同一 HBM 区间 prefetch", "C.context_done 后从同一 HBM 区间 prefetch"]}, "说明": "三个独立根 A/B/C 的 repeat=(10, 100, 1)；D 仍等待三个 distinct producer。"}
builtin.module {
  tile.program @prog_A (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_B (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_99">
    tile.await %compute_99
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_C (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.pow.async bytes = 8192 exponent = 2 pow_ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_D (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %i2: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v2 = tile.subview %i2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l2 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    %load_0_2 = tile.load.async %v2 into %l2 : !tile.event<"load_0_2">
    tile.await %load_0_0, %load_0_1, %load_0_2
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_A = nest.alloc slot = "input_A" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_A = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_A = nest.alloc slot = "W_A" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_A = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_A = nest.alloc slot = "A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_A = nest.dma.prefetch.async %h_input_A into %b_input_A : !nest.event<"pref_input_A">
    %pref_W_A = nest.dma.prefetch.async %h_W_A into %b_W_A : !nest.event<"pref_W_A">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A, %read_A, %ready_A = nest.dispatch.tasks.async @prog_A tasks(%tasks) globals() ins(%b_input_A, %b_W_A, %b_A) outs(%b_input_A, %b_W_A, %b_A) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_A, %pref_W_A) : (!nest.event<"grid_A">, !nest.event<"read_A">, !nest.event<"ready_A">)
    nest.release %b_input_A depends_on(%read_A)
    nest.release %b_W_A depends_on(%read_A)
    %store_A_0 = nest.dma.store.async %b_A into %h_A depends_on(%ready_A) : !nest.event<"store_A_0">
    nest.release %b_A depends_on(%store_A_0)
    nest.await %grid_A, %store_A_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_B = nest.alloc slot = "input_B" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_B = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_B = nest.alloc slot = "B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_B = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_B = nest.dma.prefetch.async %h_input_B into %b_input_B : !nest.event<"pref_input_B">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_B, %read_B, %ready_B = nest.dispatch.tasks.async @prog_B tasks(%tasks) globals() ins(%b_input_B, %b_B) outs(%b_input_B, %b_B) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_B) : (!nest.event<"grid_B">, !nest.event<"read_B">, !nest.event<"ready_B">)
    nest.release %b_input_B depends_on(%read_B)
    %store_B_0 = nest.dma.store.async %b_B into %h_B depends_on(%ready_B) : !nest.event<"store_B_0">
    nest.release %b_B depends_on(%store_B_0)
    nest.await %grid_B, %store_B_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_C = nest.alloc slot = "input_C" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_C = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_C = nest.alloc slot = "C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_C = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_C = nest.dma.prefetch.async %h_input_C into %b_input_C : !nest.event<"pref_input_C">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_C, %read_C, %ready_C = nest.dispatch.tasks.async @prog_C tasks(%tasks) globals() ins(%b_input_C, %b_C) outs(%b_input_C, %b_C) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_C) : (!nest.event<"grid_C">, !nest.event<"read_C">, !nest.event<"ready_C">)
    nest.release %b_input_C depends_on(%read_C)
    %store_C_0 = nest.dma.store.async %b_C into %h_C depends_on(%ready_C) : !nest.event<"store_C_0">
    nest.release %b_C depends_on(%store_C_0)
    nest.await %grid_C, %store_C_0
    nest.return
  }
  nest.context @ctx_3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_A = nest.alloc slot = "A" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_B = nest.alloc slot = "B" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_B = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_C = nest.alloc slot = "C" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_C = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_D = nest.alloc slot = "D" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_D = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_A = nest.dma.prefetch.async %h_A into %b_A : !nest.event<"pref_A">
    %pref_B = nest.dma.prefetch.async %h_B into %b_B : !nest.event<"pref_B">
    %pref_C = nest.dma.prefetch.async %h_C into %b_C : !nest.event<"pref_C">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_D, %read_D, %ready_D = nest.dispatch.tasks.async @prog_D tasks(%tasks) globals() ins(%b_A, %b_B, %b_C, %b_D) outs(%b_A, %b_B, %b_C, %b_D) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_A, %pref_B, %pref_C) : (!nest.event<"grid_D">, !nest.event<"read_D">, !nest.event<"ready_D">)
    nest.release %b_A depends_on(%read_D)
    nest.release %b_B depends_on(%read_D)
    nest.release %b_C depends_on(%read_D)
    %store_D_0 = nest.dma.store.async %b_D into %h_D depends_on(%ready_D) : !nest.event<"store_D_0">
    nest.release %b_D depends_on(%store_D_0)
    nest.await %grid_D, %store_D_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_0, %done_1, %done_2
    %done_3 = nexus.submit_context.async @ctx_3(%arena) : !nexus.event<"done_3">
    nexus.await %done_0, %done_1, %done_2, %done_3
    nexus.return
  }
}
