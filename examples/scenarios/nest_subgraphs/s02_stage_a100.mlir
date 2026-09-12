// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "s02",
//       "name": "s02_stage_a100",
//       "nodes": ["A0", "A1", "A2", "B0", "B1", "C0", "C1", "C2"],
//       "edges": {
//         "A1": ["A0"],
//         "A2": ["A1"],
//         "B1": ["B0"],
//         "C1": ["C0"],
//         "C2": ["C1"]
//       },
//       "context_partition": [
//         ["A0", "A1", "A2"],
//         ["B0", "B1"],
//         ["C0", "C1", "C2"]
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
//       "tensors": {
//         "input_A0": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 32768
//         },
//         "W_A0": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 32768
//         },
//         "W_A1": {
//           "index": 2,
//           "offset_elements": 262144,
//           "bytes": 32768
//         },
//         "W_A2": {
//           "index": 3,
//           "offset_elements": 393216,
//           "bytes": 32768
//         },
//         "input_B0": {
//           "index": 4,
//           "offset_elements": 524288,
//           "bytes": 32768
//         },
//         "input_C0": {
//           "index": 5,
//           "offset_elements": 655360,
//           "bytes": 32768
//         },
//         "A0": {
//           "index": 6,
//           "offset_elements": 786432,
//           "bytes": 32768
//         },
//         "A1": {
//           "index": 7,
//           "offset_elements": 917504,
//           "bytes": 32768
//         },
//         "A2": {
//           "index": 8,
//           "offset_elements": 1048576,
//           "bytes": 32768
//         },
//         "B0": {
//           "index": 9,
//           "offset_elements": 1179648,
//           "bytes": 32768
//         },
//         "B1": {
//           "index": 10,
//           "offset_elements": 1310720,
//           "bytes": 32768
//         },
//         "C0": {
//           "index": 11,
//           "offset_elements": 1441792,
//           "bytes": 32768
//         },
//         "C1": {
//           "index": 12,
//           "offset_elements": 1572864,
//           "bytes": 32768
//         },
//         "C2": {
//           "index": 13,
//           "offset_elements": 1703936,
//           "bytes": 32768
//         }
//       },
//       "node_programs": {
//         "A0": {
//           "program": "prog_A0",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 100,
//           "loads": 1
//         },
//         "A1": {
//           "program": "prog_A1",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "A2": {
//           "program": "prog_A2",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "B0": {
//           "program": "prog_B0",
//           "pin": 1,
//           "engine": "evu",
//           "op": "pow",
//           "repeat": 1,
//           "loads": 1
//         },
//         "B1": {
//           "program": "prog_B1",
//           "pin": 1,
//           "engine": "evu",
//           "op": "pow",
//           "repeat": 1,
//           "loads": 1
//         },
//         "C0": {
//           "program": "prog_C0",
//           "pin": 2,
//           "engine": "evu",
//           "op": "relu",
//           "repeat": 1,
//           "loads": 1
//         },
//         "C1": {
//           "program": "prog_C1",
//           "pin": 2,
//           "engine": "evu",
//           "op": "relu",
//           "repeat": 1,
//           "loads": 1
//         },
//         "C2": {
//           "program": "prog_C2",
//           "pin": 2,
//           "engine": "evu",
//           "op": "relu",
//           "repeat": 1,
//           "loads": 1
//         }
//       },
//       "consumer_ready": {
//         "A0": ["input_A0.prefetch", "W_A0.prefetch"],
//         "A1": ["A0.output_ready", "W_A1.prefetch"],
//         "A2": ["A1.output_ready", "W_A2.prefetch"],
//         "B0": ["input_B0.prefetch"],
//         "B1": ["B0.output_ready"],
//         "C0": ["input_C0.prefetch"],
//         "C1": ["C0.output_ready"],
//         "C2": ["C1.output_ready"]
//       },
//       "说明": "三条独立链连续 submit；链内只保留真实前驱。"
//     }
builtin.module {
  tile.program @prog_A0(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>,
    %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_99">
    tile.await %compute_99
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_A1(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>,
    %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_A2(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>,
    %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_B0(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  tile.program @prog_B1(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  tile.program @prog_C0(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_C1(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_C2(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_A0 = nest.alloc slot = "input_A0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_A0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_A0 = nest.alloc slot = "W_A0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_A0 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_A0 = nest.alloc slot = "A0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_A0 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_A1 = nest.alloc slot = "W_A1" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_A1 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_A1 = nest.alloc slot = "A1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_A1 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_A2 = nest.alloc slot = "W_A2" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_A2 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_A2 = nest.alloc slot = "A2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_A2 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input_A0 = nest.dma.prefetch.async %h_input_A0 into %b_input_A0
      : !nest.event<"pref_input_A0">
    %pref_W_A0 = nest.dma.prefetch.async %h_W_A0 into %b_W_A0 : !nest.event<"pref_W_A0">
    %pref_W_A1 = nest.dma.prefetch.async %h_W_A1 into %b_W_A1 : !nest.event<"pref_W_A1">
    %pref_W_A2 = nest.dma.prefetch.async %h_W_A2 into %b_W_A2 : !nest.event<"pref_W_A2">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A0, %read_A0, %ready_A0 = nest.dispatch.tasks.async @prog_A0 context = 0 tasks(%tasks)
      globals() bindings(%b_input_A0, %b_W_A0, %b_A0) ins(%b_input_A0, %b_W_A0) outs(%b_A0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input_A0, %pref_W_A0)
      : (!nest.event<"grid_A0">, !nest.event<"read_A0">, !nest.event<"ready_A0">)
    %grid_A1, %read_A1, %ready_A1 = nest.dispatch.tasks.async @prog_A1 context = 0 tasks(%tasks)
      globals() bindings(%b_A0, %b_W_A1, %b_A1) ins(%b_A0, %b_W_A1) outs(%b_A1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_A0, %pref_W_A1)
      : (!nest.event<"grid_A1">, !nest.event<"read_A1">, !nest.event<"ready_A1">)
    %grid_A2, %read_A2, %ready_A2 = nest.dispatch.tasks.async @prog_A2 context = 0 tasks(%tasks)
      globals() bindings(%b_A1, %b_W_A2, %b_A2) ins(%b_A1, %b_W_A2) outs(%b_A2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_A1, %pref_W_A2)
      : (!nest.event<"grid_A2">, !nest.event<"read_A2">, !nest.event<"ready_A2">)
    nest.release %b_input_A0 depends_on(%read_A0, %pref_input_A0)
    nest.release %b_W_A0 depends_on(%read_A0, %pref_W_A0)
    nest.release %b_W_A1 depends_on(%read_A1, %pref_W_A1)
    nest.release %b_W_A2 depends_on(%read_A2, %pref_W_A2)
    %store_A0_0 = nest.dma.store.async %b_A0 into %h_A0 depends_on(%ready_A0)
      : !nest.event<"store_A0_0">
    nest.release %b_A0 depends_on(%read_A1, %store_A0_0)
    %store_A1_0 = nest.dma.store.async %b_A1 into %h_A1 depends_on(%ready_A1)
      : !nest.event<"store_A1_0">
    nest.release %b_A1 depends_on(%read_A2, %store_A1_0)
    %store_A2_0 = nest.dma.store.async %b_A2 into %h_A2 depends_on(%ready_A2)
      : !nest.event<"store_A2_0">
    nest.release %b_A2 depends_on(%store_A2_0)
    nest.await %grid_A0, %grid_A1, %grid_A2, %store_A0_0, %store_A1_0, %store_A2_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_B0 = nest.alloc slot = "input_B0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_B0 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_B0 = nest.alloc slot = "B0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_B0 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_B1 = nest.alloc slot = "B1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_B1 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input_B0 = nest.dma.prefetch.async %h_input_B0 into %b_input_B0
      : !nest.event<"pref_input_B0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_B0, %read_B0, %ready_B0 = nest.dispatch.tasks.async @prog_B0 context = 1 tasks(%tasks)
      globals() bindings(%b_input_B0, %b_B0) ins(%b_input_B0) outs(%b_B0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input_B0)
      : (!nest.event<"grid_B0">, !nest.event<"read_B0">, !nest.event<"ready_B0">)
    %grid_B1, %read_B1, %ready_B1 = nest.dispatch.tasks.async @prog_B1 context = 1 tasks(%tasks)
      globals() bindings(%b_B0, %b_B1) ins(%b_B0) outs(%b_B1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_B0)
      : (!nest.event<"grid_B1">, !nest.event<"read_B1">, !nest.event<"ready_B1">)
    nest.release %b_input_B0 depends_on(%read_B0, %pref_input_B0)
    %store_B0_0 = nest.dma.store.async %b_B0 into %h_B0 depends_on(%ready_B0)
      : !nest.event<"store_B0_0">
    nest.release %b_B0 depends_on(%read_B1, %store_B0_0)
    %store_B1_0 = nest.dma.store.async %b_B1 into %h_B1 depends_on(%ready_B1)
      : !nest.event<"store_B1_0">
    nest.release %b_B1 depends_on(%store_B1_0)
    nest.await %grid_B0, %grid_B1, %store_B0_0, %store_B1_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_C0 = nest.alloc slot = "input_C0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_C0 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_C0 = nest.alloc slot = "C0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_C0 = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_C1 = nest.alloc slot = "C1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_C1 = nest.subview %arena offsets = [1572864] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_C2 = nest.alloc slot = "C2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_C2 = nest.subview %arena offsets = [1703936] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input_C0 = nest.dma.prefetch.async %h_input_C0 into %b_input_C0
      : !nest.event<"pref_input_C0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_C0, %read_C0, %ready_C0 = nest.dispatch.tasks.async @prog_C0 context = 2 tasks(%tasks)
      globals() bindings(%b_input_C0, %b_C0) ins(%b_input_C0) outs(%b_C0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input_C0)
      : (!nest.event<"grid_C0">, !nest.event<"read_C0">, !nest.event<"ready_C0">)
    %grid_C1, %read_C1, %ready_C1 = nest.dispatch.tasks.async @prog_C1 context = 2 tasks(%tasks)
      globals() bindings(%b_C0, %b_C1) ins(%b_C0) outs(%b_C1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_C0)
      : (!nest.event<"grid_C1">, !nest.event<"read_C1">, !nest.event<"ready_C1">)
    %grid_C2, %read_C2, %ready_C2 = nest.dispatch.tasks.async @prog_C2 context = 2 tasks(%tasks)
      globals() bindings(%b_C1, %b_C2) ins(%b_C1) outs(%b_C2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_C1)
      : (!nest.event<"grid_C2">, !nest.event<"read_C2">, !nest.event<"ready_C2">)
    nest.release %b_input_C0 depends_on(%read_C0, %pref_input_C0)
    %store_C0_0 = nest.dma.store.async %b_C0 into %h_C0 depends_on(%ready_C0)
      : !nest.event<"store_C0_0">
    nest.release %b_C0 depends_on(%read_C1, %store_C0_0)
    %store_C1_0 = nest.dma.store.async %b_C1 into %h_C1 depends_on(%ready_C1)
      : !nest.event<"store_C1_0">
    nest.release %b_C1 depends_on(%read_C2, %store_C1_0)
    %store_C2_0 = nest.dma.store.async %b_C2 into %h_C2 depends_on(%ready_C2)
      : !nest.event<"store_C2_0">
    nest.release %b_C2 depends_on(%store_C2_0)
    nest.await %grid_C0, %grid_C1, %grid_C2, %store_C0_0, %store_C1_0, %store_C2_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_0, %done_1, %done_2
    nexus.return
  }
}
