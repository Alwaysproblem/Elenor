// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// 容量必须以 group_sram_bytes=196608 运行；三个 Context 连续 submit，中间没有 await。
// Holder 输入释放形成只有 2U 的窗口；Head(4U) 位于 FIFO 队头，Tail(2U) 不得绕过。
// case: {
//       "id": "N04",
//       "name": "n04_fifo_hol",
//       "nodes": ["Holder", "Head", "Tail"],
//       "edges": {},
//       "context_partition": [
//         ["Holder"],
//         ["Head"],
//         ["Tail"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device_contexts": 4,
//         "fidelity": "full_memory",
//         "num_dma_channels": 2,
//         "arena_binding": "arena=0x1000000:8388608:rw",
//         "group_sram_bytes": 196608,
//         "placement": 15
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "Holder 以 input2U+output4U 占满 6U；释放 input2U 时 Head 需要4U而不能接纳，Tail 虽只需2U也不得绕过；Holder output4U 释放后 admission 顺序 Head→Tail",
//         "数值": "未建模；engine descriptor 与未初始化的合成输出只用于时序",
//         "Liveness": "所有已接纳 Context 完成",
//         "Scheduling Quality": "仅记录实际 service/等待，不声称最优调度"
//       },
//       "forbidden_dependencies": "严格 FIFO 队头阻塞窗口禁止 Tail 绕过 Head；不得以不同 submit 时刻制造伪空闲",
//       "tensors": {
//         "Holder_input": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 65536,
//           "arena_interval_bytes": [0, 65536],
//           "role": "external_input"
//         },
//         "Head_input": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 65536,
//           "arena_interval_bytes": [262144, 327680],
//           "role": "external_input"
//         },
//         "Tail_input": {
//           "index": 2,
//           "offset_elements": 262144,
//           "bytes": 32768,
//           "arena_interval_bytes": [524288, 557056],
//           "role": "external_input"
//         },
//         "Holder_output": {
//           "index": 3,
//           "offset_elements": 393216,
//           "bytes": 131072,
//           "arena_interval_bytes": [786432, 917504],
//           "role": "output"
//         },
//         "Head_output": {
//           "index": 4,
//           "offset_elements": 524288,
//           "bytes": 65536,
//           "arena_interval_bytes": [1048576, 1114112],
//           "role": "output"
//         },
//         "Tail_output": {
//           "index": 5,
//           "offset_elements": 655360,
//           "bytes": 32768,
//           "arena_interval_bytes": [1310720, 1343488],
//           "role": "output"
//         }
//       },
//       "phase": {
//         "Holder": ["pref_Holder_input"],
//         "Head": ["fifo_admission_after_Holder_output_release"],
//         "Tail": ["fifo_admission_after_Head"]
//       },
//       "pins": {
//         "Holder": "device-slot",
//         "Head": "device-slot",
//         "Tail": "device-slot"
//       },
//       "node_programs": {
//         "Holder": {
//           "program": "prog_Holder",
//           "engine": "evu:expand",
//           "repeat": 100,
//           "pin": "device-slot"
//         },
//         "Head": {
//           "program": "prog_Head",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": "device-slot"
//         },
//         "Tail": {
//           "program": "prog_Tail",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": "device-slot"
//         }
//       }
//     }
builtin.module {
  tile.program @prog_Holder(
    %task: !nest.task, %i0: !nest.l2_buffer<4x128x64xbf16>, %out: !nest.l2_buffer<4x256x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %l0 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x256x64xbf16>
    %lout = tile.alloc shape = [256, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<256x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_99">
    tile.await %compute_99
    tile.free %l0
    %stored = tile.store.async %lout into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_Head(
    %task: !nest.task, %i0: !nest.l2_buffer<4x128x64xbf16>, %out: !nest.l2_buffer<4x128x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %l0 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_Tail(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_holder (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input = nest.alloc slot = "holder_input" role = "in" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %b_output = nest.alloc slot = "holder_output" role = "inout" shape = [4, 256, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x256x64xbf16>
    %h_input = nest.subview %arena offsets = [0] sizes = [32768] strides = [1]
      : !nest.global_view<32768xbf16>
    %h_output = nest.subview %arena offsets = [393216] sizes = [65536] strides = [1]
      : !nest.global_view<65536xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_Holder_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Holder, %read_Holder, %ready_Holder = nest.dispatch.tasks.async @prog_Holder tasks(%tasks)
      globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input)
      : (!nest.event<"grid_Holder">, !nest.event<"read_Holder">, !nest.event<"ready_Holder">)
    nest.release %b_input depends_on(%read_Holder, %pref_input)
    %store_Holder = nest.dma.store.async %b_output into %h_output depends_on(%ready_Holder)
      : !nest.event<"store_Holder">
    nest.release %b_output depends_on(%store_Holder)
    nest.await %grid_Holder, %store_Holder
    nest.return
  }
  nest.context @ctx_head (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input = nest.alloc slot = "head_input" role = "in" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %b_output = nest.alloc slot = "head_output" role = "inout" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %h_input = nest.subview %arena offsets = [131072] sizes = [32768] strides = [1]
      : !nest.global_view<32768xbf16>
    %h_output = nest.subview %arena offsets = [524288] sizes = [32768] strides = [1]
      : !nest.global_view<32768xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_Head_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Head, %read_Head, %ready_Head = nest.dispatch.tasks.async @prog_Head tasks(%tasks)
      globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input)
      : (!nest.event<"grid_Head">, !nest.event<"read_Head">, !nest.event<"ready_Head">)
    nest.release %b_input depends_on(%read_Head, %pref_input)
    %store_Head = nest.dma.store.async %b_output into %h_output depends_on(%ready_Head)
      : !nest.event<"store_Head">
    nest.release %b_output depends_on(%store_Head)
    nest.await %grid_Head, %store_Head
    nest.return
  }
  nest.context @ctx_tail (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input = nest.alloc slot = "tail_input" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_output = nest.alloc slot = "tail_output" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_output = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_Tail_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Tail, %read_Tail, %ready_Tail = nest.dispatch.tasks.async @prog_Tail tasks(%tasks)
      globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input)
      : (!nest.event<"grid_Tail">, !nest.event<"read_Tail">, !nest.event<"ready_Tail">)
    nest.release %b_input depends_on(%read_Tail, %pref_input)
    %store_Tail = nest.dma.store.async %b_output into %h_output depends_on(%ready_Tail)
      : !nest.event<"store_Tail">
    nest.release %b_output depends_on(%store_Tail)
    nest.await %grid_Tail, %store_Tail
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_holder = nexus.submit_context.async @ctx_holder(%arena) : !nexus.event<"done_holder">
    %done_head = nexus.submit_context.async @ctx_head(%arena) : !nexus.event<"done_head">
    %done_tail = nexus.submit_context.async @ctx_tail(%arena) : !nexus.event<"done_tail">
    nexus.await %done_holder, %done_head, %done_tail
    nexus.return
  }
}
