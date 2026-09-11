// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// B/C 先读共享 X；D 先以私有 Gate×Gate 执行 BOA100，之后才读 X，再执行 EVU100。
// B/C 私有 output Store 可先回收；X 必须等待真实最后 reader D，但不必持有到 D.compute_end。
// case: {"id":"N02","name":"n02_delayed_read","nodes":["B","C","D"],"edges":{"B":["X"],"C":["X"],"D":["Gate","X"]},"context_partition":[["B","C","D"]],"resource_config":{"uce":4,"device_contexts":4,"fidelity":"full_memory","num_dma_channels":2,"arena_binding":"arena=0x1000000:8388608:rw","placement":15},"expected":{"时序/生命周期 Correctness":"X 的 release 依赖 B/C/D 三个 input_released；D 在 Gate×Gate BOA100 后才 load X，且 release(X) 应早于 D 的 EVU100 结束","数值":"未建模；engine descriptor 与未初始化的合成输出只用于时序","Liveness":"所有已接纳 Context 完成","Scheduling Quality":"仅记录实际 service/等待，不声称最优调度"},"forbidden_dependencies":"禁止 C/B 的完成替代 D 的最后读取；禁止把 Gate 隐去后声称 D 仅依赖 X","tensors":{"X":{"index":0,"offset_elements":0,"bytes":32768,"arena_interval_bytes":[0,32768],"role":"shared_external_input"},"Gate":{"index":1,"offset_elements":131072,"bytes":32768,"arena_interval_bytes":[262144,294912],"role":"private_external_gate"},"B_out":{"index":2,"offset_elements":262144,"bytes":32768,"arena_interval_bytes":[524288,557056],"role":"output"},"C_out":{"index":3,"offset_elements":393216,"bytes":32768,"arena_interval_bytes":[786432,819200],"role":"output"},"D_out":{"index":4,"offset_elements":524288,"bytes":32768,"arena_interval_bytes":[1048576,1081344],"role":"output"}},"phase":{"B":["pref_X"],"C":["pref_X"],"D":["pref_Gate","pref_X"],"release_X":["read_B","read_C","read_D"]},"pins":{"B":0,"C":1,"D":2},"node_programs":{"B":{"program":"prog_B","engine":"evu:relu","repeat":1,"pin":0},"C":{"program":"prog_C","engine":"evu:relu","repeat":1,"pin":1},"D":{"program":"prog_D","engine":["boa:matmul","evu:relu"],"repeat":{"gate_boa":100,"post_read_evu":100},"pin":2}}}
builtin.module {
  tile.program @prog_B (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  tile.program @prog_C (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  tile.program @prog_D (%task: !nest.task, %gate: !nest.l2_buffer<4x64x64xbf16>, %x: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %vg = tile.subview %gate task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %vx = tile.subview %x task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %lg0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %lg1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %lx = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_gate0 = tile.load.async %vg into %lg0 : !tile.event<"load_gate0">
    %load_gate1 = tile.load.async %vg into %lg1 : !tile.event<"load_gate1">
    tile.await %load_gate0, %load_gate1
    %gate_compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"gate_compute_0">
    tile.await %gate_compute_0
    %gate_compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_1">
    tile.await %gate_compute_1
    %gate_compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_2">
    tile.await %gate_compute_2
    %gate_compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_3">
    tile.await %gate_compute_3
    %gate_compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_4">
    tile.await %gate_compute_4
    %gate_compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_5">
    tile.await %gate_compute_5
    %gate_compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_6">
    tile.await %gate_compute_6
    %gate_compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_7">
    tile.await %gate_compute_7
    %gate_compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_8">
    tile.await %gate_compute_8
    %gate_compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_9">
    tile.await %gate_compute_9
    %gate_compute_10 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_10">
    tile.await %gate_compute_10
    %gate_compute_11 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_11">
    tile.await %gate_compute_11
    %gate_compute_12 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_12">
    tile.await %gate_compute_12
    %gate_compute_13 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_13">
    tile.await %gate_compute_13
    %gate_compute_14 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_14">
    tile.await %gate_compute_14
    %gate_compute_15 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_15">
    tile.await %gate_compute_15
    %gate_compute_16 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_16">
    tile.await %gate_compute_16
    %gate_compute_17 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_17">
    tile.await %gate_compute_17
    %gate_compute_18 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_18">
    tile.await %gate_compute_18
    %gate_compute_19 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_19">
    tile.await %gate_compute_19
    %gate_compute_20 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_20">
    tile.await %gate_compute_20
    %gate_compute_21 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_21">
    tile.await %gate_compute_21
    %gate_compute_22 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_22">
    tile.await %gate_compute_22
    %gate_compute_23 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_23">
    tile.await %gate_compute_23
    %gate_compute_24 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_24">
    tile.await %gate_compute_24
    %gate_compute_25 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_25">
    tile.await %gate_compute_25
    %gate_compute_26 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_26">
    tile.await %gate_compute_26
    %gate_compute_27 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_27">
    tile.await %gate_compute_27
    %gate_compute_28 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_28">
    tile.await %gate_compute_28
    %gate_compute_29 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_29">
    tile.await %gate_compute_29
    %gate_compute_30 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_30">
    tile.await %gate_compute_30
    %gate_compute_31 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_31">
    tile.await %gate_compute_31
    %gate_compute_32 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_32">
    tile.await %gate_compute_32
    %gate_compute_33 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_33">
    tile.await %gate_compute_33
    %gate_compute_34 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_34">
    tile.await %gate_compute_34
    %gate_compute_35 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_35">
    tile.await %gate_compute_35
    %gate_compute_36 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_36">
    tile.await %gate_compute_36
    %gate_compute_37 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_37">
    tile.await %gate_compute_37
    %gate_compute_38 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_38">
    tile.await %gate_compute_38
    %gate_compute_39 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_39">
    tile.await %gate_compute_39
    %gate_compute_40 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_40">
    tile.await %gate_compute_40
    %gate_compute_41 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_41">
    tile.await %gate_compute_41
    %gate_compute_42 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_42">
    tile.await %gate_compute_42
    %gate_compute_43 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_43">
    tile.await %gate_compute_43
    %gate_compute_44 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_44">
    tile.await %gate_compute_44
    %gate_compute_45 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_45">
    tile.await %gate_compute_45
    %gate_compute_46 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_46">
    tile.await %gate_compute_46
    %gate_compute_47 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_47">
    tile.await %gate_compute_47
    %gate_compute_48 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_48">
    tile.await %gate_compute_48
    %gate_compute_49 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_49">
    tile.await %gate_compute_49
    %gate_compute_50 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_50">
    tile.await %gate_compute_50
    %gate_compute_51 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_51">
    tile.await %gate_compute_51
    %gate_compute_52 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_52">
    tile.await %gate_compute_52
    %gate_compute_53 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_53">
    tile.await %gate_compute_53
    %gate_compute_54 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_54">
    tile.await %gate_compute_54
    %gate_compute_55 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_55">
    tile.await %gate_compute_55
    %gate_compute_56 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_56">
    tile.await %gate_compute_56
    %gate_compute_57 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_57">
    tile.await %gate_compute_57
    %gate_compute_58 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_58">
    tile.await %gate_compute_58
    %gate_compute_59 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_59">
    tile.await %gate_compute_59
    %gate_compute_60 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_60">
    tile.await %gate_compute_60
    %gate_compute_61 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_61">
    tile.await %gate_compute_61
    %gate_compute_62 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_62">
    tile.await %gate_compute_62
    %gate_compute_63 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_63">
    tile.await %gate_compute_63
    %gate_compute_64 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_64">
    tile.await %gate_compute_64
    %gate_compute_65 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_65">
    tile.await %gate_compute_65
    %gate_compute_66 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_66">
    tile.await %gate_compute_66
    %gate_compute_67 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_67">
    tile.await %gate_compute_67
    %gate_compute_68 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_68">
    tile.await %gate_compute_68
    %gate_compute_69 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_69">
    tile.await %gate_compute_69
    %gate_compute_70 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_70">
    tile.await %gate_compute_70
    %gate_compute_71 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_71">
    tile.await %gate_compute_71
    %gate_compute_72 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_72">
    tile.await %gate_compute_72
    %gate_compute_73 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_73">
    tile.await %gate_compute_73
    %gate_compute_74 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_74">
    tile.await %gate_compute_74
    %gate_compute_75 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_75">
    tile.await %gate_compute_75
    %gate_compute_76 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_76">
    tile.await %gate_compute_76
    %gate_compute_77 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_77">
    tile.await %gate_compute_77
    %gate_compute_78 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_78">
    tile.await %gate_compute_78
    %gate_compute_79 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_79">
    tile.await %gate_compute_79
    %gate_compute_80 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_80">
    tile.await %gate_compute_80
    %gate_compute_81 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_81">
    tile.await %gate_compute_81
    %gate_compute_82 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_82">
    tile.await %gate_compute_82
    %gate_compute_83 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_83">
    tile.await %gate_compute_83
    %gate_compute_84 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_84">
    tile.await %gate_compute_84
    %gate_compute_85 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_85">
    tile.await %gate_compute_85
    %gate_compute_86 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_86">
    tile.await %gate_compute_86
    %gate_compute_87 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_87">
    tile.await %gate_compute_87
    %gate_compute_88 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_88">
    tile.await %gate_compute_88
    %gate_compute_89 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_89">
    tile.await %gate_compute_89
    %gate_compute_90 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_90">
    tile.await %gate_compute_90
    %gate_compute_91 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_91">
    tile.await %gate_compute_91
    %gate_compute_92 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_92">
    tile.await %gate_compute_92
    %gate_compute_93 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_93">
    tile.await %gate_compute_93
    %gate_compute_94 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_94">
    tile.await %gate_compute_94
    %gate_compute_95 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_95">
    tile.await %gate_compute_95
    %gate_compute_96 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_96">
    tile.await %gate_compute_96
    %gate_compute_97 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_97">
    tile.await %gate_compute_97
    %gate_compute_98 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_98">
    tile.await %gate_compute_98
    %gate_compute_99 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate : !tile.event<"gate_compute_99">
    tile.await %gate_compute_99
    %load_x = tile.load.async %vx into %lx : !tile.event<"load_x">
    tile.await %load_x
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_99">
    tile.await %compute_99
    %stored = tile.store.async %lx into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_X = nest.alloc slot = "X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_Gate = nest.alloc slot = "Gate" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B = nest.alloc slot = "B_out" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_C = nest.alloc slot = "C_out" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_D = nest.alloc slot = "D_out" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %h_Gate = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %h_B = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %h_C = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %h_D = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_X = nest.dma.prefetch.async %h_X into %b_X : !nest.event<"pref_X">
    %pref_Gate = nest.dma.prefetch.async %h_Gate into %b_Gate : !nest.event<"pref_Gate">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_B, %read_B, %ready_B = nest.dispatch.tasks.async @prog_B context = 0 tasks(%tasks) globals() bindings(%b_X, %b_B) ins(%b_X) outs(%b_B) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X) : (!nest.event<"grid_B">, !nest.event<"read_B">, !nest.event<"ready_B">)
    %grid_C, %read_C, %ready_C = nest.dispatch.tasks.async @prog_C context = 1 tasks(%tasks) globals() bindings(%b_X, %b_C) ins(%b_X) outs(%b_C) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X) : (!nest.event<"grid_C">, !nest.event<"read_C">, !nest.event<"ready_C">)
    %grid_D, %read_D, %ready_D = nest.dispatch.tasks.async @prog_D context = 2 tasks(%tasks) globals() bindings(%b_Gate, %b_X, %b_D) ins(%b_Gate, %b_X) outs(%b_D) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_Gate, %pref_X) : (!nest.event<"grid_D">, !nest.event<"read_D">, !nest.event<"ready_D">)
    %store_B = nest.dma.store.async %b_B into %h_B depends_on(%ready_B) : !nest.event<"store_B">
    nest.release %b_B depends_on(%store_B)
    %store_C = nest.dma.store.async %b_C into %h_C depends_on(%ready_C) : !nest.event<"store_C">
    nest.release %b_C depends_on(%store_C)
    nest.release %b_Gate depends_on(%read_D, %pref_Gate)
    nest.release %b_X depends_on(%read_B, %read_C, %read_D, %pref_X)
    %store_D = nest.dma.store.async %b_D into %h_D depends_on(%ready_D) : !nest.event<"store_D">
    nest.release %b_D depends_on(%store_D)
    nest.await %grid_B, %grid_C, %grid_D, %store_B, %store_C, %store_D
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
