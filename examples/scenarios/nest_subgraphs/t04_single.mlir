// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {"id":"t04","name":"t04_single","nodes":["A0","B0","C0","A1","B1","C1","A2","B2","C2","A3","B3","C3"],"edges":{"B0":["A0"],"B1":["A1"],"B2":["A2"],"B3":["A3"],"C0":["B0"],"C1":["B1"],"C2":["B2"],"C3":["B3"]},"context_partition":[["A0","B0","C0","A1","B1","C1","A2","B2","C2","A3","B3","C3"]],"resource_config":{"uce_context_mode":4,"device_context_mode":1,"placement":15,"fidelity":"full_memory","bindings":{"R0":"R0=0x1000000:8388608:rw","R1":"R1=0x2000000:8388608:rw","R2":"R2=0x3000000:8388608:rw","R3":"R3=0x4000000:8388608:rw","W":"W=0x5000000:32768:r"}},"expected":{"时序/生命周期 Correctness":"检查真实 HBM/L2 数据边、input_released/output_ready、最终 Store 与释放顺序","数值":"未建模；仅使用真实搬运和合成 engine service，不作为 tensor 数值证明","Liveness":"应完成；以同次运行 trace 和退出状态确认","Scheduling Quality":"单一 Context 内先 issue R0/R1/R2 的 grid，等待 R1 的 C grid 后再启动 R3；此映射不能证明 device-slot generation"},"forbidden_dependencies":"只允许表列真数据边；UCE 共享 pin 争用不是数据依赖；跨 Context 仅使用当前 IR 可见的 context_done/HBM 可见性","tensors":{"R0.input":{"formal":"R0","index":0,"offset_elements":0,"interval_bytes":[0,32768],"bytes":32768},"R0.A":{"formal":"R0","index":1,"offset_elements":131072,"interval_bytes":[262144,294912],"bytes":32768},"R0.B":{"formal":"R0","index":2,"offset_elements":262144,"interval_bytes":[524288,557056],"bytes":32768},"R0.C":{"formal":"R0","index":3,"offset_elements":393216,"interval_bytes":[786432,819200],"bytes":32768},"R1.input":{"formal":"R1","index":0,"offset_elements":0,"interval_bytes":[0,32768],"bytes":32768},"R1.A":{"formal":"R1","index":1,"offset_elements":131072,"interval_bytes":[262144,294912],"bytes":32768},"R1.B":{"formal":"R1","index":2,"offset_elements":262144,"interval_bytes":[524288,557056],"bytes":32768},"R1.C":{"formal":"R1","index":3,"offset_elements":393216,"interval_bytes":[786432,819200],"bytes":32768},"R2.input":{"formal":"R2","index":0,"offset_elements":0,"interval_bytes":[0,32768],"bytes":32768},"R2.A":{"formal":"R2","index":1,"offset_elements":131072,"interval_bytes":[262144,294912],"bytes":32768},"R2.B":{"formal":"R2","index":2,"offset_elements":262144,"interval_bytes":[524288,557056],"bytes":32768},"R2.C":{"formal":"R2","index":3,"offset_elements":393216,"interval_bytes":[786432,819200],"bytes":32768},"R3.input":{"formal":"R3","index":0,"offset_elements":0,"interval_bytes":[0,32768],"bytes":32768},"R3.A":{"formal":"R3","index":1,"offset_elements":131072,"interval_bytes":[262144,294912],"bytes":32768},"R3.B":{"formal":"R3","index":2,"offset_elements":262144,"interval_bytes":[524288,557056],"bytes":32768},"R3.C":{"formal":"R3","index":3,"offset_elements":393216,"interval_bytes":[786432,819200],"bytes":32768},"W":{"formal":"W","index":0,"offset_elements":0,"interval_bytes":[0,32768],"bytes":32768,"readonly":true}},"node_programs":{"A0":{"program":"prog_A_slow","pin":0,"engine":"BOA:matmul","repeat":100},"B0":{"program":"prog_B","pin":0,"engine":"EVU:add","repeat":1},"C0":{"program":"prog_C","pin":0,"engine":"EVU:relu","repeat":1},"A1":{"program":"prog_A_fast","pin":1,"engine":"EVU:relu","repeat":1},"B1":{"program":"prog_B","pin":1,"engine":"EVU:add","repeat":1},"C1":{"program":"prog_C","pin":1,"engine":"EVU:relu","repeat":1},"A2":{"program":"prog_A_fast","pin":2,"engine":"EVU:relu","repeat":1},"B2":{"program":"prog_B","pin":2,"engine":"EVU:add","repeat":1},"C2":{"program":"prog_C","pin":2,"engine":"EVU:relu","repeat":1},"A3":{"program":"prog_A_fast","pin":3,"engine":"EVU:relu","repeat":1},"B3":{"program":"prog_B","pin":3,"engine":"EVU:add","repeat":1},"C3":{"program":"prog_C","pin":3,"engine":"EVU:relu","repeat":1}},"consumer_ready_granularity":{"A0->B0":"output_ready(all_tasks)","B0->C0":"output_ready(all_tasks)","A1->B1":"output_ready(all_tasks)","B1->C1":"output_ready(all_tasks)","A2->B2":"output_ready(all_tasks)","B2->C2":"output_ready(all_tasks)","A3->B3":"output_ready(all_tasks)","B3->C3":"output_ready(all_tasks)"},"notes":{"request_intervals":"每个 Rr formal 内 interval 0/1/2/3 分别为 input/A/B/C；请求间不别名","shared_weight":"W 是 4x64x64xbf16 只读 formal，所有 A 实际读取","reuse_identity":"stage 的 R1/R2/R3（uniform 还包括 R0）重复提交同一 ctx_request_fast；局部 event tag 相同，nexus done_Rr 唯一","generation_scope":"只证明正常重复 submit 的 namespace/slot generation 隔离；不伪造旧通知注入"}}
builtin.module {
  tile.program @prog_A_slow (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
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
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_A_fast (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_B (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
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
  nest.context @ctx_all_requests (%R0: !nest.global_memref<4194304xbf16>, %R1: !nest.global_memref<4194304xbf16>, %R2: !nest.global_memref<4194304xbf16>, %R3: !nest.global_memref<4194304xbf16>, %W: !nest.global_memref<4x64x64xbf16>) placement = 15 {
    %b_W = nest.alloc slot = "shared_W" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W = nest.subview %W offsets = [0, 0, 0] sizes = [4, 64, 64] strides = [1, 1, 1] : !nest.global_view<4x64x64xbf16>
    %b_R0_input = nest.alloc slot = "R0_input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0_input = nest.subview %R0 offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R0_A = nest.alloc slot = "R0_A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0_A = nest.subview %R0 offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R0_B = nest.alloc slot = "R0_B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0_B = nest.subview %R0 offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R0_C = nest.alloc slot = "R0_C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0_C = nest.subview %R0 offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R1_input = nest.alloc slot = "R1_input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R1_input = nest.subview %R1 offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R1_A = nest.alloc slot = "R1_A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R1_A = nest.subview %R1 offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R1_B = nest.alloc slot = "R1_B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R1_B = nest.subview %R1 offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R1_C = nest.alloc slot = "R1_C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R1_C = nest.subview %R1 offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R2_input = nest.alloc slot = "R2_input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R2_input = nest.subview %R2 offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R2_A = nest.alloc slot = "R2_A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R2_A = nest.subview %R2 offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R2_B = nest.alloc slot = "R2_B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R2_B = nest.subview %R2 offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R2_C = nest.alloc slot = "R2_C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R2_C = nest.subview %R2 offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R3_input = nest.alloc slot = "R3_input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R3_input = nest.subview %R3 offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R3_A = nest.alloc slot = "R3_A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R3_A = nest.subview %R3 offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R3_B = nest.alloc slot = "R3_B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R3_B = nest.subview %R3 offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R3_C = nest.alloc slot = "R3_C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R3_C = nest.subview %R3 offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_W = nest.dma.prefetch.async %h_W into %b_W : !nest.event<"pref_W">
    %pref_R0_input = nest.dma.prefetch.async %h_R0_input into %b_R0_input : !nest.event<"pref_R0_input">
    %pref_R1_input = nest.dma.prefetch.async %h_R1_input into %b_R1_input : !nest.event<"pref_R1_input">
    %pref_R2_input = nest.dma.prefetch.async %h_R2_input into %b_R2_input : !nest.event<"pref_R2_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A0, %read_A0, %ready_A0 = nest.dispatch.tasks.async @prog_A_slow context = 0 tasks(%tasks) globals() ins(%b_R0_input, %b_W, %b_R0_A) outs(%b_R0_input, %b_W, %b_R0_A) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R0_input, %pref_W) : (!nest.event<"grid_A0">, !nest.event<"read_A0">, !nest.event<"ready_A0">)
    %grid_A1, %read_A1, %ready_A1 = nest.dispatch.tasks.async @prog_A_fast context = 1 tasks(%tasks) globals() ins(%b_R1_input, %b_W, %b_R1_A) outs(%b_R1_input, %b_W, %b_R1_A) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R1_input, %pref_W) : (!nest.event<"grid_A1">, !nest.event<"read_A1">, !nest.event<"ready_A1">)
    %grid_A2, %read_A2, %ready_A2 = nest.dispatch.tasks.async @prog_A_fast context = 2 tasks(%tasks) globals() ins(%b_R2_input, %b_W, %b_R2_A) outs(%b_R2_input, %b_W, %b_R2_A) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R2_input, %pref_W) : (!nest.event<"grid_A2">, !nest.event<"read_A2">, !nest.event<"ready_A2">)
    %grid_B0, %read_B0, %ready_B0 = nest.dispatch.tasks.async @prog_B context = 0 tasks(%tasks) globals() ins(%b_R0_A, %b_R0_B) outs(%b_R0_A, %b_R0_B) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_A0) : (!nest.event<"grid_B0">, !nest.event<"read_B0">, !nest.event<"ready_B0">)
    %grid_C0, %read_C0, %ready_C0 = nest.dispatch.tasks.async @prog_C context = 0 tasks(%tasks) globals() ins(%b_R0_B, %b_R0_C) outs(%b_R0_B, %b_R0_C) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_B0) : (!nest.event<"grid_C0">, !nest.event<"read_C0">, !nest.event<"ready_C0">)
    %grid_B1, %read_B1, %ready_B1 = nest.dispatch.tasks.async @prog_B context = 1 tasks(%tasks) globals() ins(%b_R1_A, %b_R1_B) outs(%b_R1_A, %b_R1_B) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_A1) : (!nest.event<"grid_B1">, !nest.event<"read_B1">, !nest.event<"ready_B1">)
    %grid_C1, %read_C1, %ready_C1 = nest.dispatch.tasks.async @prog_C context = 1 tasks(%tasks) globals() ins(%b_R1_B, %b_R1_C) outs(%b_R1_B, %b_R1_C) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_B1) : (!nest.event<"grid_C1">, !nest.event<"read_C1">, !nest.event<"ready_C1">)
    %grid_B2, %read_B2, %ready_B2 = nest.dispatch.tasks.async @prog_B context = 2 tasks(%tasks) globals() ins(%b_R2_A, %b_R2_B) outs(%b_R2_A, %b_R2_B) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_A2) : (!nest.event<"grid_B2">, !nest.event<"read_B2">, !nest.event<"ready_B2">)
    %grid_C2, %read_C2, %ready_C2 = nest.dispatch.tasks.async @prog_C context = 2 tasks(%tasks) globals() ins(%b_R2_B, %b_R2_C) outs(%b_R2_B, %b_R2_C) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_B2) : (!nest.event<"grid_C2">, !nest.event<"read_C2">, !nest.event<"ready_C2">)
    nest.await %grid_C1
    %pref_R3_input = nest.dma.prefetch.async %h_R3_input into %b_R3_input : !nest.event<"pref_R3_input">
    %grid_A3, %read_A3, %ready_A3 = nest.dispatch.tasks.async @prog_A_fast context = 3 tasks(%tasks) globals() ins(%b_R3_input, %b_W, %b_R3_A) outs(%b_R3_input, %b_W, %b_R3_A) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R3_input, %pref_W) : (!nest.event<"grid_A3">, !nest.event<"read_A3">, !nest.event<"ready_A3">)
    %grid_B3, %read_B3, %ready_B3 = nest.dispatch.tasks.async @prog_B context = 3 tasks(%tasks) globals() ins(%b_R3_A, %b_R3_B) outs(%b_R3_A, %b_R3_B) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_A3) : (!nest.event<"grid_B3">, !nest.event<"read_B3">, !nest.event<"ready_B3">)
    %grid_C3, %read_C3, %ready_C3 = nest.dispatch.tasks.async @prog_C context = 3 tasks(%tasks) globals() ins(%b_R3_B, %b_R3_C) outs(%b_R3_B, %b_R3_C) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_B3) : (!nest.event<"grid_C3">, !nest.event<"read_C3">, !nest.event<"ready_C3">)
    nest.release %b_R0_input depends_on(%read_A0)
    nest.release %b_R1_input depends_on(%read_A1)
    nest.release %b_R2_input depends_on(%read_A2)
    nest.release %b_R3_input depends_on(%read_A3)
    nest.release %b_W depends_on(%read_A0, %read_A1, %read_A2, %read_A3)
    %store_A0 = nest.dma.store.async %b_R0_A into %h_R0_A depends_on(%ready_A0, %ready_B0) : !nest.event<"store_A0">
    nest.release %b_R0_A depends_on(%store_A0)
    %store_B0 = nest.dma.store.async %b_R0_B into %h_R0_B depends_on(%ready_B0, %ready_C0) : !nest.event<"store_B0">
    nest.release %b_R0_B depends_on(%store_B0)
    %store_C0 = nest.dma.store.async %b_R0_C into %h_R0_C depends_on(%ready_C0) : !nest.event<"store_C0">
    nest.release %b_R0_C depends_on(%store_C0)
    %store_A1 = nest.dma.store.async %b_R1_A into %h_R1_A depends_on(%ready_A1, %ready_B1) : !nest.event<"store_A1">
    nest.release %b_R1_A depends_on(%store_A1)
    %store_B1 = nest.dma.store.async %b_R1_B into %h_R1_B depends_on(%ready_B1, %ready_C1) : !nest.event<"store_B1">
    nest.release %b_R1_B depends_on(%store_B1)
    %store_C1 = nest.dma.store.async %b_R1_C into %h_R1_C depends_on(%ready_C1) : !nest.event<"store_C1">
    nest.release %b_R1_C depends_on(%store_C1)
    %store_A2 = nest.dma.store.async %b_R2_A into %h_R2_A depends_on(%ready_A2, %ready_B2) : !nest.event<"store_A2">
    nest.release %b_R2_A depends_on(%store_A2)
    %store_B2 = nest.dma.store.async %b_R2_B into %h_R2_B depends_on(%ready_B2, %ready_C2) : !nest.event<"store_B2">
    nest.release %b_R2_B depends_on(%store_B2)
    %store_C2 = nest.dma.store.async %b_R2_C into %h_R2_C depends_on(%ready_C2) : !nest.event<"store_C2">
    nest.release %b_R2_C depends_on(%store_C2)
    %store_A3 = nest.dma.store.async %b_R3_A into %h_R3_A depends_on(%ready_A3, %ready_B3) : !nest.event<"store_A3">
    nest.release %b_R3_A depends_on(%store_A3)
    %store_B3 = nest.dma.store.async %b_R3_B into %h_R3_B depends_on(%ready_B3, %ready_C3) : !nest.event<"store_B3">
    nest.release %b_R3_B depends_on(%store_B3)
    %store_C3 = nest.dma.store.async %b_R3_C into %h_R3_C depends_on(%ready_C3) : !nest.event<"store_C3">
    nest.release %b_R3_C depends_on(%store_C3)
    nest.await %grid_A0, %grid_B0, %grid_C0, %grid_A1, %grid_B1, %grid_C1, %grid_A2, %grid_B2, %grid_C2, %grid_A3, %grid_B3, %grid_C3, %store_A0, %store_B0, %store_C0, %store_A1, %store_B1, %store_C1, %store_A2, %store_B2, %store_C2, %store_A3, %store_B3, %store_C3
    nest.return
  }
  nexus.program @run (%R0: !nest.global_memref<4194304xbf16>, %R1: !nest.global_memref<4194304xbf16>, %R2: !nest.global_memref<4194304xbf16>, %R3: !nest.global_memref<4194304xbf16>, %W: !nest.global_memref<4x64x64xbf16>) {
    %done_all = nexus.submit_context.async @ctx_all_requests(%R0, %R1, %R2, %R3, %W) : !nexus.event<"done_all">
    nexus.await %done_all
    nexus.return
  }
}
