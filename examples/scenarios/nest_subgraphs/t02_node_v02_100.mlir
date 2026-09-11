// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {"id":"t02","name":"t02_node_v02_100","nodes":["V00","V01","V10","V11","V20","V02","V21","V12","V22"],"edges":{"V01":["V00"],"V10":["V00"],"V11":["V01","V10"],"V20":["V10"],"V02":["V01"],"V21":["V11","V20"],"V12":["V02","V11"],"V22":["V12","V21"]},"context_partition":[["V00"],["V01"],["V10"],["V11"],["V20"],["V02"],["V21"],["V12"],["V22"]],"resource_config":{"uce_context_mode":4,"device_context_mode":4,"placement":15,"fidelity":"full_memory","arena_binding":"arena=0x1000000:8388608:rw"},"expected":{"时序/生命周期 Correctness":"检查真实 HBM/L2 数据边、input_released/output_ready、最终 Store 与释放顺序","数值":"未建模；仅使用真实搬运和合成 engine service，不作为 tensor 数值证明","Liveness":"应完成；以同次运行 trace 和退出状态确认","Scheduling Quality":"node 基础顺序优先左下推进；V02 重 compute100，检查无关 V20 是否推进"},"forbidden_dependencies":"只允许表列真数据边；UCE 共享 pin 争用不是数据依赖；跨 Context 仅使用当前 IR 可见的 context_done/HBM 可见性","tensors":{"Input":{"formal":"arena","index":0,"offset_elements":0,"interval_bytes":[0,32768],"bytes":32768},"V00":{"formal":"arena","index":1,"offset_elements":131072,"interval_bytes":[262144,294912],"bytes":32768},"V01":{"formal":"arena","index":2,"offset_elements":262144,"interval_bytes":[524288,557056],"bytes":32768},"V10":{"formal":"arena","index":3,"offset_elements":393216,"interval_bytes":[786432,819200],"bytes":32768},"V11":{"formal":"arena","index":4,"offset_elements":524288,"interval_bytes":[1048576,1081344],"bytes":32768},"V20":{"formal":"arena","index":5,"offset_elements":655360,"interval_bytes":[1310720,1343488],"bytes":32768},"V02":{"formal":"arena","index":6,"offset_elements":786432,"interval_bytes":[1572864,1605632],"bytes":32768},"V21":{"formal":"arena","index":7,"offset_elements":917504,"interval_bytes":[1835008,1867776],"bytes":32768},"V12":{"formal":"arena","index":8,"offset_elements":1048576,"interval_bytes":[2097152,2129920],"bytes":32768},"V22":{"formal":"arena","index":9,"offset_elements":1179648,"interval_bytes":[2359296,2392064],"bytes":32768}},"node_programs":{"V00":{"program":"prog_V00","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V01":{"program":"prog_V01","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V10":{"program":"prog_V10","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V11":{"program":"prog_V11","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V20":{"program":"prog_V20","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V02":{"program":"prog_V02","pin":"跟随 device slot","engine":"EVU:add","repeat":100},"V21":{"program":"prog_V21","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V12":{"program":"prog_V12","pin":"跟随 device slot","engine":"EVU:add","repeat":1},"V22":{"program":"prog_V22","pin":"跟随 device slot","engine":"EVU:add","repeat":1}},"consumer_ready_granularity":{"Input->V00":"prefetch 完成","V00->V01":"producer context_done 后从同一 HBM 区间 prefetch","V00->V10":"producer context_done 后从同一 HBM 区间 prefetch","V01->V11":"producer context_done 后从同一 HBM 区间 prefetch","V10->V11":"producer context_done 后从同一 HBM 区间 prefetch","V10->V20":"producer context_done 后从同一 HBM 区间 prefetch","V01->V02":"producer context_done 后从同一 HBM 区间 prefetch","V11->V21":"producer context_done 后从同一 HBM 区间 prefetch","V20->V21":"producer context_done 后从同一 HBM 区间 prefetch","V02->V12":"producer context_done 后从同一 HBM 区间 prefetch","V11->V12":"producer context_done 后从同一 HBM 区间 prefetch","V12->V22":"producer context_done 后从同一 HBM 区间 prefetch","V21->V22":"producer context_done 后从同一 HBM 区间 prefetch"},"notes":{"logical_coordinates":"Vij 中 i/j 是逻辑坐标，不是固定物理 tile id","submission_order":["V00","V01","V10","V11","V20","V02","V21","V12","V22"]}}
builtin.module {
  tile.program @prog_V00 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_V01 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_V10 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_V11 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_V20 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_V02 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_9">
    tile.await %compute_9
    %compute_10 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_10">
    tile.await %compute_10
    %compute_11 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_11">
    tile.await %compute_11
    %compute_12 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_12">
    tile.await %compute_12
    %compute_13 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_13">
    tile.await %compute_13
    %compute_14 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_14">
    tile.await %compute_14
    %compute_15 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_15">
    tile.await %compute_15
    %compute_16 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_16">
    tile.await %compute_16
    %compute_17 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_17">
    tile.await %compute_17
    %compute_18 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_18">
    tile.await %compute_18
    %compute_19 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_19">
    tile.await %compute_19
    %compute_20 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_20">
    tile.await %compute_20
    %compute_21 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_21">
    tile.await %compute_21
    %compute_22 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_22">
    tile.await %compute_22
    %compute_23 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_23">
    tile.await %compute_23
    %compute_24 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_24">
    tile.await %compute_24
    %compute_25 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_25">
    tile.await %compute_25
    %compute_26 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_26">
    tile.await %compute_26
    %compute_27 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_27">
    tile.await %compute_27
    %compute_28 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_28">
    tile.await %compute_28
    %compute_29 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_29">
    tile.await %compute_29
    %compute_30 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_30">
    tile.await %compute_30
    %compute_31 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_31">
    tile.await %compute_31
    %compute_32 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_32">
    tile.await %compute_32
    %compute_33 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_33">
    tile.await %compute_33
    %compute_34 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_34">
    tile.await %compute_34
    %compute_35 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_35">
    tile.await %compute_35
    %compute_36 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_36">
    tile.await %compute_36
    %compute_37 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_37">
    tile.await %compute_37
    %compute_38 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_38">
    tile.await %compute_38
    %compute_39 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_39">
    tile.await %compute_39
    %compute_40 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_40">
    tile.await %compute_40
    %compute_41 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_41">
    tile.await %compute_41
    %compute_42 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_42">
    tile.await %compute_42
    %compute_43 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_43">
    tile.await %compute_43
    %compute_44 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_44">
    tile.await %compute_44
    %compute_45 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_45">
    tile.await %compute_45
    %compute_46 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_46">
    tile.await %compute_46
    %compute_47 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_47">
    tile.await %compute_47
    %compute_48 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_48">
    tile.await %compute_48
    %compute_49 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_49">
    tile.await %compute_49
    %compute_50 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_50">
    tile.await %compute_50
    %compute_51 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_51">
    tile.await %compute_51
    %compute_52 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_52">
    tile.await %compute_52
    %compute_53 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_53">
    tile.await %compute_53
    %compute_54 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_54">
    tile.await %compute_54
    %compute_55 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_55">
    tile.await %compute_55
    %compute_56 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_56">
    tile.await %compute_56
    %compute_57 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_57">
    tile.await %compute_57
    %compute_58 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_58">
    tile.await %compute_58
    %compute_59 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_59">
    tile.await %compute_59
    %compute_60 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_60">
    tile.await %compute_60
    %compute_61 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_61">
    tile.await %compute_61
    %compute_62 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_62">
    tile.await %compute_62
    %compute_63 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_63">
    tile.await %compute_63
    %compute_64 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_64">
    tile.await %compute_64
    %compute_65 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_65">
    tile.await %compute_65
    %compute_66 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_66">
    tile.await %compute_66
    %compute_67 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_67">
    tile.await %compute_67
    %compute_68 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_68">
    tile.await %compute_68
    %compute_69 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_69">
    tile.await %compute_69
    %compute_70 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_70">
    tile.await %compute_70
    %compute_71 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_71">
    tile.await %compute_71
    %compute_72 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_72">
    tile.await %compute_72
    %compute_73 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_73">
    tile.await %compute_73
    %compute_74 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_74">
    tile.await %compute_74
    %compute_75 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_75">
    tile.await %compute_75
    %compute_76 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_76">
    tile.await %compute_76
    %compute_77 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_77">
    tile.await %compute_77
    %compute_78 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_78">
    tile.await %compute_78
    %compute_79 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_79">
    tile.await %compute_79
    %compute_80 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_80">
    tile.await %compute_80
    %compute_81 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_81">
    tile.await %compute_81
    %compute_82 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_82">
    tile.await %compute_82
    %compute_83 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_83">
    tile.await %compute_83
    %compute_84 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_84">
    tile.await %compute_84
    %compute_85 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_85">
    tile.await %compute_85
    %compute_86 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_86">
    tile.await %compute_86
    %compute_87 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_87">
    tile.await %compute_87
    %compute_88 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_88">
    tile.await %compute_88
    %compute_89 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_89">
    tile.await %compute_89
    %compute_90 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_90">
    tile.await %compute_90
    %compute_91 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_91">
    tile.await %compute_91
    %compute_92 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_92">
    tile.await %compute_92
    %compute_93 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_93">
    tile.await %compute_93
    %compute_94 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_94">
    tile.await %compute_94
    %compute_95 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_95">
    tile.await %compute_95
    %compute_96 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_96">
    tile.await %compute_96
    %compute_97 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_97">
    tile.await %compute_97
    %compute_98 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_98">
    tile.await %compute_98
    %compute_99 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_99">
    tile.await %compute_99
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_V21 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_V12 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_V22 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_Input = nest.alloc slot = "Input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Input = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V00 = nest.alloc slot = "V00" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V00 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_Input = nest.dma.prefetch.async %h_Input into %b_Input : !nest.event<"pref_Input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V00, %read_V00, %ready_V00 = nest.dispatch.tasks.async @prog_V00 tasks(%tasks) globals() ins(%b_Input, %b_V00) outs(%b_Input, %b_V00) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_Input) : (!nest.event<"grid_V00">, !nest.event<"read_V00">, !nest.event<"ready_V00">)
    nest.release %b_Input depends_on(%read_V00)
    %store_V00 = nest.dma.store.async %b_V00 into %h_V00 depends_on(%ready_V00) : !nest.event<"store_V00">
    nest.release %b_V00 depends_on(%store_V00)
    nest.await %grid_V00, %store_V00
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V00 = nest.alloc slot = "V00" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V00 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V01 = nest.alloc slot = "V01" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V01 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V00 = nest.dma.prefetch.async %h_V00 into %b_V00 : !nest.event<"pref_V00">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V01, %read_V01, %ready_V01 = nest.dispatch.tasks.async @prog_V01 tasks(%tasks) globals() ins(%b_V00, %b_V01) outs(%b_V00, %b_V01) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V00) : (!nest.event<"grid_V01">, !nest.event<"read_V01">, !nest.event<"ready_V01">)
    nest.release %b_V00 depends_on(%read_V01)
    %store_V01 = nest.dma.store.async %b_V01 into %h_V01 depends_on(%ready_V01) : !nest.event<"store_V01">
    nest.release %b_V01 depends_on(%store_V01)
    nest.await %grid_V01, %store_V01
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V00 = nest.alloc slot = "V00" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V00 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V10 = nest.alloc slot = "V10" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V10 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V00 = nest.dma.prefetch.async %h_V00 into %b_V00 : !nest.event<"pref_V00">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V10, %read_V10, %ready_V10 = nest.dispatch.tasks.async @prog_V10 tasks(%tasks) globals() ins(%b_V00, %b_V10) outs(%b_V00, %b_V10) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V00) : (!nest.event<"grid_V10">, !nest.event<"read_V10">, !nest.event<"ready_V10">)
    nest.release %b_V00 depends_on(%read_V10)
    %store_V10 = nest.dma.store.async %b_V10 into %h_V10 depends_on(%ready_V10) : !nest.event<"store_V10">
    nest.release %b_V10 depends_on(%store_V10)
    nest.await %grid_V10, %store_V10
    nest.return
  }
  nest.context @ctx_3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V01 = nest.alloc slot = "V01" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V01 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V10 = nest.alloc slot = "V10" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V10 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V11 = nest.alloc slot = "V11" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V11 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V01 = nest.dma.prefetch.async %h_V01 into %b_V01 : !nest.event<"pref_V01">
    %pref_V10 = nest.dma.prefetch.async %h_V10 into %b_V10 : !nest.event<"pref_V10">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V11, %read_V11, %ready_V11 = nest.dispatch.tasks.async @prog_V11 tasks(%tasks) globals() ins(%b_V01, %b_V10, %b_V11) outs(%b_V01, %b_V10, %b_V11) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V01, %pref_V10) : (!nest.event<"grid_V11">, !nest.event<"read_V11">, !nest.event<"ready_V11">)
    nest.release %b_V01 depends_on(%read_V11)
    nest.release %b_V10 depends_on(%read_V11)
    %store_V11 = nest.dma.store.async %b_V11 into %h_V11 depends_on(%ready_V11) : !nest.event<"store_V11">
    nest.release %b_V11 depends_on(%store_V11)
    nest.await %grid_V11, %store_V11
    nest.return
  }
  nest.context @ctx_4 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V10 = nest.alloc slot = "V10" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V10 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V20 = nest.alloc slot = "V20" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V20 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V10 = nest.dma.prefetch.async %h_V10 into %b_V10 : !nest.event<"pref_V10">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V20, %read_V20, %ready_V20 = nest.dispatch.tasks.async @prog_V20 tasks(%tasks) globals() ins(%b_V10, %b_V20) outs(%b_V10, %b_V20) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V10) : (!nest.event<"grid_V20">, !nest.event<"read_V20">, !nest.event<"ready_V20">)
    nest.release %b_V10 depends_on(%read_V20)
    %store_V20 = nest.dma.store.async %b_V20 into %h_V20 depends_on(%ready_V20) : !nest.event<"store_V20">
    nest.release %b_V20 depends_on(%store_V20)
    nest.await %grid_V20, %store_V20
    nest.return
  }
  nest.context @ctx_5 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V01 = nest.alloc slot = "V01" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V01 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V02 = nest.alloc slot = "V02" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V02 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V01 = nest.dma.prefetch.async %h_V01 into %b_V01 : !nest.event<"pref_V01">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V02, %read_V02, %ready_V02 = nest.dispatch.tasks.async @prog_V02 tasks(%tasks) globals() ins(%b_V01, %b_V02) outs(%b_V01, %b_V02) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V01) : (!nest.event<"grid_V02">, !nest.event<"read_V02">, !nest.event<"ready_V02">)
    nest.release %b_V01 depends_on(%read_V02)
    %store_V02 = nest.dma.store.async %b_V02 into %h_V02 depends_on(%ready_V02) : !nest.event<"store_V02">
    nest.release %b_V02 depends_on(%store_V02)
    nest.await %grid_V02, %store_V02
    nest.return
  }
  nest.context @ctx_6 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V11 = nest.alloc slot = "V11" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V11 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V20 = nest.alloc slot = "V20" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V20 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V21 = nest.alloc slot = "V21" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V21 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V11 = nest.dma.prefetch.async %h_V11 into %b_V11 : !nest.event<"pref_V11">
    %pref_V20 = nest.dma.prefetch.async %h_V20 into %b_V20 : !nest.event<"pref_V20">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V21, %read_V21, %ready_V21 = nest.dispatch.tasks.async @prog_V21 tasks(%tasks) globals() ins(%b_V11, %b_V20, %b_V21) outs(%b_V11, %b_V20, %b_V21) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V11, %pref_V20) : (!nest.event<"grid_V21">, !nest.event<"read_V21">, !nest.event<"ready_V21">)
    nest.release %b_V11 depends_on(%read_V21)
    nest.release %b_V20 depends_on(%read_V21)
    %store_V21 = nest.dma.store.async %b_V21 into %h_V21 depends_on(%ready_V21) : !nest.event<"store_V21">
    nest.release %b_V21 depends_on(%store_V21)
    nest.await %grid_V21, %store_V21
    nest.return
  }
  nest.context @ctx_7 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V02 = nest.alloc slot = "V02" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V02 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V11 = nest.alloc slot = "V11" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V11 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V12 = nest.alloc slot = "V12" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V12 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V02 = nest.dma.prefetch.async %h_V02 into %b_V02 : !nest.event<"pref_V02">
    %pref_V11 = nest.dma.prefetch.async %h_V11 into %b_V11 : !nest.event<"pref_V11">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V12, %read_V12, %ready_V12 = nest.dispatch.tasks.async @prog_V12 tasks(%tasks) globals() ins(%b_V02, %b_V11, %b_V12) outs(%b_V02, %b_V11, %b_V12) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V02, %pref_V11) : (!nest.event<"grid_V12">, !nest.event<"read_V12">, !nest.event<"ready_V12">)
    nest.release %b_V02 depends_on(%read_V12)
    nest.release %b_V11 depends_on(%read_V12)
    %store_V12 = nest.dma.store.async %b_V12 into %h_V12 depends_on(%ready_V12) : !nest.event<"store_V12">
    nest.release %b_V12 depends_on(%store_V12)
    nest.await %grid_V12, %store_V12
    nest.return
  }
  nest.context @ctx_8 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_V12 = nest.alloc slot = "V12" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V12 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V21 = nest.alloc slot = "V21" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V21 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_V22 = nest.alloc slot = "V22" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_V22 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_V12 = nest.dma.prefetch.async %h_V12 into %b_V12 : !nest.event<"pref_V12">
    %pref_V21 = nest.dma.prefetch.async %h_V21 into %b_V21 : !nest.event<"pref_V21">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_V22, %read_V22, %ready_V22 = nest.dispatch.tasks.async @prog_V22 tasks(%tasks) globals() ins(%b_V12, %b_V21, %b_V22) outs(%b_V12, %b_V21, %b_V22) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_V12, %pref_V21) : (!nest.event<"grid_V22">, !nest.event<"read_V22">, !nest.event<"ready_V22">)
    nest.release %b_V12 depends_on(%read_V22)
    nest.release %b_V21 depends_on(%read_V22)
    %store_V22 = nest.dma.store.async %b_V22 into %h_V22 depends_on(%ready_V22) : !nest.event<"store_V22">
    nest.release %b_V22 depends_on(%store_V22)
    nest.await %grid_V22, %store_V22
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    nexus.await %done_0
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_1, %done_2
    %done_3 = nexus.submit_context.async @ctx_3(%arena) : !nexus.event<"done_3">
    nexus.await %done_2
    %done_4 = nexus.submit_context.async @ctx_4(%arena) : !nexus.event<"done_4">
    nexus.await %done_1
    %done_5 = nexus.submit_context.async @ctx_5(%arena) : !nexus.event<"done_5">
    nexus.await %done_3, %done_4
    %done_6 = nexus.submit_context.async @ctx_6(%arena) : !nexus.event<"done_6">
    nexus.await %done_5, %done_3
    %done_7 = nexus.submit_context.async @ctx_7(%arena) : !nexus.event<"done_7">
    nexus.await %done_7, %done_6
    %done_8 = nexus.submit_context.async @ctx_8(%arena) : !nexus.event<"done_8">
    nexus.await %done_0, %done_1, %done_2, %done_3, %done_4, %done_5, %done_6, %done_7, %done_8
    nexus.return
  }
}
