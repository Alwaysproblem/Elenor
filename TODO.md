<!-- 1. 当前的 BOA MFE 等各个模块并未提供大部分文档中的描述的功能，例如 BOA 由 reduce 模式，post scale 等等，MFE 并没有layout transformation等等
2. MFE 的queue 需要重新设计来保持 intra tile program 的IO pipeline

--- -->

## 限制

不可以改动 除了当前项目之外的任何文件，如果实在需要需要征求用户同意

## R2 需求 （已完成）

1. 当前的内存系统完全是摆设
2. 需要增加输入 nest.context, 还有需要将 prefetch，store load，都需要有 目标地址和原地址的，当前完全没有
3. 后续加入对于 HBM 的模拟等支持，gather
4. memory 以参考 /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模，对runtime的建模基本上已经符合我的预期了
5. 当前的 nest.release 没有对应的实现，具体实现思路为硬件等待所有tile.signal后再进行释放
6. tile.signal 并没有加入 context id 或者说 唯一的ID 让 L2 可以清楚的知道什么时候 进行内存的释放
7. 当前 IR 并没有对与 传入参数进行实质性的处理，也就是说，当前的 IR 并没有 输入参数的概念，需要加入输入参数的概念，并且在 IR 中进行处理，具体可以参考 reference.mlir， /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模。
8. 当前的trace 已经非常直观了，并且很好的了解当前的运行情况，memory 大小相关的，可以参考当前 queue 的状态来表示，memory latency 相关的也可以加入，但是需要注意，tile 的 memory 需要在tile 那一栏里面，L2 的 cache 需要和 L2 一起
9. 当前的 trace 中 indices_ready 的状态是 轮训的，这个需要改一下
10. 当前的 nest.dispatch.tasks.async 的 ins 和 outs 表示的并不正确，需要增加类似于 function args 这种表示，ins 和 outs 只是明确定义的输入输出。
11. 需要加入 L1 级别的 free 操作，用于释放不再使用的内存资源，确保内存的高效利用。

## R3

1. IR 层面并没有对 nest.context 的资源进行显式的配置与管理，比如像 reference.mlir 中的配置方式，当前的 IR 只是简单的将 nest.context 当作一个普通的 tile 来处理，并没有对其进行资源的显式管理
2. 需要考虑，当内存资源不足的时候（包括，L2 和 L1 的内存），下一个 L1/2 的 context 可能无法执行，需要等待，但是并不能每个cycle都进行重试，需要根据编译器给出的内存信息（包括大小和内存模式是否为cache或者scratchpad）来做比较，每次只有在有 release 的时候才进行重试。当前 L2 级别只是有初级的相关机制，L1 级别几乎没有，需要在后续设计中加入类似的机制。
3. 当前的 IR 中的 tile program 中的 load 是指 L2 -> L1 的 load，而并不是 L1 -> Register 的 load，当前的 IR 中的 store 是指 L1 -> L2 的 store，而并不是 Register -> L1 的 store，这样的设计是为了简化 IR 的设计，但是在后续的设计中，需要将 load 和 store 分为两类，一类是 L2 <-> L1 的 load/store，另一类是 L1 <-> Register 的 load/store，这样可以更好的模拟实际的硬件行为
4. 需要在 IR 中加入 bank 模式的概念，因为我们的硬件设计中，一个bank是很多种模式的，需要暴露给编译器来做优化的。
5. 允许 在 L2 开一个 shared memory 的概念，允许多个 tile 共享一个 memory，这样共享的weight 可以放在 L2 中，减少 memory IO 的占用
6. 当前的 nexus.program 需要采用 depends 机制来明确各个 tile program 之间的依赖关系，确保使用 ready-action 策略而不是wait这种策略来进行触发。
7. 需要询问 当前 next.context 的运行是不是 ready action 这种模式。

## 未来需要考虑的问题暂时先不考虑

- Transpose 和这个 Layout 变换，这边当前的思考方式是有专门的器件去做。但是呢，我后来想了一下，是不是也可以将 Transpose 这个类似这种 Layout 的东西作为一个单独的器件儿镶嵌在 BOA 或者是 EVU 这个里面来。
- 对于编译器的需求。比如说需要考虑一下拓扑排序，需要分析 liveness 的 memory 和 non-liveness 的 memory，以及 peak memory 相关的事宜，为后续的 memory addition 去做准备。tiling，以及 context 的切分，还有 L1 当前是context 总和爆内存是 fail 模式需要在编译器知道哪里超内存了，相关的吧，就是编译器这边的需要做的事情，到后面需要整理出来。
- 比如说类似于像 BOA 这种，怎么去做 register file 的 load，这一块可能得需要考虑一下硬件级别的 pipeline。比如说如果加入 transpose 的话，那就意味着 transpose input 和 BOA，之后再去接一个 transpose output。所以说这一块得需要衡量一下整体的 BOA 的大小以及功耗。理论上来说 multi-context 去把 BOA 或这种计算单元的利用率打满，所以说 BOA 内部的设计的话，还是需要一些指令级别的pipeline。
- 
