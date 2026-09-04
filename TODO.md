<!-- 1. 当前的 BOA MFE 等各个模块并未提供大部分文档中的描述的功能，例如 BOA 由 reduce 模式，post scale 等等，MFE 并没有layout transformation等等
2. MFE 的queue 需要重新设计来保持 intra tile program 的IO pipeline

--- -->

## 限制

不可以改动 除了当前项目之外的任何文件，如果实在需要需要征求用户同意

## R2 需求

1. 当前的内存系统完全是摆设
2. 需要增加输入 nest.context, 还有需要将 prefetch，store load，都需要有 目标地址和原地址的，当前完全没有
3. 后续加入对于 HBM 的模拟等支持，gather
4. memory 以参考 /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模，对runtime的建模基本上已经符合我的预期了
5. 当前的 nest.release 没有对应的实现，具体实现思路为硬件等待所有tile.signal后再进行释放
6. tile.signal 并没有加入 context id 或者说 唯一的ID 让 L2 可以清楚的知道什么时候 进行内存的释放
7. 当前 IR 并没有对与 传入参数进行实质性的处理，也就是说，当前的 IR 并没有 输入参数的概念，需要加入输入参数的概念，并且在 IR 中进行处理，具体可以参考 reference.mlir， /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模。
8. 当前的trace 已经非常直观了，并且很好的了解当前的运行情况，memory 大小相关的，可以参考当前 queue 的状态来表示，memory latency 相关的也可以加入，但是需要注意，tile 的 memory 需要在tile 那一栏里面，L2 的 cache 需要和 L2 一起

## R3

1. IR 层面并没有对 nest.context 的资源进行显式的配置与管理，比如像 reference.mlir 中的配置方式，当前的 IR 只是简单的将 nest.context 当作一个普通的 tile 来处理，并没有对其进行资源的显式管理
2. 当前的 IR 中的 tile program 中的 load 是指 L2 -> L1 的 load，而并不是 L1 -> Register 的 load，当前的 IR 中的 store 是指 L1 -> L2 的 store，而并不是 Register -> L1 的 store，这样的设计是为了简化 IR 的设计，但是在后续的设计中，需要将 load 和 store 分为两类，一类是 L2 <-> L1 的 load/store，另一类是 L1 <-> Register 的 load/store，这样可以更好的模拟实际的硬件行为
3. 需要在 IR 中加入 bank 模式的概念，因为我们的硬件设计中，一个bank是很多种模式的，需要暴露给编译器来做优化的。
4. 当前的 trace 中 indices_ready 的状态是 轮训的，这个需要改一下
5. 允许 在 L2 开一个 shared memory 的概念，允许多个 tile 共享一个 memory，这样共享的weight 可以放在 L2 中，减少 memory IO 的占用
