1. 当前的 BOA MFE 等各个模块并未提供大部分文档中的描述的功能，例如 BOA 由 reduce 模式，post scale 等等，MFE 并没有layout transformation等等
2. MFE 的queue 需要重新设计来保持 intra tile program 的IO pipeline

---

1. 当前的内存系统完全是摆设
2. 需要增加输入 nest.context, 还有需要将 prefetch，store load，都需要有 目标地址和原地址的，当前完全没有
3. 后续加入对于 HBM 的模拟等支持，gather
4. 可以参考 /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模
5. 当前的 nest.release 没有对应的实现
6. tile.signal 并没有加入 context id 或者说 唯一的ID 让 L2 可以清楚的知道什么时候 进行内存的释放
