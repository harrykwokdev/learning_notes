# Spatial Transformer Networks 

Spatial Transformer Network (STN) 是一种用于提高深度学习模型对几何变换（如旋转、缩放、平移等）鲁棒性的模块。STN 允许网络在前馈过程中对输入图像进行自适应变换，使得后续任务（如分类、检测）能更好地处理变形后的目标。

## 主要组成部分

1. **定位网络（Localization Network）**
   - 一个小型子网络（通常是 CNN），负责预测变换参数θ。

2. **参数化变换（Grid Generator）**
   - 通过预测的变换参数θ生成目标坐标网格，实现空间变换（如仿射变换）。

3. **采样器（Sampler）**
   - 使用双线性插值（bilinear interpolation）来映射变换后的图像，使梯度可传递。
