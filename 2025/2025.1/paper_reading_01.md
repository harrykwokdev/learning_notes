# BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects
#### Paper: https://arxiv.org/pdf/2303.14158 
#### Code: https://arxiv.org/pdf/2303.14158](https://github.com/NVlabs/BundleSDF

## Abstract 
- 用于从单目RGBD视频序列中对未知物进行6自由度跟踪，同时进行神经三维重建。 
  - 特点：神经物体场和位姿图优化过程同步学习。 
  - 作用：将信息累计成一个一致的三维表达，同时捕捉了物体的几何形状和外观特征。 
  - 实现方法：维护一个包含位姿信息的动态记忆帧池，用于促进处理流程之间的信息交互。

## Introduction 
- 出发点&问题：CV邻域中两个基本且紧密相关的问题，其一为从单目RGBD视频中对未知物体进行6自由度位姿跟踪，其二为对该物体进行三维重建。然而之前的方法通常假定相机位姿已知或真实的掩膜已知，此外，动态移动的相机捕捉静止物体时，无法实现完整的三维重建，实时的6自由度物体位姿估计与跟踪方法通常需要获取物体的带有纹理的三维模型进行预训练或在线模板匹配。
- 解决方案：假定物体是刚性的，并且需要视频第一帧中的二维物体掩膜，除了这两个要求之外，物体可以自由移动，乃至严重遮挡。一个在线位姿图优化过程、一个用于重建三维形状和外观的并行神经物体场，以及一个促进这两个过程之间信息交互的记忆池。
- 引入了一种混合符号距离函数（SDF）表示法，以应对在以动态物体为中心的场景中，因特别问题所导致的不确定自由空间问题，例如存在噪声的分割结果以及因交互产生的外部遮挡。

## Conclusion
- 是一种从单目RGBD视频中进行6自由度物体跟踪和三维重建的新方法。仅需在初始帧对物体进行分割，通过并行运行的两个线程分别执行在线图优化姿态估计和神经物体场表示，应对诸如快速运动、部分和完全遮挡、缺乏纹理以及镜面高光等挑战性场景。

## Related Work
- 自由度物体姿态估计与跟踪
  - 利用第一帧姿态估计，另外利用时间信息来估计视频中每一帧的物体姿态。
- 同时定位与建图（SLAM）
  - 静态：跟踪相机相对于一个大型静态环境的姿态
  - 动态：通过结合颜色信息的帧模型迭代最近点（ICP）算法、概率数据关联或三维水平集最大化来跟踪动态物体。通过将新跟踪到的姿态与g观察到的RGBD数据聚合，模型会即时进行重建。
  - 本论文方法：利用神经物体场表示，专注于物体中心的设置，能够自动即时融合，同时修正历史跟踪姿态以保持多视图统一性。
- 物体重建
  - 不假设已知相机姿态或真实分割，无需专注在具有丰富纹理或几何线索的静态场景，此方法按照视频流的顺序依次进行处理，不假设对交互主体的特定知识，保证了泛化性。

## Approach

### 1. 粗略位姿初始化 (Coarse Pose Initialization)

#### 目标
为后续的在线姿态图优化提供一个良好的初始位姿估计。

#### 输入
- 当前帧 Ft
- 前一帧 Ft-1（或者记忆池中的参考帧）

#### 流程

1. **物体分割**: 
   使用一个与物体无关的视频分割网络（论文中使用的是 XMem）来获取当前帧 Ft 中物体的 2D 掩膜。这个分割网络不需要物体的先验知识。

2. **特征匹配**:
   使用一个基于 Transformer 的特征匹配网络（论文中使用的是 LoFTR，也可以用 SuperGlue 等其他网络）在当前帧 Ft 和前一帧 Ft-1 的 RGB 图像之间建立特征对应关系。这个特征匹配网络在大规模数据集上进行了预训练，具有较强的泛化能力。

3. **位姿估计**:
   将特征对应关系和对应的深度信息结合起来，利用 RANSAC（随机采样一致性）算法滤除外点，然后使用最小二乘法求解一个 6-DoF 位姿变换 ξt，这个位姿变换表示当前帧 Ft 相对于前一帧 Ft-1 的相对运动。

4. **处理第一帧和丢失跟踪**:
   - 如果是处理视频序列的第一帧, 因为没有前一帧，则当前帧和物体的初始位姿和坐标系直接初始化。
   - 如果因为目标丢失或者严重遮挡而不能计算和上一帧的关系时, 当前帧不会和前一帧进行匹配, 而是和 memory pool 里的所有关键帧匹配, 选取最接近的一帧计算当前位姿。

#### 输出
当前帧 Ft 相对于前一帧 Ft-1（或记忆池中的参考帧）的粗略位姿估计 ξt。

#### key
- 使用与物体无关的分割网络，避免了对特定物体的依赖。
- 使用预训练的特征匹配网络，提高了匹配的准确性和鲁棒性。
- 使用 RANSAC 和最小二乘法，可以有效地处理外点和噪声。
- 当目标丢失后利用和 memory pool 的关键帧对比，保证重新出现的目标可以立即恢复粗略位姿。

![img](res_paper_reading_01/fig01.jpeg)

### 2. 记忆池管理 (Memory Pool Management)

#### 目标
存储一系列关键帧及其位姿信息，用于后续的姿态图优化和神经场学习。记忆池可以有效地保留历史信息，避免灾难性遗忘，并提高跟踪的长期鲁棒性。

#### 数据结构
记忆池 P 是一个关键帧的集合，每个关键帧包含以下信息：
- RGBD 图像
- 物体掩膜
- 位姿（相对于物体坐标系）
- 一个标志位 b(F)，指示该帧的位姿是否已经通过神经物体场进行了优化

#### 流程

1. **初始化**:
   视频序列的第一帧 F0 自动添加到记忆池中，并将物体的坐标系设置为 F0 的相机坐标系。

2. **添加新帧**:
   - 对于每个新帧 Ft，首先计算其粗略位姿 ξt。
   - 然后，使用在线姿态图优化（稍后介绍）对 ξt 进行优化，得到更新后的位姿 ξ't。
   - 计算 ξ't 与记忆池中所有帧的位姿之间的差异（考虑旋转测地距离，并忽略绕相机光轴的旋转）。
   - 如果 ξ't 与记忆池中所有帧的视角差异都足够大（超过一个阈值），则将 Ft 及其相关信息添加到记忆池中。这个策略保证了记忆池中的帧具有足够的多样性，能够覆盖物体的不同视角。
   
   关于视角距离的定义，忽略图像平面内旋转 (in-plane rotation) 对于保证足够视角信息很有帮助。因为如果加入了过多的只有平面旋转的图片，对于目标 3D 信息的补充很少。

3. **更新位姿**:
   在线姿态图优化和神经物体场学习都会更新记忆池中帧的位姿。如果一个帧的位姿被神经物体场更新过，则将其标志位 b(F) 设置为 TRUE，表示其位姿已经足够精确，后续的在线姿态图优化不再更新它。

4. **大小限制**:
   记忆池的大小是有限的（论文中限制为 K=10），以保证计算效率。当超过了这个数量后, 会根据一定的标准筛选其中重要的关键帧。

#### key
- 记忆池存储了关键帧及其位姿，提供了历史信息。
- 添加新帧的策略基于视角差异，保证了记忆池的多样性。
- 使用标志位来区分位姿是否被神经物体场优化过。
- 记忆池的大小是有限的，以保证计算效率。

