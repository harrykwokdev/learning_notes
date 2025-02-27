**一、张量 (Tensor) 创建** (更全面的创建方式)

*   `torch.tensor` (核心创建函数)
*   `torch.sparse_coo_tensor` -  创建稀疏 COO 格式张量
*   `torch.sparse_csr_tensor` - 创建稀疏 CSR 格式张量
*   `torch.sparse_csc_tensor` - 创建稀疏 CSC 格式张量
*   `torch.sparse_bsr_tensor` - 创建稀疏 BSR 格式张量
*   `torch.sparse_bssr_tensor` - 创建稀疏 BSSR 格式张量
*   `torch.nested_tensor` - 创建嵌套张量
*   `torch.zeros`
*   `torch.zeros_like` -  创建与给定张量形状相同的全零张量
*   `torch.ones`
*   `torch.ones_like` -  创建与给定张量形状相同的全一张量
*   `torch.eye`
*   `torch.empty`
*   `torch.empty_like` -  创建与给定张量形状相同的未初始化张量
*   `torch.empty_strided` -  创建指定步长的未初始化张量
*   `torch.full`
*   `torch.full_like` -  创建与给定张量形状相同的填充指定值的张量
*   `torch.clone` -  克隆张量 (创建新的内存副本)
*   `torch.from_numpy`
*   `torch.asarray` -  从类数组对象创建张量 (NumPy 风格)
*   `torch.as_tensor`
*   `torch.arange`
*   `torch.range` -  （已弃用，推荐使用 `arange` 或 `linspace`）
*   `torch.linspace`
*   `torch.logspace` -  创建对数刻度均匀间隔的一维张量
*   `torch.geomspace` - 创建几何刻度均匀间隔的一维张量
*   `torch.rand`
*   `torch.rand_like` -  创建与给定张量形状相同的均匀分布随机张量
*   `torch.randn`
*   `torch.randn_like` -  创建与给定张量形状相同的标准正态分布随机张量
*   `torch.randint`
*   `torch.randint_like` - 创建与给定张量形状相同的指定范围随机整数张量
*   `torch.randperm` -  创建 0 到 n-1 的随机排列张量
*   `torch.normal` -  创建正态分布随机张量 (可指定均值和标准差)
*   `torch.bernoulli` -  创建伯努利分布 (二项分布) 随机张量
*   `torch.poisson` -  创建泊松分布随机张量
*   `torch.multinomial` -  从多项分布中抽取样本
*   `torch.binomial` -  创建二项分布随机张量 (trials 次试验，概率为 probs)
*   `torch.manual_seed` (随机种子设置，前面已列出，此处再次强调)
*   `torch.cuda.manual_seed` / `torch.cuda.manual_seed_all` (CUDA 随机种子设置)
*   `torch.seed` / `torch.initial_seed` (获取/设置随机种子)
*   `torch.random.manual_seed` / `torch.random.seed` / `torch.random.initial_seed` (随机数生成器种子设置)
*   `torch.numpy_seed` (NumPy 随机种子同步)

**二、张量基本操作** (更丰富的运算种类)

*   **逐元素运算 (Element-wise Operations)**:
    *   `torch.add` / `+` / `torch.addcmul` / `torch.addcdiv` -  加法及变体
    *   `torch.sub` / `-`
    *   `torch.mul` / `*`
    *   `torch.div` / `/` / `torch.floor_divide` (向下整除) / `torch.true_divide` (真除法)
    *   `torch.pow` / `**`
    *   `torch.exp` / `torch.exp2` / `torch.expm1`
    *   `torch.log` / `torch.log2` / `torch.log10` / `torch.log1p`
    *   `torch.sqrt` / `torch.rsqrt`
    *   `torch.sigmoid` / `torch.tanh` / `torch.relu` / `torch.elu` / `torch.selu` / `torch.gelu` / `torch.silu` (Swish) / `torch.relu6` / `torch.leaky_relu` / `torch.prelu` / `torch.threshold`
    *   `torch.abs` / `torch.neg` / `torch.sign` / `torch.signbit`
    *   `torch.round` / `torch.floor` / `torch.ceil` / `torch.trunc` (截断取整) / `torch.frac` (小数部分)
    *   `torch.clamp` / `torch.clip` (别名)
    *   `torch.lerp` (线性插值)
    *   `torch.positive` / `torch.negative` (正/负值判断)
    *   `torch.reciprocal` (倒数)
    *   `torch.digamma` / `torch.polygamma` / `torch.erf` / `torch.erfc` / `torch.erfinv` / `torch.i0` / `torch.i0e` / `torch.i1` / `torch.i1e` / `torch.sigmoid` / `torch.logsigmoid` / `torch.xlogy` / `torch.zeta` (特殊数学函数)

*   **比较运算**:
    *   `torch.eq` / `==`
    *   `torch.ne` / `!=`
    *   `torch.ge` / `>=`
    *   `torch.gt` / `>`
    *   `torch.le` / `<=`
    *   `torch.lt` / `<`
    *   `torch.isclose` (判断两个张量是否近似相等)
    *   `torch.equal`
    *   `torch.allclose`
    *   `torch.greater_equal` / `torch.greater` / `torch.less_equal` / `torch.less` (别名)

*   **位运算 (Bitwise Operations)**:
    *   `torch.bitwise_and` / `&`
    *   `torch.bitwise_or` / `|`
    *   `torch.bitwise_xor` / `^`
    *   `torch.bitwise_not` / `~`
    *   `torch.bitwise_left_shift` / `<<`
    *   `torch.bitwise_right_shift` / `>>`

*   **归约运算 (Reduction Operations)**: (更全面的归约操作)
    *   `torch.sum` / `torch.nansum`
    *   `torch.mean` / `torch.nanmean`
    *   `torch.average` (均值的别名)
    *   `torch.std` / `torch.std_mean` / `torch.var` / `torch.var_mean` (标准差/方差及带均值版本) / `torch.nanstd` / `torch.nanvar` (忽略 NaN 值的版本)
    *   `torch.max` / `torch.min` / `torch.amax` / `torch.amin`
    *   `torch.argmax` / `torch.argmin`
    *   `torch.median` / `torch.nanmedian`
    *   `torch.mode`
    *   `torch.kthvalue` / `torch.topk`
    *   `torch.quantile` / `torch.nanquantile` (分位数)
    *   `torch.count_nonzero`
    *   `torch.unique` / `torch.unique_consecutive` (连续唯一值)
    *   `torch.prod`
    *   `torch.cumprod` / `torch.cumsum`
    *   `torch.all` (判断所有元素是否为 True) / `torch.any` (判断是否存在元素为 True)
    *   `torch.min_values` / `torch.max_values` (仅返回最大/最小值，不返回索引)
    *   `torch.msort` (沿维度排序)

*   **线性代数运算 (Linear Algebra Operations)**: (更丰富的线性代数操作)
    *   `torch.matmul` / `@` / `torch.mm` / `torch.bmm`
    *   `torch.dot` / `torch.vdot` (向量点积) / `torch.inner` (内积) / `torch.outer` (外积) / `torch.cross` (向量叉积)
    *   `torch.trace` / `torch.diag` / `torch.diag_embed` (对角矩阵嵌入) / `torch.diagflat` (展平为对角矩阵) / `torch.offdiag` (提取非对角元素)
    *   `torch.det` / `torch.slogdet` / `torch.logdet` (行列式及相关)
    *   `torch.inverse` / `torch.pinverse`
    *   `torch.linalg.solve` (求解线性方程组) / `torch.linalg.lstsq` (最小二乘解) / `torch.linalg.eig` / `torch.linalg.eigh` (埃尔米特矩阵特征值) / `torch.linalg.eigvals` / `torch.linalg.eigvalsh` / `torch.linalg.svd` / `torch.linalg.svdvals` / `torch.linalg.qr` / `torch.linalg.cholesky` / `torch.linalg.lu` (LU 分解) / `torch.linalg.cond` (条件数) / `torch.linalg.matrix_norm` (矩阵范数) / `torch.linalg.vector_norm` (向量范数) / `torch.linalg.matrix_rank` / `torch.linalg.multi_dot` (多矩阵乘法) / `torch.linalg.householder_product` (Householder 反射矩阵乘积)
    *   `torch.pca_lowrank` (低秩 PCA)
    *   `torch.linalg.matrix_power` (矩阵幂)
    *   `torch.linalg.tensor_norm` (张量范数)
    *   `torch.linalg.solve_triangular` (三角矩阵方程组求解)
    *   `torch.linalg.triangular_solve` (三角矩阵方程组求解，别名)

*   **其他常用操作**:
    *   `torch.nonzero` (返回非零元素的索引)
    *   `torch.where` (条件选择)
    *   `torch.masked_select` (掩码选择)
    *   `torch.index_select` (索引选择)
    *   `torch.take`
    *   `torch.gather` / `torch.scatter` / `torch.scatter_add` / `torch.index_add` / `torch.index_copy` / `torch.index_fill` (散布/聚集操作)
    *   `torch.masked_fill` / `torch.masked_scatter` (掩码填充/散布)
    *   `torch.flip` (沿维度翻转) / `torch.rot90` (旋转 90 度)
    *   `torch.roll` (循环移位)
    *   `torch.broadcast_tensors` / `torch.broadcast_shapes` (广播相关)
    *   `torch.cdist` (计算两组向量的成对距离) / `torch.pdist` (计算输入中向量的成对距离)
    *   `torch.cosine_similarity` (余弦相似度)

**三、形状变换、索引与切片、连接与分割** (更全面的维度操作)

*   **形状变换**:
    *   `tensor.view` / `tensor.reshape` / `tensor.contiguous` (内存连续化)
    *   `tensor.squeeze` / `tensor.unsqueeze`
    *   `tensor.transpose` / `tensor.permute` / `tensor.swapaxes` (交换轴) / `tensor.moveaxis` (移动轴)
    *   `tensor.flatten` / `tensor.ravel` (展平)
    *   `tensor.expand` / `tensor.expand_as` / `tensor.tile` (平铺) / `tensor.repeat`
    *   `tensor.unfold` (滑动窗口展开)

*   **索引与切片**: (更丰富的索引方式)
    *   **基本索引与切片**: `[]`, `:`, `...` (省略号)
    *   **布尔索引**: 使用布尔张量进行条件索引
    *   **整数索引**: 使用整数张量进行索引
    *   `torch.index_select` / `torch.masked_select` / `torch.take` / `torch.gather` / `torch.nonzero` / `torch.where` / `torch.tril_indices` (下三角索引) / `torch.triu_indices` (上三角索引)

*   **连接与分割**:
    *   `torch.cat` / `torch.stack`
    *   `torch.chunk` / `torch.split` / `torch.tensor_split` (更灵活的分割) / `torch.vsplit` / `torch.hsplit` / `torch.dsplit` (沿深度分割，用于 3D 张量)
    *   `torch.vstack` (垂直堆叠) / `torch.hstack` (水平堆叠) / `torch.dstack` (深度堆叠) / `torch.column_stack` (列堆叠) / `torch.row_stack` (行堆叠)

**四、神经网络模块 (torch.nn)** (更详尽的神经网络层和模块)

*   **容器 (Containers)**:
    *   `nn.Sequential` / `nn.ModuleList` / `nn.ModuleDict` / `nn.ParameterList` / `nn.ParameterDict`

*   **卷积层 (Convolution Layers)**:
    *   `nn.Conv1d` / `nn.Conv2d` / `nn.Conv3d` / `nn.ConvTranspose1d` / `nn.ConvTranspose2d` / `nn.ConvTranspose3d` / `nn.Unfold` (展开卷积) / `nn.Fold` (折叠卷积) / `nn.LazyConv2d` (延迟初始化卷积层)

*   **池化层 (Pooling Layers)**:
    *   `nn.MaxPool1d` / `nn.MaxPool2d` / `nn.MaxPool3d` / `nn.AvgPool1d` / `nn.AvgPool2d` / `nn.AvgPool3d` / `nn.MaxUnpool1d` / `nn.MaxUnpool2d` / `nn.MaxUnpool3d` (最大池化逆操作) / `nn.FractionalMaxPool2d` (分数最大池化) / `nn.LPPool2d` (Lp 池化) / `nn.AdaptiveMaxPool1d` / `nn.AdaptiveMaxPool2d` / `nn.AdaptiveMaxPool3d` / `nn.AdaptiveAvgPool1d` / `nn.AdaptiveAvgPool2d` / `nn.AdaptiveAvgPool3d`

*   **循环层 (Recurrent Layers)**:
    *   `nn.RNN` / `nn.LSTM` / `nn.GRU` / `nn.RNNCell` / `nn.LSTMCell` / `nn.GRUCell` / `nn.Transformer` / `nn.TransformerEncoder` / `nn.TransformerDecoder` / `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer`

*   **线性层 (Linear Layers)**:
    *   `nn.Linear` / `nn.Bilinear` (双线性层) / `nn.Identity` (恒等层) / `nn.LazyLinear` (延迟初始化线性层)

*   **激活函数 (Activation Functions)**:
    *   `nn.ReLU` / `nn.ReLU6` / `nn.ELU` / `nn.SELU` / `nn.CELU` / `nn.LeakyReLU` / `nn.PReLU` / `nn.RReLU` / `nn.GELU` / `nn.Sigmoid` / `nn.Tanh` / `nn.Tanhshrink` / `nn.Softsign` / `nn.Softplus` / `nn.Softmin` / `nn.Softmax` / `nn.LogSoftmax` / `nn.Hardtanh` / `nn.Hardsigmoid` / `nn.Hardswish` / `nn.SiLU` (Swish) / `nn.Mish` / `nn.LogSigmoid`

*   **Normalization Layers (归一化层)**:
    *   `nn.BatchNorm1d` / `nn.BatchNorm2d` / `nn.BatchNorm3d` / `nn.SyncBatchNorm` (跨 GPU 同步 BatchNorm) / `nn.LayerNorm` / `nn.GroupNorm` / `nn.InstanceNorm1d` / `nn.InstanceNorm2d` / `nn.InstanceNorm3d` / `nn.LocalResponseNorm` (局部响应归一化，早期 CNN 常用)

*   **Dropout 层 (Dropout Layers)**:
    *   `nn.Dropout` / `nn.Dropout2d` / `nn.Dropout3d` / `nn.AlphaDropout` (保持均值和方差的 Dropout) / `nn.FeatureAlphaDropout` (特征 Dropout)

*   **Embedding 层 (Embedding Layer)**:
    *   `nn.Embedding` / `nn.EmbeddingBag` (用于高效处理词袋)

*   **损失函数 (Loss Functions)**: (更丰富的损失函数)
    *   `nn.L1Loss` / `nn.MSELoss` / `nn.SmoothL1Loss` (平滑 L1 损失)
    *   `nn.CrossEntropyLoss` / `nn.NLLLoss` (负对数似然损失) / `nn.PoissonNLLLoss` (泊松负对数似然损失) / `nn.KLDivLoss` (KL 散度损失) / `nn.MarginRankingLoss` (Margin Ranking Loss) / `nn.HingeEmbeddingLoss` (Hinge Embedding Loss) / `nn.MultiLabelMarginLoss` (多标签 Margin Loss) / `nn.MultiLabelSoftMarginLoss` (多标签 Soft Margin Loss) / `nn.CosineEmbeddingLoss` (Cosine Embedding Loss) / `nn.TripletMarginLoss` (Triplet Margin Loss) / `nn.TripletMarginWithDistanceLoss` (带距离度量的 Triplet Margin Loss) / `nn.BCELoss` / `nn.BCEWithLogitsLoss` / `nn.CTCLoss` / `nn.AdaptiveLossFunction` (自适应损失函数)

*   **Padding Layers (填充层)**:
    *   `nn.ZeroPad2d` / `nn.ConstantPad1d` / `nn.ConstantPad2d` / `nn.ConstantPad3d` / `nn.ReflectionPad1d` / `nn.ReflectionPad2d` / `nn.ReflectionPad3d` / `nn.ReplicationPad1d` / `nn.ReplicationPad2d` / `nn.ReplicationPad3d`

*   **Vision Layers (视觉层 -  `torchvision.ops`  和 `torchvision.models`  中更丰富)**
    *   `nn.PixelShuffle` / `nn.PixelUnshuffle` (像素反重组)

*   **Sparse Layers (稀疏层)**
    *   `nn.Embedding` (也可用于稀疏数据)

*   **Utilities (实用工具)**
    *   `nn.init` (权重初始化) / `nn.utils.parameters_to_vector` / `nn.utils.vector_to_parameters` (参数向量化/反向量化) / `nn.utils.clip_grad_norm_` (梯度裁剪) / `nn.utils.clip_grad_value_` (梯度值裁剪) / `nn.utils.weight_norm` (权重归一化) / `nn.utils.spectral_norm` (谱归一化) / `nn.utils.remove_weight_norm` / `nn.utils.remove_spectral_norm` / `nn.utils.fuse_conv_bn` (融合 Conv 和 BatchNorm)
    *   `nn.Parameter` / `nn.Buffer` (Module 的 buffer，不参与梯度计算)
    *   `nn.Module.apply` (递归应用函数到 Module 及其子 Module) / `nn.Module.children` / `nn.Module.modules` / `nn.Module.named_children` / `nn.Module.named_modules` / `nn.Module.state_dict` / `nn.Module.load_state_dict` / `nn.Module.zero_grad` / `nn.Module.train` / `nn.Module.eval` / `nn.Module.parameters` / `nn.Module.named_parameters` / `nn.Module.buffers` / `nn.Module.named_buffers` / `nn.Module.to` / `nn.Module.cuda` / `nn.Module.cpu` / `nn.Module.double` / `nn.Module.float` / `nn.Module.half` / `nn.Module.bfloat16` / `nn.Module.type` / `nn.Module.xpu` / `nn.Module.ipu` / `nn.Module.mps` / `nn.Module.save_state_dict` / `nn.Module.get_extra_state` / `nn.Module.set_extra_state`

**五、优化 (torch.optim)** (所有优化器，更完整的学习率调度)

*   **优化器 (Optimizers)**: (所有 `torch.optim` 中的优化器)
    *   `optim.Adadelta` / `optim.Adagrad` / `optim.Adam` / `optim.AdamW` / `optim.Adamax` / `optim.ASGD` / `optim.LBFGS` / `optim.NAdam` / `optim.RAdam` / `optim.RMSprop` / `optim.Rprop` / `optim.SGD` / `optim.SparseAdam` / `optim.Optimizer` (基类)

*   **学习率调度器 (lr_scheduler)**: (更丰富的调度策略)
    *   `lr_scheduler.LambdaLR` (自定义 Lambda 函数) / `lr_scheduler.MultiplicativeLR` (乘性衰减) / `lr_scheduler.StepLR` / `lr_scheduler.MultiStepLR` / `lr_scheduler.ExponentialLR` / `lr_scheduler.CosineAnnealingLR` / `lr_scheduler.ReduceLROnPlateau` / `lr_scheduler.CyclicLR` (循环学习率) / `lr_scheduler.OneCycleLR` (OneCycle 策略) / `lr_scheduler.CosineAnnealingWarmRestarts` (带重启的余弦退火) / `lr_scheduler.ChainedScheduler` (链式调度器) / `lr_scheduler.SequentialLR` (序列调度器) / `lr_scheduler.ConstantLR` (恒定学习率) / `lr_scheduler.LinearLR` (线性学习率) / `lr_scheduler.PolynomialLR` (多项式学习率)

**六、数据处理 (torch.data)** (更全面的数据集和数据加载工具，以及 `torchvision`, `torchtext`, `torchaudio`  数据处理)

*   **数据集 (Datasets)**:
    *   `data.Dataset` (抽象基类) / `data.TensorDataset` / `data.IterableDataset` (可迭代数据集基类) / `data.Subset` / `data.ConcatDataset` / `data.ChainDataset` (链式数据集) / `data.random_split` / `data.dataloader.default_collate` (默认的 batch 合并函数)

*   **数据加载器 (DataLoaders)**:
    *   `data.DataLoader` / `data.get_worker_info` (获取 DataLoader worker 信息)

*   **采样器 (Samplers)**:
    *   `data.Sampler` (抽象基类) / `data.SequentialSampler` (顺序采样) / `data.RandomSampler` (随机采样) / `data.SubsetRandomSampler` (子集随机采样) / `data.WeightedRandomSampler` (加权随机采样) / `data.BatchSampler` (批采样器) / `data.distributed.DistributedSampler` (分布式采样器)

*   **数据转换 (Transforms -  `torchvision.transforms`, `torchtext.transforms`, `torchaudio.transforms`  中更丰富)**:
    *   `transforms.Compose` / `transforms.ToTensor` / `transforms.PILToTensor` / `transforms.ConvertImageDtype` / `transforms.Normalize` / `transforms.Resize` / `transforms.CenterCrop` / `transforms.RandomCrop` / `transforms.RandomResizedCrop` / `transforms.FiveCrop` / `transforms.TenCrop` / `transforms.LinearTransformation` / `transforms.RandomHorizontalFlip` / `transforms.RandomVerticalFlip` / `transforms.RandomRotation` / `transforms.RandomAffine` / `transforms.RandomGrayscale` / `transforms.RandomPerspective` / `transforms.RandomErasing` / `transforms.GaussianBlur` / `transforms.InterpolationMode` (插值模式枚举) / `transforms.functional` (函数式 transforms) / `transforms.AutoAugment` / `transforms.RandAugment` / `transforms.TrivialAugmentWide` / `transforms.AugMix` / `transforms.Cutout` / `transforms.MixUp` / `transforms.CutMix` / `transforms.FMix` / `transforms.ToPILImage` / `transforms.Grayscale` / `transforms.Lambda` / `transforms.Pad` / `transforms.RandomOrder` / `transforms.RandomChoice` / `transforms.ColorJitter` / `transforms.ElasticTransform` / `transforms.RandomApply` / `transforms.RandomInvert` / `transforms.RandomPosterize` / `transforms.RandomSolarize` / `transforms.RandomAdjustSharpness` / `transforms.RandomAutocontrast` / `transforms.RandomEqualize` / `transforms.functional_pil` (PIL 函数式 transforms)

*   **`torchvision` 数据集 (Datasets)** (`torchvision.datasets`): (图像数据集)
    *   `datasets.ImageFolder` / `datasets.DatasetFolder` (通用文件夹数据集) / `datasets.VisionDataset` (视觉数据集基类) / `datasets.CIFAR10` / `datasets.CIFAR100` / `datasets.MNIST` / `datasets.FashionMNIST` / `datasets.KMNIST` / `datasets.EMNIST` / `datasets.QMNIST` / `datasets.SVHN` / `datasets.LSUN` / `datasets.ImageNet` / `datasets.CocoDetection` / `datasets.CocoCaptions` / `datasets.Cityscapes` / `datasets.SBD` / `datasets.VOCDetection` / `datasets.VOCSegmentation` / `datasets.Caltech101` / `datasets.Caltech256` / `datasets.StanfordCars` / `datasets.FGVCAircraft` / `datasets.DTD` / `datasets.INaturalist` / `datasets.Places365` / `datasets.SUN397` / `datasets.Kitti` / `datasets.USPS` / `datasets.SEMEION` / `datasets.PhotoTour` / `datasets.SBDataset` / `datasets.CelebA` / `datasets.CelebASplits` / `datasets.FakeData` / `datasets.Flowers102` / `datasets.Food101` / `datasets.GTSRB` / `datasets.HMDB51` / `datasets.Omniglot` / `datasets.RenderedSST2` / `datasets.SogouNews` / `datasets. Kinetics` / `datasets. UCF101` / `datasets.VisionDataset`

*   **`torchtext` 数据集 (Datasets)** (`torchtext.datasets`): (文本数据集)
    *   `datasets.IMDB` / `datasets.TextClassificationDataset` (文本分类数据集基类) / `datasets.SST2` / `datasets.CoNLL2000Chunking` / `datasets.PennTreebank` / `datasets.WikiText2` / `datasets.WikiText103` / `datasets.BillionWord` / `datasets.IWSLT2016` / `datasets.IWSLT2017` / `datasets.Multi30k` / `datasets.WMT14` / `datasets.WMT16` / `datasets.WMT18` / `datasets.WMT19` / `datasets.WMT20` / `datasets.TranslationDataset` (翻译数据集基类) / `datasets.QNLI` / `datasets.MNLI` / `datasets.MRPC` / `datasets.RTE` / `datasets.SNLI` / `datasets.SQuAD1` / `datasets.SQuAD2` / `datasets.STSBenchmark` / `datasets.YelpReviewFull` / `datasets.YelpReviewPolarity` / `datasets.YahooAnswers` / `datasets.AG_NEWS` / `datasets.AmazonReviewFull` / `datasets.AmazonReviewPolarity` / `datasets.DBpedia` / `datasets.SogouNews` / `datasets.WebText` / `datasets.EnWik9` / `datasets.GoogleBooks` / `datasets.UDPOS` / `datasets.CoNLL15NER` / `datasets.CoNLL03NER` / `datasets.SNLI` / `datasets.MultiNLI` / `datasets.QNLI` / `datasets.RTE` / `datasets.MRPC` / `datasets.SST` / `datasets.STSB` / `datasets.WNLI` / `datasets.QQP` / `datasets.SciTail` / `datasets.CB` / `datasets.CR` / `datasets.MPQA` / `datasets.MR` / `datasets.Subj` / `datasets.TREC`

*   **`torchaudio` 数据集 (Datasets)** (`torchaudio.datasets`): (音频数据集)
    *   `datasets.YESNO` / `datasets.SPEECHCOMMANDS` / `datasets.LIBRISPEECH` / `datasets.LIBRIVOX` / `datasets.COMMONVOICE` / `datasets.GTZAN` / `datasets.CHIME` / `datasets.TEDLIUM` / `datasets.VCTK` / `datasets.FLAC` / `datasets.FOLDER`

*   **`torchvision` 模型 (Models)** (`torchvision.models`): (预训练模型)
    *   `models.resnet18` / `models.resnet34` / `models.resnet50` / `models.resnet101` / `models.resnet152` / `models.resnext50_32x4d` / `models.resnext101_32x8d` / `models.wide_resnet50_2` / `models.wide_resnet101_2` / `models.alexnet` / `models.vgg11` / `models.vgg11_bn` / `models.vgg13` / `models.vgg13_bn` / `models.vgg16` / `models.vgg16_bn` / `models.vgg19` / `models.vgg19_bn` / `models.squeezenet1_0` / `models.squeezenet1_1` / `models.densenet121` / `models.densenet161` / `models.densenet169` / `models.densenet201` / `models.inception_v3` / `models.googlenet` / `models.shufflenet_v2_x0_5` / `models.shufflenet_v2_x1_0` / `models.mobilenet_v2` / `models.mobilenet_v3_small` / `models.mobilenet_v3_large` / `models.lraspp_mobilenet_v3_large` / `models.deeplabv3_resnet50` / `models.deeplabv3_resnet101` / `models.deeplabv3_mobilenet_v3_large` / `models.fcn_resnet50` / `models.fcn_resnet101` / `models.segmentation.lraspp_mobilenetv3_large` / `models.segmentation.deeplabv3_resnet101` / `models.segmentation.deeplabv3_resnet50` / `models.segmentation.deeplabv3_mobilenet_v3_large` / `models.segmentation.fcn_resnet101` / `models.segmentation.fcn_resnet50` / `models.mnasnet0_5` / `models.mnasnet0_75` / `models.mnasnet1_0` / `models.mnasnet1_3` / `models.efficientnet_b0` / `models.efficientnet_b1` / `models.efficientnet_b2` / `models.efficientnet_b3` / `models.efficientnet_b4` / `models.efficientnet_b5` / `models.efficientnet_b6` / `models.efficientnet_b7` / `models.regnet_y_400mf` / `models.regnet_y_800mf` / `models.regnet_y_1_6gf` / `models.regnet_y_3_2gf` / `models.regnet_y_8gf` / `models.regnet_y_16gf` / `models.regnet_y_32gf` / `models.regnet_x_400mf` / `models.regnet_x_800mf` / `models.regnet_x_1_6gf` / `models.regnet_x_3_2gf` / `models.regnet_x_8gf` / `models.regnet_x_16gf` / `models.regnet_x_32gf` / `models.vit_b_16` / `models.vit_b_32` / `models.vit_l_16` / `models.vit_l_32` / `models.vit_h_16` / `models.swin_t` / `models.swin_s` / `models.swin_b` / `models.convnext_tiny` / `models.convnext_small` / `models.convnext_base` / `models.convnext_large` / `models.efficientnet_v2_s` / `models.efficientnet_v2_m` / `models.efficientnet_v2_l` / `models.get_model` (模型工厂函数) / `models.list_models` (列出可用模型)

*   **`torchtext` 词汇表和工具 (Vocab & Utils)** (`torchtext.vocab`, `torchtext.utils`): (文本处理工具)
    *   `vocab.Vocab` (词汇表类) / `vocab.build_vocab_from_iterator` (从迭代器构建词汇表) / `vocab.vocab` (函数式词汇表构建) / `vocab.GloVe` / `vocab.FastText` / `vocab.CharNGram` / `vocab.Vectors` (预训练词向量)
    *   `utils.download_asset` / `utils.extract_archive` / `utils.unicode_to_str` / `utils.get_asset_local_path`

*   **`torchaudio` 功能 ( функциональности )** (`torchaudio.functional`): (音频处理功能函数)
    *   `functional.resample` (重采样) / `functional.spectrogram` (频谱图) / `functional.melscale_fbanks` (Mel 滤波器组) / `functional.mel_spectrogram` (Mel 频谱图) / `functional.mfcc` (MFCC) / `functional.convolve` (卷积) / `functional.fftconvolve` (FFT 卷积) / `functional.rnnt_loss` (RNN-T 损失) / `functional.lfilter` (线性滤波器) / `functional.create_dct` (DCT 矩阵) / `functional.mu_law_encoding` (Mu-Law 编码) / `functional.mu_law_decoding` (Mu-Law 解码) / `functional.amplitude_to_DB` (幅度转 dB) / `functional.DB_to_amplitude` (dB 转幅度) / `functional.compute_deltas` (计算 delta 特征) / `functional.detect_pitch_frequency` (音高检测) / `functional.sliding_window_cmn` (滑动窗口倒谱均值归一化)

*   **`torchaudio` 效果器 (Effects)** (`torchaudio.effects`): (音频效果器)
    *   `effects.Resample` / `effects.Delay` / `effects.Reverb` / `effects.LowPassFilter` / `effects.HighPassFilter` / `effects.AllPassFilter` / `effects.BandPassFilter` / `effects.BandStopFilter` / `effects.Phaser` / `effects.Flanger` / `effects.Chorus` / `effects.Tremolo` / `effects.Vibrato`

**七、其他常用功能 (torch)** (更丰富的工具函数)

*   **设备管理**:
    *   `torch.device` / `torch.cuda.is_available()` / `torch.cuda.device_count()` / `torch.cuda.current_device()` / `torch.cuda.get_device_name()` / `torch.cuda.set_device()` / `torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()` / `torch.cuda.memory_reserved()` / `torch.cuda.max_memory_reserved()` / `torch.cuda.empty_cache()` / `torch.cuda.ipc_collect()` / `tensor.to(device)` / `tensor.cpu()` / `tensor.cuda()` / `nn.Module.to(device)` / `nn.Module.cuda()` / `nn.Module.cpu()`

*   **数据类型**: (所有 `torch` 数据类型)
    *   `torch.float32` / `torch.float` / `torch.float64` / `torch.double` / `torch.float16` / `torch.half` / `torch.bfloat16` / `torch.complex32` / `torch.complex64` / `torch.complex128` / `torch.uint8` / `torch.int8` / `torch.int16` / `torch.short` / `torch.int32` / `torch.int` / `torch.int64` / `torch.long` / `torch.bool` / `tensor.dtype` / `tensor.type(dtype)` / `tensor.float()` / `tensor.int()` / `tensor.bool()` / `tensor.double()` / `tensor.half()` / `tensor.long()` / `tensor.short()` / `tensor.char()` / `tensor.byte()`

*   **模型保存与加载**:
    *   `torch.save` / `torch.load` / `torch.jit.save` (TorchScript 模型保存) / `torch.jit.load` (TorchScript 模型加载) / `torch.onnx.export` (ONNX 导出) / `torch.onnx.load_model` (ONNX 模型加载) / `nn.Module.state_dict()` / `nn.Module.load_state_dict()` / `nn.Module.register_buffer()` / `nn.Module.register_parameter()` / `nn.Module.get_buffer()` / `nn.Module.get_parameter()` / `nn.Module.buffers()` / `nn.Module.parameters()` / `nn.Module.named_buffers()` / `nn.Module.named_parameters()`

*   **自动求导 (Autograd)**:
    *   `torch.autograd.backward()` / `torch.autograd.grad` (计算梯度) / `torch.autograd.functional.jacobian` (雅可比矩阵) / `torch.autograd.functional.hessian` (海森矩阵) / `torch.set_grad_enabled` (全局控制梯度计算) / `torch.is_grad_enabled` (检查梯度计算是否启用) / `torch.no_grad()` / `torch.enable_grad()` / `torch.autograd.no_grad()` / `torch.autograd.enable_grad()` / `torch.autograd.profiler.profile` (性能分析器) / `torch.autograd.grad_mode.set_grad_enabled` / `torch.autograd.grad_mode.no_grad` / `torch.autograd.grad_mode.enable_grad` / `tensor.requires_grad_()` / `tensor.detach()` / `tensor.grad_fn` (获取梯度函数) / `tensor.retain_grad()` (保留中间层梯度) / `tensor.is_leaf` (是否为叶子节点) / `tensor.grad` / `torch.autograd.Variable` (Variable，已被 Tensor 替代)

*   **分布式训练 (Distributed Training -  `torch.distributed`, `torch.nn.parallel.DistributedDataParallel`)**: (分布式训练相关，更高级主题)
    *   `torch.distributed.init_process_group` (初始化进程组) / `torch.distributed.destroy_process_group` (销毁进程组) / `torch.distributed.get_rank` (获取当前进程 rank) / `torch.distributed.get_world_size` (获取世界大小) / `torch.distributed.all_gather` (All-gather 收集所有进程数据) / `torch.distributed.broadcast` (Broadcast 广播数据) / `torch.distributed.reduce` (Reduce 规约数据) / `torch.distributed.scatter` (Scatter 散布数据) / `torch.distributed.gather` (Gather 收集数据) / `torch.distributed.all_reduce` (All-reduce 全规约) / `torch.distributed.reduce_scatter` (Reduce-scatter 规约后散布) / `torch.distributed.new_group` (创建新进程组) / `torch.distributed.get_world_size` / `torch.distributed.get_rank` / `torch.distributed.barrier` (屏障同步) / `torch.distributed.is_initialized` (检查是否初始化) / `torch.distributed.get_backend` (获取后端) / `torch.distributed.all_to_all` (All-to-all 全互联) / `torch.distributed.all_to_all_single` / `torch.distributed.broadcast_multigpu` / `torch.distributed.gather_multigpu` / `torch.distributed.reduce_multigpu` / `torch.distributed.reduce_scatter_multigpu` / `torch.distributed.scatter_multigpu` / `torch.nn.parallel.DistributedDataParallel` (DDP) / `torch.nn.parallel.data_parallel` (数据并行，单机多 GPU) / `torch.nn.parallel.DistributedDataParallelCPU` (CPU DDP) / `torch.distributed.launch` (分布式启动工具) / `torch.distributed.run` (分布式运行工具) / `torch.distributed.elastic.rendezvous` (Rendezvous 服务) / `torch.distributed.rpc.rpc_async` (RPC 远程过程调用) / `torch.distributed.rpc.remote` / `torch.distributed.rpc.RRef` (远程引用)

*   **性能分析和调试 (Profiling & Debugging)**:
    *   `torch.profiler.profile` (性能分析器，前面 autograd 部分也列出) / `torch.profiler.record_function` (记录函数) / `torch.autograd.set_detect_anomaly` (异常检测) / `torch.autograd.detect_anomaly` / `torch.utils.checkpoint.checkpoint` (checkpointing 内存优化) / `torch.utils.backcompat.broadcast_warning` (广播警告控制)

*   **JIT 编译和 TorchScript (JIT Compilation & TorchScript)**:
    *   `torch.jit.script` (Script 编译) / `torch.jit.trace` (Trace 追踪编译) / `torch.jit.save` / `torch.jit.load` / `torch.jit.freeze` (冻结模型) / `torch.jit.optimized_execution` (优化执行控制) / `torch.jit.ignore` (忽略 JIT 编译) / `torch.jit.is_scripting` (检查是否在 Scripting 模式) / `torch.jit.is_tracing` (检查是否在 Tracing 模式) / `torch.jit.annotations.var` (类型注解变量) / `torch.jit.unused` (标记未使用变量) / `torch.jit.fork` / `torch.jit.wait` / `torch.jit.if_then_else` / `torch.jit.while_loop` / `torch.jit.isinstance` / `torch.jit.export_opname` / `torch.jit.Attribute` / `torch.jit.CompilationUnit` / `torch.jit.Error` / `torch.jit.Graph` / `torch.jit.InplaceOrViewType` / `torch.jit.IRModule` / `torch.jit.ScriptModule` / `torch.jit.ScriptFunction` / `torch.jit.Value` / `torch.jit.FinalModule` / `torch.jit.LastModule` / `torch.jit._fork` / `torch.jit._wait` / `torch.jit._if_then_else` / `torch.jit._while_loop` / `torch.jit._isinstance` / `torch.jit._export_opname` / `torch.jit._Attribute` / `torch.jit._CompilationUnit` / `torch.jit._Error` / `torch.jit._Graph` / `torch.jit._InplaceOrViewType` / `torch.jit._IRModule` / `torch.jit._ScriptModule` / `torch.jit._ScriptFunction` / `torch.jit._Value` / `torch.jit._FinalModule` / `torch.jit._LastModule`

*   **量化 (Quantization - `torch.quantization`)**: (模型量化相关，更高级主题)
    *   `torch.quantization.quantize` (动态量化) / `torch.quantization.quantize_dynamic` (动态量化) / `torch.quantization.prepare` (准备量化) / `torch.quantization.convert` (转换量化模型) / `torch.quantization.default_qconfig` (默认量化配置) / `torch.quantization.float_qparams_weight_only_qconfig` / `torch.quantization.get_default_qconfig` / `torch.quantization.get_observer_dict` / `torch.quantization.get_qconfig_dict` / `torch.quantization.get_weight_only_quant_modules` / `torch.quantization.is_activation_quantized` / `torch.quantization.is_dynamic_quantized` / `torch.quantization.is_float_qparams_quantized_module` / `torch.quantization.is_graph_mode_ptq` / `torch.quantization.is_per_channel_quantized` / `torch.quantization.is_per_layer_quantized` / `torch.quantization.is_quantized` / `torch.quantization.is_quantized_dynamic` / `torch.quantization.load_qconfig` / `torch.quantization.max_observer` / `torch.quantization.min_max_observer` / `torch.quantization.observer.default_observer` / `torch.quantization.observer.fake_quantize` / `torch.quantization.observer.histogram_observer` / `torch.quantization.observer.logger_observer` / `torch.quantization.observer.min_max_observer` / `torch.quantization.observer.moving_average_min_max_observer` / `torch.quantization.observer.observer_base` / `torch.quantization.observer.per_channel_min_max_observer` / `torch.quantization.observer.percentile_observer` / `torch.quantization.qconfig.default_qconfig` / `torch.quantization.qconfig.float16_dynamic_qconfig` / `torch.quantization.qconfig.float_qparams_weight_only_qconfig` / `torch.quantization.qconfig.get_default_qconfig` / `torch.quantization.qconfig.get_qconfig_dict` / `torch.quantization.qconfig.qconfig` / `torch.quantization.qconfig_mapping.default_qconfig_mapping` / `torch.quantization.qconfig_mapping.get_default_qat_qconfig_mapping` / `torch.quantization.qconfig_mapping.get_default_qconfig_mapping` / `torch.quantization.qconfig_mapping.get_qat_modules_mapping` / `torch.quantization.qconfig_mapping.qconfig_mapping` / `torch.quantization.QConfigDynamic` / `torch.quantization.QConfigMapping` / `torch.quantization.QuantStub` / `torch.quantization.dequantize` / `torch.quantization.disable_observer` / `torch.quantization.enable_observer` / `torch.quantization.get_default_qat_qconfig_mapping` / `torch.quantization.get_default_static_quant_qconfig_mapping` / `torch.quantization.get_ethq_config` / `torch.quantization.get_graph_mode_ptq_backend` / `torch.quantization.get_observer_dict` / `torch.quantization.get_qconfig_dict` / `torch.quantization.get_weight_only_quant_modules` / `torch.quantization.graph_mode_fx.convert` / `torch.quantization.graph_mode_fx.prepare` / `torch.quantization.graph_mode_ptq.convert` / `torch.quantization.graph_mode_ptq.prepare` / `torch.quantization.is_activation_post_process_module` / `torch.quantization.is_activation_quantized_tensor` / `torch.quantization.is_annotated` / `torch.quantization.is_dynamic_quantized` / `torch.quantization.is_fx_qat` / `torch.quantization.is_graph_mode_ptq` / `torch.quantization.is_per_channel_quantized` / `torch.quantization.is_per_layer_quantized` / `torch.quantization.is_ptq_backend_config_dict_set` / `torch.quantization.is_quantized` / `torch.quantization.is_quantized_dynamic` / `torch.quantization.observer` / `torch.quantization.propagate_qconfig_recursive` / `torch.quantization.ptq.convert` / `torch.quantization.ptq.prepare` / `torch.quantization.qconfig` / `torch.quantization.qconfig_mapping` / `torch.quantization.QConfig` / `torch.quantization.QConfigDynamic` / `torch.quantization.QConfigMapping` / `torch.quantization.quant_type` / `torch.quantization.QuantType` / `torch.quantization.QuantStub` / `torch.quantization.QATConvert` / `torch.quantization.QATPrepare` / `torch.quantization.QuantizeWrapper` / `torch.quantization.default_observer` / `torch.quantization.default_weight_observer` / `torch.quantization.dtype` / `torch.quantization.FakeQuantize` / `torch.quantization.FixedQParamsFakeQuantize` / `torch.quantization.HistogramObserver` / `torch.quantization.MovingAverageHistogramObserver` / `torch.quantization.MovingAverageMinMaxObserver` / `torch.quantization.ObserverBase` / `torch.quantization.PerChannelMinMaxObserver` / `torch.quantization.PercentileObserver` / `torch.quantization.PythonQuantizer` / `torch.quantization.RecordHistogramObserver` / `torch.quantization.MinMaxObserver` / `torch.quantization.MovingAverageMinMaxObserver`

*   **ONNX 相关 (ONNX)**: (模型导出与互操作，更高级主题)
    *   `torch.onnx.export` / `torch.onnx.load_model` / `torch.onnx.is_in_onnx_export` / `torch.onnx.dynamo_export` / `torch.onnx.symbolic_override` / `torch.onnx.utils.extract_init_net` / `torch.onnx.utils.extract_param_dict` / `torch.onnx.utils.verify` / `torch.onnx.checker.check_model` / `torch.onnx.optimizer.optimize` / `torch.onnx.shape_inference.infer_shapes` / `torch.onnx.validation.validate_model` / `torch.onnx.OperatorExportTypes` / `torch.onnx.TrainingMode`

*   **C++ 扩展 (C++ Extensions)**: (自定义 C++ 扩展，高级主题)
    *   `torch.utils.cpp_extension.load` / `torch.utils.cpp_extension.CUDAExtension` / `torch.utils.cpp_extension.CppExtension` / `torch.utils.cpp_extension.BuildExtension` / `torch.utils.cpp_extension.get_cmake_args` / `torch.utils.cpp_extension.get_ninja_args` / `torch.utils.cpp_extension.verify_ninja_availability`

*   **其他工具**:
    *   `torch.version.__version__` / `torch.__version__` / `torch.show_config()` (显示 PyTorch 配置) / `torch.typename()` / `torch.set_printoptions()` / `torch.is_deterministic()` / `torch.use_deterministic_algorithms()` / `torch.are_deterministic_algorithms_enabled()` / `torch.set_anomaly_enabled()` / `torch.is_anomaly_enabled()` / `torch.get_default_dtype()` / `torch.set_default_dtype()` / `torch.get_default_device()` / `torch.set_default_device()` / `torch.set_flush_denormal` / `torch.get_flush_denormal` / `torch.set_num_threads()` / `torch.get_num_threads()` / `torch.set_num_interop_threads()` / `torch.get_num_interop_threads()` / `torch.backends.cudnn.benchmark` / `torch.backends.cudnn.deterministic` / `torch.backends.cudnn.enabled` / `torch.backends.mkldnn.enabled` / `torch.backends.mkl.is_available` / `torch.backends.openmp.is_available` / `torch.backends.openmp.max_threads` / `torch.backends.cudnn.version` / `torch.backends.cuda.is_built` / `torch.backends.mps.is_available` / `torch.backends.mps.is_built` / `torch.backends.backend` / `torch.backends.cpu.is_reading_cpu_count_from_env`
