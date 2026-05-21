---
title: "大模型的相关内容"
date: 2026-05-21T22:30:00+08:00
draft: false
---



# 实现LLaMA2 大模型 

参考内容  [Happy-LLM](https://datawhalechina.github.io/happy-llm/#/)



背景内容暂不详细罗列，只对代码进行相关总结

## LLaMA2 

### 1. 超参数定义



```python
# 自定义一个 ModelConfig 类 储存和记录超参数

from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = 'Tink-K'
    def __init__(
            self,
            dim: int = 768, # 模型维度
            n_layers: int = 12, # Transformer 层数
            n_heads: int = 16, # 注意力机制的头数
            n_kv_heads: int = 8, # 键值头的数量
            vocab_size: int = 6144, # 词汇表大小
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64, 
            norm_eps: float = 1e-5, # 归一化层的eps
            max_seq_len: int = 512, #最大序列长度
            dropout: float = 0.0, # dropout 概率
            flash_attn: bool = True, # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
        


# 更加合理的版本 ： 

# from transformers import PretrainedConfig

# class ModelConfig(PretrainedConfig):
#     model_type = 'Tink-K'
    
#     def __init__(
#             self,
#             dim: int = 768,          # 模型维度
#             n_layers: int = 12,      # Transformer 层数
#             n_heads: int = 16,       # 注意力机制的头数
#             n_kv_heads: int = 8,     # 键值头的数量 (MQA/GQA 机制)
#             vocab_size: int = 6144,   # 词汇表大小
#             hidden_dim: int = None,  # 隐藏层维度
#             multiple_of: int = 64,   # 隐藏层维度对齐的倍数
#             norm_eps: float = 1e-5,  # 归一化层的 eps
#             max_seq_len: int = 512,  # 最大序列长度
#             dropout: float = 0.0,    # dropout 概率
#             flash_attn: bool = True, # 是否使用 Flash Attention
#             **kwargs,
#     ):
#         self.dim = dim
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.n_kv_heads = n_kv_heads
#         self.vocab_size = vocab_size
#         self.multiple_of = multiple_of
#         self.norm_eps = norm_eps
#         self.max_seq_len = max_seq_len
#         self.dropout = dropout
#         self.flash_attn = flash_attn
        
#         # 完善 hidden_dim 的计算逻辑：如果为 None，则基于 dim 自动计算并对齐到 multiple_of 的倍数
#         if hidden_dim is None:
#             # 经典的 4/3 * dim 并在 SwiGLU 激活函数下优化的 FFN 维度计算
#             hidden_dim = int(2 * (4 * dim / 3) / 2)
#             hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
#         self.hidden_dim = hidden_dim
        
#         # 修正拼写错误，正确调用父类的初始化
#         super().__init__(**kwargs)


```

更加合理的版本继承自 Hugging Face 的 `PretrainedConfig`.



### 2. 构建 RMSNorm

其中 RMSNorm 公式为： 
$$
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^nx_i^2+\epsilon}}\cdot\gamma
$$
代码实现：

```python
#  实现 LLaMA2 中的 RMSNorm 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps 防止分母为 0 
        self.eps = eps
        # weight 是一个可学习的参数, 初始化为 1 
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMSNorm 核心部分
        # x.pow(2).mean(-1, keepdim = True) 计算输入 x 的平方的均值
        # torch.rsqrt 是平方根的倒数, 这样就得到了RMSNorm 的分母部分, + eps 防止为 0  
        # 最后乘 x, 得到 RMSNorm 结果
        
        # 计算过程：
        # 1. x.pow(2)  --> x 平方
        # 2.   .mean(-1, keepdim = True) --> 在最后一个维度(dim = 768)，求平均值。 
        # keepdim = True 确保求完均值后，张量的形状从 [1, 50, 768] ---> [1, 50, 1]
        # 3. torch.rsqrt() 求平方根倒数。 
        
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim = True) + self.eps )
    def forward(self, x):
        # 将 x 转为  float 类型, 然后进行RMSNorm, 最后再转回原来的数据类型
        # 最后乘 weight, 这是 RMSNorm 的 一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    


# a simple test
norm = RMSNorm(768, 1e-5)
x = torch.randn(1, 50, 768)
output = norm(x)
print(output.shape)

```

输出结果

```bash
---> [1, 50, 768]
```





### 3.构建LLaMA2 Attention 部分

选择使用 GQA (Grouped-Query Attention，分组查询注意力)  构建 LLaMA Attention 模块。

为了减少显存占用并加速推理，**Query（查询）的头数通常比 Key/Value（键/值）的头数多**。在计算注意力权重之前，我们必须把 Key 和 Value 的头数“复制、扩张”到和 Query 一样多，这就是这个函数的核心任务。

#### 3.1 repeat_kv

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple  # python版本问题 
```



```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # acquire Tensor shape : 
    # batch_size, seq_len, key/Value number, every head dim
    
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # if repeat = 1, return initional Tensor, No repeat 
    # 
    if n_rep == 1:
        return x
    # expand Tensor and  reshape Tensor to repeat  key-value
    # 
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )

```

如果 n_rep == 1  Query 头数和 KV 头数刚好相等， 说明不需要复制，直接返回原张量。

当 n_rep > 1 ，进入变换。  `x[:, :, :, None, :]`

利用 `None`（等价于 `np.newaxis`），在倒数第二维（即 `n_kv_heads` 和 `head_dim` 之间）插入一个长度为 `1`的新维度。



### 3.2 旋转嵌入

旋转嵌入为注意力机制提供更强的上下文信息，从而提高模型的性能。

首先，构造获得旋转嵌入的实部和虚部的函数：

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
```

预计算 RoPE 的函数， 传统的 GPT 采用直接“加”一个位置向量( 绝对位置编码)，而 LLaMA 则采用**将向量在二维空间中旋转一个角度**的方法。

RoPE 计算公式

 对于特征维度（`head_dim`）中的第 i  组双通道，其基础旋转频率（旋转速度）为：
$$
\theta_i = \theta^{-\frac{2i}{d}}, \quad i \in [0, 1, \dots, \frac{d}{2}-1]
$$
**`torch.outer(t, freqs)`**：**外乘积（Outer Product）**。它让位置向量 `t`（形状为 `[end]`）与频率向量 `freqs`（形状为 `[dim // 2]`）进行交叉相乘。

矩阵的行代表 位置  t （第几个字）, 矩阵的列代表**通道 i**（特征的哪一部分）。网格里的每一个值就是 :

t * \theta_i，即**第 t 个位置的 Token 在第  i 个通道应该旋转的绝对弧度**。

最后根据欧拉公式，得到相对应的正余弦投影。 $t \cdot \theta_i$

**最终输出**：两个形状完全相同的二维矩阵，形状均为 **`[end, dim // 2]`**



