"""EEG预训练模型向后兼容性模块

该模块提供了向原始API的兼容性，所有实际实现都在modeling_eegpt.py中。
这个文件只是一个兼容层，以确保依赖于旧API的代码仍然能够工作。
新代码应该直接使用modeling_eegpt.py中的实现。
"""

import warnings
from modeling_eegpt import (
    CHANNEL_DICT,
    apply_mask,
    apply_mask_t,
    _no_grad_trunc_normal_,
    trunc_normal_,
    rotate_half,
    apply_rotary_emb,
    RotaryEmbedding,
    DropPath,
    MLP,
    Attention,
    Block,
    PatchEmbed,
    PatchNormEmbed,
    EEGPTModel,
    EEGPTPredictor,
    EEGPTReconstructor
)

warnings.warn(
    "使用来自 modeling_pretraining.py 的类和函数已被弃用。"
    "请直接从 modeling_eegpt.py 导入相应的类和函数。",
    DeprecationWarning,
    stacklevel=2
)

# 为了向后兼容，我们保留原始名称
EEGTransformer = EEGPTModel
EEGTransformerPredictor = EEGPTPredictor
EEGTransformerReconstructor = EEGPTReconstructor