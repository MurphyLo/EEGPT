"""
此模块已被废弃，请改用新的实现。

LitEEGPT类已移至run_pretraining.py文件中。
所有模型实现已移至modeling_eegpt.py文件中。
配置类已移至configuration_eegpt.py文件中。
"""

import warnings

warnings.warn(
    "engine_pretraining.py模块已被废弃。"
    "LitEEGPT类已移至run_pretraining.py中。"
    "模型实现已移至modeling_eegpt.py中。"
    "配置类已移至configuration_eegpt.py中。",
    DeprecationWarning,
    stacklevel=2
)

# 保留原始导入以便向后兼容
from run_pretraining import LitEEGPT, seed_torch
from modeling_pretraining import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor

# 保留通道列表以便向后兼容
use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                           'AF3', 'AF4', 
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
         'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                  'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                           'O1', 'OZ', 'O2', ] 