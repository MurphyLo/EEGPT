"""EEGPT模型配置

这个文件包含EEGPT模型的各种配置参数，包括数据加载、训练参数和模型结构参数。
"""

import torch
import torchvision
import math
import random

def load_fn(x):
    """
    加载EEG数据的函数，用于数据集加载器
    
    Args:
        x: EEG数据文件路径
        
    Returns:
        tensor: 加载并处理后的EEG数据
    """
    x = torch.load(x)
    
    window_length = 4*256  # 4秒的数据，采样率256Hz
    data_length = x.shape[1]  

    # 计算窗口可以开始的最大索引
    max_start_index = data_length - window_length

    # 随机选择一个起始点
    if max_start_index > 0:
        index = random.randint(0, max_start_index)
        x = x[:, index:index+window_length]
    x = x.to(torch.float)
    return x

# 训练参数
max_epochs = 200  # 最大训练轮次
max_lr = 5e-4     # 最大学习率
batch_size = 64   # 批大小
devices = [0]     # 使用的GPU设备

# 数据集和数据加载器
train_dataset = torchvision.datasets.DatasetFolder(
    root="../datasets/pretrain/merged/TrainFolder/", 
    loader=load_fn,  
    extensions=['.edf']
)

valid_dataset = torchvision.datasets.DatasetFolder(
    root="../datasets/pretrain/merged/ValidFolder/", 
    loader=load_fn, 
    extensions=['.edf']
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    num_workers=0, 
    shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    num_workers=0, 
    shuffle=False
)

# 计算每轮的步数
steps_per_epoch = math.ceil(len(train_loader)/len(devices))

# 模型配置
tag = "tiny1"  # 使用的模型大小标签
variant = "D"   # 模型变体

# 不同大小的模型配置
MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim": 64, 
        "embed_num": 1, 
        "depth": [2, 2, 4], 
        "num_heads": 4
    },
    "tiny2": {
        "embed_dim": 64, 
        "embed_num": 4, 
        "depth": [2, 2, 4], 
        "num_heads": 4
    },
    "tiny3": {
        "embed_dim": 64, 
        "embed_num": 4, 
        "depth": [8, 8, 8], 
        "num_heads": 4
    },
    "little": {
        "embed_dim": 128, 
        "embed_num": 4, 
        "depth": [8, 8, 8], 
        "num_heads": 4
    },
    "base1": {
        "embed_dim": 256, 
        "embed_num": 1, 
        "depth": [6, 6, 6], 
        "num_heads": 4
    },
    "base2": {
        "embed_dim": 256, 
        "embed_num": 4, 
        "depth": [8, 8, 8], 
        "num_heads": 4
    },
    "base3": {
        "embed_dim": 512, 
        "embed_num": 1, 
        "depth": [6, 6, 6], 
        "num_heads": 8
    },
    "large": {
        "embed_dim": 512, 
        "embed_num": 4, 
        "depth": [8, 8, 8], 
        "num_heads": 8
    },
}

def get_config(embed_dim=512, embed_num=4, depth=[8, 8, 8], num_heads=4):
    """
    获取模型配置
    
    Args:
        embed_dim: 嵌入维度
        embed_num: 嵌入数量
        depth: 各部分的深度 [encoder_depth, predictor_depth, reconstructor_depth]
        num_heads: 注意力头数
        
    Returns:
        dict: 模型配置字典
    """
    models_configs = {
        'encoder': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'depth': depth[0],
            'num_heads': num_heads,
        },
        'predictor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'predictor_embed_dim': embed_dim,
            'depth': depth[1],
            'num_heads': num_heads,
        },
        'reconstructor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'reconstructor_embed_dim': embed_dim,
            'depth': depth[2],
            'num_heads': num_heads,
        },
    }
    return models_configs



        