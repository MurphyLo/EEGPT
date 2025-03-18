"""EEGPT模型配置类

该文件定义了EEGPT模型的配置类，用于控制模型的结构和参数。
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import json
import os


@dataclass
class EEGPTConfig:
    """EEGPT模型配置类

    该类定义了EEGPT模型的所有配置参数，用于初始化和控制模型行为。

    Attributes:
        model_type (str): 模型类型标识符
        img_size (Tuple[int, int]): EEG数据的形状 [通道数, 时间步]
        patch_size (int): Patch的大小
        patch_stride (Optional[int]): Patch的步长，如果为None则使用patch_size
        embed_dim (int): 嵌入维度
        embed_num (int): 嵌入数量
        depth (int): 编码器Transformer层数
        predictor_depth (int): 预测器Transformer层数
        reconstructor_depth (int): 重建器Transformer层数
        num_heads (int): 多头注意力的头数
        mlp_ratio (float): MLP中隐藏维度与输入维度的比率
        qkv_bias (bool): 是否在QKV投影中使用偏置
        predictor_embed_dim (int): 预测器嵌入维度
        reconstructor_embed_dim (int): 重建器嵌入维度
        drop_rate (float): Dropout比率
        attn_drop_rate (float): 注意力Dropout比率
        drop_path_rate (float): 随机路径Dropout比率
        use_rope (bool): 是否使用旋转位置编码
        interpolate_factor (float): 位置插值因子
        init_std (float): 参数初始化标准差
        max_epochs (int): 最大训练轮次
        max_lr (float): 最大学习率
        batch_size (int): 批处理大小
        use_part_pred (bool): 是否使用部分预测
    """
    model_type: str = "eegpt"
    
    # 图像和Patch参数
    img_size: Tuple[int, int] = (58, 256*4)  # EEG数据的形状 [通道数, 时间步]
    patch_size: int = 32*2
    patch_stride: Optional[int] = None
    
    # Transformer参数
    embed_dim: int = 768
    embed_num: int = 1
    depth: int = 12
    predictor_depth: int = 6
    reconstructor_depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    
    # 预测器和重建器参数
    predictor_embed_dim: int = 384
    reconstructor_embed_dim: int = 384
    
    # 正则化参数
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    
    # 其他参数
    use_rope: bool = True
    use_part_pred: bool = True
    interpolate_factor: float = 2.0
    init_std: float = 0.02
    
    # 训练参数
    max_epochs: int = 200
    max_lr: float = 5e-4
    batch_size: int = 64
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典形式

        Returns:
            Dict[str, Any]: 包含配置的字典
        """
        output = {}
        for key, value in self.__dict__.items():
            output[key] = value
        return output
    
    def to_json_string(self) -> str:
        """将配置转换为JSON字符串

        Returns:
            str: JSON格式的配置字符串
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True)
    
    def to_json_file(self, json_file_path: str) -> None:
        """将配置保存到JSON文件

        Args:
            json_file_path (str): JSON文件路径
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EEGPTConfig":
        """从字典创建配置

        Args:
            config_dict (Dict[str, Any]): 配置字典

        Returns:
            EEGPTConfig: 配置对象
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, json_file_path: str) -> "EEGPTConfig":
        """从JSON文件加载配置

        Args:
            json_file_path (str): JSON文件路径

        Returns:
            EEGPTConfig: 配置对象
        """
        with open(json_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "EEGPTConfig":
        """从预训练模型加载配置

        Args:
            pretrained_model_name_or_path (str): 预训练模型名称或路径
            **kwargs: 其他参数

        Returns:
            EEGPTConfig: 配置对象
        """
        config_dict = kwargs.pop("config_dict", None)
        # 如果提供了config_dict，直接使用
        if config_dict is not None:
            return cls.from_dict(config_dict)
        
        # 否则尝试从文件加载
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                return cls.from_json_file(config_file)
        
        # 如果是模型名称，可以从Hugging Face hub加载
        # 这里简化处理，实际应该添加从hub下载的逻辑
        raise ValueError(
            f"无法从{pretrained_model_name_or_path}加载配置，请提供有效的模型路径或配置字典"
        ) 