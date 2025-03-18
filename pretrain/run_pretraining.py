"""使用transformers风格的EEGPT模型进行预训练"""

import os
import torch
import random
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from configuration_eegpt import EEGPTConfig
from modeling_eegpt import EEGPTForPreTraining
from configs import MODELS_CONFIGS, tag, variant, train_loader, valid_loader, max_epochs, steps_per_epoch

# 设置float32矩阵乘法精度
torch.set_float32_matmul_precision("medium")

def seed_torch(seed=1029):
    """设置随机种子以确保实验可重现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机种子
seed_torch(7)

# 从配置中获取模型参数
model_config = MODELS_CONFIGS[tag]
config = EEGPTConfig(
    embed_dim=model_config["embed_dim"],
    embed_num=model_config["embed_num"],
    depth=model_config["depth"][0],
    predictor_depth=model_config["depth"][1],
    reconstructor_depth=model_config["depth"][2],
    num_heads=model_config["num_heads"],
    use_part_pred=True,
    max_epochs=max_epochs
)

# 如果需要，将配置保存到文件
config_dir = f"./configs/EEGPT_{tag}_{variant}"
os.makedirs(config_dir, exist_ok=True)
config.to_json_file(os.path.join(config_dir, "config.json"))

# 使用Lightning包装器包装模型
class LitEEGPT(pl.LightningModule):
    def __init__(self, config, use_loss_a=True, use_ln=True, use_skip=True):
        super().__init__()
        
        self.USE_LOSS_A = use_loss_a
        self.USE_LN = use_ln
        self.USE_SKIP = use_skip
        
        # 创建预训练模型
        self.model = EEGPTForPreTraining(config)
        
        # 准备通道ID
        use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2']
        self.chans_id = self.model.encoder.prepare_chan_ids(use_channels_names)
        
        # 损失函数
        self.loss_fn = torch.nn.MSELoss()
        
    def make_masks(self, num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):
        """创建用于预训练的掩码"""
        C, N = num_patchs
        
        while True:
            mask_x = []  # mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random() > p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x) == 0: continue
            if len(mask_y_bx) == 0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape) < p_c_y]
            if len(mask_y_bx) == 0: continue
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y + [mask_y_bx], dim=0)
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.model.encoder.num_patches)
        
        outputs = self.model(x, chan_ids=self.chans_id.to(x), mask_x=mask_x, mask_y=mask_y)
        
        loss1 = outputs["contrast_loss"]
        loss2 = outputs["reconstruct_loss"]
        
        if self.USE_LOSS_A:
            loss = loss1 + loss2
        else:
            loss = loss2
        
        # 记录损失
        self.log('valid_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.model.encoder.num_patches)
        
        outputs = self.model(x, chan_ids=self.chans_id.to(x), mask_x=mask_x, mask_y=mask_y)
        
        loss1 = outputs["contrast_loss"]
        loss2 = outputs["reconstruct_loss"]
        
        if self.USE_LOSS_A:
            loss = loss1 + loss2
        else:
            loss = loss2
        
        # 记录损失
        self.log('train_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        self.wd_scheduler.step()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        from utils import grad_logger
        
        # 记录梯度统计信息
        grad_stats = grad_logger(self.model.encoder.named_parameters())
        self.log('grad_stats.first_layer', grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min', grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max', grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)
        
        # 目标编码器的动量更新
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.model.encoder.parameters(), self.model.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        from utils import CosineWDSchedule
        from configs import max_lr
        
        # 参数分组
        param_groups = [
            {
                'params': (p for n, p in self.model.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.model.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.model.reconstructor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.model.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.model.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.model.reconstructor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
        
        # 创建优化器
        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)
        
        # 创建学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            steps_per_epoch=steps_per_epoch, 
            epochs=max_epochs,
            div_factor=2,
            final_div_factor=8,
            pct_start=0.2
        )
        
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }
        
        # 创建权重衰减调度器
        self.wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=1e-6,
            final_wd=1e-6,
            T_max=int(max_epochs*steps_per_epoch)
        )
        
        # 创建动量调度器
        ema = [0.996, 1.0]
        self.momentum_scheduler = (
            ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
            for i in range(int(steps_per_epoch*max_epochs)+1)
        )
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}


# 初始化模型
model = LitEEGPT(
    config,
    use_loss_a=(variant != "A"),
    use_ln=(variant != "B"),
    use_skip=(variant != "C")
)

# 设置回调
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

# 设置Logger
logger = [
    pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{tag}_{variant}_tb"), 
    pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{tag}_{variant}_csv")
]

# 创建Trainer并开始训练
trainer = pl.Trainer(
    strategy='auto',
    devices=[0],  # 根据实际情况配置
    max_epochs=max_epochs,
    callbacks=callbacks,
    logger=logger
)

# 开始训练
trainer.fit(model, train_loader, valid_loader)