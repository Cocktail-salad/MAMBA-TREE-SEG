import functools
import os
import torch
from collections import OrderedDict
from timm.scheduler import CosineLRScheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0
# 这个函数定义了一个判断是否能整除的函数。它接收两个参数:num:需要判断的数字multiple:除数

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def cuda_cast(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper


def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16):
    if hasattr(model, 'module'):
        model = model.module
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    checkpoint = {
        'net': weights_to_cpu(model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f)

    # remove previous checkpoints unless they are a power of 2 or a multiple of save_freq
    epoch = epoch - 1
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq):
            os.remove(f)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False):# load_checkpoint的作用是从检查点文件中加载模型参数和优化器状态到内存中
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    # if not (torch.equal(torch.tensor(state_dict['net']['input_conv.0.weight'].shape), torch.tensor([32, 3, 3, 3, 7]))):
    #     state_dict['net']['input_conv.0.weight'] = torch.permute(state_dict['net']['input_conv.0.weight'], (3, 0, 1, 2, 4))
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining 在预训练的情况下跳过不匹配大小的张量
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(
            f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

    # load optimizer
    if optimizer is not None:
        assert 'optimizer' in state_dict
        optimizer.load_state_dict(state_dict['optimizer'])

    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
    else:
        epoch = 0
    return epoch + 1


def build_optimizer(model, optim_cfg):# 这个build_optimizer函数的作用是根据配置信息构建优化器对象
    assert 'type' in optim_cfg
    _optim_cfg = optim_cfg.copy()# 拷贝一下optim_cfg字典
    optim_type = _optim_cfg.pop('type')# 这行代码从字典_optim_cfg中取出键为'type'的值,并赋值给变量optim_type
    optim = getattr(torch.optim, optim_type)# 用getattr可以根据字符串动态获取属性,这就实现了根据配置文件选择不同优化器的目的
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)# 返回一个已经初始化完的优化器对象


def build_cosine_scheduler(cfg, optimizer):# 函数首先定义一个名为CosineLRScheduler的类。这个类可以帮助我们按照余弦函数的波动规律来自动调整学习率。
    scheduler = CosineLRScheduler(optimizer,
                t_initial=cfg.t_initial,
                lr_min=cfg.lr_min,
                cycle_decay=cfg.cycle_decay,
                warmup_lr_init=cfg.warmup_lr_init,
                warmup_t=cfg.warmup_t,
                cycle_limit=cfg.cycle_limit,
                t_in_epochs=cfg.t_in_epochs)
    return scheduler


def build_dataloader(dataset, batch_size=1, num_workers=1, training=True):# 输入参数包括数据集dataset,每批样本数batch_size,以及工作线程数num_workers等
    shuffle = training
    sampler = None
    
    if sampler is not None:
        shuffle = False
    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True)
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True)
# 这段代码是一个函数,它的功能是构建DataLoader。
# DataLoader负责从数据集中 efficiently地读取批次数据送入模型训练。

@cuda_cast
# def point_wise_loss(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, n_points=None):
# # 这个point_wise_loss函数定义了计算语义预测和偏移预测的点wise损失的方法。
#     if n_points is not None and len(offset_predictions) >= n_points:# .采样点数n_points设置了值,且样本数量大于n_points,则进入这部分逻辑
#         permuted_indices_sem = torch.randperm(len(semantic_prediction_logits))# 对语义预测和偏移预测样本分别进行随机排列,生成随机索引。semantic_prediction_logits是模型对输入样本做出的语义预测结果张量
#         # permuted_indices_sem表示语义预测结果样本的随机排列索引，它可以用于后续根据索引高效提取随机采样的样本计算损失
#         permuted_indices_off = torch.randperm(len(offset_predictions))
#         ind_sem = permuted_indices_sem[:n_points]
#         ind_off = permuted_indices_off[:n_points]
#     else:
#         ind_sem = torch.arange(len(semantic_prediction_logits))
#         ind_off = torch.arange(len(offset_predictions))
#
#     if len(semantic_prediction_logits) == 0:
#         semantic_loss = 0 * semantic_labels.sum()
#     else:
#         # semantic_loss
#         semantic_loss = F.cross_entropy(
#             semantic_prediction_logits[ind_sem], semantic_labels[ind_sem], reduction='sum') / len(semantic_prediction_logits[ind_sem])
#         # 计算类别权重，例如使用标签分布的反比例
#         # unique_labels, counts = torch.unique(semantic_labels, return_counts=True)
#         # class_weights = 1.0 / counts.float()
#         # class_weights = class_weights / class_weights.sum()  # 归一化
#         #
#         # # 计算带权重的语义损失
#         # semantic_loss = F.cross_entropy(
#         #     semantic_prediction_logits[ind_sem],
#         #     semantic_labels[ind_sem],
#         #     weight=class_weights.to(semantic_prediction_logits.device),
#         #     reduction='sum'
#         # ) / len(semantic_prediction_logits[ind_sem])
#
#     if len(offset_predictions) == 0:
#         offset_loss = 0 * offset_predictions.sum()
#     else:
#         # offset loss
#         offset_losses = (offset_predictions[ind_off] - offset_labels[ind_off]).pow(2).sum(1).sqrt()
#         offset_loss = offset_losses.mean()
#         # offset_losses = F.smooth_l1_loss(
#         #     offset_predictions[ind_off], offset_labels[ind_off], reduction='none'
#         # )
#         # offset_loss = offset_losses.mean()
#
#     return semantic_loss, offset_loss

def point_wise_loss(semantic_prediction_logits, offset_predictions, masks_sem, masks_off, semantic_labels,
                    offset_labels, weights=None):
    if masks_sem.sum() == 0:
        semantic_loss = 0 * semantic_prediction_logits.sum()
    else:
        if weights is None:
            # semantic_loss
            semantic_loss = F.cross_entropy(
                semantic_prediction_logits[masks_sem], semantic_labels[masks_sem], reduction='sum') / len(
                semantic_prediction_logits[masks_sem])
        else:
            # semantic_loss
            semantic_loss = (F.cross_entropy(
                semantic_prediction_logits[masks_sem], semantic_labels[masks_sem],
                reduction='none') * weights).sum() / len(semantic_prediction_logits[masks_sem])

    if masks_off.sum() == 0:
        offset_loss = 0 * offset_predictions.sum()
    else:
        # offset loss
        offset_losses = (offset_predictions[masks_off] - offset_labels[masks_off]).pow(2).sum(1).sqrt()
        offset_loss = offset_losses.mean()

    return semantic_loss, offset_loss