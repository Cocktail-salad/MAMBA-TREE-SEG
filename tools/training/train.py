import sys
sys.path.append('/home/jjgw/The project/TreeLearn-main/')
import torch
import torch.nn.parallel
import tqdm
import numpy as np
import time
from collections import defaultdict
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer,
                            point_wise_loss, get_eval_res_components, get_segmentation_metrics, build_dataloader)
from tree_learn.model import TreeLearn
from tree_learn.dataset import TreeDataset

TREE_CLASS_IN_DATASET = 0 # semantic label for tree class in pytorch dataset
NON_TREE_CLASS_IN_DATASET = 1 # semantic label for non-tree class in pytorch dataset
TREE_CONF_THRESHOLD = 0.5 # minimum confidence for tree prediction






def train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer):
    model.train() # 将模型设置为训练模式
    start = time.time() # 记录训练起始时间初
    losses_dict = defaultdict(list) # 始化一个有序字典来记录各损失项的值


    for i, batch in enumerate(train_loader, start=1):
        # break after a fixed number of samples have been passed 在通过一定数量的采样后中断
        if config.examples_per_epoch < (i * config.dataloader.train.batch_size):
            break
        
        scheduler.step(epoch)
        # scheduler表示学习率调整策略的对象 step(epoch)方法是触发学习率变化的接口,让学习率根据当前轮数自动调整 这样可以实现学习率随训练过程逐步下降,帮助优化收敛。
        optimizer.zero_grad()
        # optimizer是优化器对象,如Adam等。它是在主训练函数中构建好的。zero_grad()是PyTorch优化器中的一个方法。
        with torch.cuda.amp.autocast(enabled=config.fp16):
        # PyTorch提供了torch.cuda.amp模块,用于实现混合精度训练技术AMP(Automatic Mixed Precision)。enabled=config.fp16指定是否开启AMP混合精度功能。
        # AMP可以在训练中自动选择将算子和训练动作转换为half精度(fp16),从而加速训练速度。
        # 但forward和backward passes仍然保持在float32的高精度,以保证训练稳定性。
            # forward
            loss, loss_dict = model(batch, return_loss=True)
            # loss, loss_dict返回的包括总损失和各子损失。model(batch)会对batch数据进行前向传播,计算预测等。return_loss=True的意思是返回loss
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())
            # 将模型前向计算得到的各个损失项提取出来,并记录到losses_dict字典中,方便后续统计和使用。
            # detach():将张量从计算图分离cpu():将张量从GPU移动到CPU
            # item():将张量转化为Python数值。append()来自Python内置list
        # backward
        scaler.scale(loss).backward()
        # 利用混合精度API对损失进行自动缩放,启动梯度下降的反向传播计算,既提高速度又保证正确性。
        if config.grad_norm_clip:# 检查是否开启梯度裁剪功能，目的是避免梯度爆炸问题,控制模型更新的幅度在一个合理范围。
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip, norm_type=2)
            # torch.nn.utils.clip_grad_norm_ 是PyTorch提供的用于梯度裁剪的函数。
            # model.parameters()获取当前模型的参数tensors。
            # config.grad_norm_clip指定最大允许的梯度norm值。
        scaler.step(optimizer)# 使用混合精度信息unscale梯度,并调用优化器更新参数
        scaler.update()# 同时更新AMP管理的混合精度状态,为后续训练做好准备

    epoch_time = time.time() - start# 通过开始和结束时间戳的差别,计算一个epoch的训练耗时,方便后续统计和分析。
    lr = optimizer.param_groups[0]['lr']
    # 这行代码获取到当前批次的学习率值,后面会记录到日志或者tensorboard中,方便观察学习率的折线图,看是否按预期进行衰减调整。
    # 通过优化器对象可以实时获取动态学习率,记录它将有助于调试和分析训练是否正常进行。
    writer.add_scalar('train/learning_rate', lr, epoch)
    # 这行代码的目的是将学习率数值记录到TensorBoard中,为后续分析提供参考,帮助优化训练过程。
    average_losses_dict = {k: sum(v) / len(v) for k, v in losses_dict.items()}
    # 计算每个损失项在一个epoch中的平均值,构建一个包含损失名称和损失平均值的字典average_losses_dict。
    for k, v in average_losses_dict.items():
        writer.add_scalar(f'train/{k}', v, epoch)
    # 所以这段代码循环记录每个损失项的平均值到TensorBoard,方便可视化各损失在训练过程中的变化。
    log_str = f'[TRAINING] [{epoch}/{config.epochs}], time {epoch_time:.2f}s'
    # 这行代码定义了一个格式化的日志字符串模板,它包含了训练必需的基础指标,方便后续输出完整的训练监控日志。
    for k, v in average_losses_dict.items():
        log_str += f', {k}: {v:.2f}'
    # 利用上面预先定义的日志模板,简单地通过字符串拼接的方式输出了一个全面且易读的训练日志。
    logger.info(log_str)
    # 将预先定义好的日志字符串log_str通过logger.info域记录到日志中。logger.info域提供日志记录服务,可以将训练进展信息以标准形式保存下来。
    checkpoint_save(epoch, model, optimizer, config.work_dir, config.save_frequency)
    # 在每config.save_frequency个周期保存一次检查点。
    # 它会将当前的模型model状态,优化器optimizer状态保存到文件中,默认保存路径为config.work_dir。
    # 通过检查点机制可以将训练过程中模型的状态定期保存下来,在发生异常时可以从检查点继续训练,防止训练结果损失。

# def validate(config, epoch, model, val_loader, logger, writer):  # 函数实现：验证集上模型预测和评估计算，指标结果的记录和监控
#     with torch.no_grad():# torch.no_grad()是一个上下文管理器,它可以暂时取消梯度计算。在这段代码中,我们是在验证模式下进行模型推理,不需要计算和更新模型参数,所以使用no_grad可以提高效率。
#         model.eval()
#         semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels = [], [], [], [], [], []
#         # semantic_prediction_logits:存放每批语义预测的logits值
#         # offset_predictions:存放每批偏移值的预测结果，semantic_labels:存放每批样本的语义 ground truth 标签
#         # offset_labels:存放每批样本的偏移 ground truth 值
#         # coords:存放每批样本的坐标信息，instance_labels:存放每批样本对应的实例id标签
#         for batch in tqdm.tqdm(val_loader):# val_loader是一个数据加载器,它会将验证集一次批量加载成多个batch
#
#             # forward
#             output = model(batch, return_loss=False)
#             # model(batch)表示对当前batch输入数据进行前向传播,通过深度学习模型计算预测结果
#             offset_prediction, semantic_prediction_logit = output['offset_predictions'], output['semantic_prediction_logits']
#
#             batch['coords'] = batch['coords'] + batch['centers']
#             # batch['coords']获取当前batch内所有样本的坐标信息。
#             # batch['centers']获取当前batch内所有样本的中心点信息。
#             # 行为batch['coords'] = batch['coords'] + batch['centers']表示:
#             semantic_prediction_logits.append(semantic_prediction_logit[batch['masks_sem']])
#             # 通过mask提取每个batch真实任务点的语义预测，将不同batch连接成一个列表,统一开展后续验证过程
#             semantic_labels.append(batch['semantic_labels'][batch['masks_sem']])
#             # 使用mask只提取每个batch内有效点的语义gt标签，将不同batch的有效点gt通过append聚合到一个列表中
#             offset_predictions.append(offset_prediction[batch['masks_sem']])
#             # 每个batch的有效点的偏移预测结果到列表中,为后续偏移误差评估提供数据支持。
#             offset_labels.append(batch['offset_labels'][batch['masks_sem']])
#             # 使用掩码 mask 只从每个batch中提取有效点的偏移ground truth，将不同batch有效点偏移gt通过append函数汇总成一个offset_labels列表
#             coords.append(batch['coords'][batch['masks_sem']]),
#             instance_labels.append(batch['instance_labels'][batch['masks_sem']])
#             # 每个batch有效点的坐标信息coords，每个batch有效点的实例标签instance_labels
#     semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0), torch.cat(semantic_labels, 0)
#     offset_predictions, offset_labels = torch.cat(offset_predictions, 0), torch.cat(offset_labels, 0)
#     # 这两行代码使用torch.cat函数将多个列表 Concatenating 成一个tensor,目的是将各batch结果连接成一个全量结果供后续用来计算评估指标。
#     coords, instance_labels = torch.cat(coords, 0), torch.cat(instance_labels).cpu().numpy()
#     # 这行代码使用torch.cat将实例标签列表instance_labels中的所有标签张量进行拼接,然后调用.cpu()和.numpy()进行设备与格式转换为numpy
#
#     # split valset into 2 parts along y=0 沿 y=0 将 valset 分割成 2 部分
#     mask_y_greater_zero = coords[:, 1] > 0
#     mask_y_not_greater_zero = torch.logical_not(mask_y_greater_zero)
#     # mask_y_greater_zero掩码标识y坐标大于0的点
#     # mask_y_not_greater_zero掩码标识y坐标不大于或等于0的点
#
#     # pointwise eval y_greater_zero 评估
#     pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
#                           config, epoch, writer, logger, 'full')
#
#     pointwise_eval(semantic_prediction_logits[mask_y_greater_zero], offset_predictions[mask_y_greater_zero], semantic_labels[mask_y_greater_zero], offset_labels[mask_y_greater_zero],
#                           config, epoch, writer, logger, 'y_greater_zero')
#
#     pointwise_eval(semantic_prediction_logits[mask_y_not_greater_zero], offset_predictions[mask_y_not_greater_zero], semantic_labels[mask_y_not_greater_zero], offset_labels[mask_y_not_greater_zero],
#                           config, epoch, writer, logger, 'y_not_greater_zero')
#     # 这几行代码调用pointwise_eval函数对完整数据集和根据y坐标区分的两个子集数据进行评估, 目的是为了:
#     # 比较模型在不同y坐标范围内的预测效果, 找到模型的潜在弱点。
#     # 全量数据评估看模型整体水平, 子集评估看不同区域预测如何。
#     # 使用y坐标掩码选择子集点数据, 实现针对不同子区域的精细化评估。


def validate(config, epoch, model, val_loader, logger, writer):
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels = [], [], [], [], [], []
        for batch in tqdm.tqdm(val_loader):

            # forward
            output = model(batch, return_loss=False)
            offset_prediction, semantic_prediction_logit = output['offset_predictions'], output['semantic_prediction_logits']

            batch['coords'] = batch['coords'] + batch['centers']
            semantic_prediction_logits.append(semantic_prediction_logit[batch['masks_sem']])
            semantic_labels.append(batch['semantic_labels'][batch['masks_sem']])
            offset_predictions.append(offset_prediction[batch['masks_sem']])
            offset_labels.append(batch['offset_labels'][batch['masks_sem']])
            coords.append(batch['coords'][batch['masks_sem']]),
            instance_labels.append(batch['instance_labels'][batch['masks_sem']])

    # concatenate all batches
    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0), torch.cat(semantic_labels, 0)
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0), torch.cat(offset_labels, 0)
    coords, instance_labels = torch.cat(coords, 0), torch.cat(instance_labels).cpu().numpy()

    # evaluate semantic and offset predictions
    pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
                          config, epoch, writer, logger)


def pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, config, epoch,
                   writer, logger):
    # get offset loss
    masks_sem = torch.ones_like(semantic_labels).bool()
    masks_off = semantic_labels == TREE_CLASS_IN_DATASET
    _, offset_loss = point_wise_loss(semantic_prediction_logits.float(), offset_predictions.float(),
                                     masks_sem, masks_off, semantic_labels, offset_labels)

    # get semantic accuracy of classification into tree and non-tree 获得树状和非树状分类的语义精度
    semantic_prediction_logits, semantic_labels = semantic_prediction_logits.cpu().numpy(), semantic_labels.cpu().numpy()
    tree_pred_mask = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)[:,TREE_CLASS_IN_DATASET] >= TREE_CONF_THRESHOLD
    tree_pred_mask = tree_pred_mask.numpy()
    tree_mask = semantic_labels == TREE_CLASS_IN_DATASET
    tp, fp, tn, fn = get_eval_components(tree_pred_mask, tree_mask)
    acc = (tp + tn) / (tp + fp + fn + tn)

    # log and write to tensorboard
    logger.info(
        f'[VALIDATION] [{epoch}/{config.epochs}] val/semantic_acc {acc * 100:.2f}, val/offset_loss {offset_loss.item():.3f}')
    writer.add_scalar(f'val/acc', acc if not np.isnan(acc) else 0, epoch)
    writer.add_scalar(f'val/Offset_MAE', offset_loss, epoch)


def get_eval_components(preds_mask, labels_mask):
    assert len(preds_mask) == len(labels_mask)

    tp = (preds_mask & labels_mask).sum()
    fp = (preds_mask & np.logical_not(labels_mask)).sum()
    fn = (np.logical_not(preds_mask) & labels_mask).sum()
    tn = (np.logical_not(preds_mask) & np.logical_not(labels_mask)).sum()

    return tp, fp, tn, fn


# 这个代码块定义了一个pointwise_eval函数,它实现了含有语义分割和偏移预测任务的模型在验证集上的点wise评估
# def pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, config, epoch, writer, logger, eval_name):
#     _, offset_loss = point_wise_loss(semantic_prediction_logits.float(), offset_predictions[semantic_labels != NON_TREE_CLASS_IN_DATASET].float(),
#                                                     semantic_labels, offset_labels[semantic_labels != NON_TREE_CLASS_IN_DATASET])
#     #semantic_prediction_logits和semantic_labels分别为语义预测结果和标签作为输入。
#     #semantic_prediction_logits表示语义分割或分类模型对输入样本做出的语义预测结果，offset_predictions表示模型预测的位置偏移值。
#     semantic_prediction_logits, semantic_labels = semantic_prediction_logits.cpu().numpy(), semantic_labels.cpu().numpy()
#     # 将语义预测结果和标签从GPU拷贝到CPU
#     # 将它们转换为ndarray数组格式
#     tree_pred_mask = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)[:, TREE_CLASS_IN_DATASET] >= TREE_CONF_THRESHOLD
#     # tree_pred_mask表示语义预测结果中树类预测的高置信mask，就是对于semantic_prediction_logits中，预测点为树概率高的点进行一次简化，将其置换为1和0
#     # TREE_CLASS_IN_DATASET是当前数据集中表示"树"这一语义类的编号索引，TREE_CONF_THRESHOLD是设定的"树"类预测概率的置信度阈值
#     tree_pred_mask = tree_pred_mask.numpy()# tree_pred_mask从pytorch tensor格式转换为numpy数组格式
#     tree_mask = semantic_labels == TREE_CLASS_IN_DATASET
#
#     tp, fp, tn, fn = get_eval_res_components(tree_pred_mask, tree_mask)
#     # get_eval_res_components函数通过一系列数学公式，将预测和标签进行操作，对后续的损失计算进行操作
#     segmentation_res = get_segmentation_metrics(tp, fp, tn, fn)
#     acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate = segmentation_res
#     # 根据预测结果和标签计算语义分割模型各项评价指标
#     # 提取出这些指标值,为后续 model 对比和分析做准备
#     writer.add_scalar(f'{eval_name}/acc', acc if not np.isnan(acc) else 0, epoch)
#     writer.add_scalar(f'{eval_name}/Offset_MAE', offset_loss, epoch)
#
#     logger.info(f'[VALIDATION] [{epoch}/{config.epochs}] {eval_name}/semantic_acc {acc*100:.2f}, {eval_name}/offset_loss {offset_loss.item():.3f}')


def main():
    args, config = get_args_and_cfg()# 读取配置
    logger, writer = init_train_logger(config, args)# 生成日志

    # training objects
    model = TreeLearn(**config.model).cuda()
    # TreeLearn()定义了一个点云分类及分割神经网络模型的类，**config.model表示将这些参数以关键字参数的形式传递给TreeLearn的构造函数，.cuda()将模型复制到GPU加速计算中
    optimizer = build_optimizer(model, config.optimizer)# 根据模型和配置,调用构建函数,从配置中提取信息,构建出一个为该模型量化的参数优化器对象。
    scheduler = build_cosine_scheduler(config.scheduler, optimizer)# 初始化scheduler学习率调整器
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    # 这行代码是创建了一个GradScaler实例并赋值给scaler变量。GradScaler类来自pytorch的torch.cuda.amp模块,它用于混合精度训练。
    train_set = TreeDataset(**config.dataset_train, logger=logger)
    # 这行代码创建了一个TreeDataset实例,并赋值给train_set变量。TreeDataset类我们前面已经介绍过,它是用来处理点云数据集的。config对象读取了配置文件中dataset_train部分的配置参数
    val_set = TreeDataset(**config.dataset_test, logger=logger)
    # train_set以及val_set就是训练和验证数据集实例化
    train_loader = build_dataloader(train_set, training=True, **config.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, **config.dataloader.test)
    # 这两行代码是通过build_dataloader函数构建训练和验证的数据加载器train_loader和val_loader。
    # optionally pretrain or resume 可选择预训或复训
    start_epoch = 1
    if args.resume:# 复训练走这个循环
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif config.pretrain:# 加载预训练权重
        logger.info(f'Load pretrain from {config.pretrain}')
        load_checkpoint(config.pretrain, logger, model)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, config.epochs + 1):
        # 训练过程
        train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer)
        if is_multiple(epoch, config.validation_frequency):# 每过validation_frequency次循环进行验证
            optimizer.zero_grad() # 使用optimizer对象的zero_grad()方法,将参数的梯度清零，验证集上的前向计算不需要梯度,仅进行预测,不进行模型更新。保留梯度可能会影响预测结果。
            logger.info('Validation')
            validate(config, epoch, model, val_loader, logger, writer)
        writer.flush()
        # writer.flush()这行代码的作用是将tensorboard写入缓存中的日志内容进行强制flush(刷新)写入硬盘。循环一次写一次




if __name__ == '__main__':
    main()