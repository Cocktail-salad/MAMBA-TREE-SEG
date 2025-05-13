import functools
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel

from .blocks import MLP, ResidualBlock, UBlock , TensorResidualBlock
from tree_learn.util.train import cuda_cast, point_wise_loss
from .ssm import Tensor_Mamba

LOSS_MULTIPLIER_SEMANTIC = 50  # multiply semantic loss for similar magnitude with offset loss 将类似幅度的语义损失与偏移损失相乘
N_POINTS = None  # only calculate loss for specified number of randomly sampled points; use all points if set to None 只计算指定随机抽样点数的损失；如果设置为 "无"，则使用所有点数


# class MambaLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         # self.norm = nn.LayerNorm(input_dim).to("cuda")
#         self.mamba = Tensor_Mamba(
#             d_model=input_dim,  # Model dimension d_model
#             d_state=d_state,  # SSM state expansion factor
#             d_conv=d_conv,  # Local convolution width
#             expand=expand,  # Block expansion factor
#         ).to("cuda")
#         # self.proj = nn.Linear(input_dim, output_dim)
#         self.skip_scale = nn.Parameter(torch.ones(1))
#
#     def forward(self, x):
#         # 处理浮点类型
#         if x.dtype == torch.float16:
#             x = x.type(torch.float32)
#
#         # 获取输入张量的维度
#         B = x.shape[0]  # Batch size
#         C = x.shape[2]  # 通道数
#         seq_len = x.shape[1]  # 最后一维（长度）
#
#         assert C == self.input_dim, f"Expected input_dim={self.input_dim}, but got C={C}"
#
#         # 处理输入为 (B, C, L) 的情况
#         n_tokens = seq_len
#         img_dims = (seq_len,)  # 空间维度替代
#
#         # 调整 x 的形状
#         x_flat = x.reshape(B, C,n_tokens ).transpose(-1, -2)  # (B, L, C)
#         # x_norm = self.norm(x_flat).to("cuda")  # 归一化
#         x_mamba = self.mamba(x_flat) + self.skip_scale * x_flat  # 加 skip connection
#         # x_mamba = self.norm(x_mamba)
#         # x_mamba = self.proj(x_mamba)
#
#         # 还原形状
#         out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
#         out = out.permute(0, 2, 1)
#         return out
#
#
# def get_mamba_layer(
#     spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
# ):
#     mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
#     if stride != 1:
#         if spatial_dims==2:
#             return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
#         if spatial_dims==3:
#             return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
#     return mamba_layer

class TreeLearn(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 kernel_size=3,
                 dim_coord=3,
                 dim_feat=1,
                 fixed_modules=[],
                 use_feats=True,
                 use_coords=False,
                 spatial_shape=None,
                 max_num_points_per_voxel=3,
                 voxel_size=0.1,
                 **kwargs):
        # channels (整数, 默认32): 定义了网络中的通道数或特征数 num_blocks (整数, 默认7): 指定了网络中块的数量
        # dim_coord (整数, 默认3): 指坐标的维度，通常3D点云数据的坐标是三维的 (X, Y, Z)。dim_feat (整数, 默认1): 表示除坐标外，每个点可能带有的额外特征的维度
        # fixed_modules (列表, 默认为空): 指定不需要在训练过程中更新权重的模块名称列表。
        # use_feats (布尔值, 默认为True): 是否在处理数据时使用点的额外特征。
        # spatial_shape (可选, 默认为None): 如果提供，这是一个定义了数据的空间形状（如网格大小）的参数，用于某些类型的网络操作
        # max_num_points_per_voxel (整数, 默认为3): 每个体素最大的点数，这个参数用于体素化过程中限制每个体素包含点的数量。
        # voxel_size (浮点数, 默认为0.1): 体素的大小，定义了空间网格中单个体素的边长。
        super().__init__()
        self.voxel_size = voxel_size
        self.fixed_modules = fixed_modules
        self.use_feats = use_feats
        self.use_coords = use_coords
        self.spatial_shape = spatial_shape
        self.max_num_points_per_voxel = max_num_points_per_voxel
        # self.mambaRes = TensorResidualBlock(in_channels=4, out_channels=4)
        self.unet_model = Tensor_Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=4, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=2,    # Block expansion factor
                    ).to("cuda")
        # self.norm = nn.LayerNorm(4) # 归一化层
        # self.activation = nn.ReLU() # 激活函数

        # self.mlp_model = Tensor_Mamba(
        #             d_model=2,
        #             d_state=16,
        #             d_conv=4,
        #             expand=2
        #             ).to("cuda")
        # self.mlp_model_off = Tensor_Mamba(
        #             d_model=3,
        #             d_state=16,
        #             d_conv=4,
        #             expand=2
        # ).to("cuda")
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        # 主要作用是定义一个自定义的批归一化函数，这个函数在初始化时就设定了 eps 和 momentum 两个参数

        # backbone 稀疏卷积神经网络
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                dim_coord + dim_feat, channels, kernel_size=kernel_size, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]  # 输出特征图列表
        self.unet = UBlock(block_channels, norm_fn, 2, ResidualBlock, kernel_size, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # head
        self.semantic_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        self.init_weights()

        # weight init
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False
        # 这段代码是在神经网络的构造函数中用来固定特定模块的参数，不让它们在训练过程中更新。这通常用于迁移学习或微调（fine-tuning）网络时，
        # 当你想保持某些已经预训练好的层不变，而专注于训练网络的其他部分。遍历固定模块，获取模块对象，禁止梯度更新

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                # 将批归一化层的权重初始化为 1。这是因为批归一化层的权重（通常表示为 gamma）控制着输入特征的缩放，初始化为 1 意味着在训练开始时不改变输入的缩放。
                nn.init.constant_(m.bias, 0)
                # 将批归一化层的偏置（通常表示为 beta）初始化为 0。这确保了在训练初期，批归一化层不会对特征进行平移。
            elif isinstance(m, MLP):
                m.init_weights()

    # manually set batchnorms in fixed modules to eval mode
    # 将固定模块中的批处理模式手动设置为评估模式
    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss):
        backbone_output, v2p_map = self.forward_backbone(**batch)
        output = self.forward_head(backbone_output, v2p_map)
        if return_loss:
            output = self.get_loss(model_output=output, **batch)

        return output

    @cuda_cast
    def forward_backbone(self, coords, input_feats, batch_ids, batch_size, **kwargs):
        voxel_feats, voxel_coords, v2p_map, spatial_shape = voxelize(torch.hstack([coords, input_feats]), batch_ids,
                                                                     batch_size, self.voxel_size, self.use_coords,
                                                                     self.use_feats,
                                                                     max_num_points_per_voxel=self.max_num_points_per_voxel)
        if self.spatial_shape is not None:
            spatial_shape = torch.tensor(self.spatial_shape, device=voxel_coords.device)
        # voxel_feats = voxel_feats.unsqueeze(0).expand(batch_size,-1,-1)
        # voxel_feats = self.unet_model(voxel_feats)
        # # voxel_feats = self.norm(voxel_feats)
        # # voxel_feats = self.activation(voxel_feats)
        # voxel_feats = voxel_feats.squeeze(0)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        return output, v2p_map

    def forward_head(self, backbone_output, v2p_map):
        output = dict()
        backbone_feats = backbone_output.features[v2p_map]
        output['backbone_feats'] = backbone_feats
        # in_channels = backbone_feats.shape[-1]
        # self.conv1 = get_mamba_layer(
        #     spatial_dims=1, in_channels=in_channels, out_channels=in_channels
        # ).to("cuda")
        # backbone_feats = self.conv1(backbone_feats .unsqueeze(0))
        # backbone_feats = backbone_feats.squeeze(0)
        output['semantic_prediction_logits'] = self.semantic_linear(backbone_feats)
        # semantic_prediction_logits = self.semantic_linear(backbone_feats)
        # semantic_prediction_logits = self.mlp_model(semantic_prediction_logits.unsqueeze(0))
        # output['semantic_prediction_logits'] = semantic_prediction_logits.squeeze(0)
        output['offset_predictions'] = self.offset_linear(backbone_feats)
        # offset_predictions = self.offset_linear(backbone_feats)
        # offset_predictions = self.mlp_model_off(offset_predictions.unsqueeze(0))
        # output['offset_predictions'] = offset_predictions.squeeze(0)
        return output

    # def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_sem, **kwargs):
    #     loss_dict = dict()
    #     semantic_loss, offset_loss = point_wise_loss(model_output['semantic_prediction_logits'][masks_sem].float(),
    #                                                  model_output['offset_predictions'][masks_off].float(),
    #                                                  semantic_labels[masks_sem], offset_labels[masks_off],
    #                                                  n_points=N_POINTS)
    #     loss_dict['semantic_loss'] = semantic_loss * LOSS_MULTIPLIER_SEMANTIC
    #     loss_dict['offset_loss'] = offset_loss
    #
    #     loss = sum(_value for _value in loss_dict.values())
    #     return loss, loss_dict

    @cuda_cast
    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_sem, **kwargs):
        loss_dict = dict()

        # Define variables
        semantic_prediction_logits = model_output['semantic_prediction_logits'].float()
        offset_predictions = model_output['offset_predictions'].float()

        # semantic and offset losses
        semantic_loss, offset_loss = point_wise_loss(
            semantic_prediction_logits,
            offset_predictions,
            masks_sem, masks_off,
            semantic_labels, offset_labels
        )
        loss_dict['semantic_loss'] = semantic_loss * LOSS_MULTIPLIER_SEMANTIC
        loss_dict['offset_loss'] = offset_loss

        # Sum all losses
        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict


def voxelize(feats, batch_ids, batch_size, voxel_size, use_coords, use_feats, max_num_points_per_voxel, epsilon=1):
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0
    for i in range(batch_size):
        feats_one_element = feats[batch_ids == i]
        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon
        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size],
            coors_range_xyz=min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1],
            max_num_voxels=len(feats),
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=feats.device)
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        assert torch.sum(v2p_map == -1) == 0
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]
        voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=feats.device) * i, voxel_coord), dim=1)

        # get mean feature of voxel
        zero_rows = torch.sum(voxel_feat == 0, dim=2) == voxel_feat.shape[2]
        voxel_feat[zero_rows] = float("nan")
        voxel_feat = torch.nanmean(voxel_feat, dim=1)
        if not use_coords:
            voxel_feat[:, :3] = torch.ones_like(voxel_feat[:, :3])
        if not use_feats:
            voxel_feat[:, 3:] = torch.ones_like(voxel_feat[:, 3:])
        voxel_feat = torch.hstack([voxel_feat[:, 3:], voxel_feat[:, :3]])

        voxel_coords.append(voxel_coord)
        voxel_feats.append(voxel_feat)
        v2p_maps.append(v2p_map + total_len_voxels)
        total_len_voxels += len(voxel_coord)
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_feats = torch.cat(voxel_feats, dim=0)
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = voxel_coords.max(dim=0).values + 1

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]

