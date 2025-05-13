from collections import OrderedDict
import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn
from .ssm import Tensor_Mamba


class TensorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn=nn.BatchNorm1d, kernel_size=3):
        """
        ResidualBlock 适用于形状 (1, X, C) 的输入，C 为通道数。
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            norm_fn (nn.Module): 规格化函数，例如 BatchNorm1d。
            kernel_size (int): 卷积核大小，默认为 3。
        """
        super(TensorResidualBlock, self).__init__()

        # 跳跃连接分支（如果通道数不一致，使用 1x1 卷积映射）
        if in_channels == out_channels:
            self.i_branch = nn.Identity()
        else:
            self.i_branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_fn(out_channels)
            )

        # 主分支：2 个卷积块
        self.conv_branch = nn.Sequential(
            norm_fn(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            norm_fn(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        )

    def forward(self, x):
        """
        Forward 过程：
        输入 x 形状: (1, X, C)
        """
        # 将输入转换为 (B, C, X) 格式以适配 Conv1d
        x = x.permute(0, 2, 1)  # (1, X, C) -> (1, C, X)

        # 主分支和跳跃连接
        identity = self.i_branch(x)
        out = self.conv_branch(x)

        # 残差连接 + 激活
        out += identity
        out = torch.relu(out)

        # 转回 (1, X, C) 格式
        out = out.permute(0, 2, 1)  # (1, C, X) -> (1, X, C)
        return out


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):

        # # 使用 MAMBA 作为 MLP 的一部分
        # self.mamba = Mamba(d_model=in_channels)

        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

    # def forward(self, x):
    #     # 首先通过 MAMBA 模块处理
    #     x = self.mamba(x)
    #     # 然后通过标准的 MLP 层
    #     return self.layers(x)

class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

# class SparseTransformerModule(nn.Module):
#     def __init__(self, channels, heads=8, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(channels)
#         self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout)
#         self.norm2 = nn.LayerNorm(channels)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(channels, dim_feedforward),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim_feedforward, channels),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         x = x.features
#         x = x.permute(1, 0).unsqueeze(0)  # MultiheadAttention expects input as (L, N, E)
#         x = self.norm1(x)
#         x, _ = self.attention(x, x, x)
#         x = x.squeeze(0).permute(1, 0) + x.features
#         x = self.norm2(x)
#         x = self.feed_forward(x) + x
#         return spconv.SparseConvTensor(x, x.indices, x.spatial_shape, x.batch_size)

class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, kernel_size, indice_key=None):
    # in_channels 和 out_channels 参数分别定义了这个模块输入和输出的“特征通道数” norm_fn 参数传入“规格化”处理
        super().__init__()

        # residual connection either for unchanged number of channels or for increased number of channels 剩余连接可用于保持频道数量不变或增加频道数量
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        # 2 subsequent conv blocks, RF = 3x3x3 each
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=int((kernel_size-1)/2),
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=int((kernel_size-1)/2),
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output

class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, kernel_size, indice_key_id=1):
    # nPlanes参数决定了这个网络每一级的通道数 norm_fn设置了归一化处理 block_reps决定了每一级会重复使用多少个微小模块来提取特征
    # block参数传入具体的残差块结构定义 kernel_size和indice_key_id设置卷积操作的核大小和索引键等细节
        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, kernel_size, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }

        blocks = OrderedDict(blocks)
        # 2 residual blocks with RF 5x5x5 each; RF += 8 ()
        self.blocks = spconv.SparseSequential(blocks)
        #self.transformer = SparseTransformerModule(nPlanes[0])

        # # 正确引入 MAMBA 模块
        # self.mamba = Tensor_Mamba(
        #     d_model=channels,
        #     d_state=64,  # SSM 状态扩展因子
        #     d_conv=4,  # 局部卷积宽度
        #     expand=2  # 块扩展因子
        # )

        # 使用 MAMBA 增强 U-Net 的特征提取能力

        # self.mamba = Mamba(d_model=nPlanes[0])



        if len(nPlanes) > 1:

            # downsample by factor 2; RF *= 2
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, kernel_size, indice_key_id=indice_key_id + 1)

            # used to upsample higher level size to current level size 用于将较高级别尺寸升采样到当前级别尺寸
            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            # used to combine higher level and current level features into feature map of size nPlanes[0] 用于将高层和当前层的特征合并为大小为 nPlanes[0] 的特征图
            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    kernel_size,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        # input = self.mamba(input)
        output = self.blocks(input)
        # # 通过 MAMBA 模块增强特征
        # output_features = self.mamba(output.features)  # 使用 Mamba 模块处理特征
        # output = spconv.SparseConvTensor(output_features, output.indices, output.spatial_shape, output.batch_size)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            # 获取输入张量的通道数
            # out_feats = out_feats.unsqueeze(0)
            # batch_size, seq_len, channel = out_feats.shape  # 从 SparseConvTensor 动态获取通道数
            # # out_feats = self.mamba(out_feats.unsqueeze(0))
            # # out_feats = self.res(out_feats)
            # # 假设 mamba 支持的通道数最大为 max_channels_per_block
            # max_channels_per_block = 16
            # # 切割为小块的逻辑
            # split_outputs = []
            # # 切割通道维度
            # for start_idx in range(0, channel, max_channels_per_block):
            #     end_idx = min(start_idx + max_channels_per_block, channel)
            #     sub_out_feats = out_feats[:, :, start_idx:end_idx]  # 提取通道块
            #     batch_size, seq_len, channels = sub_out_feats.shape
            #     self.mamba = Tensor_Mamba(
            #         d_model=channels,
            #         d_state=64,
            #         d_conv=4,
            #         expand=2
            #     ).to("cuda")
            #     self.res = TensorResidualBlock(in_channels=channels, out_channels=channels).to("cuda")
            #     # 对小块应用 mamba 和残差模块
            #     sub_out_feats = self.mamba(sub_out_feats)
            #     sub_out_feats = self.res(sub_out_feats)
            #     split_outputs.append(sub_out_feats)
            #     del self.mamba, self.res, sub_out_feats
            #     torch.cuda.empty_cache()  # 清理显存
            #
            # # 拼接回原大小
            # out_feats = torch.cat(split_outputs, dim=-1)  # 按通道拼接
            # out_feats = out_feats.squeeze(0)
            # self.conv1 = get_mamba_layer(
            #     spatial_dims=1, in_channels=in_channels, out_channels=in_channels
            # ).to("cuda")
            # out_feats = self.conv1(out_feats.unsqueeze(0))
            # out_feats = out_feats.squeeze(0)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output

# The following calculations do not account for the input convolutions which further increase the receptive field.下面的计算并没有考虑到输入卷积会进一步增加感受野。
# receptive field of ublock with n = len(nPlanes): (1 + block_reps * ((kernel_size - 1) * 2) * 2^(n-1)) + block_reps * ((kernel_size - 1) * 2) * (2^(n-1) - 1) (proof by induction)

# derivation for block_reps = 2 and kernel_size = 3
# derivation: 9 * 2 + 8
# derivation: (9 * 2 + 8) * 2 + 8
# derivation: ((9 * 2 + 8) * 2 + 8) * 2 + 8
# derivation: (((9 * 2 + 8) * 2 + 8) * 2 + 8) * 2 + 8
# etc ...

# for standard configuration of n = 7, it follows that RF = 9 * 2^6 + 8 * (2^6 - 1) = 1080x1080x1080
# for kernel size of 5 instead of 3 it follows for standard configuration that RF = 17 * 2^6 + 16 * (2^6 - 1)
# for kernel size of 3 and n = 2, it follows that RF = 9 * 2^1 + 8 * (2^1 - 1) = 26
