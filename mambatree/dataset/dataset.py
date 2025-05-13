import math
import numpy as np
import torch
import os
from torch.utils.data import Dataset


INSTANCE_LABEL_IGNORE_IN_RAW_DATA = -1 # label for unlabeled in raw data 标注原始数据中的未标注
NON_TREE_CLASS_IN_RAW_DATA = 0 # label for non-trees in raw data 原始数据中的非树标签
NON_TREE_CLASS_IN_PYTORCH_DATASET = 1 # semantic label for non-tree in pytorch dataset  pytorch数据集中的非树语义标签
TREE_CLASS_IN_PYTORCH_DATASET = 0 # semantic label for tree in pytorch dataset  pytorch数据集中树的语义标签


class TreeDataset(Dataset):
    def __init__(self,
                 data_root,
                 inner_square_edge_length,
                 training,
                 logger,
                 data_augmentations=None):

        self.data_paths = [os.path.join(data_root, path) for path in os.listdir(data_root)]
        self.inner_square_edge_length = inner_square_edge_length
        self.logger = logger
        self.training = training
        self.data_augmentations = data_augmentations
        mode = 'train' if training else 'test'
        self.logger.info(f'Load {mode} dataset: {len(self.data_paths)} scans')


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, index):
        # load data
        data_path = self.data_paths[index]
        data = np.load(data_path)
        
        # get entries
        xyz = data['points']
        input_feat = data['feat']
        verticality = input_feat[:, -1] # verticality feature computed in last column of data['feat'] 数据['特征']最后一列计算的垂直度特征

        instance_label = data['instance_label']
        semantic_label = np.empty(len(instance_label))
        semantic_label[instance_label == NON_TREE_CLASS_IN_RAW_DATA] = NON_TREE_CLASS_IN_PYTORCH_DATASET
        semantic_label[instance_label != NON_TREE_CLASS_IN_RAW_DATA] = TREE_CLASS_IN_PYTORCH_DATASET

        # get masks for loss calculation 获取用于计算损失的掩码
        mask_inner = self.get_mask_inner(xyz)
        mask_not_ignore = self.get_mask_not_ignore(instance_label)
        mask_off = mask_inner & mask_not_ignore & (semantic_label != NON_TREE_CLASS_IN_PYTORCH_DATASET)
        mask_sem = mask_inner & mask_not_ignore
        
        # get center of chunk (only for test) 获取块中心（仅用于测试）
        if self.training:
            center = np.ones_like(xyz) # dummy value in training (used for stitching tiles back together) 训练中的虚拟值（用于重新拼接瓦片）
        else:
            center = np.ones_like(xyz) * data['center']

        # transform data
        xyz = self.transform_train(xyz) if self.training else self.transform_test(xyz)

        # get offset
        pt_offset_label = self.getOffset(xyz, instance_label, semantic_label, verticality)

        xyz = torch.from_numpy(xyz)
        instance_label = torch.from_numpy(instance_label)
        semantic_label = torch.from_numpy(semantic_label)
        mask_inner = torch.from_numpy(mask_inner)
        mask_off = torch.from_numpy(mask_off)
        mask_sem = torch.from_numpy(mask_sem)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        input_feat = torch.from_numpy(input_feat)
        center = torch.from_numpy(center)

        return xyz, input_feat, instance_label, semantic_label, pt_offset_label, center, mask_inner, mask_off, mask_sem


    def get_mask_not_ignore(self, instance_label):
        mask_ignore = instance_label == INSTANCE_LABEL_IGNORE_IN_RAW_DATA
        mask_not_ignore = np.logical_not(mask_ignore)
        return mask_not_ignore


    def get_mask_inner(self, xyz):
        # mask of inner square
        inf_norm = np.linalg.norm(xyz[:, :-1], ord=np.inf, axis=1)
        mask_inner = inf_norm <= (self.inner_square_edge_length/2)
        return mask_inner


    def point_jitter(self, points, sigma=0.04, clip=0.1):
        jitter = np.clip(sigma * np.random.randn(points.shape[0], 3), -1 * clip, clip)
        points += jitter
        return points


    def transform_train(self, xyz, aug_prob=0.6):

        if self.data_augmentations["point_jitter"] == True:
            if np.random.random() <= aug_prob:
                xyz = self.point_jitter(xyz)
        xyz = self.dataAugment(xyz, data_augmentations=self.data_augmentations, prob=aug_prob)
        return xyz


    def transform_test(self, xyz):
        return xyz

    def getOffset(self, xyz, instance_label, semantic_label, verticality, verticality_thresh=0.6, target_z=3):

        mask_vert = np.squeeze(verticality > verticality_thresh) # 创建垂直性掩膜的意义在于筛选出那些在竖直方向上表现良好的点，通常用于识别树木的结构特征。
        position = np.ones_like(xyz, dtype=np.float32) # 创建一个与输入点相同形状的数组，用于存储每个实例的偏移位置。
        instances = np.unique(instance_label)

        for instance in instances:
            inst_idx = np.where(instance_label == instance)
            first_idx = inst_idx[0][0]

            if semantic_label[first_idx] != NON_TREE_CLASS_IN_PYTORCH_DATASET:
                tree_points = xyz[inst_idx]
                if len(tree_points[:, 2]) > 11:
                    min_z = np.partition(tree_points[:, 2], 10)[10]
                else:
                    min_z = tree_points[:, 2].min()
                # 这里的循环是代码通过对每个实例（每棵树）的 Z 坐标（高度）进行排序，并找到第 10 个最低的值，作为 min_z。
                tree_mask_vert = mask_vert[inst_idx]
                tree_points = tree_points[tree_mask_vert]
                # 根据垂直性掩膜，筛选出符合条件的点。

                z_thresh_lower = min_z + target_z - 0.25
                z_thresh_upper = min_z + target_z + 0.25
                mask_thresh_lower = tree_points[:, 2] >= z_thresh_lower
                mask_thresh_upper = tree_points[:, 2] <= z_thresh_upper
                tree_points_of_interest = tree_points[mask_thresh_lower & mask_thresh_upper]
                if len(tree_points_of_interest) > 0:
                    position_instance = np.mean(tree_points_of_interest, axis=0)
                else:
                    position_instance = np.array([0, 0, 0])

                position[inst_idx] = position_instance
                # target_z是树基
                # 利用点的Z坐标的统计信息
                # if len(tree_points) > 0:
                #     # 计算合适的高度范围，使用标准差
                #     mean_z = np.mean(tree_points[:, 2])
                #     std_z = np.std(tree_points[:, 2])
                #     z_thresh_lower = mean_z - std_z
                #     z_thresh_upper = mean_z + std_z
                #
                #     # 根据垂直性掩膜和新的高度范围筛选点
                #     tree_mask_vert = mask_vert[inst_idx]
                #     tree_points = tree_points[tree_mask_vert]
                #     mask_thresh_lower = tree_points[:, 2] >= z_thresh_lower
                #     mask_thresh_upper = tree_points[:, 2] <= z_thresh_upper
                #
                #     tree_points_of_interest = tree_points[mask_thresh_lower & mask_thresh_upper]
                #     # 从树木点中筛选出位于设定高度范围内的点。
                #     if len(tree_points_of_interest) > 0:
                #         position_instance = np.mean(tree_points_of_interest, axis=0)
                #     else:
                #         position_instance = np.array([0, 0, 0])
                #
                # position[inst_idx] = position_instance
                # # 更新参考位置数组

        pt_offset_label = position - xyz
        return pt_offset_label

    def dataAugment(self, xyz, data_augmentations, prob=0.6):
        jitter = data_augmentations["jitter"]
        flip = data_augmentations["flip"] 
        rot = data_augmentations["rot"]
        scale = data_augmentations["scaled"]
        m = np.eye(3)

        if scale and np.random.rand() < prob:
            scale_xy = np.random.uniform(0.8, 1.2, 2)
            scale_z = np.random.uniform(0.8, 1, 1)
            scale = np.concatenate([scale_xy, scale_z])
            m = m * scale
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        return np.matmul(xyz, m)


    def collate_fn(self, batch):
        xyzs = []
        input_feats = []
        batch_ids = []
        instance_labels = []
        semantic_labels = []
        pt_offset_labels = []
        centers = []
        masks_inner = []
        masks_off = []
        masks_sem = []

        total_points_num = 0
        batch_id = 0


        for data in batch:
            xyz, input_feat, instance_label, semantic_label, pt_offset_label, center, mask_inner, mask_off, mask_sem = data
            total_points_num += len(xyz)

            xyzs.append(xyz)
            input_feats.append(input_feat)
            batch_ids.append(torch.ones(len(xyz))*batch_id)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            masks_inner.append(mask_inner)
            masks_off.append(mask_off)
            masks_sem.append(mask_sem)
            pt_offset_labels.append(pt_offset_label)
            centers.append(center)           
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        xyzs = torch.cat(xyzs, 0).to(torch.float32)
        input_feats = torch.cat(input_feats, 0).to(torch.float32)
        batch_ids = torch.cat(batch_ids, 0).long()
        semantic_labels = torch.cat(semantic_labels, 0).long()
        instance_labels = torch.cat(instance_labels, 0).long()
        masks_inner = torch.cat(masks_inner, 0).bool()
        masks_off = torch.cat(masks_off, 0).bool()
        masks_sem = torch.cat(masks_sem, 0).bool()
        pt_offset_labels = torch.cat(pt_offset_labels, 0).float()
        centers = torch.cat(centers, 0).float()


        return {
            'coords': xyzs,
            'input_feats': input_feats,
            'batch_ids': batch_ids,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'masks_inner': masks_inner,
            'masks_off': masks_off,
            'masks_sem': masks_sem,
            'offset_labels': pt_offset_labels,
            'batch_size': batch_id,
            'centers': centers
        }