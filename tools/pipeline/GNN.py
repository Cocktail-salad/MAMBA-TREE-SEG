import torch
import os
import numpy as np
import argparse
import pprint
import shutil
from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import (build_dataloader, get_root_logger, load_checkpoint, ensemble, get_config, get_pointwise_preds, make_labels_consecutive)
from tree_learn.util.data_preparation import rad_filter
import hdbscan
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1

N_JOBS = 10 # number of threads/processes to use for several functions that have multiprocessing/multithreading enabled

def get_instances(coords, offset, semantic_prediction_logits, grouping_cfg, verticality_feat, tree_class_in_dataset, non_trees_label_in_grouping, not_assigned_label_in_grouping, start_num_preds):
    cluster_coords = coords + offset
    cluster_coords = cluster_coords[:, :3]
    train_gnn = True
    semantic_prediction_probs = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)
    tree_mask = semantic_prediction_probs[:, tree_class_in_dataset] >= grouping_cfg.tree_conf_thresh
    vertical_mask = verticality_feat > grouping_cfg.tau_vert
    offset_mask = np.abs(offset[:, 2] - np.mean(cluster_coords[:, 2])) < grouping_cfg.tau_off
    mask_before_filter = tree_mask.numpy() & vertical_mask & offset_mask
    ind_before_filter = np.where(mask_before_filter)[0]
    cluster_coords_tree_before_filter = cluster_coords[ind_before_filter]

    mask_filtering = rad_filter(cluster_coords_tree_before_filter, rad=0.05, npoints_rad=2)
    ind_after_filter = ind_before_filter[mask_filtering]
    cluster_coords_tree_after_filter = cluster_coords[ind_after_filter]
    cluster_coords_tree_after_filter = cluster_coords_tree_after_filter[:, :2]

    predictions = non_trees_label_in_grouping * np.ones(len(cluster_coords))
    predictions[tree_mask] = not_assigned_label_in_grouping

    # 超体素生成步骤
    print("超体素生成步骤")
    # Replace HDBSCAN with GNN clustering
    pred_instances = group_dbscan(cluster_coords_tree_after_filter, grouping_cfg.tau_group, grouping_cfg.tau_min,
                               not_assigned_label_in_grouping, start_num_preds)

    predictions[ind_after_filter] = pred_instances
    return predictions.astype(np.int64)


def group_dbscan(cluster_coords, radius, npoint_thr, not_assigned_label_in_grouping, start_num_preds):
    # 构建图结构，基于点之间的距离
    print("构建图结构，基于点之间的距离")
    edges = kneighbors_graph(cluster_coords, n_neighbors=10, mode='distance', include_self=False)
    graph = csr_matrix(edges)

    # 找到连通组件
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # 初始化聚类标签
    clustering_labels = np.full(cluster_coords.shape[0], not_assigned_label_in_grouping)

    # 有效聚类的筛选
    ind_valid_list = []

    # 对每个连通组件单独运行 HDBSCAN
    for component in range(n_components):
        # 找到当前组件的点索引
        indices = np.where(labels == component)[0]
        if len(indices) > 1:  # 确保组件中有多个点
            # 提取当前组件的坐标
            component_coords = cluster_coords[indices]
            # 使用 HDBSCAN 聚类
            clustering = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, core_dist_n_jobs=-1).fit(component_coords)
            # 更新最终聚类标签
            print("更新最终聚类标签")
            # 筛选有效聚类
            cluster_nums, n_points = np.unique(clustering.labels_, return_counts=True)
            valid_cluster_nums = cluster_nums[(n_points >= npoint_thr) & (cluster_nums != -1)]
            ind_valid = np.isin(clustering.labels_, valid_cluster_nums)

            # 保存有效聚类的索引
            ind_valid_list.append((indices, ind_valid, clustering.labels_))

    for indices, ind_valid, clustering.labels_ in ind_valid_list:
        clustering.labels_[ind_valid] = make_labels_consecutive(clustering.labels_[ind_valid], start_num=start_num_preds)
        clustering.labels_[np.logical_not(ind_valid)] = not_assigned_label_in_grouping
        clustering_labels[indices] = clustering.labels_

    print("11")
    return clustering_labels

def run_treelearn_pipeline(config, config_path=None):
    # # make dirs
    plot_name = os.path.basename(config.forest_path)[:-4]
    base_dir = os.path.dirname(os.path.dirname(config.forest_path))
    documentation_dir = os.path.join(base_dir, 'documentation')
    # unvoxelized_data_dir = os.path.join(base_dir, 'forest')
    # voxelized_data_dir = os.path.join(base_dir, f'forest_voxelized{config.sample_generation.voxel_size}')
    # tiles_dir = os.path.join(base_dir, 'tiles')
    # results_dir = os.path.join(base_dir, 'results')
    # os.makedirs(documentation_dir, exist_ok=True)
    # os.makedirs(unvoxelized_data_dir, exist_ok=True)
    # os.makedirs(voxelized_data_dir, exist_ok=True)
    # os.makedirs(tiles_dir, exist_ok=True)
    # os.makedirs(results_dir, exist_ok=True)
    #
    # # documentation 文件
    logger = get_root_logger(os.path.join(documentation_dir, 'log_pipeline'))
    logger.info(pprint.pformat(config, indent=2))
    if config_path is not None:
        shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))

    # # generate tiles used for inference and specify path to it in dataset config 生成用于推理的瓦片，并在数据集配置中指定其路径
    # config.dataset_test.data_root = os.path.join(tiles_dir, 'npz')
    # if config.tile_generation:
    #     logger.info('#################### generating tiles ####################')
    #     generate_tiles(config.sample_generation, config.forest_path, logger)

    # Make pointwise predictions with pretrained model 使用预训练模型进行定点预测
    logger.info(f'{plot_name}: #################### getting pointwise predictions ####################')
    model = TreeLearn(**config.model).cuda()
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    dataloader = build_dataloader(dataset, training=False, **config.dataloader)
    load_checkpoint(config.pretrain, logger, model)
    pointwise_results = get_pointwise_preds(model, dataloader, config.model, logger)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    del model

    # ensemble predictions from overlapping tiles 重叠瓦片的集合预测
    logger.info(f'{plot_name}: #################### ensembling predictions ####################')
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions,
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data

    # get mask of inner coords if outer points should be removed 如果要删除外层点，则获取内层坐标掩码
    # if config.shape_cfg.outer_remove:
    #     logger.info(f'{plot_name}: #################### remove outer points ####################')
    #     hull_buffer_large = get_hull_buffer(coords, config.shape_cfg.alpha, buffersize=config.shape_cfg.outer_remove)
    #     mask_coords_within_hull_buffer_large = get_coords_within_shape(coords, hull_buffer_large)
    #     masks_inner_coords = np.logical_not(mask_coords_within_hull_buffer_large)

    # get tree detections 获取树木预测
    logger.info(f'{plot_name}: #################### getting predicted instances ####################')
    print("1231231")
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    instance_preds_after_initial_clustering = np.copy(instance_preds)
    return instance_preds_after_initial_clustering

if __name__ == '__main__':
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config', type=str, help='path to config file for pipeline')
    args = parser.parse_args()
    if not args.config:
        args.config = 'configs/pipeline/gnn.yaml'
    config = get_config(args.config)
    run_treelearn_pipeline(config, args.config)