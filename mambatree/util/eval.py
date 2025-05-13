import numpy as np
import pandas as pd
import scipy
import torch
import laspy
from .data_preparation import load_data


def get_unique_instance_labels_for_splits(coords, instance_labels, unique_instance_labels_to_consider):
    unique_instance_labels_y_greater_zero = []
    unique_instance_labels_y_not_greater_zero = []

    for instance_label in unique_instance_labels_to_consider:
        coords_instance = coords[instance_labels == instance_label]
        z_instance = coords_instance[:, 2]
        z_min_instance = np.min(z_instance)
        coords_instance_smaller_than_min_plus_30cm = coords_instance[z_instance < (z_min_instance + 0.3)]
        position = np.mean(coords_instance_smaller_than_min_plus_30cm, 0)
        if position[1] > 0:
            unique_instance_labels_y_greater_zero.append(instance_label)
        else:
            unique_instance_labels_y_not_greater_zero.append(instance_label)

    return unique_instance_labels_y_greater_zero, unique_instance_labels_y_not_greater_zero


# def get_detections(instance_labels, instance_preds, unique_instance_labels, unique_instance_preds, min_iou_match, non_tree_label):
#     iou_matrix = np.zeros((len(unique_instance_preds), len(unique_instance_labels)))
#     for instance_pred in unique_instance_preds:
#         instance_pred_mask = instance_preds == instance_pred
#         instance_labels_part_of_instance_pred = np.unique(instance_labels[instance_pred_mask])
#         instance_labels_part_of_instance_pred = instance_labels_part_of_instance_pred[instance_labels_part_of_instance_pred != non_tree_label]
#
#         for instance_label_part_of_instance_pred in instance_labels_part_of_instance_pred:
#             instance_label_mask = instance_labels == instance_label_part_of_instance_pred
#
#             tp, fp, tn, fn = get_eval_res_components(instance_pred_mask, instance_label_mask)
#             acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate = get_segmentation_metrics(tp, fp, tn, fn)
#             iou_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = iou
#
#     # get matches and denoise them
#     matched_preds_preliminary, matched_gts_preliminary = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)
#     mask_satisfies_match_condition = iou_matrix[matched_preds_preliminary, matched_gts_preliminary] > min_iou_match # min_iou_match >> 0 (e.g. 0.2) recommended to avoid zero iou matching
#     matched_preds, matched_gts = matched_preds_preliminary[mask_satisfies_match_condition], matched_gts_preliminary[mask_satisfies_match_condition]
#     return matched_gts, matched_preds, iou_matrix

def get_detections(instance_labels, instance_preds, min_iou_match, non_tree_label):
    iou_matrix = np.zeros((np.max(instance_preds) + 1, np.max(instance_labels) + 1))
    precision_matrix = np.zeros((np.max(instance_preds) + 1, np.max(instance_labels) + 1))
    recall_matrix = np.zeros((np.max(instance_preds) + 1, np.max(instance_labels) + 1))

    ################## calculate iou, precision and recall for each instance_pred and instance_label
    for instance_pred in np.arange(np.max(instance_preds) + 1):
        instance_pred_mask = instance_preds == instance_pred
        instance_labels_part_of_instance_pred = np.unique(instance_labels[instance_pred_mask])
        instance_labels_part_of_instance_pred = instance_labels_part_of_instance_pred[
            instance_labels_part_of_instance_pred != non_tree_label]

        for instance_label_part_of_instance_pred in instance_labels_part_of_instance_pred:
            instance_label_mask = instance_labels == instance_label_part_of_instance_pred

            tp, fp, tn, fn = get_eval_components(instance_pred_mask, instance_label_mask)
            prec, rec, iou = get_segmentation_metrics(tp, fp, fn)
            iou_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = iou
            precision_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = prec
            recall_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = rec

    ################## Perform hungarian matching between instance_preds and instance_labels based on iou_matrix and filter matches based on minimum iou
    matched_preds_preliminary, matched_gts_preliminary = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)
    mask_satisfies_match_condition = iou_matrix[matched_preds_preliminary, matched_gts_preliminary] > min_iou_match
    matched_preds, matched_gts = matched_preds_preliminary[mask_satisfies_match_condition], matched_gts_preliminary[
        mask_satisfies_match_condition]
    return matched_gts, matched_preds, iou_matrix, precision_matrix, recall_matrix

def get_eval_components(preds_mask, labels_mask):
    assert len(preds_mask) == len(labels_mask)

    tp = (preds_mask & labels_mask).sum()
    fp = (preds_mask & np.logical_not(labels_mask)).sum()
    fn = (np.logical_not(preds_mask) & labels_mask).sum()
    tn = (np.logical_not(preds_mask) & np.logical_not(labels_mask)).sum()

    return tp, fp, tn, fn

# def get_detection_failures(matched_gts, matched_preds, unique_instance_labels, unique_instance_preds, iou_matrix):
#
#     # get non matched preds and gts
#     assert (iou_matrix[matched_preds, matched_gts] > 0).sum() == len(matched_preds), 'a zero iou correspondence has been matched'
#     non_matched_preds = np.array(list(set(unique_instance_preds) - set(matched_preds))).astype(np.int64)
#     non_matched_gts = np.array(list(set(unique_instance_labels) - set(matched_gts))).astype(np.int64)
#     non_matched_preds_corresponding_gt = []
#
#     # get additional info for non matched preds and gts
#     for non_matched_pred in non_matched_preds:
#         if iou_matrix[non_matched_pred].max() == 0:
#             non_matched_preds_corresponding_gt.append(np.nan)
#         else:
#             non_matched_preds_corresponding_gt.append(iou_matrix[non_matched_pred].argmax())
#     non_matched_preds_corresponding_gt = np.array(non_matched_preds_corresponding_gt)
#
#     non_matched_gts_corresponding_larger_tree = []
#     for non_matched_gt in non_matched_gts:
#         assert np.max(iou_matrix[:, non_matched_gt]) > 0, 'non matched ground truth does not have any corresponding prediction'
#         corresponding_pred = np.argmax(iou_matrix[:, non_matched_gt])
#         non_matched_gts_corresponding_larger_tree.append(np.argmax(iou_matrix[corresponding_pred, :]))
#     non_matched_gts_corresponding_larger_tree = np.array(non_matched_gts_corresponding_larger_tree)
#
#     return non_matched_gts, non_matched_preds, non_matched_preds_corresponding_gt, non_matched_gts_corresponding_larger_tree

def get_detection_failures(matched_gts, matched_preds, unique_instance_labels, unique_instance_preds, iou_matrix,
                           precision_matrix, recall_matrix, min_precision_for_pred, min_recall_for_gt):
    ################## get non matched preds and gts
    assert (iou_matrix[matched_preds, matched_gts] > 0).sum() == len(
        matched_preds), 'a zero iou correspondence has been matched'
    non_matched_preds = np.array(list(set(unique_instance_preds) - set(matched_preds))).astype(np.int64)
    non_matched_gts = np.array(list(set(unique_instance_labels) - set(matched_gts))).astype(np.int64)

    ################## get additional info for non matched preds
    non_matched_preds_corresponding_gt = []
    for non_matched_pred in non_matched_preds:
        # at least min_precision_for_pred percent of a non-matched pred should belong to gts in order to be considered as a commission error.
        # Otherwise it might be a detection of an unlabeled tree or bush, which is not counted as a commission error.
        if precision_matrix[non_matched_pred].sum() < min_precision_for_pred:
            non_matched_preds_corresponding_gt.append(np.nan)
        else:
            non_matched_preds_corresponding_gt.append(precision_matrix[non_matched_pred].argmax())
    non_matched_preds_corresponding_gt = np.array(non_matched_preds_corresponding_gt)

    ################## get additional info for non matched gts
    # If there is a prediction that has a sufficiently high recall (> min_recall_for_gt) with the non_matched_gt, we consider this a case of undersegmentation.
    # In this case, this part of the code identifies the prediction that represents the undersegmented trees (non_matched_gts_corresponding_pred),
    # and the matched ground truth tree that corresponds to this prediction (non_matched_gts_corresponding_other_tree).
    non_matched_gts_corresponding_pred = []
    non_matched_gts_corresponding_other_tree = []
    for non_matched_gt in non_matched_gts:
        if recall_matrix[:,
           non_matched_gt].max() < min_recall_for_gt:  # no undersegmentation error, e.g. in case that the tree is not detected at all
            non_matched_gts_corresponding_other_tree.append(np.nan)
            non_matched_gts_corresponding_pred.append(np.nan)
        else:  # identify corresponding pred and other tree with highest recall > min_recall_for_gt (if it exists)
            corresponding_pred = np.argmax(recall_matrix[:, non_matched_gt])
            non_matched_gts_corresponding_pred.append(corresponding_pred)
            other_gts = np.delete(np.arange(recall_matrix.shape[1]), non_matched_gt)
            argmin_recall_other_gts = recall_matrix[corresponding_pred, other_gts].argmax()

            if recall_matrix[corresponding_pred, other_gts][argmin_recall_other_gts] < min_recall_for_gt:
                non_matched_gts_corresponding_other_tree.append(np.nan)
            else:
                non_matched_gts_corresponding_other_tree.append(other_gts[argmin_recall_other_gts])

    non_matched_gts_corresponding_pred = np.array(non_matched_gts_corresponding_pred)
    non_matched_gts_corresponding_other_tree = np.array(non_matched_gts_corresponding_other_tree)
    return non_matched_gts, non_matched_preds, non_matched_preds_corresponding_gt, non_matched_gts_corresponding_pred, non_matched_gts_corresponding_other_tree


def filter_detection_stuff(array_used_for_filtering, other_array, instance_labels_to_compare_to):
    mask = np.isin(array_used_for_filtering, instance_labels_to_compare_to)
    array_used_for_filtering_subset = array_used_for_filtering[mask]
    other_array_subset = other_array[mask]

    return array_used_for_filtering_subset, other_array_subset


# def evaluate_instance_segmentation(instance_preds, instance_labels, matched_gts, matched_preds, coords,
#                                    xy_partition_relative, xy_partition_absolute, z_partition_relative, z_partition_absolute):
#
#     no_partition = evaluate_no_partition(instance_preds, instance_labels, matched_gts, matched_preds)
#     if xy_partition_relative:
#         xy_relative = evaluate_xy_partition(instance_preds, instance_labels, matched_gts, matched_preds, coords, xy_partition_relative)
#     else:
#         xy_relative = None
#     if xy_partition_absolute:
#         xy_absolute = evaluate_xy_partition(instance_preds, instance_labels, matched_gts, matched_preds, coords, xy_partition_absolute)
#     else:
#         xy_absolute = None
#     if z_partition_relative:
#         z_relative = evaluate_z_partition(instance_preds, instance_labels, matched_gts, matched_preds, coords, z_partition_relative)
#     else:
#         z_relative = None
#     if z_partition_absolute:
#         z_absolute = evaluate_z_partition(instance_preds, instance_labels, matched_gts, matched_preds, coords, z_partition_absolute)
#     else:
#         z_absolute = None
#
#     return no_partition, xy_relative, xy_absolute, z_relative, z_absolute


def evaluate_instance_segmentation(instance_preds, instance_labels, unique_gts, unique_preds, coords,
                                   mapping_to_original_gt_nums, mapping_to_original_pred_nums,
                                   xy_partition, z_partition):
    no_partition = evaluate_no_partition(instance_preds, instance_labels, unique_gts, unique_preds,
                                         mapping_to_original_gt_nums, mapping_to_original_pred_nums)
    if xy_partition:
        xy = evaluate_xy_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, xy_partition,
                                   mapping_to_original_gt_nums, mapping_to_original_pred_nums)
    else:
        xy = None
    if z_partition:
        z = evaluate_z_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, z_partition,
                                 mapping_to_original_gt_nums, mapping_to_original_pred_nums)
    else:
        z = None
    return no_partition, xy, z


# def evaluate_no_partition(instance_preds, instance_labels, matched_gts, matched_preds):
#     val_res = dict()
#     val_res['instance_pred'] = []
#     val_res['instance_label'] = []
#     val_res['prec'] = []
#     val_res['rec'] = []
#     val_res['f1'] = []
#     val_res['iou'] = []
#     val_res['fdr'] = []
#     val_res['fnr'] = []
#     val_res['one_minus_f1'] = []
#     val_res['fp_error_rate'] = []
#     val_res['fn_error_rate'] = []
#     val_res['error_rate'] = []
#
#     # get results
#     for instance_pred, instance_label in zip(matched_preds, matched_gts):
#         val_res['instance_pred'].append(instance_pred)
#         val_res['instance_label'].append(instance_label)
#
#         ind_pred_positive = instance_preds == instance_pred
#         ind_positive = instance_labels == instance_label
#
#         tp, fp, tn, fn = get_eval_res_components(ind_pred_positive, ind_positive)
#         acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate = get_segmentation_metrics(tp, fp, tn, fn)
#         val_res['prec'].append(prec)
#         val_res['rec'].append(rec)
#         val_res['f1'].append(f1)
#         val_res['fdr'].append(fdr)
#         val_res['fnr'].append(fnr)
#         val_res['one_minus_f1'].append(one_minus_f1)
#         val_res['iou'].append(iou)
#         val_res['fp_error_rate'].append(fp_error_rate)
#         val_res['fn_error_rate'].append(fn_error_rate)
#         val_res['error_rate'].append(error_rate)
#
#     val_res_df = pd.DataFrame.from_dict(val_res)
#     return val_res_df


################################## Instance segmentation evaluation for complete trees (no partition)
def evaluate_no_partition(instance_preds, instance_labels, unique_gts, unique_preds, mapping_to_original_gt_nums,
                          mapping_to_original_pred_nums):
    val_res = dict()
    val_res['instance_pred'] = []
    val_res['instance_label'] = []
    val_res['prec'] = []
    val_res['rec'] = []
    val_res['iou'] = []

    # get results
    for instance_pred, instance_label in zip(unique_preds, unique_gts):
        val_res['instance_pred'].append(mapping_to_original_pred_nums[instance_pred])
        val_res['instance_label'].append(mapping_to_original_gt_nums[instance_label])

        ind_pred_positive = instance_preds == instance_pred
        ind_positive = instance_labels == instance_label

        tp, fp, tn, fn = get_eval_components(ind_pred_positive, ind_positive)
        prec, rec, iou = get_segmentation_metrics(tp, fp, fn)
        val_res['prec'].append(prec)
        val_res['rec'].append(rec)
        val_res['iou'].append(iou)

    val_res_df = pd.DataFrame.from_dict(val_res)
    return val_res_df


# def evaluate_xy_partition(instance_preds, instance_labels, matched_gts, matched_preds, coords, intvls):
#     val_res = dict()
#     val_res['instance_pred'] = []
#     val_res['instance_label'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'f1_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fdr_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fnr_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'one_minus_f1_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fp_error_rate_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fn_error_rate_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'error_rate_intvl{intvls[i]}_{intvls[i+1]}'] = []
#
#     # get results
#     for instance_pred, instance_label in zip(matched_preds, matched_gts):
#         val_res['instance_pred'].append(instance_pred)
#         val_res['instance_label'].append(instance_label)
#
#         ind_pred_positive = instance_preds == instance_pred
#         ind_positive = instance_labels == instance_label
#
#         # calculate tree position and center xy-coordinates according to it
#         tree_coords = coords[ind_positive]
#         min_z = np.min(tree_coords[:, 2])
#         z_thresh = min_z + 0.30
#         lowest_points = tree_coords[tree_coords[:, 2] <= z_thresh]
#         position = np.mean(lowest_points, axis=0)
#         position = position[:2]
#         coords_centered = coords[:, :2] - position
#
#         # distance to seedpoint either relative or absolute
#         if intvls[-1] > 1:
#             distance_from_seedpoint = np.linalg.norm(coords_centered, ord=None, axis=1)
#         else:
#             distance_from_seedpoint = np.linalg.norm(coords_centered, ord=None, axis=1)
#             distance_from_seedpoint_tree = distance_from_seedpoint[ind_positive]
#             sorted_inds = distance_from_seedpoint_tree.argsort()
#             regularized_max = distance_from_seedpoint_tree[sorted_inds[-5]]
#             distance_from_seedpoint = distance_from_seedpoint / regularized_max
#
#         # get precision, recall and F1-score radial
#         for i in range(len(intvls) - 1):
#             ind_gte_min = distance_from_seedpoint >= intvls[i]
#             ind_lt_max = distance_from_seedpoint < intvls[i+1]
#             ind_orbit = ind_gte_min & ind_lt_max
#             ind_positive_orbit = ind_positive[ind_orbit]
#             ind_pred_positive_orbit = ind_pred_positive[ind_orbit]
#             tp, fp, tn, fn = get_eval_res_components(ind_pred_positive_orbit, ind_positive_orbit)
#             acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate = get_segmentation_metrics(tp, fp, tn, fn)
#
#             # append scores to validation results
#             val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'].append(prec)
#             val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'].append(rec)
#             val_res[f'f1_intvl{intvls[i]}_{intvls[i+1]}'].append(f1)
#             val_res[f'fdr_intvl{intvls[i]}_{intvls[i+1]}'].append(fdr)
#             val_res[f'fnr_intvl{intvls[i]}_{intvls[i+1]}'].append(fnr)
#             val_res[f'one_minus_f1_intvl{intvls[i]}_{intvls[i+1]}'].append(one_minus_f1)
#             val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'].append(iou)
#             val_res[f'fp_error_rate_intvl{intvls[i]}_{intvls[i+1]}'].append(fp_error_rate)
#             val_res[f'fn_error_rate_intvl{intvls[i]}_{intvls[i+1]}'].append(fn_error_rate)
#             val_res[f'error_rate_intvl{intvls[i]}_{intvls[i+1]}'].append(error_rate)
#
#     val_res_df = pd.DataFrame.from_dict(val_res)
#     return val_res_df


################################## Instance segmentation evaluation for the xy partition of trees
def evaluate_xy_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, intvls,
                          mapping_to_original_gt_nums, mapping_to_original_pred_nums):
    val_res = dict()
    val_res['instance_pred'] = []
    val_res['instance_label'] = []
    for i in range(len(intvls) - 1):
        val_res[f'prec_intvl{intvls[i]}_{intvls[i + 1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'rec_intvl{intvls[i]}_{intvls[i + 1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'iou_intvl{intvls[i]}_{intvls[i + 1]}'] = []

    # get results
    for instance_pred, instance_label in zip(unique_preds, unique_gts):
        val_res['instance_pred'].append(mapping_to_original_pred_nums[instance_pred])
        val_res['instance_label'].append(mapping_to_original_gt_nums[instance_label])

        ind_pred_positive = instance_preds == instance_pred
        ind_positive = instance_labels == instance_label

        # calculate tree position and center xy-coordinates according to it
        tree_coords = coords[ind_positive]
        min_z = np.min(tree_coords[:, 2])
        z_thresh = min_z + 0.30
        lowest_points = tree_coords[tree_coords[:, 2] <= z_thresh]
        position = np.mean(lowest_points, axis=0)
        position = position[:2]
        coords_centered = coords[:, :2] - position

        # relative distance to seedpoint (0=seedpoint, 1=most distant point)
        distance_from_seedpoint = np.linalg.norm(coords_centered, ord=None, axis=1)
        distance_from_seedpoint_tree = distance_from_seedpoint[ind_positive]
        sorted_inds = distance_from_seedpoint_tree.argsort()
        # regularized_max = distance_from_seedpoint_tree[sorted_inds[-5]]
        # 检查 sorted_inds 的大小，如果小于5，使用最大的距离作为 regularized_max
        if len(sorted_inds) > 1:  # 如果有多个点
            regularized_max = distance_from_seedpoint_tree[sorted_inds[-5]]
            distance_from_seedpoint = distance_from_seedpoint / regularized_max
        else:
            # 跳过或特殊处理这个实例
            continue
        distance_from_seedpoint = distance_from_seedpoint / regularized_max

        # get precision, recall and F1-score radial
        for i in range(len(intvls) - 1):
            ind_gte_min = distance_from_seedpoint >= intvls[i]
            ind_lt_max = distance_from_seedpoint < intvls[i + 1]
            ind_orbit = ind_gte_min & ind_lt_max
            ind_positive_orbit = ind_positive[ind_orbit]
            ind_pred_positive_orbit = ind_pred_positive[ind_orbit]
            tp, fp, tn, fn = get_eval_components(ind_pred_positive_orbit, ind_positive_orbit)
            prec, rec, iou = get_segmentation_metrics(tp, fp, fn)

            # append scores to validation results
            val_res[f'prec_intvl{intvls[i]}_{intvls[i + 1]}'].append(prec)
            val_res[f'rec_intvl{intvls[i]}_{intvls[i + 1]}'].append(rec)
            val_res[f'iou_intvl{intvls[i]}_{intvls[i + 1]}'].append(iou)
    # 确保所有数组长度一致
    max_len = max(len(lst) for lst in val_res.values())
    for key in val_res:
        while len(val_res[key]) < max_len:
            val_res[key].append(np.nan)  # 或者使用 np.nan

    val_res_df = pd.DataFrame.from_dict(val_res)
    return val_res_df


# def evaluate_z_partition(instance_preds, instance_labels, matched_gts, matched_preds, coords, intvls):
#     val_res = dict()
#     val_res['instance_pred'] = []
#     val_res['instance_label'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'f1_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fdr_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fnr_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'one_minus_f1_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fp_error_rate_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'fn_error_rate_intvl{intvls[i]}_{intvls[i+1]}'] = []
#     for i in range(len(intvls) - 1):
#         val_res[f'error_rate_intvl{intvls[i]}_{intvls[i+1]}'] = []
#
#     # get results
#     for instance_pred, instance_label in zip(matched_preds, matched_gts):
#         val_res['instance_pred'].append(instance_pred)
#         val_res['instance_label'].append(instance_label)
#
#         ind_pred_positive = instance_preds == instance_pred
#         ind_positive = instance_labels == instance_label
#         tree_coords = coords[ind_positive]
#
#         # get non-normalized relative or absolute version of z value of coords_gt
#         if intvls[-1] > 1:
#             coords_temp = coords - np.array([0, 0, np.min(tree_coords[:, -1])])
#             coords_temp_z = coords_temp[:, -1]
#         else:
#             coords_temp = coords - np.array([0, 0, np.min(tree_coords[:, -1])])
#             coords_temp_z = coords_temp[:, -1]
#
#             sorted_inds = tree_coords[:, 2].argsort()
#             regularized_max = tree_coords[:, 2][sorted_inds[-5]]
#             coords_temp_z = coords_temp_z / (regularized_max - np.min(tree_coords[:, -1]))
#
#         # get precision, recall and F1-score vertical
#         for i in range(len(intvls) - 1):
#             ind_gte_min = coords_temp_z >= intvls[i]
#             ind_lt_max = coords_temp_z < intvls[i+1]
#             ind_layer = ind_gte_min & ind_lt_max
#             ind_positive_layer = ind_positive[ind_layer]
#             ind_pred_positive_layer = ind_pred_positive[ind_layer]
#             tp, fp, tn, fn = get_eval_res_components(ind_pred_positive_layer, ind_positive_layer)
#             acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate = get_segmentation_metrics(tp, fp, tn, fn)
#
#             # append scores to validation results
#             val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'].append(prec)
#             val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'].append(rec)
#             val_res[f'f1_intvl{intvls[i]}_{intvls[i+1]}'].append(f1)
#             val_res[f'fdr_intvl{intvls[i]}_{intvls[i+1]}'].append(fdr)
#             val_res[f'fnr_intvl{intvls[i]}_{intvls[i+1]}'].append(fnr)
#             val_res[f'one_minus_f1_intvl{intvls[i]}_{intvls[i+1]}'].append(one_minus_f1)
#             val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'].append(iou)
#             val_res[f'fp_error_rate_intvl{intvls[i]}_{intvls[i+1]}'].append(fp_error_rate)
#             val_res[f'fn_error_rate_intvl{intvls[i]}_{intvls[i+1]}'].append(fn_error_rate)
#             val_res[f'error_rate_intvl{intvls[i]}_{intvls[i+1]}'].append(error_rate)
#
#     val_res_df = pd.DataFrame.from_dict(val_res)
#     return val_res_df


################################## Instance segmentation evaluation for the z partition of trees
def evaluate_z_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, intvls, mapping_to_original_gt_nums, mapping_to_original_pred_nums):
    val_res = dict()
    val_res['instance_pred'] = []
    val_res['instance_label'] = []
    for i in range(len(intvls) - 1):
        val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'] = []

    # get results
    for instance_pred, instance_label in zip(unique_preds, unique_gts):
        val_res['instance_pred'].append(mapping_to_original_pred_nums[instance_pred])
        val_res['instance_label'].append(mapping_to_original_gt_nums[instance_label])

        ind_pred_positive = instance_preds == instance_pred
        ind_positive = instance_labels == instance_label
        tree_coords = coords[ind_positive]

        if len(tree_coords) < 5:
            continue  # Skip if there are fewer than 5 points

        # get relative distance to lowest point (0=lowest point, 1=highest point)
        coords_temp = coords - np.array([0, 0, np.min(tree_coords[:, -1])])
        coords_temp_z = coords_temp[:, -1]

        sorted_inds = tree_coords[:, 2].argsort()
        # regularized_max = tree_coords[:, 2][sorted_inds[-5]]
        # coords_temp_z = coords_temp_z / (regularized_max - np.min(tree_coords[:, -1]))
        # Ensure there are enough points to access the -5th element
        if len(sorted_inds) >= 5:
            regularized_max = tree_coords[:, 2][sorted_inds[-5]]
            coords_temp_z = coords_temp_z / (regularized_max - np.min(tree_coords[:, -1]))
        else:
            # If there are fewer than 5 points, use the maximum value of the sorted coordinates as the regularized max
            regularized_max = np.max(tree_coords[:, 2])
            coords_temp_z = coords_temp_z / (regularized_max - np.min(tree_coords[:, -1]))

        # get precision, recall and F1-score vertical
        for i in range(len(intvls) - 1):
            ind_gte_min = coords_temp_z >= intvls[i]
            ind_lt_max = coords_temp_z < intvls[i+1]
            ind_layer = ind_gte_min & ind_lt_max
            ind_positive_layer = ind_positive[ind_layer]
            ind_pred_positive_layer = ind_pred_positive[ind_layer]
            tp, fp, tn, fn = get_eval_components(ind_pred_positive_layer, ind_positive_layer)
            prec, rec, iou = get_segmentation_metrics(tp, fp, fn)

            # append scores to validation results
            val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'].append(prec)
            val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'].append(rec)
            val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'].append(iou)

    max_len = max(len(lst) for lst in val_res.values())
    for key in val_res:
        while len(val_res[key]) < max_len:
            val_res[key].append(np.nan)  # 或者使用 np.nan

    val_res_df = pd.DataFrame.from_dict(val_res)
    return val_res_df


def get_eval_res_components(preds_mask, labels_mask):
    assert len(preds_mask) == len(labels_mask)

    tp = (preds_mask & labels_mask).sum()
    fp = (preds_mask & np.logical_not(labels_mask)).sum()
    fn = (np.logical_not(preds_mask) & labels_mask).sum()
    tn = (np.logical_not(preds_mask) & np.logical_not(labels_mask)).sum()

    return tp, fp, tn, fn


# def get_segmentation_metrics(tp, fp, tn, fn):
#     assert not (np.isnan(tp) or np.isnan(fp) or np.isnan(fn)), 'one of the inputs is nan'
#     # the following nan cases specified by if statements can happen for partitions
#
#     # accuracy
#     acc = (tp + tn) / (tp + fp + fn + tn)
#
#     # iou
#     if tp == 0 and fp == 0 and fn == 0:
#         iou = np.nan
#         fp_error_rate = np.nan
#         fn_error_rate = np.nan
#     else:
#         iou = tp / (tp + fp + fn)
#         fp_error_rate = fp / (tp + fp + fn)
#         fn_error_rate = fn / (tp + fp + fn)
#
#     # rec
#     if tp + fn == 0:
#         rec = np.nan
#     else:
#         rec = tp / (tp + fn)
#
#     # prec
#     if tp + fp == 0:
#         prec = np.nan
#     else:
#         prec = tp / (tp + fp)
#
#     # f1
#     if not np.isnan(prec) and not np.isnan(rec) and not (prec == 0 and rec == 0):
#         f1 = 2 * (prec * rec) / (prec + rec)
#     else:
#         f1 = np.nan
#
#
#     # define other metrics
#     fdr = 1 - prec
#     fnr = 1 - rec
#     one_minus_f1 = 1 - f1
#     error_rate = 1 - iou
#
#     return acc, prec, rec, f1, fdr, fnr, one_minus_f1, iou, fp_error_rate, fn_error_rate, error_rate


################################## Calculate segmentation metrics
def get_segmentation_metrics(tp, fp, fn):
    assert not (np.isnan(tp) or np.isnan(fp) or np.isnan(fn)), 'one of the inputs is nan'
    # iou
    if tp == 0 and fp == 0 and fn == 0:
        iou = np.nan
    else:
        iou = tp / (tp + fp + fn)
    # rec
    if tp + fn == 0:
        rec = np.nan
    else:
        rec = tp / (tp + fn)
    # prec
    if tp + fp == 0:
        prec = np.nan
    else:
        prec = tp / (tp + fp)

    return prec, rec, iou


#############################################################################################
###################### FUNCTIONS FOR EVALUATION NOTEBOOK#####################################
#############################################################################################



def load_results(instance_evaluation_path, benchmark_forest_path, unlabeled_class_in_instance_labels):
    instance_evaluation = torch.load(instance_evaluation_path)

    benchmark_forest = load_data(benchmark_forest_path)
    outpoints_trees = laspy.read(benchmark_forest_path).treeID
    benchmark_forest[:, 3][outpoints_trees != 0] = outpoints_trees[outpoints_trees != 0]
    benchmark_forest = benchmark_forest[benchmark_forest[:, -1] != unlabeled_class_in_instance_labels]
    instance_labels = benchmark_forest[:, 3]
    instance_preds = instance_evaluation['instance_preds_propagated_to_benchmark_pointcloud']

    return instance_evaluation, instance_labels, instance_preds


def get_qualitative_assessment(instance_evaluation, verbose=True):
    if verbose:
        print(f"Number of matched predictions: {len(instance_evaluation['detection_results']['matched_preds'])}")
        print(f"non_matched_predictions: {instance_evaluation['detection_results']['non_matched_preds']}; non_matched_predictions corresponding gt: {instance_evaluation['detection_results']['non_matched_preds_corresponding_gt']}")
        print(f"non_matched_gt: {instance_evaluation['detection_results']['non_matched_gts']}; non_matched_gts_corresponding_larger_tree: {instance_evaluation['detection_results']['non_matched_gts_corresponding_larger_tree']}")
        print(f"non_matched_preds_where_corresponding_gt_is_nan: {instance_evaluation['detection_results']['non_matched_preds_where_corresponding_gt_is_nan']}")

    n_fp = len(instance_evaluation['detection_results']['non_matched_preds'])
    n_fn = len(instance_evaluation['detection_results']['non_matched_gts'])
    return n_fp, n_fn


def get_semantic_assessment(instance_labels, instance_preds, non_tree_class_in_instance_preds, non_tree_class_in_instance_labels):
    tree_preds = instance_preds != non_tree_class_in_instance_preds
    tree_labels = instance_labels != non_tree_class_in_instance_labels
    non_tree_preds = np.logical_not(tree_preds)
    non_tree_labels = np.logical_not(tree_labels)

    tp_tree, fp_tree, tn_tree, fn_tree = get_eval_res_components(tree_preds, tree_labels)
    tp_non_tree, fp_non_tree, tn_non_tree, fn_non_tree = get_eval_res_components(non_tree_preds, non_tree_labels)

    acc_tree, prec_tree, rec_tree, f1_tree, fdr_tree, fnr_tree, one_minus_f1_tree, iou_tree, fp_error_rate_tree, fn_error_rate_tree, error_rate_tree = get_segmentation_metrics(tp_tree, fp_tree, tn_tree, fn_tree)
    acc_non_tree, prec_non_tree, rec_non_tree, f1_non_tree, fdr_non_tree, fnr_non_tree, one_minus_f1_non_tree, iou_non_tree, fp_error_rate_non_tree, fn_error_rate_non_tree, error_rate_non_tree = get_segmentation_metrics(tp_non_tree, fp_non_tree, tn_non_tree, fn_non_tree)
    # if you want, you can also return measures for non_tree and trees separately
    
    print(f'accuracy: {acc_tree}')


def get_instance_assessment(instance_evaluation, n_instances_to_validate=54):
    assert (len(instance_evaluation['segmentation_results']['no_partition']) + len(instance_evaluation['detection_results']['non_matched_gts']) * 2) == n_instances_to_validate
    print(instance_evaluation['segmentation_results']['no_partition'][['prec', 'rec', 'f1', 'iou', 'fp_error_rate', 'fn_error_rate', 'error_rate']].mean(0))
# def get_instance_assessment(instance_evaluation, n_instances_to_validate=1):
#     # 获取“未分割的实例”和“未匹配的检测结果”的数量
#     no_partition_count = len(instance_evaluation['segmentation_results']['no_partition'])
#     unmatched_detections_count = len(instance_evaluation['detection_results']['non_matched_gts'])
#
#     # 计算实例总数
#     total_instances = no_partition_count + unmatched_detections_count * 2
#
#     # 如果实例总数不匹配，打印详细信息
#     assert total_instances == n_instances_to_validate, (
#         f"总实例数不匹配！\n"
#         f"预期值: {n_instances_to_validate}\n"
#         f"计算值: {total_instances}\n"
#         f"未分割实例数: {no_partition_count}\n"
#         f"未匹配检测数的两倍: {unmatched_detections_count * 2}"
#     )

    # 计算和打印指定指标的均值
    metrics = ['prec', 'rec', 'f1', 'iou', 'fp_error_rate', 'fn_error_rate', 'error_rate']
    mean_metrics = instance_evaluation['segmentation_results']['no_partition'][metrics].mean(0)
    print(mean_metrics)