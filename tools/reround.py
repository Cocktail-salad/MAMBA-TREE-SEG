import numpy as np
import pylas

def compute_center_of_mass(points):
    """计算点云的几何中心（质心）"""
    center = np.mean(points, axis=0)  # 计算所有点的平均值
    return center

def rotate_point_cloud_around_center(points, angle, center):
    """围绕给定的中心点进行旋转"""
    # 将角度转换为弧度
    angle_rad = np.radians(angle)

    # 创建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    # 将点云中心移到原点
    points_centered = points - center

    # 旋转点云
    rotated_points_centered = points_centered @ rotation_matrix.T

    # 将点云移回原中心
    rotated_points = rotated_points_centered + center
    return rotated_points

def save_to_las(output_path, points, classifications, tree_ids):
    """将点云保存为 .las 文件"""
    las = pylas.create()
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.add_extra_dim(name="treeID", type="int32", description="Tree Identifier")
    las.treeID = tree_ids
    las.classification = classifications
    las.write(output_path)

def process_and_save_point_cloud(input_path, output_path, input_angle=0):
    """加载点云、计算旋转、保存旋转后的点云"""
    las = pylas.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    classifications = las.classification
    tree_ids = las.treeID

    # 计算点云的中心
    center = compute_center_of_mass(points)

    # 围绕中心旋转点云
    rotated_points = rotate_point_cloud_around_center(points, input_angle, center)

    # 保存旋转后的点云
    save_to_las(output_path, rotated_points, classifications, tree_ids)

# 示例使用
input_file = "../FORinstance_dataset/SCION/plot_87_annotated.las"
output_file = "../FORinstance_dataset/SCION-T/plot_87_annotated_rotated.las"
rotation_angle = 55  # 输入旋转角度（0-360度）
process_and_save_point_cloud(input_file, output_file, input_angle=rotation_angle)
