import laspy
import numpy as np
# 读取生成的 las 文件
las = laspy.read("../data_NIBIO/pipeline/forest/merged_points.las")

# 查看维度
print("Points Shape:", las.x.shape, las.y.shape, las.z.shape)  # 输出 x, y, z 坐标的维度
print("Classifications Shape:", las.classification.shape)  # 输出分类标签的维度
print("TreeID Shape:", las.treeID.shape)  # 输出 TreeID 的维度
# print("TreeID Shape:", las.treeID.shape)  #输出 TreeID 的维度

# 查看数据信息
print("Number of Points:", len(las.x))  # 点云的总数
print("Classification Labels:", np.unique(las.classification))  # 不同分类标签的值

unique_tree_ids = np.unique(las.treeID)
treeid = las.treeID
unique_classfication = np.unique(las.classification)

# 输出唯一的 treeID 数量
print("Number of unique Tree IDsS:", len(unique_tree_ids))
print("Number of unique classification:", len(unique_classfication))
# 输出额外维度的名称
