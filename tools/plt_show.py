import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 假设数据存储在 evaluation_results 中
file_path = "../plt/Out11.csv"
columns = ["index", "instance_pred", "instance_label", "prec", "rec", "iou"]
data = pd.read_csv(file_path, header=None, names=columns)

# 转换为DataFrame并命名列（根据您的描述）
df = pd.DataFrame(data, columns=["index", "instance_pred", "instance_label", "prec", "rec", "iou"])

# 创建可视化画布
plt.figure(figsize=(12, 6))

# 绘制三条指标曲线
plt.plot(df["instance_label"], df["prec"], marker='o', label="Precision", color='royalblue')
plt.plot(df["instance_label"], df["rec"], marker='s', label="Recall", color='darkorange')
plt.plot(df["instance_label"], df["iou"], marker='^', label="IoU", color='forestgreen')

# # 指定中文字体 宋体
# chinese_font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"  # 适用于 Linux
# chinese_font = fm.FontProperties(fname=chinese_font_path, size=20)
# chinese_font_title = fm.FontProperties(fname=chinese_font_path, size=25)
# 使用 Liberation Sans 字体
plt.rcParams['font.family'] = 'Liberation Sans'
# 添加标签和标题
# plt.title("CULS-Treelearn", pad=10, fontproperties=chinese_font_title)
# plt.xlabel("Instance Label", fontsize=20, fontproperties=chinese_font)
# plt.ylabel("Metric Value", fontsize=20, fontproperties=chinese_font)
# plt.legend(loc='lower left', fontsize=20, prop=chinese_font)
plt.title("Treelearn dataset-Ours", pad=10, fontsize=20)
plt.xlabel("Instance Label", fontsize=20)
plt.ylabel("Metric Value", fontsize=20)
plt.legend(loc='lower left', fontsize=20)
plt.grid(True, alpha=0.3)
# 设置坐标轴刻度的字体大小
plt.xticks(fontsize=20)  # 设置 x 轴刻度的字体大小
plt.yticks(fontsize=20)  # 设置 y 轴刻度的字体大小

# 设置 y 轴范围和间隔
plt.ylim(0.15, 1.05)  # 设置 y 轴范围为 0.9 到 1.0
plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# 设置 x 轴范围和间隔
plt.xlim(-15, 250)  # 设置 y 轴范围为 0.9 到 1.0
plt.xticks([1, 25, 50, 75, 100, 125, 150, 175, 200, 225])

# 保存图表为文件
output_path = "../plt/segmentation_metrics_ARIAL_treelearn_ours_111.png"  # 保存路径
plt.savefig(output_path, dpi=1000, bbox_inches='tight')  # dpi 设置分辨率，bbox_inches 确保保存完整内容
# 显示图表
plt.tight_layout()
plt.show()