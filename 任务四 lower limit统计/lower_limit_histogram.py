"""
任务四：lower limit统计
功能：读取数据表格中的"lower limit"列，绘制柱状统计图
横坐标：自旋值a（范围-1到1，每0.1一份）
纵坐标：自旋值处于该范围内的源的数量
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 设置路径 ==========
# 获取当前工作目录（项目根目录）
script_dir = os.getcwd()
project_root = script_dir

# 如果当前在子目录中，向上查找项目根目录（包含data文件夹的目录）
while not os.path.exists(os.path.join(project_root, 'data')) and project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)

# 创建输出文件夹
output_dir = os.path.join(project_root, '任务四 lower limit统计')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出文件夹: {output_dir}")

# 数据文件路径
data_file = os.path.join(project_root, 'data', 'lower limit数据.xlsx')

# 检查文件是否存在
if not os.path.exists(data_file):
    print(f"错误：找不到数据文件 {data_file}")
    print("请确保'lower limit数据.xlsx'位于项目根目录的'data'文件夹中")
    print(f"当前项目根目录: {project_root}")
    exit(1)

# 读取数据
print(f"正在读取数据文件: {data_file}")
df = pd.read_excel(data_file, sheet_name='Sheet1')

# 提取源名称和lower limit列
source_col = '源'
spin_col = 'lower limit'

# 检查列是否存在
if source_col not in df.columns:
    possible_source_cols = ['源', 'source', 'Source', '名称']
    for col in possible_source_cols:
        if col in df.columns:
            source_col = col
            break
    else:
        print(f"错误：找不到'源'列，现有列: {df.columns.tolist()}")
        exit(1)

if spin_col not in df.columns:
    possible_spin_cols = ['lower limit', 'lower_limit', 'Lower limit', '下限']
    for col in possible_spin_cols:
        if col in df.columns:
            spin_col = col
            break
    else:
        print(f"错误：找不到'lower limit'列，现有列: {df.columns.tolist()}")
        exit(1)

print(f"使用列: 源='{source_col}', 自旋下限='{spin_col}'")

# 提取有效数据（排除缺失值）
valid_data = df[[source_col, spin_col]].dropna(subset=[spin_col])
valid_data[spin_col] = pd.to_numeric(valid_data[spin_col], errors='coerce')
valid_data = valid_data.dropna(subset=[spin_col])

print(f"有效数据行数: {len(valid_data)}")
print(f"自旋值范围: {valid_data[spin_col].min():.2f} - {valid_data[spin_col].max():.2f}")

# 定义区间（从-1到1，每0.1一份，共20份）
bins = np.arange(-1.0, 1.1, 0.1)  # 区间边界: -1.0, -0.9, ..., 1.0
bin_labels = [f'[{bins[i]:.1f}, {bins[i+1]:.1f})' for i in range(len(bins)-1)]
# 最后一个区间改为闭区间
bin_labels[-1] = f'[{bins[-2]:.1f}, {bins[-1]:.1f}]'

# 统计每个区间的数量
counts, bin_edges = np.histogram(valid_data[spin_col], bins=bins)

# 创建图形，设置更大的尺寸以确保文字显示完整
fig, ax = plt.subplots(figsize=(16, 8))

# 绘制柱状图
bars = ax.bar(range(len(bin_labels)), counts, width=0.8, color='steelblue', 
              edgecolor='black', alpha=0.8, linewidth=1.2)

# 移除柱子上方的数值标签（已删除）

# 设置x轴标签和刻度
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=10)
# 确保x轴标签显示完整
plt.subplots_adjust(bottom=0.15)

# 设置标签和标题
ax.set_xlabel('自旋值 a (lower limit)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('源的数目', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title(f'自旋下限 (lower limit) 统计直方图\n(有效数据: {len(valid_data)}个源)', 
             fontsize=15, fontweight='bold', pad=20)

# 设置y轴范围
y_max = max(counts) * 1.15 if max(counts) > 0 else 5
ax.set_ylim(0, y_max)

# 设置y轴刻度为整数
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 添加网格线
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# 设置边框样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 调整布局，确保所有文字完整显示
plt.tight_layout()

# 保存图片（高清）
output_file = os.path.join(output_dir, 'lower_limit_histogram.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图片已保存: {output_file}")

# 显示图片
plt.show()

# 打印详细统计结果到控制台
print("\n" + "="*70)
print("详细统计结果:")
print("="*70)
print(f"{'区间':<22} {'数量':<8} {'比例':<10}")
print("-"*50)
for label, count in zip(bin_labels, counts):
    if count > 0:
        ratio = count / len(valid_data) * 100
        print(f"{label:<22} {count:<8} {ratio:.1f}%")
print("="*70)

# 导出统计结果到CSV
stats_df = pd.DataFrame({
    '区间': bin_labels,
    '数量': counts,
    '比例(%)': [count/len(valid_data)*100 for count in counts]
})
stats_file = os.path.join(output_dir, 'lower_limit_statistics.csv')
stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"\n统计结果已保存: {stats_file}")

# 导出源数据列表（按自旋值排序）
source_data = valid_data[[source_col, spin_col]].sort_values(by=spin_col)
source_file = os.path.join(output_dir, 'lower_limit_sources.csv')
source_data.to_csv(source_file, index=False, encoding='utf-8-sig')
print(f"源数据列表已保存: {source_file}")