"""
任务四：lower limit统计 - 恒星质量黑洞自旋下限镜像分布图
功能：绘制恒星质量黑洞自旋下限的镜像水平条形图
- 所有条形都向左延伸（镜像显示）
- 横坐标：源的数量（统一向左）
- 纵坐标：自旋参数a（-1到1，每0.1一份，-1在最底下，1在最上面）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 设置路径 ==========
script_dir = os.getcwd()
project_root = script_dir

# 向上查找项目根目录（包含data文件夹的目录）
while not os.path.exists(os.path.join(project_root, 'data')) and project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)

# 输出文件夹
output_dir = os.path.join(project_root, '任务四 lower limit统计')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出文件夹: {output_dir}")

# ============================================================
# 读取并处理本研究的lower limit数据
# ============================================================
data_file = os.path.join(project_root, 'data', 'lower limit数据.xlsx')

if not os.path.exists(data_file):
    print(f"错误：找不到数据文件 {data_file}")
    exit(1)

print(f"正在读取数据文件: {data_file}")
df = pd.read_excel(data_file, sheet_name='Sheet1')

# 提取列
source_col = '源'
spin_col = 'lower limit'

# 检查列是否存在
if source_col not in df.columns:
    possible_cols = ['源', 'source', 'Source', '名称']
    for col in possible_cols:
        if col in df.columns:
            source_col = col
            break

if spin_col not in df.columns:
    possible_cols = ['lower limit', 'lower_limit', 'Lower limit', '下限']
    for col in possible_cols:
        if col in df.columns:
            spin_col = col
            break

print(f"使用列: 源='{source_col}', 自旋下限='{spin_col}'")

# 提取有效数据（采用最小自旋下限原则）
valid_data = df[[source_col, spin_col]].dropna(subset=[spin_col])
valid_data[spin_col] = pd.to_numeric(valid_data[spin_col], errors='coerce')
valid_data = valid_data.dropna(subset=[spin_col])

# 对同一源取最小自旋下限
valid_data = valid_data.groupby(source_col)[spin_col].min().reset_index()

print(f"有效源数量: {len(valid_data)}")
print(f"自旋值范围: {valid_data[spin_col].min():.2f} - {valid_data[spin_col].max():.2f}")

# 定义区间（从-1到1，每0.1一份，共20份）
bins = np.arange(-1.0, 1.1, 0.1)
bin_labels = []
for i in range(len(bins)-1):
    if i == len(bins)-2:
        bin_labels.append(f'[{bins[i]:.1f}, {bins[i+1]:.1f}]')
    else:
        bin_labels.append(f'[{bins[i]:.1f}, {bins[i+1]:.1f})')

# 统计每个区间的数量
counts, bin_edges = np.histogram(valid_data[spin_col], bins=bins)

# 所有条形都向左延伸（取负值）
negative_counts = -counts.copy()

# ============================================================
# 绘制镜像水平条形图（所有条形向左，-1在底部，1在顶部）
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))

# 纵坐标位置（从小到大排列，即-1对应位置0，1对应位置19）
y_pos = np.arange(len(bin_labels))

# 使用不同颜色区分正负自旋
colors = ['indianred' if i < 10 else 'steelblue' for i in range(len(bin_labels))]

# 绘制水平条形图（所有条形向左延伸）
bars = ax.barh(y_pos, negative_counts, height=0.7, color=colors, 
               edgecolor='black', alpha=0.8, linewidth=1.0)

# 设置纵坐标（不反转，让-1在底部，1在顶部）
ax.set_yticks(y_pos)
ax.set_yticklabels(bin_labels, fontsize=10)
ax.set_ylim(-0.5, len(bin_labels) - 0.5)
# 注意：移除了 ax.invert_yaxis()，这样区间按原有顺序显示
# 原有顺序：索引0=[-1.0,-0.9) 在底部，索引19=[0.9,1.0] 在顶部

# 设置x轴（所有条形向左，x轴负值）
max_count = max(counts)
x_limit = max_count * 1.15 if max_count > 0 else 5
ax.set_xlim(-x_limit, 0)

# 设置x轴刻度标签（显示绝对值）
x_ticks = np.arange(-int(x_limit), 1, 1)
ax.set_xticks(x_ticks)
ax.set_xticklabels([str(abs(x)) for x in x_ticks], fontsize=10)

# 添加垂直零线（右边界）
ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')

# 设置标签和标题
ax.set_xlabel('Number of Sources', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Spin Parameter a (lower limit)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title(f'Stellar-Mass Black Holes Spin Lower Limit Distribution\n(Mirrored, N = {len(valid_data)} sources)', 
             fontsize=14, fontweight='bold', pad=20)

# 添加网格线
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 调整布局
plt.tight_layout()

# 保存图片
output_file = os.path.join(output_dir, 'stmbh_lower_limit_mirror.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n图片已保存: {output_file}")

# 显示图片
plt.show()

# ============================================================
# 打印详细统计结果
# ============================================================
print("\n" + "="*70)
print("恒星质量黑洞 lower limit 统计结果:")
print("="*70)
print(f"{'区间':<22} {'数量':<8} {'比例':<10}")
print("-"*50)
for label, count in zip(bin_labels, counts):
    if count > 0:
        ratio = count / len(valid_data) * 100
        print(f"{label:<22} {count:<8} {ratio:.1f}%")
print("="*70)
print(f"总计: {len(valid_data)} 个源")
print(f"负自旋源 (a < 0): {(valid_data[spin_col] < 0).sum()} 个")
print(f"正自旋源 (a ≥ 0): {(valid_data[spin_col] >= 0).sum()} 个")

# 导出统计数据
stats_df = pd.DataFrame({
    '区间': bin_labels,
    '数量': counts,
    '比例(%)': [count/len(valid_data)*100 for count in counts]
})
stats_file = os.path.join(output_dir, 'stmbh_lower_limit_statistics.csv')
stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"\n统计结果已保存: {stats_file}")

# 导出源数据列表
source_data = valid_data[[source_col, spin_col]].sort_values(by=spin_col)
source_file = os.path.join(output_dir, 'stmbh_lower_limit_sources.csv')
source_data.to_csv(source_file, index=False, encoding='utf-8-sig')
print(f"源数据列表已保存: {source_file}")

print("\n✓ 完成！所有条形向左镜像的分布图已生成（-1在底部，1在顶部）。")