import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
df = pd.read_excel('data/数据收集 表3.xlsx', sheet_name='Sheet1')
print(f"成功读取 {len(df)} 行数据")

# 清洗数据：向前填充黑洞名称
df['源'] = df['源'].ffill()
df = df.dropna(subset=['源'])

# 获取第一个黑洞（4U 1543-475 作为示例，因为参考图就是这个）
# 如果你想要测试 4U 1630-472，可以改成 '4U 1630-472'
first_source = '4U 1543-475'
source_df = df[df['源'] == first_source].copy()
source_df = source_df.reset_index(drop=True)

print(f"\n黑洞: {first_source}")
print(f"共有 {len(source_df)} 个数据点")
print("\n数据预览:")
print(source_df[['文献来源', '自旋值i', '自旋值i -', '自旋值i +', '拟合模型', '爆发时间']])

# 定义颜色映射
color_map = {
    'combining': 'green',
    'reflection': 'red',
    'continuum-fitting': 'blue'
}
other_color = 'gray'

# 定义模型顺序（从上到下）
model_order = ['combining', 'reflection', 'continuum-fitting']

# 辅助函数：从文献来源提取年份
def get_lit_year(literature):
    if pd.isna(literature):
        return 9999
    lit_str = str(literature)
    import re
    years = re.findall(r'\b(19|20)\d{2}\b', lit_str)
    if years:
        return int(years[0])
    return 9999

# 辅助函数：处理爆发年份（用于排序）
def get_year_sort_key(year_str):
    if pd.isna(year_str):
        return 9999
    year_str = str(year_str).strip()
    if '+' in year_str:
        first_year = year_str.split('+')[0]
        if first_year.isdigit():
            return int(first_year)
    if year_str.isdigit():
        return int(year_str)
    return 9999

# 添加辅助列
source_df['lit_year'] = source_df['文献来源'].apply(get_lit_year)
source_df['year_sort'] = source_df['爆发时间'].apply(get_year_sort_key)

# 按模型、爆发年份、文献年份排序
sorted_indices = []
for model in model_order:
    model_df = source_df[source_df['拟合模型'] == model]
    if len(model_df) > 0:
        # 按爆发年份排序（早的在上）
        model_df = model_df.sort_values(['year_sort', 'lit_year'])
        sorted_indices.extend(model_df.index.tolist())

# 处理其他模型（不在 model_order 中的）
other_models = [m for m in source_df['拟合模型'].unique() if m not in model_order]
for model in other_models:
    model_df = source_df[source_df['拟合模型'] == model]
    if len(model_df) > 0:
        model_df = model_df.sort_values(['year_sort', 'lit_year'])
        sorted_indices.extend(model_df.index.tolist())

source_df = source_df.loc[sorted_indices].reset_index(drop=True)

# 创建图形
fig, ax = plt.subplots(figsize=(10, max(6, len(source_df) * 0.6)))

# 绘制每个数据点
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    model = row['拟合模型']
    lit = row['文献来源']
    burst_year = row['爆发时间']
    
    # 确定颜色
    color = color_map.get(model, other_color)
    
    # 检查是否有有效误差
    has_error = False
    if not pd.isna(error_min) and not pd.isna(error_max):
        if error_min > 0 or error_max > 0:
            has_error = True
    
    # 绘制数据点和误差棒
    if has_error:
        xerr = [[error_min], [error_max]]
        ax.errorbar(a_star, i, xerr=xerr, fmt='o', color=color,
                   capsize=4, markersize=8, elinewidth=1.5,
                   ecolor=color, markeredgecolor=color, markerfacecolor=color)
    else:
        ax.plot(a_star, i, 'o', color=color, markersize=8)

# 设置y轴标签
y_labels = []
for i, row in source_df.iterrows():
    model = row['拟合模型']
    lit = row['文献来源']
    burst_year = row['爆发时间']
    
    # 格式化标签：模型 爆发年份 文献来源
    # 例如: combining 2002 Morningstar & Miller (2014)
    label = f"{model} {burst_year} {lit}"
    y_labels.append(label)

ax.set_yticks(range(len(source_df)))
ax.set_yticklabels(y_labels, fontsize=9)

# 设置x轴
ax.set_xlim(0.0, 1.0)
ax.set_xlabel(r'$a_*$', fontsize=14)
ax.set_title('spin parameters comparison', fontsize=14, fontweight='bold')

# 添加网格
ax.grid(axis='x', linestyle='--', alpha=0.5)

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
           markersize=10, label='combining'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=10, label='reflection'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
           markersize=10, label='continuum-fitting'),
]

ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)

# 添加自旋值文本标注（类似参考图的效果）
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    
    if not pd.isna(error_min) and not pd.isna(error_max):
        # 格式化误差文本，类似 0.30^{+0.10}_{-0.10}
        if error_min > 0 or error_max > 0:
            text = f"{a_star:.3f}$^{{+{error_max:.3f}}}_{{-{error_min:.3f}}}$"
        else:
            text = f"{a_star:.3f}"
    else:
        text = f"{a_star:.3f}"
    
    # 在数据点右侧添加文本
    ax.text(a_star + 0.02, i, text, fontsize=8, va='center')

plt.tight_layout()

# 保存图片
os.makedirs('output', exist_ok=True)
safe_name = first_source.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
output_path = f'output/spin_comparison_{safe_name}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存到: {output_path}")

# 显示图片
plt.show()