import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_error_type(error_min, error_max):
    """判断误差类型，用于范围值（如 <0.8 或 >0.9）"""
    # 处理范围值（如 <0.8）：error_min 为 NaN，error_max 为 0
    if pd.isna(error_min) and error_max == 0:
        return 'less_than'
    # 处理范围值（如 >0.9）：error_min 为 0，error_max 为 NaN
    if error_min == 0 and pd.isna(error_max):
        return 'greater_than'
    return 'normal'

# 读取数据
print("正在读取数据...")
# 数据文件在根目录的 data 文件夹中
df = pd.read_excel(r'C:\Users\王雨珊\Desktop\blackhole_spin_visualization\data\数据收集 表3 画图用.xlsx', sheet_name='Sheet1')
print(f"成功读取 {len(df)} 行数据")

# 清洗数据：向前填充黑洞名称
df['源'] = df['源'].ffill()
df = df.dropna(subset=['源'])

# 获取指定黑洞（可以修改为你想测试的黑洞名称）
first_source = '4U 1543-475'
source_df = df[df['源'] == first_source].copy()
source_df = source_df.reset_index(drop=True)

print(f"\n黑洞: {first_source}")
print(f"共有 {len(source_df)} 个数据点")

# 定义颜色映射
color_map = {
    'combining': 'green',
    'reflection': 'red',
    'continuum-fitting': 'blue'
}
other_color = 'gray'

# 定义模型顺序（从上到下：combining在最顶部，reflection中间，continuum-fitting最底部）
model_order = ['combining', 'reflection', 'continuum-fitting']

# 辅助函数：从文献来源提取年份和作者（补全完整年份）
def parse_literature(lit_str):
    """解析文献来源，返回 (作者, 完整年份)"""
    if pd.isna(lit_str):
        return ('Unknown', 'Unknown')
    lit_str = str(lit_str).strip()
    
    # 使用非捕获组匹配完整四位年份
    years = re.findall(r'\b(?:19|20)\d{2}\b', lit_str)
    year = years[0] if years else 'Unknown'
    
    # 提取作者（取第一个单词作为第一作者）
    author_part = re.sub(r'\b(?:19|20)\d{2}\b', '', lit_str).strip()
    # 如果有 "et al." 或 "et al"，保留
    if 'et al' in author_part.lower():
        author = author_part.split()[0] + ' et al.'
    else:
        # 取第一个单词作为作者名
        author = author_part.split()[0] + ' et al.'
    
    return (author, year)

def get_lit_year(literature):
    if pd.isna(literature):
        return 9999
    lit_str = str(literature)
    years = re.findall(r'\b(?:19|20)\d{2}\b', lit_str)
    if years:
        return int(years[0])
    return 9999

def get_year_sort_key(year_str):
    """处理爆发年份，提取完整年份"""
    if pd.isna(year_str):
        return 9999
    year_str = str(year_str).strip()
    # 处理 2013+2015 格式，取第一个年份
    if '+' in year_str:
        first_year = year_str.split('+')[0]
        if first_year.isdigit():
            return int(first_year)
    # 处理 2013-2015 格式
    if '-' in year_str and len(year_str) > 5:
        first_year = year_str.split('-')[0]
        if first_year.isdigit():
            return int(first_year)
    # 处理纯数字
    if year_str.isdigit():
        return int(year_str)
    return 9999

def format_burst_year(year_str):
    """格式化爆发年份，补全完整年份显示"""
    if pd.isna(year_str):
        return 'Unknown'
    year_str = str(year_str).strip()
    # 如果已经是完整格式，直接返回
    if '+' in year_str or '-' in year_str:
        return year_str
    # 如果是纯数字年份，补全为4位
    if year_str.isdigit():
        if len(year_str) == 2:
            # 假设20世纪或21世纪，这里简单处理，根据实际数据调整
            if int(year_str) >= 90:
                return '19' + year_str
            else:
                return '20' + year_str
        return year_str
    return year_str

# 添加辅助列
source_df['lit_year'] = source_df['文献来源'].apply(get_lit_year)
source_df['year_sort'] = source_df['爆发时间'].apply(get_year_sort_key)
source_df['burst_year_display'] = source_df['爆发时间'].apply(format_burst_year)
source_df['author'], source_df['lit_year_str'] = zip(*source_df['文献来源'].apply(parse_literature))

# 按模型、爆发年份、文献年份排序
sorted_indices = []
for model in model_order:
    model_df = source_df[source_df['拟合模型'] == model]
    if len(model_df) > 0:
        # 按爆发年份排序（早的在上）
        model_df = model_df.sort_values(['year_sort', 'lit_year'])
        sorted_indices.extend(model_df.index.tolist())

other_models = [m for m in source_df['拟合模型'].unique() if m not in model_order]
for model in other_models:
    model_df = source_df[source_df['拟合模型'] == model]
    if len(model_df) > 0:
        model_df = model_df.sort_values(['year_sort', 'lit_year'])
        sorted_indices.extend(model_df.index.tolist())

source_df = source_df.loc[sorted_indices].reset_index(drop=True)

# 设置均匀的垂直间距
n_points = len(source_df)
y_min = -0.5
y_max = n_points - 0.5

# 创建图形（放大图片尺寸）
fig, ax = plt.subplots(figsize=(14, max(7, n_points * 0.8)))

# 第一遍：绘制数据点和误差棒（包括范围值的箭头）
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    model = row['拟合模型']
    
    # 确定颜色
    color = color_map.get(model, other_color)
    
    # 判断误差类型
    error_type = get_error_type(error_min, error_max)
    
    if error_type == 'less_than':
        # 向左箭头：从 a_star-0.12 到 a_star
        ax.arrow(a_star - 0.12, i, 0.10, 0,
                head_width=0.1, head_length=0.02,
                fc=color, ec=color, alpha=0.8, linewidth=1.0)
        ax.plot(a_star, i, 'o', color=color, markersize=8)
        
    elif error_type == 'greater_than':
        # 向右箭头：从 a_star 到 a_star+0.12
        ax.arrow(a_star, i, 0.10, 0,
                head_width=0.1, head_length=0.02,
                fc=color, ec=color, alpha=0.8, linewidth=1.0)
        ax.plot(a_star, i, 'o', color=color, markersize=8)
        
    else:
        # 正常误差棒
        has_error = False
        if not pd.isna(error_min) and not pd.isna(error_max):
            if (error_min > 0 or error_max > 0) and not np.isinf(error_min) and not np.isinf(error_max):
                has_error = True
        
        if has_error:
            xerr = [[error_min], [error_max]]
            ax.errorbar(a_star, i, xerr=xerr, fmt='o', color=color,
                       capsize=5, markersize=10, elinewidth=2,
                       ecolor=color, markeredgecolor=color, markerfacecolor=color)
        else:
            ax.plot(a_star, i, 'o', color=color, markersize=10)

# 计算 x 轴范围（自动识别是否需要扩展到负数）
all_values = []
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    error_type = get_error_type(error_min, error_max)
    
    all_values.append(a_star)
    
    if error_type == 'normal':
        if not pd.isna(error_min) and not np.isinf(error_min) and error_min > 0:
            all_values.append(a_star - error_min)
        if not pd.isna(error_max) and not np.isinf(error_max) and error_max > 0:
            all_values.append(a_star + error_max)
    elif error_type == 'less_than':
        all_values.append(a_star - 0.15)  # 箭头向左延伸
    elif error_type == 'greater_than':
        all_values.append(a_star + 0.15)  # 箭头向右延伸

min_val = min(all_values)
max_val = max(all_values)

# 判断是否需要扩展到负数
if min_val < 0:
    x_min = -1.05
else:
    x_min = 0

x_max = max(1.05, max_val + 0.1)

# 设置 x 轴
ax.set_xlim(x_min, x_max)

# 计算 x 轴范围用于标签定位
x_range = x_max - x_min

# 添加标注（数值标签和文献标签）
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    model = row['拟合模型']
    author = row['author']
    burst_year = row['burst_year_display']
    lit_year = row['lit_year_str']
    color = color_map.get(row['拟合模型'], other_color)
    
    error_type = get_error_type(error_min, error_max)
    
    # 数值标签（放在点正上方）
    if error_type == 'less_than':
        upper_text = f"<{a_star:.3f}"
    elif error_type == 'greater_than':
        upper_text = f">{a_star:.3f}"
    else:
        if not pd.isna(error_min) and not pd.isna(error_max):
            if (error_min > 0 or error_max > 0) and not np.isinf(error_min) and not np.isinf(error_max):
                upper_text = f"{a_star:.3f}$^{{+{error_max:.3f}}}_{{-{error_min:.3f}}}$"
            else:
                upper_text = f"{a_star:.3f}"
        else:
            upper_text = f"{a_star:.3f}"
    
    ax.text(a_star, i + 0.2, upper_text, fontsize=10, va='bottom', ha='center', 
            color=color, fontweight='bold')
    
    # 文献标签 - 智能定位，避免遮挡
    lower_text = f"{model} {burst_year} {author} ({lit_year})"
    
    # 优先放在右侧
    right_space = x_max - a_star
    left_space = a_star - x_min
    
    # 检查下方是否有数据点
    y_offset = 0
    if i < n_points - 1:
        next_a_star = source_df.iloc[i+1]['自旋值i']
        # 如果自旋值相近，增加垂直偏移避免遮挡
        if abs(a_star - next_a_star) < 0.3:
            y_offset = -0.15
    
    if right_space > x_range * 0.12:
        # 放在右侧
        x_pos = a_star + x_range * 0.06
        ha = 'left'
        va = 'center'
        ax.text(x_pos, i + y_offset, lower_text, fontsize=8, 
                va=va, ha=ha, color=color, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    elif left_space > x_range * 0.12:
        # 放在左侧
        x_pos = a_star - x_range * 0.08
        ha = 'right'
        va = 'center'
        ax.text(x_pos, i + y_offset, lower_text, fontsize=8, 
                va=va, ha=ha, color=color, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    else:
        # 空间不足，放在下方
        x_pos = a_star
        ha = 'center'
        va = 'top'
        ax.text(x_pos, i - 0.45, lower_text, fontsize=8, 
                va=va, ha=ha, color=color, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# 完全隐藏 y 轴（刻度线和刻度标签）
ax.set_yticks([])
ax.set_ylim(y_min, y_max)

# 设置 x 轴标签（增加 labelpad 避免被截断）
ax.set_xlabel(r'$a_*$', fontsize=16, ha='center', fontweight='bold', labelpad=12)

# 添加网格
ax.grid(axis='x', linestyle='--', alpha=0.5, linewidth=0.8)

# 设置标题为源名称
ax.set_title(first_source, fontsize=16, fontweight='bold')

# 设置坐标轴刻度字体大小
ax.tick_params(axis='x', labelsize=12)

# 调整边距，确保 x 轴标签完整显示
plt.subplots_adjust(left=0.2, bottom=0.18)

# 【问题1】保存图片到 output/test/ 文件夹
output_dir = 'output/test'
os.makedirs(output_dir, exist_ok=True)
safe_name = first_source.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
output_path = os.path.join(output_dir, f'spin_comparison_{safe_name}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存到: {output_path}")

# 显示图片
plt.show()