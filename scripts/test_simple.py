import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

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

# 获取指定黑洞
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

# 绘制每个数据点
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    model = row['拟合模型']
    
    # 确定颜色
    color = color_map.get(model, other_color)
    
    # 检查是否有有效误差
    has_error = False
    if not pd.isna(error_min) and not pd.isna(error_max):
        if (error_min > 0 or error_max > 0) and not np.isinf(error_min) and not np.isinf(error_max):
            has_error = True
    
    # 绘制数据点和误差棒
    if has_error:
        xerr = [[error_min], [error_max]]
        ax.errorbar(a_star, i, xerr=xerr, fmt='o', color=color,
                   capsize=5, markersize=10, elinewidth=2,
                   ecolor=color, markeredgecolor=color, markerfacecolor=color)
    else:
        ax.plot(a_star, i, 'o', color=color, markersize=10)

# 添加标注（放大字体）
for i, row in source_df.iterrows():
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    model = row['拟合模型']
    author = row['author']
    burst_year = row['burst_year_display']
    lit_year = row['lit_year_str']
    color = color_map.get(row['拟合模型'], other_color)
    
    # 上方标签：自旋值及其误差（放在点正上方，垂直偏移量减小为0.2）
    if not pd.isna(error_min) and not pd.isna(error_max):
        if (error_min > 0 or error_max > 0) and not np.isinf(error_min) and not np.isinf(error_max):
            upper_text = f"{a_star:.3f}$^{{+{error_max:.3f}}}_{{-{error_min:.3f}}}$"
        else:
            upper_text = f"{a_star:.3f}"
    else:
        upper_text = f"{a_star:.3f}"
    
    # 左侧标签：模型 爆发年份 作者 (年份)（放在点左侧，x坐标右移为-0.05）
    lower_text = f"{model} {burst_year} {author} ({lit_year})"
    
    # 数值标签放在点正上方（垂直偏移0.2）
    ax.text(a_star, i + 0.2, upper_text, fontsize=11, va='bottom', ha='center', 
            color=color, fontweight='bold')
    
    # 文献标签放在点左侧（x坐标改为-0.05，向右移动）
    ax.text(-0.05, i, lower_text, fontsize=10, va='center', ha='right', 
            color=color, alpha=0.9)

# 设置y轴：隐藏刻度，只保留均匀间距
ax.set_yticks(range(n_points))
ax.set_yticklabels([])
ax.set_ylim(y_min, y_max)

# 设置x轴（范围0-1）
ax.set_xlim(0.0, 1.0)
ax.set_xlabel(r'$a_*$', fontsize=16, ha='center', fontweight='bold')

# 添加网格
ax.grid(axis='x', linestyle='--', alpha=0.5, linewidth=0.8)

# 调整x轴范围，给左侧标签留出空间（因为标签右移，左边界可以适当减小）
ax.set_xlim(-0.35, 1.05)

# 设置标题（放大字体）
ax.set_title('spin parameters comparison', fontsize=16, fontweight='bold')

# 设置坐标轴刻度字体大小
ax.tick_params(axis='x', labelsize=12)

plt.tight_layout()

# 保存图片
os.makedirs('output', exist_ok=True)
safe_name = first_source.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
output_path = f'output/spin_comparison_{safe_name}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存到: {output_path}")

# 显示图片
plt.show()