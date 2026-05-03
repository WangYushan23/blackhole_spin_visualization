import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色映射
color_map = {
    'combining': 'green',
    'reflection': 'red',
    'continuum-fitting': 'blue'
}
other_color = 'gray'

# 定义模型顺序（从上到下）
model_order = ['combining', 'reflection', 'continuum-fitting']

def get_error_type(error_min, error_max):
    """判断误差类型，用于范围值（如 <0.8 或 >0.9）"""
    def is_nan(val):
        if val is None:
            return True
        if isinstance(val, float) and np.isnan(val):
            return True
        if pd.isna(val):
            return True
        return False
    
    is_min_nan = is_nan(error_min)
    is_max_nan = is_nan(error_max)
    
    # 处理 < 类型：error_min 为 NaN，error_max 为 0 或 NaN
    if is_min_nan and (error_max == 0 or is_max_nan):
        return 'less_than'
    # 处理 > 类型：error_max 为 NaN，error_min 为 0 或 NaN
    if is_max_nan and (error_min == 0 or is_min_nan):
        return 'greater_than'
    return 'normal'

def parse_literature(lit_str):
    """解析文献来源，返回 (作者, 完整年份)"""
    if pd.isna(lit_str):
        return ('Unknown', 'Unknown')
    lit_str = str(lit_str).strip()
    years = re.findall(r'\b(?:19|20)\d{2}\b', lit_str)
    year = years[0] if years else 'Unknown'
    author_part = re.sub(r'\b(?:19|20)\d{2}\b', '', lit_str).strip()
    if 'et al' in author_part.lower():
        author = author_part.split()[0] + ' et al.'
    else:
        author = author_part.split()[0] + ' et al.'
    return (author, year)

def get_lit_year(literature):
    if pd.isna(literature):
        return 9999
    lit_str = str(literature)
    years = re.findall(r'\b(?:19|20)\d{2}\b', lit_str)
    return int(years[0]) if years else 9999

def get_year_sort_key(year_str):
    """处理爆发年份，提取排序用的年份（最早年份）"""
    if pd.isna(year_str):
        return 9999
    year_str = str(year_str).strip()
    if '+' in year_str:
        first = year_str.split('+')[0]
        if first.isdigit():
            return int(first)
    if '-' in year_str and len(year_str) > 5:
        first = year_str.split('-')[0]
        if first.isdigit():
            return int(first)
    if year_str.isdigit():
        return int(year_str)
    return 9999

def format_burst_year(year_str):
    """格式化爆发年份用于显示（保持原样，但补全两位数年份为四位数）"""
    if pd.isna(year_str):
        return 'Unknown'
    year_str = str(year_str).strip()
    if '+' in year_str or '-' in year_str:
        return year_str
    if year_str.isdigit():
        if len(year_str) == 2:
            if int(year_str) >= 90:
                return '19' + year_str
            else:
                return '20' + year_str
        return year_str
    return year_str

def estimate_text_width(text, fontsize=8):
    """估算文本宽度（单位：x轴坐标单位）"""
    # 中文字符算2个单位宽度，英文字符算1个单位宽度
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese_chars
    # 每个字符约占总坐标的 0.008 倍（经验值）
    return (chinese_chars * 2 + english_chars) * 0.0065 * (fontsize / 8)

def prepare_source_df(df, source_name):
    """为指定黑洞准备并排序数据"""
    source_df = df[df['源'] == source_name].copy()
    if len(source_df) == 0:
        return None
    
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
            model_df = model_df.sort_values(['year_sort', 'lit_year'])
            sorted_indices.extend(model_df.index.tolist())
    other_models = [m for m in source_df['拟合模型'].unique() if m not in model_order]
    for model in other_models:
        model_df = source_df[source_df['拟合模型'] == model]
        if len(model_df) > 0:
            model_df = model_df.sort_values(['year_sort', 'lit_year'])
            sorted_indices.extend(model_df.index.tolist())
    
    source_df = source_df.loc[sorted_indices].reset_index(drop=True)
    return source_df

def plot_single_source(source_df, source_name, output_dir):
    """绘制单个黑洞的图表"""
    n_points = len(source_df)
    y_min = -0.5
    y_max = n_points - 0.5
    
    fig, ax = plt.subplots(figsize=(14, max(7, n_points * 0.8)))
    
    # 第一遍：绘制数据点和误差棒（包括范围值的箭头）
    for i, row in source_df.iterrows():
        a_star = row['自旋值i']
        error_min = row['自旋值i -']
        error_max = row['自旋值i +']
        model = row['拟合模型']
        color = color_map.get(model, other_color)
        
        error_type = get_error_type(error_min, error_max)
        
        if error_type == 'less_than':
            # 向左箭头：从数据点 a_star 向左指
            ax.arrow(a_star, i, -0.10, 0,
                    head_width=0.1, head_length=0.02,
                    fc=color, ec=color, alpha=0.8, linewidth=1.0)
            ax.plot(a_star, i, 'o', color=color, markersize=8)
        elif error_type == 'greater_than':
            # 向右箭头：从数据点 a_star 向右指
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
    
    # 计算 x 轴范围
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
            all_values.append(a_star - 0.15)
        elif error_type == 'greater_than':
            all_values.append(a_star + 0.15)
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    if min_val < 0:
        x_min = -1.05
    else:
        x_min = 0
    
    x_max = max(1.05, max_val + 0.1)
    ax.set_xlim(x_min, x_max)
    
    x_range = x_max - x_min
    
    # 第二遍：添加标注（数值标签和文献标签）
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
        
        # 数值标签
        if error_type == 'less_than':
            upper_text = f"<{a_star:.3f}"
            ax.text(a_star - 0.12, i + 0.2, upper_text, fontsize=10, va='bottom', ha='center', 
                    color=color, fontweight='bold')
        elif error_type == 'greater_than':
            upper_text = f">{a_star:.3f}"
            ax.text(a_star + 0.12, i + 0.2, upper_text, fontsize=10, va='bottom', ha='center', 
                    color=color, fontweight='bold')
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
        
        # 文献标签 - 根据文本长度动态调整位置
        lower_text = f"{model} {burst_year} {author} ({lit_year})"
        
        # 估算文本宽度
        text_width = estimate_text_width(lower_text, fontsize=8)
        min_space_needed = text_width + 0.08  # 需要的最小空间
        
        right_space = x_max - a_star
        left_space = a_star - x_min
        
        # 检查是否有数据点紧邻
        right_data_distance = min([abs(a_star - row2['自旋值i']) for _, row2 in source_df.iterrows() if abs(a_star - row2['自旋值i']) > 0.01] + [1.0])
        
        # 计算垂直偏移，避免上下重叠
        y_offset = 0
        # 检查相邻数据点是否在附近
        for other_i in range(max(0, i-2), min(n_points, i+3)):
            if other_i != i:
                other_a = source_df.iloc[other_i]['自旋值i']
                if abs(a_star - other_a) < 0.15:
                    y_offset -= 0.12 * (1 if other_i < i else 0)
        
        # 智能定位
        if right_space > min_space_needed and right_data_distance > 0.1:
            # 放在右侧
            x_pos = a_star + 0.04
            ha = 'left'
            va = 'center'
            final_y = i + y_offset
        elif left_space > min_space_needed and right_data_distance > 0.1:
            # 放在左侧
            x_pos = a_star - 0.04
            ha = 'right'
            va = 'center'
            final_y = i + y_offset
        else:
            # 空间不足，放在下方
            x_pos = a_star
            ha = 'center'
            va = 'top'
            final_y = i - 0.45
        
        ax.text(x_pos, final_y, lower_text, fontsize=8, 
                va=va, ha=ha, color=color, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 隐藏 y 轴
    ax.set_yticks([])
    ax.set_ylim(y_min, y_max)
    
    # 设置 x 轴
    ax.set_xlabel(r'$a_*$', fontsize=16, ha='center', fontweight='bold', labelpad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_title(source_name, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    
    # 调整边距
    plt.subplots_adjust(left=0.2, bottom=0.18)
    plt.tight_layout()
    
    # 保存图片
    safe_name = source_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
    output_path = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已生成: {safe_name}.png ({len(source_df)} 个数据点)")

def main():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(r'C:\Users\王雨珊\Desktop\blackhole_spin_visualization\data\数据收集 表3 画图用.xlsx', sheet_name='Sheet1')
    print(f"成功读取 {len(df)} 行数据")
    
    # 清洗数据：转换数值并处理范围值
    df['自旋值i -'] = pd.to_numeric(df['自旋值i -'], errors='coerce')
    df['自旋值i +'] = pd.to_numeric(df['自旋值i +'], errors='coerce')
    
    # 清洗：向前填充黑洞名称
    df['源'] = df['源'].ffill()
    df = df.dropna(subset=['源'])
    print(f"清洗后剩余 {len(df)} 行数据")
    
    # 获取所有黑洞名称
    sources = df['源'].unique()
    print(f"共发现 {len(sources)} 个黑洞")
    
    # 创建输出目录
    output_dir = '任务一 同一黑洞多个自旋值画图/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 逐个黑洞绘图
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] 处理黑洞: {source}")
        source_df = prepare_source_df(df, source)
        if source_df is None:
            print("  警告: 无有效数据，跳过")
            continue
        plot_single_source(source_df, source, output_dir)
    
    print(f"\n全部完成！图片已保存到 {output_dir}/ 文件夹")

if __name__ == '__main__':
    main()