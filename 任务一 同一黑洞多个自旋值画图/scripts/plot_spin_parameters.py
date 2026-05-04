import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

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
    
    if is_min_nan and (error_max == 0 or is_max_nan):
        return 'less_than'
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
    """格式化爆发年份用于显示"""
    if pd.isna(year_str):
        return ''
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

def format_spin_text(row):
    """格式化自旋值文本（用于显示）"""
    a_star = row['自旋值i']
    error_min = row['自旋值i -']
    error_max = row['自旋值i +']
    error_type = get_error_type(error_min, error_max)
    
    if error_type == 'less_than':
        return f"<{a_star:.3f}"
    elif error_type == 'greater_than':
        return f">{a_star:.3f}"
    else:
        if not pd.isna(error_min) and not pd.isna(error_max):
            if (error_min > 0 or error_max > 0) and not np.isinf(error_min) and not np.isinf(error_max):
                return f"${a_star:.3f}^{{+{error_max:.3f}}}_{{-{error_min:.3f}}}$"
        return f"{a_star:.3f}"

def prepare_source_df(df, source_name):
    """为指定黑洞准备并排序数据"""
    source_df = df[df['源'] == source_name].copy()
    if len(source_df) == 0:
        return None
    
    source_df['lit_year'] = source_df['文献来源'].apply(get_lit_year)
    source_df['year_sort'] = source_df['爆发时间'].apply(get_year_sort_key)
    source_df['burst_year_display'] = source_df['爆发时间'].apply(format_burst_year)
    source_df['author'], source_df['lit_year_str'] = zip(*source_df['文献来源'].apply(parse_literature))
    
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
    """绘制单个黑洞的图表 - 简化版本，确保文字向下移动"""
    n_points = len(source_df)
    
    # 检查是否有负值
    has_negative = any(source_df['自旋值i'] < 0)
    
    # 计算x轴范围
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
    
    # 设置x轴范围
    if has_negative:
        x_min = -1.0
        x_max = 1.0
    else:
        x_min = 0.0
        x_max = 1.0
    
    # 确保所有数据点在范围内
    if min_val < x_min:
        x_min = min_val - 0.05
    if max_val > x_max:
        x_max = max_val + 0.15
    
    # 统一使用较大的行间距，确保文字不遮挡
    y_spacing = 1.6      # 增加行间距
    y_start = 0.8        # 增加顶部空间
    
    # 计算每行的y位置
    y_positions = [y_start + i * y_spacing for i in range(n_points)]
    
    # 计算图表尺寸
    fig_width = 14
    fig_height = max(6, n_points * 0.9 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 设置黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # 设置y轴范围 - 给底部留更多空间
    y_min = -0.5
    y_max = max(y_positions) + 1.0
    ax.set_ylim(y_min, y_max)
    
    # 绘制数据点和误差棒
    for idx, (i, row) in enumerate(source_df.iterrows()):
        y_pos = y_positions[idx]
        a_star = row['自旋值i']
        error_min = row['自旋值i -']
        error_max = row['自旋值i +']
        model = row['拟合模型']
        color = color_map.get(model, other_color)
        
        error_type = get_error_type(error_min, error_max)
        
        if error_type == 'less_than':
            ax.arrow(a_star, y_pos, -0.08, 0,
                    head_width=0.08, head_length=0.015,
                    fc=color, ec=color, alpha=0.8, linewidth=1.0,
                    length_includes_head=True)
            ax.plot(a_star, y_pos, 'o', color=color, markersize=8, zorder=10)
        elif error_type == 'greater_than':
            ax.arrow(a_star, y_pos, 0.08, 0,
                    head_width=0.08, head_length=0.015,
                    fc=color, ec=color, alpha=0.8, linewidth=1.0,
                    length_includes_head=True)
            ax.plot(a_star, y_pos, 'o', color=color, markersize=8, zorder=10)
        else:
            has_error = False
            if not pd.isna(error_min) and not pd.isna(error_max):
                if (error_min > 0 or error_max > 0) and not np.isinf(error_min) and not np.isinf(error_max):
                    has_error = True
            
            if has_error:
                xerr = [[error_min], [error_max]]
                ax.errorbar(a_star, y_pos, xerr=xerr, fmt='o', color=color,
                           capsize=5, markersize=8, elinewidth=1.5,
                           ecolor=color, markeredgecolor=color, 
                           markerfacecolor=color, zorder=10)
            else:
                ax.plot(a_star, y_pos, 'o', color=color, markersize=8, zorder=10)
    
    # 为每个数据点添加标签 - 关键修改：所有文本都放在数据点下方
    for idx, row in source_df.iterrows():
        y_pos = y_positions[idx]
        a_star = row['自旋值i']
        model = row['拟合模型']
        author = row['author']
        burst_year = row['burst_year_display']
        lit_year = row['lit_year_str']
        color = color_map.get(row['拟合模型'], other_color)
        
        # 格式化自旋值文本
        spin_text = format_spin_text(row)
        
        # 构建完整的左侧文本
        if burst_year and burst_year != 'Unknown':
            left_text = f"{model} {burst_year} {author} ({lit_year})"
        else:
            left_text = f"{model} {author} ({lit_year})"
        
        # 关键修改：所有参考文献文本都放在数据点下方，向下偏移0.6
        # 自旋值放在数据点正上方，向上偏移0.4
        ax.text(x_min + 0.05, y_pos - 0.6, left_text, fontsize=9, 
               va='top', ha='left', color=color, alpha=0.85,
               fontweight='normal')
        
        # 自旋值放在数据点正上方
        ax.text(a_star, y_pos + 0.4, spin_text, fontsize=10, 
               va='bottom', ha='center', color=color, alpha=0.9,
               fontweight='bold')
    
    # 设置坐标轴
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    
    # 设置x轴标签
    ax.set_xlabel(r'$a_*$', fontsize=14, ha='center', fontweight='bold', labelpad=10)
    ax.tick_params(axis='x', labelsize=11)
    
    # 添加网格线
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 设置标题
    ax.set_title(source_name, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # 保存图片
    safe_name = source_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
    output_path = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已生成: {safe_name}.png ({len(source_df)} 个数据点)")

def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    
    # 任务一文件夹 = scripts的父目录
    task1_dir = script_dir.parent
    
    # 根目录 = 任务一文件夹的父目录
    root_dir = task1_dir.parent
    
    # data文件夹在根目录下
    data_dir = root_dir / 'data'
    data_file = data_dir / '数据收集 表3 画图用.xlsx'
    
    # output文件夹在任务一文件夹下（与scripts同级）
    output_dir = task1_dir / 'output'
    
    print(f"脚本目录: {script_dir}")
    print(f"任务一目录: {task1_dir}")
    print(f"根目录: {root_dir}")
    print(f"数据文件路径: {data_file}")
    print(f"输出目录: {output_dir}")
    
    # 检查数据文件是否存在
    if not data_file.exists():
        print(f"错误：找不到数据文件 {data_file}")
        print("请确保目录结构为：")
        print("blackhole_spin_visualization/")
        print("├── data/")
        print("│   └── 数据收集 表3 画图用.xlsx")
        print("└── 任务一 同一黑洞多个自旋值画图/")
        print("    ├── scripts/")
        print("    │   └── plot_spin_parameters.py")
        print("    └── output/")
        return
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(data_file, sheet_name='Sheet1')
    print(f"成功读取 {len(df)} 行数据")
    
    df['自旋值i -'] = pd.to_numeric(df['自旋值i -'], errors='coerce')
    df['自旋值i +'] = pd.to_numeric(df['自旋值i +'], errors='coerce')
    df['源'] = df['源'].ffill()
    df = df.dropna(subset=['源'])
    print(f"清洗后剩余 {len(df)} 行数据")
    
    sources = df['源'].unique()
    print(f"共发现 {len(sources)} 个黑洞")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
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