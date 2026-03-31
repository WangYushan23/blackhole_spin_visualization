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

def has_valid_error(error_min, error_max):
    """判断误差是否有效"""
    if pd.isna(error_min) or pd.isna(error_max):
        return False
    try:
        if np.isinf(error_min) or np.isinf(error_max):
            return False
        if error_min == 0 and error_max == 0:
            return False
        # 检查是否为有效数字
        if np.isnan(error_min) or np.isnan(error_max):
            return False
        return True
    except:
        return False

def prepare_source_df(df, source_name):
    """为指定黑洞准备并排序数据"""
    source_df = df[df['源'] == source_name].copy()
    if len(source_df) == 0:
        return None
    
    # 过滤掉自旋值为无效的数据
    source_df = source_df[pd.notna(source_df['自旋值i'])]
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
    
    # 计算所有数据点（包括误差）的最小值和最大值
    all_values = []
    for _, row in source_df.iterrows():
        a_star = row['自旋值i']
        
        # 跳过无效的自旋值
        if pd.isna(a_star) or np.isinf(a_star):
            print(f"  警告: 跳过无效自旋值 {a_star}")
            continue
            
        error_min = row['自旋值i -']
        error_max = row['自旋值i +']
        
        # 添加中心值
        all_values.append(a_star)
        
        # 如果有有效误差，添加误差边界
        if has_valid_error(error_min, error_max):
            try:
                lower_bound = a_star - error_min
                upper_bound = a_star + error_max
                if not np.isnan(lower_bound) and not np.isinf(lower_bound):
                    all_values.append(lower_bound)
                if not np.isnan(upper_bound) and not np.isinf(upper_bound):
                    all_values.append(upper_bound)
            except:
                pass
    
    # 如果没有有效值，使用默认范围
    if not all_values:
        print(f"  警告: 黑洞 {source_name} 没有有效数据点，使用默认范围")
        x_min = 0
        x_max = 1.05
    else:
        # 过滤掉无效值
        valid_values = [v for v in all_values if not np.isnan(v) and not np.isinf(v)]
        if not valid_values:
            x_min = 0
            x_max = 1.05
        else:
            x_min = min(valid_values)
            x_max = max(valid_values)
            
            # 添加一些边距（10%）
            x_range = x_max - x_min
            if x_range > 0:
                x_min = x_min - x_range * 0.1
                x_max = x_max + x_range * 0.1
            else:
                # 如果所有值相同，添加对称边距
                x_min = x_min - 0.1
                x_max = x_max + 0.1
            
            # 确保x_min不超过负太多（至少保留到-0.2，给负数留空间）
            x_min = min(x_min, -0.2)
            
            # 如果最小值大于0，可以从0开始
            if min(valid_values) >= 0:
                x_min = min(x_min, 0)
            
            # x_max至少到1，如果有大于1的值则扩展
            x_max = max(x_max, 1.05)
            
            # 如果最大值小于1，可以扩展到1
            if max(valid_values) <= 1:
                x_max = max(x_max, 1)
    
    # 绘制数据点和误差棒
    for i, row in source_df.iterrows():
        a_star = row['自旋值i']
        
        # 跳过无效的自旋值
        if pd.isna(a_star) or np.isinf(a_star):
            print(f"  警告: 跳过无效数据点 {a_star}")
            continue
            
        error_min = row['自旋值i -']
        error_max = row['自旋值i +']
        model = row['拟合模型']
        color = color_map.get(model, other_color)
        
        try:
            if has_valid_error(error_min, error_max):
                xerr = [[error_min], [error_max]]
                ax.errorbar(a_star, i, xerr=xerr, fmt='o', color=color,
                           capsize=5, markersize=10, elinewidth=2,
                           ecolor=color, markeredgecolor=color, markerfacecolor=color)
            else:
                ax.plot(a_star, i, 'o', color=color, markersize=10)
        except Exception as e:
            print(f"  警告: 绘制数据点失败: {e}")
            ax.plot(a_star, i, 'o', color=color, markersize=10)
    
    # 添加标注
    for i, row in source_df.iterrows():
        a_star = row['自旋值i']
        
        # 跳过无效的自旋值
        if pd.isna(a_star) or np.isinf(a_star):
            continue
            
        error_min = row['自旋值i -']
        error_max = row['自旋值i +']
        model = row['拟合模型']
        author = row['author']
        burst_year = row['burst_year_display']
        lit_year = row['lit_year_str']
        color = color_map.get(row['拟合模型'], other_color)
        
        # 数值标签
        try:
            if has_valid_error(error_min, error_max):
                upper_text = f"{a_star:.3f}$^{{+{error_max:.3f}}}_{{-{error_min:.3f}}}$"
            else:
                upper_text = f"{a_star:.3f}"
            ax.text(a_star, i + 0.2, upper_text, fontsize=11, va='bottom', ha='center',
                    color=color, fontweight='bold')
        except:
            upper_text = f"{a_star:.3f}"
            ax.text(a_star, i + 0.2, upper_text, fontsize=11, va='bottom', ha='center',
                    color=color, fontweight='bold')
        
        # 文献标签
        lower_text = f"{model} {burst_year} {author} ({lit_year})"
        ax.text(-0.08, i, lower_text, fontsize=11, va='center', ha='right',
                color=color, alpha=0.9)
    
    # 隐藏 y 轴
    ax.set_yticks([])
    ax.set_ylim(y_min, y_max)
    
    # 设置 x 轴
    try:
        ax.set_xlim(x_min, x_max)
    except ValueError as e:
        print(f"  错误: 设置x轴范围失败 {x_min} 到 {x_max}: {e}")
        ax.set_xlim(0, 1.05)
    
    ax.set_xlabel(r'$a_*$', fontsize=16, ha='center', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 标题
    ax.set_title(source_name, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    
    # 添加一条参考线在 x=0 处（如果x_min < 0）
    if x_min < 0:
        ax.axvline(x=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # 为左侧标签预留空间
    plt.subplots_adjust(left=0.2)
    plt.tight_layout()
    
    # 保存图片
    safe_name = source_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
    output_path = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {safe_name}.png ({len(source_df)} 个数据点)")

def main():
    import os
    
    # 获取脚本所在目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # scripts的上一级
    
    # 读取数据
    print("正在读取数据...")
    data_path = os.path.join(project_root, 'data', '数据收集 表3.xlsx')
    print(f"数据文件路径: {data_path}")
    
    try:
        df = pd.read_excel(data_path, sheet_name='Sheet1')
        print(f"成功读取 {len(df)} 行数据")
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        print("请确保 data/数据收集 表3.xlsx 文件存在")
        return
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return
    
    # 清洗：向前填充黑洞名称
    df['源'] = df['源'].ffill()
    df = df.dropna(subset=['源'])
    print(f"清洗后剩余 {len(df)} 行数据")
    
    # 获取所有黑洞名称
    sources = df['源'].unique()
    print(f"共发现 {len(sources)} 个黑洞")
    
    # 创建输出目录（在项目根目录）
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
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