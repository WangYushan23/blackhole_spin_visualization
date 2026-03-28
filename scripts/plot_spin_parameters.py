import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色映射（拟合模型 -> 颜色）
color_map = {
    'reflection': 'red',
    'continuum-fitting': 'blue',
    'combining': 'green'
}
# 其他模型统一用灰色
other_color = 'gray'

# 定义年份类别和形状映射
def get_year_category(year_str):
    """根据爆发时间字符串返回年份类别"""
    if pd.isna(year_str):
        return 'unknown'
    year_str = str(year_str).strip()
    # 如果包含 + 或 - 或 , 或空格，视为多年份
    if '+' in year_str or '-' in year_str and len(year_str) > 5 or ',' in year_str:
        return 'multi_year'
    # 如果是纯数字年份
    if year_str.isdigit() and len(year_str) == 4:
        return year_str
    return 'unknown'

def get_year_sort_key(year_str):
    """返回用于排序的年份数值"""
    if pd.isna(year_str):
        return 9999  # 未知年份排最后
    year_str = str(year_str).strip()
    if '+' in year_str:
        # 多年份取第一个年份
        first_year = year_str.split('+')[0]
        if first_year.isdigit():
            return int(first_year)
    if '-' in year_str and len(year_str) > 5:
        first_year = year_str.split('-')[0]
        if first_year.isdigit():
            return int(first_year)
    if year_str.isdigit():
        return int(year_str)
    return 9999

# 定义形状映射
shape_map = {
    'single_year': 'o',      # 单一年份用圆形
    'multi_year': '^',       # 多年份用三角形
    'unknown': 's'           # 未知年份用方形
}

def get_shape_category(year_str):
    """返回形状类别"""
    if pd.isna(year_str):
        return 'unknown'
    year_str = str(year_str).strip()
    if '+' in year_str or ('-' in year_str and len(year_str) > 5) or ',' in year_str:
        return 'multi_year'
    if year_str.isdigit():
        return 'single_year'
    return 'unknown'

def get_literature_year(literature):
    """从文献来源提取年份"""
    if pd.isna(literature):
        return 9999
    lit_str = str(literature)
    # 提取4位数字年份
    import re
    years = re.findall(r'\b(19|20)\d{2}\b', lit_str)
    if years:
        return int(years[0])
    return 9999

def has_valid_error(error_min, error_max):
    """判断误差是否有效（非NaN、非inf、非0）"""
    if pd.isna(error_min) or pd.isna(error_max):
        return False
    if np.isinf(error_min) or np.isinf(error_max):
        return False
    if error_min == 0 and error_max == 0:
        return False
    return True

def prepare_data(df):
    """准备和清洗数据"""
    # 复制数据
    df_clean = df.copy()
    
    # 向前填充黑洞名称（处理空值）
    df_clean['源'] = df_clean['源'].fillna(method='ffill')
    
    # 删除仍然没有源名称的行
    df_clean = df_clean.dropna(subset=['源'])
    
    # 重置索引
    df_clean = df_clean.reset_index(drop=True)
    
    # 添加处理后的列
    df_clean['year_sort'] = df_clean['爆发时间'].apply(get_year_sort_key)
    df_clean['year_category'] = df_clean['爆发时间'].apply(get_year_category)
    df_clean['shape_category'] = df_clean['爆发时间'].apply(get_shape_category)
    df_clean['lit_year'] = df_clean['文献来源'].apply(get_literature_year)
    
    return df_clean

def get_y_position(df, source_name):
    """为指定黑洞的数据点分配y坐标"""
    # 筛选该黑洞的数据
    source_df = df[df['源'] == source_name].copy()
    
    # 按模型分组排序
    model_order = ['reflection', 'continuum-fitting', 'combining']
    other_models = [m for m in source_df['拟合模型'].unique() if m not in model_order]
    
    # 构建排序键
    def get_sort_key(row):
        model = row['拟合模型']
        # 模型优先级
        if model in model_order:
            model_priority = model_order.index(model)
        else:
            model_priority = len(model_order)  # 其他模型排最后
        
        # 爆发年份（早的在上）
        year = row['year_sort']
        # 文献年份（早的在上）
        lit_year = row['lit_year']
        
        return (model_priority, year, lit_year)
    
    # 排序
    source_df = source_df.sort_values(by=['拟合模型', 'year_sort', 'lit_year'], 
                                       key=lambda x: x.apply(get_sort_key if x.name == '拟合模型' else (lambda y: y)))
    
    # 重新排序
    sorted_indices = []
    for model in model_order + other_models:
        model_df = source_df[source_df['拟合模型'] == model]
        if len(model_df) > 0:
            # 按年份和文献年份排序
            model_df = model_df.sort_values(['year_sort', 'lit_year'])
            sorted_indices.extend(model_df.index.tolist())
    
    source_df = source_df.loc[sorted_indices] if sorted_indices else source_df
    
    # 分配y坐标（从上到下，0,1,2,...）
    y_positions = {}
    for i, idx in enumerate(source_df.index):
        y_positions[idx] = i
    
    return y_positions

def plot_single_source(df, source_name, y_positions, output_dir):
    """为单个黑洞绘制图表"""
    source_df = df[df['源'] == source_name]
    
    if len(source_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(source_df) * 0.4)))
    
    # 绘制每个数据点
    for idx, row in source_df.iterrows():
        if idx not in y_positions:
            continue
        
        y = y_positions[idx]
        a_star = row['自旋值i']
        error_min = row['自旋值i -']
        error_max = row['自旋值i +']
        model = row['拟合模型']
        shape_cat = row['shape_category']
        
        # 确定颜色
        color = color_map.get(model, other_color)
        
        # 确定形状
        marker = shape_map.get(shape_cat, 'o')
        
        # 确定是否画误差棒
        has_error = has_valid_error(error_min, error_max)
        
        if has_error:
            # 不对称误差
            xerr = [[error_min], [error_max]]
            ax.errorbar(a_star, y, xerr=xerr, fmt=marker, color=color,
                       capsize=4, markersize=8, elinewidth=1.5,
                       label=None)
        else:
            # 只画点
            ax.plot(a_star, y, marker=marker, color=color, 
                   markersize=8, linestyle='None')
    
    # 设置y轴
    y_max = len(source_df)
    ax.set_ylim(-0.5, y_max - 0.5)
    
    # 设置y轴标签（文献来源）
    y_labels = []
    for idx, row in source_df.iterrows():
        if idx in y_positions:
            y_pos = y_positions[idx]
            # 标签格式：文献来源 (年份)
            label = f"{row['文献来源']} ({row['爆发时间']})"
            y_labels.append((y_pos, label))
    
    y_labels.sort(key=lambda x: x[0])
    ax.set_yticks([y for y, _ in y_labels])
    ax.set_yticklabels([label for _, label in y_labels], fontsize=9)
    
    # 设置x轴
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r'自旋参数 $a_*$', fontsize=12)
    
    # 设置标题
    ax.set_title(f'{source_name} - 自旋参数比较', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # 模型图例
    for model, color in color_map.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, 
                                      label=model))
    # 其他模型
    other_models = set(source_df['拟合模型']) - set(color_map.keys())
    if other_models:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=other_color, markersize=10, 
                                      label='其他模型'))
    
    # 年份形状图例
    legend_elements.append(Line2D([0], [0], marker='o', color='k', 
                                  linestyle='None', markersize=8, 
                                  label='单一年份'))
    legend_elements.append(Line2D([0], [0], marker='^', color='k', 
                                  linestyle='None', markersize=8, 
                                  label='多年份'))
    legend_elements.append(Line2D([0], [0], marker='s', color='k', 
                                  linestyle='None', markersize=8, 
                                  label='未知年份'))
    
    ax.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    safe_name = source_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_')
    output_path = os.path.join(output_dir, f'spin_comparison_{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成: {safe_name}.png ({len(source_df)} 个数据点)")

def main():
    """主函数"""
    # 设置路径
    data_path = 'data/数据收集 表3.xlsx'
    output_dir = 'output'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(data_path, sheet_name='Sheet1')
    print(f"成功读取 {len(df)} 行数据")
    
    # 清洗数据
    print("正在清洗数据...")
    df_clean = prepare_data(df)
    print(f"清洗后剩余 {len(df_clean)} 行数据")
    
    # 获取所有黑洞名称
    sources = df_clean['源'].unique()
    print(f"共 {len(sources)} 个黑洞")
    
    # 为每个黑洞生成图片
    print("\n开始生成图片...")
    for i, source in enumerate(sources, 1):
        # 获取y轴位置
        y_positions = get_y_position(df_clean, source)
        
        # 绘图
        plot_single_source(df_clean, source, y_positions, output_dir)
        
        if i % 10 == 0:
            print(f"已完成 {i}/{len(sources)} 个黑洞")
    
    print(f"\n完成！图片已保存到 {output_dir}/ 文件夹")

if __name__ == '__main__':
    main()