# plot_comparison_figures.py
# 该代码用于生成两张四合一的对比图：
# 图1: 四个数据集的自旋-倾角关系对比
# 图2: 四个数据集的自旋-距离关系对比

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def get_x_bounds(a_star, error_min, error_max, error_type, x_min=-1, x_max=1, model_name=None):
    """计算 x 方向的边界（左边界，右边界）
    
    对于 reflection 和 combining 模型，自旋值边界在 -0.998 和 0.998 处截断
    """
    # 对于 reflection 和 combining 模型，强制使用 -0.998 和 0.998 作为边界
    if model_name in ['reflection', 'combining']:
        bound_min = -0.998
        bound_max = 0.998
    else:
        bound_min = x_min
        bound_max = x_max
    
    if error_type == 'normal':
        if pd.isna(error_min) or np.isinf(error_min):
            left = a_star
        else:
            left = a_star - error_min
        if pd.isna(error_max) or np.isinf(error_max):
            right = a_star
        else:
            right = a_star + error_max
    elif error_type == 'less_than':
        # < 类型：向左延伸到 bound_min
        left = bound_min
        right = a_star
    elif error_type == 'greater_than':
        # > 类型：向右延伸到 bound_max
        left = a_star
        right = bound_max
    else:
        left = a_star
        right = a_star
    
    # 确保边界在合理范围内
    left = max(left, bound_min)
    right = min(right, bound_max)
    
    return left, right

def get_y_bounds_inclination(y_value, error_min, error_max, error_type, y_min_val=0, y_max_val=90):
    """计算倾角方向的边界（下边界，上边界）"""
    if error_type == 'normal':
        if pd.isna(error_min) or np.isinf(error_min):
            bottom = y_value
        else:
            bottom = y_value - error_min
        if pd.isna(error_max) or np.isinf(error_max):
            top = y_value
        else:
            top = y_value + error_max
    elif error_type == 'less_than':
        bottom = y_min_val
        top = y_value
    elif error_type == 'greater_than':
        bottom = y_value
        top = y_max_val
    else:
        bottom = y_value
        top = y_value
    
    # 确保边界在合理范围内
    bottom = max(bottom, y_min_val)
    top = min(top, y_max_val)
    
    return bottom, top

def get_y_bounds_distance(y_value, error_min, error_max, error_type, y_min_val=0, y_max_val=15):
    """计算距离方向的边界（下边界，上边界）- 固定范围 0-15 kpc"""
    if error_type == 'normal':
        if pd.isna(error_min) or np.isinf(error_min):
            bottom = y_value
        else:
            bottom = y_value - error_min
        if pd.isna(error_max) or np.isinf(error_max):
            top = y_value
        else:
            top = y_value + error_max
    elif error_type == 'less_than':
        bottom = y_min_val
        top = y_value
    elif error_type == 'greater_than':
        bottom = y_value
        top = y_max_val if not pd.isna(error_max) else y_value
    else:
        bottom = y_value
        top = y_value
    
    # 确保边界在合理范围内
    bottom = max(bottom, y_min_val)
    top = min(top, y_max_val)
    
    return bottom, top

def check_zero_errors(error_min, error_max):
    """检查误差是否都是0（即倾角或距离误差为0.000的情况）"""
    def is_zero(val):
        if pd.isna(val):
            return False
        if isinstance(val, (int, float)):
            return abs(val) < 1e-6
        return False
    
    return is_zero(error_min) and is_zero(error_max)

def plot_single_dataset(ax, df, x_range, y_range, y_type, source_color_map, 
                         show_points=True, alpha=0.4, model_name=None, subplot_label=None):
    """
    绘制单个数据集的散点图
    y_type: 'inclination' 或 'distance'
    show_points: 是否显示中心圆点
    model_name: 模型名称，用于确定自旋值边界
    subplot_label: 子图标签（如 'a', 'b', 'c', 'd'）
    """
    for source in df['源'].unique():
        source_df = df[df['源'] == source]
        color = source_color_map[source]
        
        for idx, row in source_df.iterrows():
            a_star = row[spin_col]
            
            if y_type == 'inclination':
                y_value = row[inclination_col]
                y_err_min = row[inclination_err_min_col]
                y_err_max = row[inclination_err_max_col]
                y_min_val, y_max_val = y_range
                
                # 检查倾角误差是否都为0
                y_zero_errors = check_zero_errors(y_err_min, y_err_max)
                
                if not y_zero_errors:
                    y_bottom, y_top = get_y_bounds_inclination(y_value, y_err_min, y_err_max, 
                                                                get_error_type(y_err_min, y_err_max),
                                                                y_min_val, y_max_val)
                else:
                    # 如果倾角误差为0，则只使用自旋值的矩形
                    y_bottom, y_top = y_value, y_value
                    
            else:  # distance
                y_value = row[distance_col]
                y_err_min = row[distance_err_min_col]
                y_err_max = row[distance_err_max_col]
                y_min_val, y_max_val = y_range
                
                # 检查距离误差是否都为0
                y_zero_errors = check_zero_errors(y_err_min, y_err_max)
                
                if not y_zero_errors:
                    y_bottom, y_top = get_y_bounds_distance(y_value, y_err_min, y_err_max,
                                                             get_error_type(y_err_min, y_err_max),
                                                             y_min_val, y_max_val)
                else:
                    # 如果距离误差为0，则只使用自旋值的矩形
                    y_bottom, y_top = y_value, y_value
            
            a_err_min = row[spin_err_min_col]
            a_err_max = row[spin_err_max_col]
            a_error_type = get_error_type(a_err_min, a_err_max)
            
            # 调用 get_x_bounds，传入 model_name 以确保正确的边界截断
            x_left, x_right = get_x_bounds(a_star, a_err_min, a_err_max, a_error_type, 
                                           x_range[0], x_range[1], model_name)
            
            # 绘制矩形（表示误差范围）
            width = x_right - x_left
            height = y_top - y_bottom
            
            # 只有当宽度或高度大于0时才绘制矩形
            if width > 0 and height > 0:
                rect = patches.Rectangle((x_left, y_bottom), width, height,
                                         fill=True, alpha=alpha, facecolor=color,
                                         linewidth=1, edgecolor=color, linestyle='-')
                ax.add_patch(rect)
            elif width > 0 and height == 0:
                # 如果只有宽度有值，绘制一条水平线表示自旋误差
                ax.hlines(y=y_value, xmin=x_left, xmax=x_right, 
                         colors=color, linewidth=2, alpha=alpha*0.8)
            
            # 绘制中心点（可选）
            if show_points:
                ax.scatter(a_star, y_value, color=color, s=50, marker='o',
                          zorder=5, edgecolor='black', linewidth=0.5)
    
    # 设置坐标轴 - 不显示标签名称，只保留刻度
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    # 不设置 xlabel 和 ylabel
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    
    # 添加子图标签 (a), (b), (c), (d) - 去掉黑框
    if subplot_label:
        ax.text(0.02, 0.98, f'({subplot_label})', transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

# 读取数据
print("正在读取数据...")
file_path = 'data/数据收集 表2 画图用.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"成功读取 {len(df)} 行数据")

# 定义列名
spin_col = '自旋值i'
spin_err_min_col = '自旋值i -'
spin_err_max_col = '自旋值i +'
inclination_col = '倾角i（deg）'
inclination_err_min_col = '倾角i（deg）-'
inclination_err_max_col = '倾角i（deg）+'
distance_col = '距离d(kpc)'
distance_err_min_col = '距离d(kpc)-'
distance_err_max_col = '距离d(kpc)+'
model_col = '拟合模型'

# 清洗数据：向前填充黑洞名称和拟合模型
df['源'] = df['源'].ffill()
df[model_col] = df[model_col].ffill()
df = df.dropna(subset=['源'])

# 转换数值类型
df[spin_col] = pd.to_numeric(df[spin_col], errors='coerce')
df[spin_err_min_col] = pd.to_numeric(df[spin_err_min_col], errors='coerce')
df[spin_err_max_col] = pd.to_numeric(df[spin_err_max_col], errors='coerce')
df[inclination_col] = pd.to_numeric(df[inclination_col], errors='coerce')
df[inclination_err_min_col] = pd.to_numeric(df[inclination_err_min_col], errors='coerce')
df[inclination_err_max_col] = pd.to_numeric(df[inclination_err_max_col], errors='coerce')
df[distance_col] = pd.to_numeric(df[distance_col], errors='coerce')
df[distance_err_min_col] = pd.to_numeric(df[distance_err_min_col], errors='coerce')
df[distance_err_max_col] = pd.to_numeric(df[distance_err_max_col], errors='coerce')

# 获取所有黑洞名称（用于颜色映射）
sources = df['源'].unique()
print(f"共发现 {len(sources)} 个黑洞")

# 为每个源分配固定颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(sources)))
source_color_map = {source: colors[i] for i, source in enumerate(sources)}

# 定义四个数据集
datasets = {
    'all_data': {'df': df, 'model_name': None, 'title': 'All Data'},
    'continuum-fitting': {'df': df[df[model_col] == 'continuum-fitting'], 
                          'model_name': 'continuum-fitting', 'title': 'Continuum-fitting'},
    'reflection': {'df': df[df[model_col] == 'reflection'], 
                   'model_name': 'reflection', 'title': 'Reflection'},
    'combining': {'df': df[df[model_col] == 'combining'], 
                  'model_name': 'combining', 'title': 'Combining'}
}

# 子图配置（按顺序：a, b, c, d）
subplot_configs = [
    {'dataset_key': 'all_data', 'subplot_label': 'a', 'title': 'All Data'},
    {'dataset_key': 'continuum-fitting', 'subplot_label': 'b', 'title': 'Continuum-fitting'},
    {'dataset_key': 'reflection', 'subplot_label': 'c', 'title': 'Reflection'},
    {'dataset_key': 'combining', 'subplot_label': 'd', 'title': 'Combining'}
]

# 定义x轴范围和标签
x_range = (-1, 1)

# 创建输出目录
output_base_dir = '任务二 所有数据总图'
output_dir = os.path.join(output_base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("开始生成对比图...")
print("="*60)

# ============================================================
# 第一张图：四个数据集的自旋-倾角关系对比
# ============================================================
print("\n📊 正在生成图1: 自旋-倾角关系对比图...")
print("-"*40)

# 使用更紧密的布局，减小子图间距
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))  # 减小画布尺寸使子图更紧密
fig1.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.15, hspace=0.15)
fig1.suptitle('不同数据集下黑洞自旋值与倾角关系对比', fontsize=16, fontweight='bold')

for i, config in enumerate(subplot_configs):
    row, col = i // 2, i % 2
    ax = axes1[row, col]
    
    dataset_info = datasets[config['dataset_key']]
    dataset_df = dataset_info['df']
    model_name = dataset_info['model_name']
    
    # 筛选有倾角数据的数据点
    plot_df = dataset_df.dropna(subset=[spin_col, inclination_col])
    
    if plot_df.empty:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(0, 90)
        ax.tick_params(axis='both', labelsize=10)
        ax.text(0.02, 0.98, f'({config["subplot_label"]})', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        continue
    
    # 绘制图形
    plot_single_dataset(ax, plot_df, x_range, (0, 90), 'inclination', 
                        source_color_map, show_points=True, alpha=0.4, 
                        model_name=model_name, subplot_label=config['subplot_label'])
    
    # 设置子图标题
    ax.set_title(f'{config["title"]}', fontsize=12, fontweight='bold', pad=5)

# 保存图片
output_path1 = os.path.join(output_dir, 'comparison_spin_inclination.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✅ 图1已保存: {output_path1}")
plt.close(fig1)

# ============================================================
# 第二张图：四个数据集的自旋-距离关系对比
# ============================================================
print("\n📊 正在生成图2: 自旋-距离关系对比图...")
print("-"*40)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))  # 减小画布尺寸使子图更紧密
fig2.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.15, hspace=0.15)
fig2.suptitle('不同数据集下黑洞自旋值与距离关系对比', fontsize=16, fontweight='bold')

for i, config in enumerate(subplot_configs):
    row, col = i // 2, i % 2
    ax = axes2[row, col]
    
    dataset_info = datasets[config['dataset_key']]
    dataset_df = dataset_info['df']
    model_name = dataset_info['model_name']
    
    # 筛选有距离数据的数据点
    plot_df = dataset_df.dropna(subset=[spin_col, distance_col])
    
    if plot_df.empty:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(0, 15)
        ax.tick_params(axis='both', labelsize=10)
        ax.text(0.02, 0.98, f'({config["subplot_label"]})', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        continue
    
    # 绘制图形
    plot_single_dataset(ax, plot_df, x_range, (0, 15), 'distance', 
                        source_color_map, show_points=True, alpha=0.4, 
                        model_name=model_name, subplot_label=config['subplot_label'])
    
    # 设置子图标题
    ax.set_title(f'{config["title"]}', fontsize=12, fontweight='bold', pad=5)

# 保存图片
output_path2 = os.path.join(output_dir, 'comparison_spin_distance.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ 图2已保存: {output_path2}")
plt.close(fig2)

print("\n" + "="*60)
print("🎉 全部完成！")
print("="*60)
print(f"\n📁 生成的文件:")
print(f"  图1 (自旋-倾角对比): {os.path.abspath(output_path1)}")
print(f"  图2 (自旋-距离对比): {os.path.abspath(output_path2)}")
print(f"\n图片保存位置: {os.path.abspath(output_dir)}")
print("="*60)

# 生成统计信息
print("\n📈 统计信息:")
for dataset_key, dataset_info in datasets.items():
    dataset_df = dataset_info['df']
    if not dataset_df.empty:
        incl_count = dataset_df.dropna(subset=[spin_col, inclination_col]).shape[0]
        dist_count = dataset_df.dropna(subset=[spin_col, distance_col]).shape[0]
        print(f"  {dataset_info['title']}: 总数据点 {len(dataset_df)}, 倾角数据 {incl_count}, 距离数据 {dist_count}")
print(f"  黑洞总数: {len(sources)}")
print("="*60)