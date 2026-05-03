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

def get_x_bounds(a_star, error_min, error_max, error_type, x_min=0, x_max=1):
    """计算 x 方向的边界（左边界，右边界）"""
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
        # < 类型：向左延伸到 x_min
        left = x_min
        right = a_star
    elif error_type == 'greater_than':
        # > 类型：向右延伸到 x_max
        left = a_star
        right = x_max
    else:
        left = a_star
        right = a_star
    
    # 确保边界在合理范围内
    left = max(left, x_min)
    right = min(right, x_max)
    
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

def plot_dataset(ax, df, x_range, y_range, y_type, source_color_map, title, x_label, y_label, 
                 show_points=True, alpha=0.4):
    """
    绘制单个数据集
    y_type: 'inclination' 或 'distance'
    show_points: 是否显示中心圆点
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
                y_bottom, y_top = get_y_bounds_inclination(y_value, y_err_min, y_err_max, 
                                                            get_error_type(y_err_min, y_err_max),
                                                            y_min_val, y_max_val)
            else:  # distance
                y_value = row[distance_col]
                y_err_min = row[distance_err_min_col]
                y_err_max = row[distance_err_max_col]
                y_min_val, y_max_val = y_range
                y_bottom, y_top = get_y_bounds_distance(y_value, y_err_min, y_err_max,
                                                         get_error_type(y_err_min, y_err_max),
                                                         y_min_val, y_max_val)
            
            a_err_min = row[spin_err_min_col]
            a_err_max = row[spin_err_max_col]
            a_error_type = get_error_type(a_err_min, a_err_max)
            x_left, x_right = get_x_bounds(a_star, a_err_min, a_err_max, a_error_type, x_range[0], x_range[1])
            
            # 绘制矩形（表示误差范围）
            width = x_right - x_left
            height = y_top - y_bottom
            
            if width > 0 and height > 0:
                rect = patches.Rectangle((x_left, y_bottom), width, height,
                                         fill=True, alpha=alpha, facecolor=color,
                                         linewidth=1, edgecolor=color, linestyle='-')
                ax.add_patch(rect)
            
            # 绘制中心点（可选）
            if show_points:
                ax.scatter(a_star, y_value, color=color, s=50, marker='o',
                          zorder=5, edgecolor='black', linewidth=0.5)
    
    # 设置坐标轴
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

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
    'all_data': df,
    'continuum-fitting': df[df[model_col] == 'continuum-fitting'],
    'reflection': df[df[model_col] == 'reflection'],
    'combining': df[df[model_col] == 'combining']
}

# 定义绘图配置
plot_configs = [
    {'name': 'spin_vs_inclination_full', 'x_range': (0, 1), 'y_range': (0, 90),
     'x_label': r'自旋值 $a_*$', 'y_label': r'倾角 $i$ (度)', 'y_type': 'inclination',
     'title_suffix': '自旋-倾角关系 (全范围)'},
    {'name': 'spin_vs_inclination_zoom', 'x_range': (0.25, 1), 'y_range': (0, 90),
     'x_label': r'自旋值 $a_*$', 'y_label': r'倾角 $i$ (度)', 'y_type': 'inclination',
     'title_suffix': '自旋-倾角关系 (自旋 ≥ 0.25)'},
    {'name': 'spin_vs_distance_full', 'x_range': (0, 1), 'y_range': (0, 15),
     'x_label': r'自旋值 $a_*$', 'y_label': r'距离 $d$ (kpc)', 'y_type': 'distance',
     'title_suffix': '自旋-距离关系 (全范围)'},
    {'name': 'spin_vs_distance_zoom', 'x_range': (0.25, 1), 'y_range': (0, 15),
     'x_label': r'自旋值 $a_*$', 'y_label': r'距离 $d$ (kpc)', 'y_type': 'distance',
     'title_suffix': '自旋-距离关系 (自旋 ≥ 0.25)'}
]

# 两个版本：带圆点和不带圆点
versions = [
    {'name': 'with_points', 'show_points': True, 'suffix': '带数据点'},
    {'name': 'without_points', 'show_points': False, 'suffix': '不带数据点'}
]

# 创建基础输出目录
output_base_dir = '任务二 所有数据总图'
for version in versions:
    version_dir = os.path.join(output_base_dir, version['name'])
    for dataset_name in datasets.keys():
        output_dir = os.path.join(version_dir, dataset_name, 'output')
        os.makedirs(output_dir, exist_ok=True)
        # 同时创建scripts目录
        scripts_dir = os.path.join(version_dir, dataset_name, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)

print("\n开始生成图片...")
print("="*60)

# 为每个版本生成图片
for version in versions:
    print(f"\n📊 正在生成 {version['suffix']} 版本...")
    print("-"*40)
    
    # 为每个数据集生成图片
    for dataset_name, dataset_df in datasets.items():
        if dataset_df.empty:
            print(f"⚠️ 数据集 '{dataset_name}' 为空，跳过")
            continue
        
        print(f"\n  处理数据集: {dataset_name} (共 {len(dataset_df)} 个数据点)")
        
        # 筛选有效数据
        df_inclination = dataset_df.dropna(subset=[spin_col, inclination_col])
        df_distance = dataset_df.dropna(subset=[spin_col, distance_col])
        
        print(f"    倾角有效数据点: {len(df_inclination)}")
        print(f"    距离有效数据点: {len(df_distance)}")
        
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'{dataset_name.replace("-", " ").title()} 数据集 - {version["suffix"]}', 
                     fontsize=18, fontweight='bold')
        
        for i, config in enumerate(plot_configs):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # 选择对应的数据
            if config['y_type'] == 'inclination':
                plot_df = df_inclination
            else:
                plot_df = df_distance
            
            if plot_df.empty:
                ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_xlim(config['x_range'][0], config['x_range'][1])
                ax.set_ylim(config['y_range'][0], config['y_range'][1])
                continue
            
            # 绘制图形
            title = f'{config["title_suffix"]}'
            plot_dataset(ax, plot_df, config['x_range'], config['y_range'], 
                        config['y_type'], source_color_map, title,
                        config['x_label'], config['y_label'], 
                        show_points=version['show_points'], alpha=0.4)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存组合图
        output_dir = os.path.join(output_base_dir, version['name'], dataset_name, 'output')
        output_path = os.path.join(output_dir, f'{dataset_name}_all_plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✅ 组合图已保存: {output_path}")
        plt.close(fig)
    
    print(f"\n  ✅ {version['suffix']} 版本的组合图生成完成！")

# ============================================================
# 生成单独的图片文件（每个数据集4张，共32张）
# ============================================================
print("\n" + "="*60)
print("开始生成单独的图片文件（共32张）...")
print("="*60)

total_plots = 0
for version in versions:
    print(f"\n📊 生成 {version['suffix']} 版本的单独图片...")
    print("-"*40)
    
    for dataset_name, dataset_df in datasets.items():
        if dataset_df.empty:
            continue
        
        print(f"\n  处理数据集: {dataset_name}")
        
        # 筛选有效数据
        df_inclination = dataset_df.dropna(subset=[spin_col, inclination_col])
        df_distance = dataset_df.dropna(subset=[spin_col, distance_col])
        
        for config in plot_configs:
            # 选择对应的数据
            if config['y_type'] == 'inclination':
                plot_df = df_inclination
            else:
                plot_df = df_distance
            
            if plot_df.empty:
                print(f"    ⚠️ 跳过 {config['name']} (无有效数据)")
                continue
            
            # 创建单张图
            fig, ax = plt.subplots(figsize=(12, 10))
            
            title = f'{dataset_name.replace("-", " ").title()} - {config["title_suffix"]}'
            plot_dataset(ax, plot_df, config['x_range'], config['y_range'],
                        config['y_type'], source_color_map, title,
                        config['x_label'], config['y_label'], 
                        show_points=version['show_points'], alpha=0.4)
            
            # 保存图片
            output_dir = os.path.join(output_base_dir, version['name'], dataset_name, 'output')
            output_path = os.path.join(output_dir, f'{dataset_name}_{config["name"]}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    ✅ 已保存: {dataset_name}_{config['name']}.png")
            plt.close(fig)
            total_plots += 1

print("\n" + "="*60)
print("🎉 全部完成！")
print("="*60)
print(f"\n📁 文件结构:")
print(f"{output_base_dir}/")
for version in versions:
    print(f"  📁 {version['name']}/ ({version['suffix']})")
    for dataset_name in datasets.keys():
        if not datasets[dataset_name].empty:
            print(f"    📁 {dataset_name}/")
            print(f"      📁 output/ (包含该数据集的5张图片: 1张组合图 + 4张单图)")
            print(f"      📁 scripts/")
print(f"\n📊 总计生成图片数: {total_plots} 张单图 + {len(versions) * sum(1 for d in datasets.values() if not d.empty)} 张组合图 = {total_plots + len(versions) * sum(1 for d in datasets.values() if not d.empty)} 张图片")
print(f"\n图片保存位置: {os.path.abspath(output_base_dir)}")
print("="*60)

# 生成统计信息
print("\n📈 统计信息:")
for dataset_name, dataset_df in datasets.items():
    if not dataset_df.empty:
        print(f"  {dataset_name}: {len(dataset_df)} 个数据点")
print(f"  黑洞总数: {len(sources)}")
print(f"  图片版本: 2个版本 (带数据点 / 不带数据点)")
print(f"  每个版本: {sum(1 for d in datasets.values() if not d.empty)} 个数据集 × 4张图 = {sum(1 for d in datasets.values() if not d.empty) * 4} 张单图 + {sum(1 for d in datasets.values() if not d.empty)} 张组合图")
print("="*60)