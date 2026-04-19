import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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

def get_x_bounds(a_star, error_min, error_max, error_type):
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
        # < 类型：向左延伸到 -1
        left = -1
        right = a_star
    elif error_type == 'greater_than':
        # > 类型：向右延伸到 1
        left = a_star
        right = 1
    else:
        left = a_star
        right = a_star
    
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
        # < 类型：向下延伸到 y_min_val
        bottom = y_min_val
        top = y_value
    elif error_type == 'greater_than':
        # > 类型：向上延伸到 y_max_val
        bottom = y_value
        top = y_max_val
    else:
        bottom = y_value
        top = y_value
    
    # 确保边界在合理范围内
    bottom = max(bottom, y_min_val)
    top = min(top, y_max_val)
    
    return bottom, top

def get_y_bounds_distance(y_value, error_min, error_max, error_type):
    """计算距离方向的边界（下边界，上边界）- 自动适应数据范围"""
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
        # < 类型：向下延伸到 0
        bottom = 0
        top = y_value
    elif error_type == 'greater_than':
        # > 类型：向上延伸，上限根据数据自动调整
        bottom = y_value
        top = y_value + error_max if not pd.isna(error_max) else y_value
    else:
        bottom = y_value
        top = y_value
    
    # 确保下边界不为负数
    bottom = max(bottom, 0)
    
    return bottom, top

# 读取数据
print("正在读取数据...")
file_path = 'data/数据收集 表2 画图用.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"成功读取 {len(df)} 行数据")

# 定义列名（根据实际列名）
spin_col = '自旋值i'
spin_err_min_col = '自旋值i -'
spin_err_max_col = '自旋值i +'
inclination_col = '倾角i（deg）'
inclination_err_min_col = '倾角i（deg）-'
inclination_err_max_col = '倾角i（deg）+'
distance_col = '距离d(kpc)'
distance_err_min_col = '距离d(kpc)-'
distance_err_max_col = '距离d(kpc)+'

# 清洗数据：向前填充黑洞名称
df['源'] = df['源'].ffill()
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

# 获取所有黑洞名称
sources = df['源'].unique()
print(f"共发现 {len(sources)} 个黑洞")

# 为每个源分配固定颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(sources)))
source_color_map = {source: colors[i] for i, source in enumerate(sources)}

# ============================================================
# 第一张图：自旋值 vs 倾角
# ============================================================
print("\n正在绘制第一张图：自旋值 vs 倾角...")

# 筛选有倾角数据的数据点
df_inclination = df.dropna(subset=[spin_col, inclination_col])
print(f"  倾角数据点: {len(df_inclination)} 个")

fig1, ax1 = plt.subplots(figsize=(14, 10))

for source in sources:
    source_df = df_inclination[df_inclination['源'] == source]
    color = source_color_map[source]
    
    for idx, row in source_df.iterrows():
        a_star = row[spin_col]
        y_value = row[inclination_col]
        a_err_min = row[spin_err_min_col]
        a_err_max = row[spin_err_max_col]
        y_err_min = row[inclination_err_min_col]
        y_err_max = row[inclination_err_max_col]
        
        # 判断误差类型
        a_error_type = get_error_type(a_err_min, a_err_max)
        y_error_type = get_error_type(y_err_min, y_err_max)
        
        # 计算边界
        x_left, x_right = get_x_bounds(a_star, a_err_min, a_err_max, a_error_type)
        y_bottom, y_top = get_y_bounds_inclination(y_value, y_err_min, y_err_max, y_error_type, y_min_val=0, y_max_val=90)
        
        # 确保边界有效
        width = x_right - x_left
        height = y_top - y_bottom
        
        if width > 0 and height > 0:
            # 绘制填充矩形
            rect = plt.Rectangle((x_left, y_bottom), width, height,
                                 fill=True, alpha=0.4, facecolor=color, 
                                 linewidth=1, edgecolor=color, linestyle='-')
            ax1.add_patch(rect)
        
        # 绘制中心点
        ax1.scatter(a_star, y_value, color=color, s=50, marker='o', 
                   zorder=5, edgecolor='black', linewidth=0.5)

# 设置坐标轴范围（倾角固定 0-90 度）
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(0, 100)

# 设置标签
ax1.set_xlabel(r'自旋值 $a_*$', fontsize=14, fontweight='bold')
ax1.set_ylabel(r'倾角 $i$ (度)', fontsize=14, fontweight='bold')

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# 设置标题
ax1.set_title('黑洞自旋值与倾角关系图', fontsize=16, fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存第一张图
output_dir = '任务二 所有数据总图/output'
os.makedirs(output_dir, exist_ok=True)
output_path1 = os.path.join(output_dir, 'spin_vs_inclination.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✅ 第一张图已保存到: {output_path1}")
plt.close(fig1)

# ============================================================
# 第二张图：自旋值 vs 距离（自动调整 y 轴范围）
# ============================================================
print("\n正在绘制第二张图：自旋值 vs 距离...")

# 筛选有距离数据的数据点
df_distance = df.dropna(subset=[spin_col, distance_col])
print(f"  距离数据点: {len(df_distance)} 个")

# 计算距离数据的实际范围（用于设置 y 轴）
all_distance_values = []
for _, row in df_distance.iterrows():
    y_value = row[distance_col]
    y_err_min = row[distance_err_min_col]
    y_err_max = row[distance_err_max_col]
    y_error_type = get_error_type(y_err_min, y_err_max)
    
    all_distance_values.append(y_value)
    
    if y_error_type == 'normal':
        if not pd.isna(y_err_min):
            all_distance_values.append(y_value - y_err_min)
        if not pd.isna(y_err_max):
            all_distance_values.append(y_value + y_err_max)
    elif y_error_type == 'less_than':
        all_distance_values.append(0)
    elif y_error_type == 'greater_than':
        if not pd.isna(y_err_max):
            all_distance_values.append(y_value + y_err_max)

distance_min = min(all_distance_values)
distance_max = max(all_distance_values)

# 添加 10% 的边距
y_margin = (distance_max - distance_min) * 0.1
y_min = max(0, distance_min - y_margin)
y_max = distance_max + y_margin

print(f"  距离范围: {y_min:.1f} - {y_max:.1f} kpc")

fig2, ax2 = plt.subplots(figsize=(14, 10))

for source in sources:
    source_df = df_distance[df_distance['源'] == source]
    color = source_color_map[source]
    
    for idx, row in source_df.iterrows():
        a_star = row[spin_col]
        y_value = row[distance_col]
        a_err_min = row[spin_err_min_col]
        a_err_max = row[spin_err_max_col]
        y_err_min = row[distance_err_min_col]
        y_err_max = row[distance_err_max_col]
        
        # 判断误差类型
        a_error_type = get_error_type(a_err_min, a_err_max)
        y_error_type = get_error_type(y_err_min, y_err_max)
        
        # 计算边界
        x_left, x_right = get_x_bounds(a_star, a_err_min, a_err_max, a_error_type)
        y_bottom, y_top = get_y_bounds_distance(y_value, y_err_min, y_err_max, y_error_type)
        
        # 确保边界有效
        width = x_right - x_left
        height = y_top - y_bottom
        
        if width > 0 and height > 0:
            # 绘制填充矩形
            rect = plt.Rectangle((x_left, y_bottom), width, height,
                                 fill=True, alpha=0.4, facecolor=color, 
                                 linewidth=1, edgecolor=color, linestyle='-')
            ax2.add_patch(rect)
        
        # 绘制中心点
        ax2.scatter(a_star, y_value, color=color, s=50, marker='o', 
                   zorder=5, edgecolor='black', linewidth=0.5)

# 设置坐标轴范围（距离自动适应数据范围）
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(y_min, y_max)

# 设置标签
ax2.set_xlabel(r'自旋值 $a_*$', fontsize=14, fontweight='bold')
ax2.set_ylabel(r'距离 $d$ (kpc)', fontsize=14, fontweight='bold')

# 添加网格
ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# 设置标题
ax2.set_title('黑洞自旋值与距离关系图', fontsize=16, fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存第二张图
output_path2 = os.path.join(output_dir, 'spin_vs_distance.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ 第二张图已保存到: {output_path2}")
plt.close(fig2)

print("\n全部完成！")
print(f"图片保存位置: {os.path.abspath(output_dir)}")