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

def get_y_bounds(y_value, error_min, error_max, error_type, y_min_val=0, y_max_val=90):
    """计算 y 方向的边界（下边界，上边界）"""
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

# 只保留有自旋值和倾角的数据
df = df.dropna(subset=[spin_col, inclination_col])
print(f"清洗后剩余 {len(df)} 行数据")

# 转换数值类型
df[spin_col] = pd.to_numeric(df[spin_col], errors='coerce')
df[spin_err_min_col] = pd.to_numeric(df[spin_err_min_col], errors='coerce')
df[spin_err_max_col] = pd.to_numeric(df[spin_err_max_col], errors='coerce')
df[inclination_col] = pd.to_numeric(df[inclination_col], errors='coerce')
df[inclination_err_min_col] = pd.to_numeric(df[inclination_err_min_col], errors='coerce')
df[inclination_err_max_col] = pd.to_numeric(df[inclination_err_max_col], errors='coerce')

# 删除无效数据
df = df.dropna(subset=[spin_col, inclination_col])

# 获取所有黑洞名称
sources = df['源'].unique()
print(f"共发现 {len(sources)} 个黑洞")

# 为每个源分配固定颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(sources)))
source_color_map = {source: colors[i] for i, source in enumerate(sources)}

# 创建图形（第一张图：自旋值 vs 倾角）
print("\n正在绘制第一张图：自旋值 vs 倾角...")
fig1, ax1 = plt.subplots(figsize=(14, 10))

# 存储所有矩形的 zorder，用于控制绘制顺序
rectangles = []

# 绘制数据点（先收集所有矩形信息）
for source in sources:
    source_df = df[df['源'] == source]
    color = source_color_map[source]
    
    for idx, row in source_df.iterrows():
        a_star = row[spin_col]
        i_value = row[inclination_col]
        a_err_min = row[spin_err_min_col]
        a_err_max = row[spin_err_max_col]
        i_err_min = row[inclination_err_min_col]
        i_err_max = row[inclination_err_max_col]
        
        # 判断误差类型
        a_error_type = get_error_type(a_err_min, a_err_max)
        i_error_type = get_error_type(i_err_min, i_err_max)
        
        # 计算边界
        x_left, x_right = get_x_bounds(a_star, a_err_min, a_err_max, a_error_type)
        y_bottom, y_top = get_y_bounds(i_value, i_err_min, i_err_max, i_error_type, y_min_val=0, y_max_val=90)
        
        # 确保边界有效
        width = x_right - x_left
        height = y_top - y_bottom
        
        if width > 0 and height > 0:
            # 绘制填充矩形（长方形/正方形）
            rect = plt.Rectangle((x_left, y_bottom), width, height,
                                 fill=True, alpha=0.4, color=color, 
                                 linewidth=1, edgecolor=color, linestyle='-')
            ax1.add_patch(rect)
            rectangles.append((rect, color, a_star, i_value, width, height))
        
        # 绘制中心点
        ax1.scatter(a_star, i_value, color=color, s=50, marker='o', 
                   zorder=5, edgecolor='black', linewidth=0.5)

# 设置坐标轴范围
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(0, 100)

# 设置标签
ax1.set_xlabel(r'自旋值 $a_*$', fontsize=14, fontweight='bold')
ax1.set_ylabel(r'倾角 $i$ (度)', fontsize=14, fontweight='bold')

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# 设置标题
ax1.set_title('黑洞自旋值与倾角关系图（测试版）', fontsize=16, fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存第一张图
output_dir = '任务二 所有数据总图/output'
os.makedirs(output_dir, exist_ok=True)
output_path1 = os.path.join(output_dir, 'test_spin_vs_inclination.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✅ 第一张图已保存到: {output_path1}")

# 显示第一张图
plt.show()

print("\n测试完成！")