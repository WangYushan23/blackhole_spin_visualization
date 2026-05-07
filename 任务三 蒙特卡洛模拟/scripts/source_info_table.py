"""
黑洞源信息汇总表 - 最终版
功能：读取数据表格，生成每个源在不同模型下的自旋值汇总表
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径设置 ====================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / '数据收集 表3 蒙特卡洛模拟用.xlsx'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

print(f"数据文件路径: {DATA_PATH}")
print(f"数据文件存在: {DATA_PATH.exists()}")
print(f"输出目录: {OUTPUT_DIR}\n")

# ==================== 读取数据 ====================
df = pd.read_excel(DATA_PATH, sheet_name='Sheet1', usecols='B:I')
df.columns = ['源', '文献来源', '自旋值i', '自旋值i-', '自旋值i+', '置信度_sigma', '拟合模型', '爆发时间']
df = df.dropna(subset=['自旋值i'])

print(f"成功读取数据，共 {len(df)} 条记录")

# ==================== 格式化自旋值字符串 ====================
def format_spin_value(row):
    spin = row['自旋值i']
    err_minus = row['自旋值i-']
    err_plus = row['自旋值i+']
    
    if abs(spin) < 0.1:
        decimal_places = 4
    elif abs(spin) < 1:
        decimal_places = 3
    else:
        decimal_places = 2
    
    spin_str = f"{spin:.{decimal_places}f}"
    err_plus_str = f"{err_plus:.{decimal_places}f}"
    err_minus_str = f"{err_minus:.{decimal_places}f}"
    
    return f"{spin_str} (+{err_plus_str}) (-{err_minus_str})"

df['formatted_spin'] = df.apply(format_spin_value, axis=1)

# ==================== 按源和模型汇总 ====================
sources = sorted(df['源'].unique())
models = ['reflection', 'continuum-fitting']
model_names = {'reflection': 'Reflection 模型', 'continuum-fitting': 'Continuum-fitting 模型'}

# 构建表格数据
table_data = []
row_heights = []

for src in sources:
    src_df = df[df['源'] == src]
    
    for model in models:
        model_df = src_df[src_df['拟合模型'] == model]
        if len(model_df) > 0:
            spin_values = '\n'.join(model_df['formatted_spin'].tolist())
            line_count = len(model_df['formatted_spin'].tolist())
        else:
            spin_values = '-'
            line_count = 1
        
        table_data.append([src, model_names[model], spin_values])
        # 行高：基础0.4 + 每多一行增加0.25
        row_heights.append(0.4 + (line_count - 1) * 0.25)

# ==================== 计算总高度 ====================
total_height = sum(row_heights) + 0.8

# ==================== 创建图形 ====================
fig, ax = plt.subplots(figsize=(14, max(6, total_height)))
ax.axis('off')

# 创建表格（不合并单元格，通过修改文字来实现合并效果）
table = ax.table(
    cellText=table_data,
    colLabels=['源名称', '模型', '自旋值测量结果'],
    cellLoc='center',
    loc='center',
    colWidths=[0.22, 0.24, 0.54]
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.2)

# 表头样式
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=10)
    table[(0, i)].set_edgecolor('#CCCCCC')

# 设置所有数据单元格样式
for i in range(1, len(table_data) + 1):
    for j in range(3):
        table[(i, j)].set_facecolor('#FFFFFF')
        table[(i, j)].set_edgecolor('#CCCCCC')
        if j == 2:
            table[(i, j)].set_text_props(ha='left', va='top')
        else:
            table[(i, j)].set_text_props(ha='center', va='center')
    
    # 设置行高
    table[(i, 0)].set_height(row_heights[i-1])
    table[(i, 1)].set_height(row_heights[i-1])
    table[(i, 2)].set_height(row_heights[i-1])

# ==================== 合并源名称列（通过清空重复单元格的文字） ====================
current_src = None
for i in range(1, len(table_data) + 1):
    src = table_data[i-1][0]
    if src == current_src:
        # 清空重复的源名称文字
        table[(i, 0)].get_text().set_text('')
        # 设置这些单元格的垂直对齐为居中，并隐藏边框
        table[(i, 0)].set_facecolor('#FFFFFF')
        # 设置该单元格的边距
        table[(i, 0)].set_text_props(ha='center', va='center')
    else:
        current_src = src
        # 第一行的源名称设置垂直居中
        table[(i, 0)].set_text_props(ha='center', va='center')

# 设置标题
plt.title('各黑洞源自旋值测量结果汇总表', fontsize=16, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()

# 保存图片
save_path = OUTPUT_DIR / 'source_info_table.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"表格图片已保存: {save_path}")

# 不显示窗口，直接关闭
plt.close()
print("\n表格生成完成！请查看 output 文件夹中的 source_info_table.png")