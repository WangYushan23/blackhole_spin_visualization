import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色映射 - 使用柔和色系
color_map = {
    'reflection': '#E8A0A0',           # 柔和红色/粉红色
    'continuum-fitting': '#A0C8E8',     # 柔和蓝色
    'combining': '#A8D8A8',             # 柔和绿色
    'other': '#D0D0D0'                  # 柔和灰色
}

def plot_model_distribution():
    """绘制拟合模型分布的饼状图"""
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    
    # 任务一文件夹 = scripts的父目录
    task1_dir = script_dir.parent
    
    # 根目录 = 任务一文件夹的父目录
    root_dir = task1_dir.parent
    
    # data文件夹在根目录下
    data_file = root_dir / 'data' / '数据收集 表3 画图用.xlsx'
    
    # output文件夹在任务一文件夹下
    output_dir = task1_dir / 'output'
    
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
        print("    │   └── plot_pie_chart.py")
        print("    └── output/")
        return
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(data_file, sheet_name='Sheet1')
    print(f"成功读取 {len(df)} 行数据")
    
    # 清洗数据：向前填充黑洞名称
    df['源'] = df['源'].ffill()
    df = df.dropna(subset=['源'])
    
    # 统计各模型的数量
    model_column = '拟合模型'
    model_counts = df[model_column].value_counts()
    
    print("\n模型统计：")
    for model, count in model_counts.items():
        print(f"  {model}: {count} 个数据点")
    
    # 定义要单独显示的模型
    main_models = ['reflection', 'continuum-fitting', 'combining']
    
    # 分类统计
    categories = {}
    categories['reflection'] = 0
    categories['continuum-fitting'] = 0
    categories['combining'] = 0
    categories['other'] = 0
    
    for model, count in model_counts.items():
        if model in main_models:
            categories[model] = count
        else:
            categories['other'] += count
    
    # 准备饼状图数据
    labels = []
    sizes = []
    colors = []
    model_keys = []
    
    # 按照固定顺序添加
    model_order_display = [
        ('continuum-fitting', 'Continuum-fitting'),
        ('reflection', 'Reflection'),
        ('combining', 'Combining'),
        ('other', 'Other')
    ]
    
    for model_key, display_name in model_order_display:
        if categories[model_key] > 0:
            labels.append(display_name)
            sizes.append(categories[model_key])
            colors.append(color_map[model_key])
            model_keys.append(model_key)
    
    # 计算总数据点数
    total = sum(sizes)
    
    # 准备图例文本（只保留百分比）
    legend_labels = []
    for i, (label, size) in enumerate(zip(labels, sizes)):
        percentage = size/total*100
        legend_labels.append(f'{label}: {percentage:.1f}%')
    
    # ========== 生成第一张图：带图例 ==========
    print("\n生成第一张图（带图例）...")
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    
    # 绘制饼状图
    wedges1, texts1, autotexts1 = ax1.pie(
        sizes, 
        labels=None,
        colors=colors,
        autopct='',
        startangle=90,
        explode=None,
        shadow=False,
        textprops={'fontsize': 12}
    )
    
    # 在扇区上添加百分比标注
    for i, (wedge, size) in enumerate(zip(wedges1, sizes)):
        percentage = f'{size/total*100:.1f}%'
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 0.6 * np.cos(np.radians(angle))
        y = 0.6 * np.sin(np.radians(angle))
        ax1.text(x, y, percentage, ha='center', va='center', 
                fontsize=11, fontweight='bold')
    
    # 放置图例在右侧
    ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=10, frameon=False)
    
    # 设置标题
    ax1.set_title('拟合模型分布统计', fontsize=14, fontweight='bold', pad=10)
    
    # 确保饼状图是圆的
    ax1.axis('equal')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存第一张图
    output_path1 = output_dir / 'model_distribution_pie_chart_with_legend.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  已保存: {output_path1}")
    
    # ========== 生成第二张图：不带图例，所有标签都在饼图内部 ==========
    print("生成第二张图（不带图例，所有标签在饼图内部）...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # 绘制饼状图
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes, 
        labels=None,
        colors=colors,
        autopct='',
        startangle=90,
        explode=None,
        shadow=False,
        textprops={'fontsize': 12}
    )
    
    # 为每个扇区添加模型名称和百分比，都放在饼图内部
    for i, (wedge, size, label) in enumerate(zip(wedges2, sizes, labels)):
        percentage_value = size/total*100
        percentage_text = f'{percentage_value:.1f}%'
        
        # 计算扇区的中间角度
        angle = (wedge.theta2 + wedge.theta1) / 2
        
        # 根据扇区大小决定内部布局
        if percentage_value > 10:
            # 大扇区：模型名称在外圈半径0.65，百分比在内圈半径0.45
            name_radius = 0.65
            pct_radius = 0.45
        else:
            # 小扇区：两个文本都放在稍近的位置
            name_radius = 0.55
            pct_radius = 0.7
        
        # 计算模型名称位置
        name_x = name_radius * np.cos(np.radians(angle))
        name_y = name_radius * np.sin(np.radians(angle))
        
        # 计算百分比位置
        pct_x = pct_radius * np.cos(np.radians(angle))
        pct_y = pct_radius * np.sin(np.radians(angle))
        
        # 根据角度微调对齐方式，避免文字重叠
        # 水平对齐
        if abs(angle) < 80 or abs(angle - 180) < 80:
            ha_name = 'center'
            ha_pct = 'center'
        elif angle > 0:
            ha_name = 'left'
            ha_pct = 'left'
        else:
            ha_name = 'right'
            ha_pct = 'right'
        
        # 垂直对齐
        if angle > 0 and angle < 180:
            va_name = 'bottom'
            va_pct = 'bottom'
        else:
            va_name = 'top'
            va_pct = 'top'
        
        # 放置模型名称
        ax2.text(name_x, name_y, label, ha=ha_name, va=va_name, 
                fontsize=9, fontweight='bold', color='#333333')
        
        # 放置百分比
        ax2.text(pct_x, pct_y, percentage_text, ha=ha_pct, va=va_pct, 
                fontsize=10, fontweight='bold', color='#333333')
    
    # 设置标题
    ax2.set_title('拟合模型分布统计', fontsize=14, fontweight='bold', pad=10)
    
    # 确保饼状图是圆的
    ax2.axis('equal')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存第二张图
    output_path2 = output_dir / 'model_distribution_pie_chart_without_legend.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  已保存: {output_path2}")
    
    # 输出统计信息
    print(f"\n饼状图生成完成！")
    print(f"总数据点数: {total}")
    for i, (label, size) in enumerate(zip(labels, sizes)):
        print(f"  {label}: {size} 个 ({size/total*100:.1f}%)")
    print(f"\n生成的文件：")
    print(f"  1. model_distribution_pie_chart_with_legend.png (带图例)")
    print(f"  2. model_distribution_pie_chart_without_legend.png (不带图例，所有标签在饼图内部)")

def main():
    plot_model_distribution()

if __name__ == '__main__':
    main()