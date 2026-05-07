"""
黑洞自旋值蒙特卡洛模拟
功能：对不同文献的黑洞自旋测量值进行高斯抽样，生成统计分布图
作者：黑洞研究专家
日期：2026-04-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import lcm
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
BASE_MULTIPLIER = 100
BINS = np.arange(-1.0, 1.01, 0.1)
CONF_TO_Z = {1.0: 1.0, 1.645: 1.645, 2.576: 2.576, 3.0: 3.0}

# 统一纵坐标范围（根据 Reflection 模型最高约 225,346，设置为 250,000 留有余量）
Y_LIMIT = 250000

np.random.seed(42)

# ==================== 路径设置 ====================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / '数据收集 表3 蒙特卡洛模拟用.xlsx'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

# ==================== 清理旧的输出文件 ====================
if OUTPUT_DIR.exists():
    print(f"检测到旧的输出文件夹，正在清理...")
    shutil.rmtree(OUTPUT_DIR)
    print(f"已清理旧文件")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"已创建新的输出文件夹: {OUTPUT_DIR}\n")

print(f"数据文件路径: {DATA_PATH}")
print(f"数据文件存在: {DATA_PATH.exists()}")

# ==================== 读取数据 ====================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

df = pd.read_excel(DATA_PATH, sheet_name='Sheet1', usecols='B:I')
df.columns = ['源', '文献来源', '自旋值i', '自旋值i-', '自旋值i+', '置信度_sigma', '拟合模型', '爆发时间']
df = df.dropna(subset=['自旋值i'])

print(f"成功读取数据，共 {len(df)} 条记录")
print(f"黑洞源列表: {df['源'].unique()}")
print(f"拟合模型类型: {df['拟合模型'].unique()}")

# ==================== 计算 sigma_dist ====================
def calc_sigma(row):
    e_min = min(row['自旋值i-'], row['自旋值i+'])
    k = row['置信度_sigma']
    z = CONF_TO_Z.get(k, 1.0)
    return e_min / z

df['sigma_dist'] = df.apply(calc_sigma, axis=1)

print("\nsigma_dist计算示例：")
print(df[['源', '自旋值i', '自旋值i-', '自旋值i+', '置信度_sigma', 'sigma_dist']].head(10))

# ==================== 计算全局 L_global ====================
sources = df['源'].unique()
source_stats = {}

for src in sources:
    src_df = df[df['源'] == src]
    n_all = len(src_df)
    n_cf = len(src_df[src_df['拟合模型'] == 'continuum-fitting'])
    n_ref = len(src_df[src_df['拟合模型'] == 'reflection'])
    
    nums = [n for n in (n_all, n_cf, n_ref) if n > 0]
    if len(nums) == 1:
        L_i = nums[0]
    else:
        L_i = 1
        for num in nums:
            L_i = lcm(L_i, num)
    
    source_stats[src] = {'n_all': n_all, 'n_cf': n_cf, 'n_ref': n_ref, 'L_i': L_i}

L_list = [stats['L_i'] for stats in source_stats.values()]
L_global = 1
for L_val in L_list:
    L_global = lcm(L_global, L_val)

print(f"\n各源统计:")
for src, stats in source_stats.items():
    print(f"  {src}: 全={stats['n_all']}, CF={stats['n_cf']}, REF={stats['n_ref']}, L_i={stats['L_i']}")
print(f"\nL_global = {L_global}")
print(f"每个数据集总样本数 = {L_global} × {BASE_MULTIPLIER} = {L_global * BASE_MULTIPLIER}")

# ==================== 抽样 ====================
total_samples_per_dataset = L_global * BASE_MULTIPLIER

samples_all = []
samples_cf = []
samples_ref = []

print("\n开始蒙特卡洛抽样...")
for src, stats in source_stats.items():
    src_df = df[df['源'] == src]
    
    all_df = src_df
    cf_df = src_df[src_df['拟合模型'] == 'continuum-fitting']
    ref_df = src_df[src_df['拟合模型'] == 'reflection']
    
    samples_per_row_all = total_samples_per_dataset // stats['n_all'] if stats['n_all'] > 0 else 0
    samples_per_row_cf = total_samples_per_dataset // stats['n_cf'] if stats['n_cf'] > 0 else 0
    samples_per_row_ref = total_samples_per_dataset // stats['n_ref'] if stats['n_ref'] > 0 else 0
    
    def draw_samples(df_sub, samples_per_row):
        if df_sub.empty or samples_per_row == 0:
            return []
        samples = []
        for _, row in df_sub.iterrows():
            mu = row['自旋值i']
            sigma = row['sigma_dist']
            s = np.random.normal(mu, sigma, samples_per_row)
            s = np.clip(s, -1, 1)
            samples.extend(s)
        return samples
    
    s_all = draw_samples(all_df, samples_per_row_all)
    s_cf = draw_samples(cf_df, samples_per_row_cf)
    s_ref = draw_samples(ref_df, samples_per_row_ref)
    
    samples_all.extend(s_all)
    samples_cf.extend(s_cf)
    samples_ref.extend(s_ref)
    
    print(f"  {src}: 全={len(s_all)}, CF={len(s_cf)}, REF={len(s_ref)}")

print(f"\n抽样完成！")
print(f"  全数据集总样本数: {len(samples_all)}")
print(f"  Continuum-fitting总样本数: {len(samples_cf)}")
print(f"  Reflection总样本数: {len(samples_ref)}")

# ==================== 绘图 ====================
def plot_single_hist(samples, title, filename):
    """绘制单个柱状图并保存（只有一张图，没有其他子图）"""
    counts, _ = np.histogram(samples, bins=BINS)
    bin_centers = (BINS[:-1] + BINS[1:]) / 2
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(bin_centers, counts, width=0.08, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlim(-1, 1)
    plt.ylim(0, Y_LIMIT)
    plt.xlabel('黑洞自旋值 a', fontsize=14)
    plt.ylabel('MC预测值', fontsize=14)  # 修改为"MC预测值"
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 在柱子上方添加数值标签
    for center, count in zip(bin_centers, counts):
        if count > 0:
            plt.text(center, count + max(counts) * 0.01, str(int(count)), 
                    ha='center', va='bottom', fontsize=9)
    
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {save_path}")

# 生成三张独立图片（每张只有一个柱状图，纵坐标统一）
plot_single_hist(samples_all, '所有测量数据（所有源）', 'spin_distribution_all.png')
plot_single_hist(samples_cf, '仅 Continuum-fitting 模型', 'spin_distribution_cf.png')
plot_single_hist(samples_ref, '仅 Reflection 模型', 'spin_distribution_ref.png')

# 生成三图并排的对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def plot_on_ax(samples, title, ax):
    """在指定坐标轴上绘制柱状图"""
    counts, _ = np.histogram(samples, bins=BINS)
    bin_centers = (BINS[:-1] + BINS[1:]) / 2
    ax.bar(bin_centers, counts, width=0.08, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, Y_LIMIT)
    ax.set_xlabel('黑洞自旋值 a', fontsize=12)
    ax.set_ylabel('MC预测值', fontsize=12)  # 修改为"MC预测值"
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 添加数值标签（可选）
    for center, count in zip(bin_centers, counts):
        if count > 0:
            ax.text(center, count + max(counts) * 0.01, str(int(count)), 
                   ha='center', va='bottom', fontsize=8)

plot_on_ax(samples_all, '所有测量数据', axes[0])
plot_on_ax(samples_cf, 'Continuum-fitting 模型', axes[1])
plot_on_ax(samples_ref, 'Reflection 模型', axes[2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'spin_distribution_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"图片已保存: {OUTPUT_DIR / 'spin_distribution_combined.png'}")

print(f"\n所有图片已保存到: {OUTPUT_DIR}")
print("  1. spin_distribution_all.png - 所有测量数据")
print("  2. spin_distribution_cf.png - Continuum-fitting模型")
print("  3. spin_distribution_ref.png - Reflection模型")
print("  4. spin_distribution_combined.png - 三图对比")

# ==================== 统计信息 ====================
def print_statistics(samples, name):
    if len(samples) == 0:
        print(f"\n{name} 无数据")
        return
    print(f"\n{name} 统计信息:")
    print(f"  样本总数: {len(samples)}")
    print(f"  均值: {np.mean(samples):.4f}")
    print(f"  中位数: {np.median(samples):.4f}")
    print(f"  标准差: {np.std(samples):.4f}")
    print(f"  68% CI: [{np.percentile(samples, 16):.4f}, {np.percentile(samples, 84):.4f}]")

print_statistics(samples_all, "所有数据")
print_statistics(samples_cf, "Continuum-fitting")
print_statistics(samples_ref, "Reflection")

# ==================== 保存抽样结果 ====================
pd.DataFrame({'samples_all': samples_all}).to_csv(OUTPUT_DIR / 'sampled_spins_all.csv', index=False)
pd.DataFrame({'samples_cf': samples_cf}).to_csv(OUTPUT_DIR / 'sampled_spins_cf.csv', index=False)
pd.DataFrame({'samples_ref': samples_ref}).to_csv(OUTPUT_DIR / 'sampled_spins_ref.csv', index=False)
print(f"\n抽样数据已保存到: {OUTPUT_DIR}")

print("\n蒙特卡洛模拟完成！")