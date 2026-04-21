import pandas as pd
import numpy as np
import os

# 设置随机种子，确保结果可重复
np.random.seed(42)

def is_range_value(error_min, error_max):
    """判断是否为范围值（如 <0.8 或 >0.9）"""
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
    
    # 范围值：< 类型 或 > 类型
    if is_min_nan and (error_max == 0 or is_max_nan):
        return True
    if is_max_nan and (error_min == 0 or is_min_nan):
        return True
    return False

def sample_from_distribution(mean, error_min, error_max, n_samples):
    """
    从给定分布中抽样
    使用正态分布，误差范围作为 1-sigma 范围
    """
    # 计算标准差（不对称误差取平均）
    if not pd.isna(error_min) and not pd.isna(error_max):
        sigma = (abs(error_min) + abs(error_max)) / 2
    elif not pd.isna(error_min):
        sigma = abs(error_min)
    elif not pd.isna(error_max):
        sigma = abs(error_max)
    else:
        sigma = abs(mean) * 0.1
    
    # 避免 sigma 为 0
    if sigma == 0:
        sigma = 0.01
    
    # 从正态分布抽样
    samples = np.random.normal(mean, sigma, n_samples)
    
    # 截断到 [-1, 1] 范围（自旋值的物理范围）
    samples = np.clip(samples, -1, 1)
    
    return samples

def calculate_weights(source_df, confidence_col='自旋值i 置信度(σ)', default_weight=1.0):
    """
    根据置信度计算每条记录的抽样权重
    缺失置信度的数据使用默认权重 1
    """
    # 获取置信度列
    confidences = source_df[confidence_col].values
    
    # 处理缺失值：将 NaN 替换为默认权重
    confidences = np.array([default_weight if pd.isna(c) else c for c in confidences])
    
    # 确保权重为正数
    confidences = np.maximum(confidences, 0.01)
    
    return confidences

def simulate_source_weighted(source_df, source_name, total_samples=1000):
    """
    加权蒙特卡洛模拟（考虑置信度）
    返回字典：{'all': samples, 'reflection': samples, ...}
    """
    # 定义四种模拟类型
    simulation_types = {
        'all': source_df,  # 所有数据
        'reflection': source_df[source_df['拟合模型'] == 'reflection'],
        'continuum-fitting': source_df[source_df['拟合模型'] == 'continuum-fitting'],
        'combining': source_df[source_df['拟合模型'] == 'combining']
    }
    
    results = {}
    
    for sim_name, sim_df in simulation_types.items():
        if len(sim_df) == 0:
            print(f"  警告: {source_name} 没有 {sim_name} 类型的数据，跳过")
            results[sim_name] = np.array([])
            continue
        
        # 过滤掉范围值（< 或 > 类型）
        valid_df = sim_df.copy()
        valid_df['is_range'] = valid_df.apply(
            lambda row: is_range_value(row['自旋值i -'], row['自旋值i +']), 
            axis=1
        )
        valid_df = valid_df[~valid_df['is_range']]
        
        if len(valid_df) == 0:
            print(f"  警告: {source_name} 的 {sim_name} 类型数据全是范围值，跳过")
            results[sim_name] = np.array([])
            continue
        
        # 计算每条记录的权重（基于置信度）
        weights = calculate_weights(valid_df, confidence_col='自旋值i 置信度(σ)', default_weight=1.0)
        
        # 归一化权重，计算每条记录的抽样数量
        sample_counts = (weights / weights.sum() * total_samples).astype(int)
        
        # 处理取整误差：将剩余样本分配给权重最大的记录
        remainder = total_samples - sample_counts.sum()
        if remainder > 0:
            # 找到权重最大的记录（如果有多个，取第一个）
            max_idx = np.argmax(weights)
            sample_counts[max_idx] += remainder
        
        # 存储所有抽样结果
        all_samples = []
        
        for idx, (_, row) in enumerate(valid_df.iterrows()):
            n = sample_counts[idx]
            if n <= 0:
                continue
            
            mean = row['自旋值i']
            error_min = row['自旋值i -']
            error_max = row['自旋值i +']
            
            samples = sample_from_distribution(mean, error_min, error_max, n)
            all_samples.extend(samples)
        
        results[sim_name] = np.array(all_samples)
    
    return results

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 数据文件路径：从 scripts 文件夹向上两级到根目录，再进入 data 文件夹
file_path = os.path.join(script_dir, '../../data/数据收集 表3 蒙特卡洛模拟用.xlsx')

print("=" * 60)
print("蒙特卡洛模拟程序（考虑置信度权重）")
print("=" * 60)

print("\n正在读取数据...")
print(f"数据文件路径: {file_path}")
df = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"成功读取 {len(df)} 行数据")

# 清洗数据：向前填充黑洞名称
df['源'] = df['源'].ffill()
df = df.dropna(subset=['源'])

# 转换数值类型
df['自旋值i'] = pd.to_numeric(df['自旋值i'], errors='coerce')
df['自旋值i -'] = pd.to_numeric(df['自旋值i -'], errors='coerce')
df['自旋值i +'] = pd.to_numeric(df['自旋值i +'], errors='coerce')
df['自旋值i 置信度(σ)'] = pd.to_numeric(df['自旋值i 置信度(σ)'], errors='coerce')

# 删除自旋值为空的行
df = df.dropna(subset=['自旋值i'])

# 获取所有黑洞名称
sources = df['源'].unique()
print(f"共发现 {len(sources)} 个黑洞")

# 设置总抽样数量
TOTAL_SAMPLES = 1000

# 创建输出目录（在任务三文件夹下的 output）
output_dir = os.path.join(script_dir, '../output')
os.makedirs(output_dir, exist_ok=True)

# 对每个源进行模拟
print("\n开始蒙特卡洛模拟...")
print(f"每个源每次模拟抽样 {TOTAL_SAMPLES} 个点")
print("权重分配：基于置信度（缺失置信度默认权重为 1）\n")

for source in sources:
    print(f"处理黑洞: {source}")
    source_df = df[df['源'] == source].copy()
    
    # 显示该源的置信度信息（用于调试）
    conf_values = source_df['自旋值i 置信度(σ)'].values
    n_missing = sum(1 for c in conf_values if pd.isna(c))
    if n_missing > 0:
        print(f"  注意: 有 {n_missing} 条记录置信度缺失，使用默认权重 1")
    
    # 模拟
    results = simulate_source_weighted(source_df, source, TOTAL_SAMPLES)
    
    # 保存结果到 CSV 文件
    safe_name = source.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_').replace(' ', '')
    
    for sim_type, samples in results.items():
        if len(samples) == 0:
            continue
        
        # 创建 DataFrame
        result_df = pd.DataFrame({
            '模拟值': samples,
            '模拟类型': sim_type,
            '源': source
        })
        
        # 保存到 CSV
        filename = f"{safe_name}_{sim_type}.csv"
        filepath = os.path.join(output_dir, filename)
        result_df.to_csv(filepath, index=False)
        print(f"  ✓ 已保存: {filename} ({len(samples)} 个样本)")
    
    print()

print("\n" + "=" * 60)
print("蒙特卡洛模拟完成！")
print(f"结果保存在: {os.path.abspath(output_dir)}")
print("=" * 60)

# 生成汇总统计
print("\n生成汇总统计...")

summary_data = []
for source in sources:
    safe_name = source.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('\n', '_').replace(' ', '')
    
    for sim_type in ['all', 'reflection', 'continuum-fitting', 'combining']:
        filename = f"{safe_name}_{sim_type}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            df_sample = pd.read_csv(filepath)
            samples = df_sample['模拟值']
            
            # 计算统计量
            mean_val = samples.mean()
            median_val = samples.median()
            std_val = samples.std()
            min_val = samples.min()
            max_val = samples.max()
            p5 = np.percentile(samples, 5)
            p95 = np.percentile(samples, 95)
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
            
            summary_data.append({
                '源': source,
                '模拟类型': sim_type,
                '样本数': len(samples),
                '均值': mean_val,
                '中位数': median_val,
                '标准差': std_val,
                '最小值': min_val,
                '最大值': max_val,
                '5%分位数': p5,
                '95%分位数': p95,
                '95%置信区间下限': ci_lower,
                '95%置信区间上限': ci_upper
            })

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_dir, '汇总统计.csv')
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(f"✓ 汇总统计已保存: 汇总统计.csv")

# 额外生成一个权重分配说明文件
weight_info = pd.DataFrame({
    '说明': [
        '本模拟采用加权蒙特卡洛方法',
        '权重依据：自旋值i 置信度(σ) 列',
        '缺失置信度的数据使用默认权重 1',
        '每个源每种模拟类型总抽样数: 1000',
        '抽样数量与置信度成正比（置信度越高，抽样越多）'
    ]
})
weight_info_path = os.path.join(output_dir, '权重分配说明.csv')
weight_info.to_csv(weight_info_path, index=False, encoding='utf-8-sig')
print(f"✓ 权重分配说明已保存: 权重分配说明.csv")

print("\n全部完成！")