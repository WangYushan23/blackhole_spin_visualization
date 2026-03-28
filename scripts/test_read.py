import pandas as pd

# 注意：文件名有空格，需要用引号括起来
df = pd.read_excel('data/数据收集 表3.xlsx', sheet_name='Sheet1')
print(f"成功读取，共 {len(df)} 行数据")
print("\n列名：")
print(df.columns.tolist())
print("\n前5行数据：")
print(df.head())