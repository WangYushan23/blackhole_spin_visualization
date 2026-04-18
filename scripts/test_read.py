import pandas as pd

# 读取桌面上的新数据文件（使用正确的用户名：王雨珊）
df = pd.read_excel(r'C:\Users\王雨珊\Desktop\数据收集 表3 画图用.xlsx', sheet_name='Sheet1')

print("=" * 50)
print("文件读取成功！")
print("=" * 50)
print(f"\n数据行数：{len(df)} 行")
print(f"\n列名列表：")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")
print(f"\n前3行数据预览:")
print(df.head(3))
print("\n数据类型:")
print(df.dtypes)