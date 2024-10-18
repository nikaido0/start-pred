# 导入Pandas库，用于处理CSV文件中的数据
import pandas as pd

# 定义CSV文件路径
csv_file_path = '../data/test.csv'

# 定义输出的两个FASTA格式文件路径
fasta_ref_path = 'test_ref-seq.fasta'  # 用于保存参考序列REF_seq
fasta_alt_path = 'test_alt-seq.fasta'  # 用于保存替代序列ALT_seq

# 使用Pandas读取CSV文件，并将数据加载为DataFrame
df = pd.read_csv(csv_file_path)

# 保存REF_seq列的数据到REF文件
# 'with'语句用于打开文件，并保证在操作结束后自动关闭文件
with open(fasta_ref_path, 'w') as f_ref:
    # 使用iterrows()逐行遍历DataFrame
    for _, row in df.iterrows():
        # 将每一行的REF_seq数据写入到fasta_ref文件中，并添加换行符
        f_ref.write(f'{row["REF_seq"]}\n')

# 保存ALT_seq列的数据到ALT文件
with open(fasta_alt_path, 'w') as f_alt:
    # 逐行遍历DataFrame
    for _, row in df.iterrows():
        # 将每一行的ALT_seq数据写入到fasta_alt文件中，并添加换行符
        f_alt.write(f'{row["ALT_seq"]}\n')
