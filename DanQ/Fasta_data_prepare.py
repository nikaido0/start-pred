# 将 CSV 文件中的参考序列和替代序列分别提取并保存为 FASTA 格式文件

# 导入Pandas库，用于处理数据
import pandas as pd

# 定义CSV文件的路径
csv_file_path = '../data/test.csv'

# 定义FASTA格式文件的路径
fasta_ref_path = 'test_ref-seq.fasta'  # 用于存储参考序列的FASTA文件
fasta_alt_path = 'test_alt-seq.fasta'  # 用于存储替代序列的FASTA文件

# 使用Pandas读取CSV文件，并将数据存储为DataFrame
df = pd.read_csv(csv_file_path)

# 将REF_seq中的数据写入REF文件
# 使用with语句打开文件，确保文件在使用后自动关闭
with open(fasta_ref_path, 'w') as f_ref:
    # 遍历DataFrame中的每一行
    for _, row in df.iterrows():
        # 将每行中的REF_seq写入fasta_ref文件，末尾加换行符
        f_ref.write(f'{row["REF_seq"]}\n')

# 将ALT_seq中的数据写入ALT文件
with open(fasta_alt_path, 'w') as f_alt:
    # 遍历DataFrame中的每一行
    for _, row in df.iterrows():
        # 将每行中的ALT_seq写入fasta_alt文件，末尾加换行符
        f_alt.write(f'{row["ALT_seq"]}\n')
