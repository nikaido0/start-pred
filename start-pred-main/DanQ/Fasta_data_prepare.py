# ����Pandas�⣬���ڴ���CSV�ļ��е�����
import pandas as pd

# ����CSV�ļ�·��
csv_file_path = '../data/test.csv'

# �������������FASTA��ʽ�ļ�·��
fasta_ref_path = 'test_ref-seq.fasta'  # ���ڱ���ο�����REF_seq
fasta_alt_path = 'test_alt-seq.fasta'  # ���ڱ����������ALT_seq

# ʹ��Pandas��ȡCSV�ļ����������ݼ���ΪDataFrame
df = pd.read_csv(csv_file_path)

# ����REF_seq�е����ݵ�REF�ļ�
# 'with'������ڴ��ļ�������֤�ڲ����������Զ��ر��ļ�
with open(fasta_ref_path, 'w') as f_ref:
    # ʹ��iterrows()���б���DataFrame
    for _, row in df.iterrows():
        # ��ÿһ�е�REF_seq����д�뵽fasta_ref�ļ��У�����ӻ��з�
        f_ref.write(f'{row["REF_seq"]}\n')

# ����ALT_seq�е����ݵ�ALT�ļ�
with open(fasta_alt_path, 'w') as f_alt:
    # ���б���DataFrame
    for _, row in df.iterrows():
        # ��ÿһ�е�ALT_seq����д�뵽fasta_alt�ļ��У�����ӻ��з�
        f_alt.write(f'{row["ALT_seq"]}\n')
