#!/usr/bin/env python
#coding=gbk

import csv
import os
import time
import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import estimate
from config import sta_config
from models.StartPred import StartPred
from train import DataTrain, predict, CosineScheduler

torch.manual_seed(20230226)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bases='ATCG'


def one_hot_encode_dna(sequence, bases='ATCG', seq_length=1001):
    """
    ��DNA���н��ж��ȱ���
    sequence: �б�����DNA���е��ַ���  
    bases: �ַ���������DNA������ַ���  
    seq_length: ��������ʾ���ȱ�������г���
    """
    # ���ÿ�����еĳ����Ƿ���seq_lengthһ��  
    for seq in sequence:  
        if len(seq) != seq_length:  
            raise ValueError(f"���г��ȴ����������еĳ��ȱ������{seq_length}")
                
    one_hot = np.zeros((len(sequence), seq_length, len(bases)), dtype=np.float32)

    for i, seq in enumerate(sequence):
        for j, base in enumerate(seq):
            if base in bases:  # ȷ�������Ԥ�����bases��
                one_hot[i, j, bases.index(base)] = 1
            else:
                # �����������bases�еļ��������ѡ����Ի��׳�����
                # ����ѡ�����
                pass
                
    return one_hot


# ���ڼ��������ļ��ʹ���ǩ�������ļ�
# ����DNA����ģ�͵������ļ�
# �������ļ�Ϊ.pth�ļ�����ʹ��torch.load���������ļ�Ϊ.npy�ļ�����ʹ��np.load
def getSequenceData(direction, chrom_direction, label_direction):
    # ����ļ���չ�����������ط���
    if direction.endswith('.pth'):
        data = torch.load(direction)
    elif direction.endswith('.npy'):
        data = torch.from_numpy(np.load(direction)).float()  # ����.npy�ļ��е������Ǹ����͵�
    else:
        raise ValueError(f"Unsupported file format: {direction}")

# ����ͻ�����м�������ǩ����
    Frame = pd.read_csv(label_direction)
    sequence = Frame["ALT_seq"].values #ͻ������
    label = torch.tensor(Frame["Label"].values, dtype=torch.long) #����ǩ���ļ��С�Label����Ϊ��ǩ����
    
# ��DNA���н��б���
    #���ȱ���   
    one_hot_sequences = one_hot_encode_dna(sequence)
    #�����ȱ��������ת��ΪPyTorch����  
    one_hot_sequences_tensor = torch.tensor(one_hot_sequences, dtype=torch.float32)
    

# ����Ⱦɫ�������������
    chrom = h5py.File(chrom_direction, 'r')
    alt = np.array(chrom['feat_alt'])
    chrom_fea = torch.tensor(alt)

    return data, one_hot_sequences_tensor, chrom_fea, label


def data_load(train_direction, chrom_train_direction, train_label_direction, test_direction, chrom_test_direction, test_label_direction, batch, encode='embedding', cv=True, SH=True):

    dataset_train, dataset_test = [], []
    dataset_va = None
    assert encode in ['embedding', 'sequence'], 'There is no such representation!!!'

    if cv:
        dataset_va = []
        encode_data, sequence_data, chrom_data, encode_label = getSequenceData(train_direction, chrom_train_direction, train_label_direction)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
        for i, (train_index, test_index) in enumerate(cv.split(encode_data, encode_label)):
            data_train, sequence_train, chrom_train, label_train = encode_data[train_index], sequence_data[train_index], chrom_data[train_index], encode_label[train_index]
            data_test, sequence_test, chrom_test, label_test = encode_data[test_index], sequence_data[test_index], chrom_data[test_index], encode_label[test_index]
            train_data = TensorDataset(torch.tensor(data_train), torch.tensor(sequence_train), torch.tensor(chrom_train), torch.tensor(label_train))
            test_data = TensorDataset(torch.tensor(data_test), torch.tensor(sequence_test), torch.tensor(chrom_test), torch.tensor(label_test))
            dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=SH))
            dataset_va.append(DataLoader(test_data, batch_size=batch, shuffle=SH))
    else:

        print("encode train")
        x_train, seq_train, chr_train, y_train = getSequenceData(train_direction, chrom_train_direction, train_label_direction)
#        print(type(x_train), type(seq_train), type(y_train))
#        print((x_train.shape), (seq_train.shape), (y_train.shape))

        # Create datasets
        train_data = TensorDataset(x_train, seq_train, chr_train, y_train)
        dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=SH))


    print("encode test")
    x_test, seq_test, chr_test, y_test = getSequenceData(test_direction, chrom_test_direction, test_label_direction)

    # Create datasets
    test_data = TensorDataset(x_test, seq_test, chr_test, y_test)

    dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=False))

    dataset_test.append(dataset_test)

    return dataset_train, dataset_va, dataset_test


def spent_time(start, end):
    epoch_time = end - start
    minute = int(epoch_time / 60)
    secs = int(epoch_time - minute * 60)
    return minute, secs


def save_results(model_name, start, end, test_score, file_path):
    #    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
    title = ['Model', 'Recall', 'SPE', 'Precision', 'F1', 'MCC', 'Acc', 'AUC', 'AUPR',
             'RunTime', 'Test_Time']

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content = [[model_name,
                '%.3f' % test_score[0],
                '%.3f' % test_score[1],
                '%.3f' % test_score[2],
                '%.3f' % test_score[3],
                '%.3f' % test_score[4],
                '%.3f' % test_score[5],
                '%.3f' % test_score[6],
                '%.3f' % test_score[7],
                '%.3f' % (end - start),
                now]]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, encoding='gbk')
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)


def main(paths=None):

    print("doing: start-lost predition")

    Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    
    parse_file = f"result/sta_pares.txt"
    file1 = open(parse_file, 'a')
    file1.write(Time)
    file1.write('\n')
    print(args, file=file1)
    file1.write('\n')
    file1.close()
    file_path = "{}/{}.csv".format('result', 'sta_test')


    print("Data is loading......")
    train_datasets, va_datasets, test_datasets = data_load(args.train_direction, args.chrom_train_direction, args.train_label_direction, args.test_direction, args.chrom_test_direction, args.test_label_direction,
                                                           args.batch_size, cv=args.CV)
    print("Data is loaded!")
    all_test_score = 0
    start_time = time.time()
    if paths is None:
        print(f"{args.model_name} is training......")
        a = len(train_datasets)
        for i in range(len(train_datasets)):
            train_dataset = train_datasets[i]
            test_dataset = test_datasets[0]

            train_start = time.time()

            model = StartPred(args.vocab_size, args.embedding_size_DLM, args.embedding_size_seq, args.DLM_seq_len, args.sequence_length, args.filter_num, args.filter_size, args.output_size, args.dropout) 
            
            model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file2 = open(models_file, 'a')
            file2.write(model_time)
            file2.write('\n')
            print(model, file=file2)
            file2.write('\n')
            file2.close()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500)
            criterion = torch.nn.BCEWithLogitsLoss()


            Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)


            if va_datasets is None:
                Train.train_step(train_dataset, test_dataset, args.model_name, args.epochs,
                                 threshold=args.threshold)
            else:
                test_dataset = va_datasets[i]            
                Train.train_step(train_dataset, va_datasets, args.model_name, args.epochs,
                                 threshold=args.threshold)

            PATH = os.getcwd()
            each_model = os.path.join(PATH, 'saved_models', args.model_name + '.pth')
            torch.save(model.state_dict(), each_model)


            model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
            
            result = pd.DataFrame(model_predictions) ###���Ԥ�����
            result.to_csv('./result/test_pred_score.txt', sep='\t', index=False, header=False)
            
            test_score = estimate.scores(model_predictions, true_labels, args.threshold)


            train_end = time.time()
            if len(train_datasets) > 1:
                save_results(args.model_name + "fold " + str(i), train_start, train_end, test_score, file_path)
            else:
                save_results(args.model_name, train_start, train_end, test_score, file_path)


            print(f"{args.model_name}, test set:")
            metric = ["Recall", "SPE", "Precision", "F1", "MCC", "Acc", "AUC", "AUPR"]
            for k in range(len(metric)):
                print(f"{metric[k]}: {test_score[k]}\n")
            run_time = time.time()
            save_results('average', start_time, run_time, test_score, file_path)


#             # ���¼������ݼ������ÿ�����������ѧϰģ�ͱ���������
#            print("Data is reloading......")       
#            train_datasets, va_datasets, test_datasets = data_load(args.train_direction, args.chrom_train_direction, args.train_label_direction, args.test_direction, args.chrom_test_direction, args.test_label_direction,
#                                                                  args.batch_size, cv=args.CV, SH=False)
#            train_dataset = train_datasets[0]
#            train_fea = feature(model, train_dataset)            
#            # torch.save(train_fea, "./save_feature/1-training_TextCNN.h5")
#            train_fea = pd.DataFrame(train_fea.cpu().detach().numpy()) ###����Ϊnumpy����
#            train = pd.read_csv(args.train_label_direction)   ###���¼��ذ�����ǩ������
#            train_label = train["Label"] ##��ȡ��ǩ
#            train_fea = pd.concat([train_label, train_fea], axis=1)   ##�����������ͱ�ǩƴ����һ����������Ϊ����������������        
#            train_fea.to_csv('./save_feature/1-training_TextCNN.txt', sep='\t', index=False, header=False)
#            
#            test_dataset = test_datasets[0]
#            test_fea = feature(model, test_dataset)
#            # torch.save(test_fea, "./save_feature/2-testing_TextCNN.h5")
#            test_fea = pd.DataFrame(test_fea.cpu().detach().numpy())
#            test = pd.read_csv(args.test_label_direction)
#            test_label = test["Label"]
#            test_fea = pd.concat([test_label, test_fea], axis=1)            
#            test_fea.to_csv('./save_feature/2-testing_TextCNN.txt', sep='\t', index=False, header=False)


if __name__ == '__main__':

    models_file = f'result/model_details.txt'
    args = sta_config.get_config()
    main()
