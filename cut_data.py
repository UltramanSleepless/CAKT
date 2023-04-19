import pandas as pd
import numpy as np
import csv
import argparse
# 代码第23行 修改，逆置列表元素，探讨序列顺序对于模型的影响


def count_row(path):
    total = len(open(path+".csv").readlines())
    print('The total lines is ', total)
    return total


def cut_data(ratio, path):
    f = open(path+".csv", "r")
    reader = csv.reader(f)
    data_len = count_row(path)
    data_len = data_len*ratio
    f = open(path+"_"+str(ratio)+".csv", "w")
    csv_writer = csv.writer(f)
    for line_id, row in enumerate(reader):
        if line_id < data_len:
            csv_writer.writerow(row)
            # if line_id % 3 == 0:
            #     csv_writer.writerow(row)

            # else:
            #     row = list(reversed(row))
            #     # 逆置列表元素
            #     csv_writer.writerow(row)
            #     # print(line_id)
            #     # print(row)
    f.close()


if __name__ == '__main__':
    ratio = 0.1

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to cut data')
    # Basic Parameters
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    params = parser.parse_args()
    dataset = params.dataset
    if dataset in {"assist2009_pid"}:
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    if dataset in {"assist2017_pid"}:
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    if dataset in {"assist2015"}:
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    if dataset in {"statics"}:
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    for i in range(5):
        print("第", i+1, "折数据集切割。")

        train_data_path = params.data_dir + "/" + \
            params.data_name + "_train"+str(i+1)
        valid_data_path = params.data_dir + "/" + \
            params.data_name + "_valid"+str(i+1)

        cut_data(ratio, train_data_path)
        cut_data(ratio, valid_data_path)
