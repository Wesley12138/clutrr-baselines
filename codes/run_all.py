import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil


def create_csv(model_name, dataset, train_file, test_file, embedding_dim, hidden_dim):
    """
    create csv file for later data input
    :param dir:
    :param train_file:
    :param test_file:
    :param embedding_dim:
    :param hidden_dim:
    :return:
    """
    base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines'
    file_dir = os.path.join(base_path, 'logs', model_name, dataset)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, f'{dataset}_{model_name}_ed_{embedding_dim}_hd_{hidden_dim}.csv')
    # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines/logs/graph_lstm/data_089907f8/data_089907f8_graph_lstm_ed_512_hd_32.csv'

    train_name = train_file.split('/')[-1].split('.csv')[0]  # i.e. 1.2,1.3_train
    test_name = [eval(f.split('.')[1].split('_')[0]) for f in test_file]  # i.e. [10, 2, 3,...]
    attr = ['Epoch', f'{train_name}_loss', f'{train_name}_acc', 'Mean_test_accuracy'] + [f'{t}_test_acc' for t in
                                                                                         sorted(test_name)]
    df = pd.DataFrame(columns=attr)
    df.set_index(['Epoch'], inplace=True)
    df.to_csv(file_path)

    return file_path


def save_to_csv(file_path, epoch, data_file, value, mode):
    """
    save each epoch results under each models with different hyper-parameters for analysis and graph drawing
    :param file_path:
    :param epoch:
    :param data_file:
    :param value:
    :param mode:
    :return:
    """
    df = pd.read_csv(file_path, index_col=0, dtype=object)

    if mode == 'val':
        file_name = data_file.split('/')[-1].split('.csv')[0]  # i.e. 1.2,1.3_train
        df.at[epoch, f'{file_name}_loss'] = value[0]
        df.at[epoch, f'{file_name}_acc'] = value[1]
    elif mode == 'test':
        for t in value:
            file_name = t[0].split('.')[1] + '_acc'  # i.e. 10_test_acc
            df.at[epoch, file_name] = t[1]

        df.at[epoch, 'Mean_test_accuracy'] = np.mean([t[1] for t in value])

    df.to_csv(file_path)


def modify_hyperparas(config, config_id, datasets, embedding_dim, hidden_dim):
    """
    modify hyper-parameters
    :param config:
    :param config_id:
    :param datasets:
    :param embedding_dim:
    :param hidden_dim:
    :return:
    """
    config['model']['embedding']['dim'] = embedding_dim
    config['model']['graph']['edge_dim'] = embedding_dim
    config['model']['encoder']['hidden_dim'] = hidden_dim
    print('*' * 50)
    print(f'model={config_id}, dataset={datasets}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}')
    print('*' * 50)

    return config


def draw_from_csv(dataset, model_name):
    """
    draw img from csv
    :param dataset:
    :param model_name:
    :return:
    """
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    file_dir = os.path.join(path, 'logs', model_name, dataset)
    file_names = os.listdir(file_dir)
    csv_datas = [pd.read_csv(os.path.join(file_dir, cf)) for cf in file_names]

    col_name = list(csv_datas[0].columns)
    df_min = df_max = df_10 = pd.DataFrame(columns=col_name)
    for i, csv_data_ in enumerate(csv_datas):
        # min val_loss
        opt_idx_min = csv_data_[col_name[1]].idxmin()
        df_min = df_min.append(csv_data_.iloc[opt_idx_min], ignore_index=True)
        # max_val_acc
        opt_idx_max = csv_data_[col_name[2]].idxmax()
        df_max = df_max.append(csv_data_.iloc[opt_idx_max], ignore_index=True)
        # max_10_test_acc
        opt_idx_10 = csv_data_[col_name[-1]].idxmax()
        df_10 = df_10.append(csv_data_.iloc[opt_idx_10], ignore_index=True)

    # min_val_loss
    min_idx = df_min[col_name[1]].astype(float).idxmin()  # corresponding to particular hyperparas
    min_val_loss = df_min.iloc[min_idx, 1]
    test_acc_min = df_min.iloc[min_idx, 4:].values
    print(f'Based on minimum val_loss={min_val_loss}: {file_names[min_idx].split(".")[0]}')
    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_min.iloc[min_idx].values)]))
    print()
    # max_val_acc
    max_idx = df_max[col_name[2]].astype(float).idxmax()
    max_val_acc = df_max.iloc[max_idx, 2]
    test_acc_max = df_max.iloc[max_idx, 4:].values
    print(f'Based on maximum val_acc={max_val_acc}: {file_names[max_idx].split(".")[0]}')
    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_max.iloc[max_idx].values)]))
    print()
    # max_10_test_acc
    max_idx_10 = df_10[col_name[-1]].astype(float).idxmax()
    max_10_test_acc = df_10.iloc[max_idx_10, -1]
    test_acc_best_10 = df_10.iloc[max_idx_10, 4:].values
    print(f'Based on maximum 10_test_acc={max_10_test_acc}: {file_names[max_idx_10].split(".")[0]}')
    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_10.iloc[max_idx_10].values)]))
    print()

    img_dir = os.path.join(path, 'plots', model_name, dataset)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        shutil.rmtree(img_dir)
        os.makedirs(img_dir)

    x = col_name[4:]
    plt.figure()
    plt.plot(x, test_acc_min)
    plt.xticks(rotation=45)
    plt.title(f'best_min_val_loss_{file_names[min_idx].split(".")[0]}')
    plt.savefig(os.path.join(img_dir, f'best_min_val_loss_{file_names[min_idx].split(".")[0]}.jpg'))

    plt.figure()
    plt.plot(x, test_acc_max)
    plt.xticks(rotation=45)
    plt.title(f'best_max_val_acc_{file_names[max_idx].split(".")[0]}')
    plt.savefig(os.path.join(img_dir, f'best_max_val_acc_{file_names[max_idx].split(".")[0]}.jpg'))

    plt.figure()
    plt.plot(x, test_acc_best_10)
    plt.xticks(rotation=45)
    plt.title(f'best_max_10_test_acc_{file_names[max_idx_10].split(".")[0]}')
    plt.savefig(os.path.join(img_dir, f'best_max_10_test_acc_{file_names[max_idx_10].split(".")[0]}.jpg'))

    # plt.show()