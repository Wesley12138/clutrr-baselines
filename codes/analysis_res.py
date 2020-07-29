import glob, os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil


def create_csv(config):
    """
    create csv file for later data input
    :return:
    """
    model_name = config.general.id
    se = config.general.seed
    dataset = config.dataset.data_path
    train_file = config.dataset.train_file
    test_file = config.dataset.test_files
    embedding_dim = config.model.embedding.dim
    hidden_dim = config.model.encoder.hidden_dim
    ned = config.model.embedding.dim
    eed = config.model.graph.edge_dim
    hd = config.model.encoder.hidden_dim
    ep = config.model.num_epochs
    fi = config.model.encoder.num_filters
    he = config.model.graph.num_reads
    hi = config.model.encoder.num_highway

    base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines'
    file_dir = os.path.join(base_path, 'logs', model_name, dataset)
    if se != 42:
        file_dir = os.path.join(file_dir, 'repeat')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if se == 42:
        file_path = os.path.join(file_dir,
                                 f'{dataset}_{model_name}_ned_{ned}_eed_{eed}_hd_{hd}_ep_{ep}_fi_{fi}_he_{he}_hi_{hi}.csv')
        # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines/logs/graph_lstm/data_089907f8/data_089907f8_graph_lstm_ed_512_hd_32.csv'
    else:
        file_path = os.path.join(file_dir,
                                 f'{se}={dataset}_{model_name}_ned_{ned}_eed_{eed}_hd_{hd}_ep_{ep}_fi_{fi}_he_{he}_hi_{hi}.csv')

    train_name = train_file.split('/')[-1].split('.csv')[0]  # i.e. 1.2,1.3_train
    if dataset == 'data_089907f8' or dataset == 'data_db9b8f04':
        test_name = [eval(f.split('.')[1].split('_')[0]) for f in test_file]  # i.e. [10, 2, 3,...]
    else:
        test_name = [eval(f.split('/')[-1].split('_')[0]) for f in test_file]
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
        dataset = file_path.split('/')[-1].split('_')[1]
        for t in value:
            if dataset == '089907f8' or dataset == 'db9b8f04':
                file_name = t[0].split('.')[1] + '_acc'  # i.e. 10_test_acc
            else:
                file_name = t[0].split('/')[-1].split('_')[0] + '_test_acc'  # 2.3_test_acc

            df.at[epoch, file_name] = t[1]

        df.at[epoch, 'Mean_test_accuracy'] = np.mean([t[1] for t in value])

    df.to_csv(file_path)


def modify_hyperparas(config, model_name, hyperparas):
    """
    modify hyper-parameters
    """
    ds, ned, eed, hd, ep, fi, he, hi, se = hyperparas
    if model_name != 'gcn' and model_name != 'gat':
        eed = ned

    config.general.id = config.model.encoder.model_name = model_name
    config.general.seed = se
    config.dataset.data_path = ds
    config.model.embedding.dim = ned
    config.model.graph.edge_dim = eed
    config.model.encoder.hidden_dim = hd
    config.model.num_epochs = ep
    config.model.encoder.num_filters = fi
    config.model.graph.num_reads = he
    config.model.encoder.num_highway = hi
    print('*' * 100)
    print(f'model={model_name}, dataset={ds}, node_embedding_dim={ned}, edge_embedding_dim={eed}, '
          f'hidden_dim={hd}, num_epochs={ep}, num_filters={fi}, num_heads={he}, num_highway={hi}, seed={se}')
    print('*' * 100)

    return config


def analysis_draw(model_name, dataset, repe, ty):
    """
    analyse results and draw the img
    :param model_name:
    :param dataset:
    :return:
    """
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    for ds in dataset:
        best_min_val_loss = []
        best_max_val_acc = []
        best_max_mean_test_acc = []
        best_max_10_test_acc = []
        df_re = pd.DataFrame(columns=['x', 'y', 'model'])

        for model in model_name:
            file_dir = os.path.join(path, 'logs', model, ds)
            if repe:
                select_type = ty  # 1,2,3,4:  're_best_min_val_loss'  're_best_max_val_acc'  're_best_max_mean_test_acc'  're_best_max_10_test_acc'
                file_dir = os.path.join(path, 'tmp', ds, select_type, 'logs', model, ds, 'repeat')

            file_names = glob.glob(os.path.join(file_dir, "*.csv"))  # find all csv files.  os.listdir(file_dir)
            csv_datas = [pd.read_csv(os.path.join(file_dir, cf)) for cf in file_names]

            if repe:
                for csv_data_ in csv_datas:
                    x = csv_data_.columns[4:].values  # [2_test_acc, 3_test_acc, ...]
                    y = csv_data_.iloc[:, 4:]
                    for x_ in x:
                        df_re = pd.concat([df_re, pd.DataFrame({'x': x_, 'y': y[x_], 'model': model})], ignore_index=True)

            # elif balabala>0:
            #     # min_val_loss
            #     df_mean_min, df_std_min = df_min.values.mean(axis=0), df_min.values.std(axis=0)
            #     mean_min_val_loss, std_min_val_loss = df_mean_min[1], df_std_min[1]
            #     mean_test_acc_min, std_test_acc_min = df_mean_min[4:], df_std_min[4:]
            #     # max_val_acc
            #     df_mean_max, df_std_max = df_max.values.mean(axis=0), df_max.values.std(axis=0)
            #     mean_max_val_acc, std_max_val_acc = df_mean_max[2], df_std_max[2]
            #     mean_test_acc_max, std_test_acc_max = df_mean_max[4:], df_std_max[4:]
            #     # max_mean_test_acc
            #     df_mean_max_mean, df_std_max_mean = df_mean.values.mean(axis=0), df_mean.values.std(axis=0)
            #     mean_max_mean_test_acc, std_max_mean_test_acc = df_mean_max_mean[3], df_std_max_mean[3]
            #     mean_test_acc_max_mean, std_test_acc_max_mean = df_mean_max_mean[4:], df_std_max_mean[4:]
            #     # max_10_test_acc
            #     df_mean_max_10, df_std_max_10 = df_10.values.mean(axis=0), df_10.values.std(axis=0)
            #     mean_max_10_test_acc, std_max_10_test_acc = df_mean_max_10[-1], df_std_max_10[-1]
            #     mean_test_acc_best_10, std_test_acc_best_10 = df_mean_max_10[4:], df_std_max_10[4:]
            #
            #     log_dir = os.path.join(path, 'logs', model, f're_{ds}_{model}.txt')
            #     with open(log_dir, 'w') as fr:
            #         print(f'Based on minimum val_loss={mean_min_val_loss}: {file_names[0].split("=")[-1].split(".")[0]}', file=fr)
            #         print(" ,".join([f"{n}: {v}({s})" for n, v, s in zip(col_name, df_mean_min, df_std_min)]), file=fr)
            #         print(file=fr)
            #         print(f'Based on maximum val_acc={mean_max_val_acc}: {file_names[0].split("=")[-1].split(".")[0]}', file=fr)
            #         print(" ,".join([f"{n}: {v}({s})" for n, v, s in zip(col_name, df_mean_max, df_std_max)]), file=fr)
            #         print(file=fr)
            #         print(f'Based on maximum mean_test_acc={mean_max_mean_test_acc}: {file_names[0].split("=")[-1].split(".")[0]}',
            #             file=fr)
            #         print(" ,".join([f"{n}: {v}({s})" for n, v, s in zip(col_name, df_mean_max_mean, df_std_max_mean)]), file=fr)
            #         print(file=fr)
            #         print(f'Based on maximum 10_test_acc={mean_max_10_test_acc}: {file_names[0].split("=")[-1].split(".")[0]}',
            #               file=fr)
            #         print(" ,".join([f"{n}: {v}({s})" for n, v, s in zip(col_name, df_mean_max_10, df_std_max_10)]), file=fr)

            else:
                col_name = list(csv_datas[0].columns)
                df_min = df_max = df_mean = df_10 = pd.DataFrame(columns=col_name)
                for i, csv_data_ in enumerate(csv_datas):
                    # min val_loss
                    opt_idx_min = csv_data_[col_name[1]].idxmin()
                    df_min = df_min.append(csv_data_.iloc[opt_idx_min], ignore_index=True)
                    # max_val_acc
                    opt_idx_max = csv_data_[col_name[2]].idxmax()
                    df_max = df_max.append(csv_data_.iloc[opt_idx_max], ignore_index=True)
                    # max_mean_test_acc
                    opt_idx_mean = csv_data_[col_name[3]].idxmax()
                    df_mean = df_mean.append(csv_data_.iloc[opt_idx_mean], ignore_index=True)
                    # max_10_test_acc
                    opt_idx_10 = csv_data_[col_name[-1]].idxmax()
                    df_10 = df_10.append(csv_data_.iloc[opt_idx_10], ignore_index=True)

                # min_val_loss
                min_idx = df_min[col_name[1]].astype(float).idxmin()  # corresponding to particular hyperparas
                min_val_loss = df_min.iloc[min_idx, 1]
                test_acc_min = df_min.iloc[min_idx, 4:].values
                # max_val_acc
                max_idx = df_max[col_name[2]].astype(float).idxmax()
                max_val_acc = df_max.iloc[max_idx, 2]
                test_acc_max = df_max.iloc[max_idx, 4:].values
                # max_mean_test_acc
                max_idx_mean = df_mean[col_name[3]].astype(float).idxmax()
                max_mean_test_acc = df_mean.iloc[max_idx_mean, 3]
                test_acc_max_mean = df_mean.iloc[max_idx_mean, 4:].values
                # max_10_test_acc
                max_idx_10 = df_10[col_name[-1]].astype(float).idxmax()
                max_10_test_acc = df_10.iloc[max_idx_10, -1]
                test_acc_best_10 = df_10.iloc[max_idx_10, 4:].values

                log_dir = os.path.join(path, 'logs', model, f'{ds}_{model}.txt')
                with open(log_dir, 'w') as fl:
                    print(f'Based on minimum val_loss={min_val_loss}: {file_names[min_idx].split(".")[0]}', file=fl)
                    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_min.iloc[min_idx].values)]), file=fl)
                    print(file=fl)
                    print(f'Based on maximum val_acc={max_val_acc}: {file_names[max_idx].split(".")[0]}', file=fl)
                    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_max.iloc[max_idx].values)]), file=fl)
                    print(file=fl)
                    print(
                        f'Based on maximum mean_test_acc={max_mean_test_acc}: {file_names[max_idx_mean].split(".")[0]}',
                        file=fl)
                    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_mean.iloc[max_idx_mean].values)]),
                          file=fl)
                    print(file=fl)
                    print(f'Based on maximum 10_test_acc={max_10_test_acc}: {file_names[max_idx_10].split(".")[0]}',
                          file=fl)
                    print(" ,".join([f"{n}: {v}" for n, v in zip(col_name, df_10.iloc[max_idx_10].values)]), file=fl)

                best_min_val_loss.append(test_acc_min)
                best_max_val_acc.append(test_acc_max)
                best_max_mean_test_acc.append(test_acc_max_mean)
                best_max_10_test_acc.append(test_acc_best_10)

        # plot img
        img_dir = os.path.join(path, 'plots', ds)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        else:
            shutil.rmtree(img_dir)
            os.makedirs(img_dir)

        if not repe:
            x = col_name[4:]
            draw_img(x, best_min_val_loss, model_name, img_dir, 'best_min_val_loss')
            draw_img(x, best_max_val_acc, model_name, img_dir, 'best_max_val_acc')
            draw_img(x, best_max_mean_test_acc, model_name, img_dir, 'best_max_mean_test_acc')
            draw_img(x, best_max_10_test_acc, model_name, img_dir, 'best_max_10_test_acc')
            # plt.show()

        else:
            plt.figure(figsize=(8, 7))
            sns.set_style("white")
            sns.set_style("ticks")
            plt.grid()
            dash_styles = ["",
                        (4, 1.5),
                        (1, 1),
                        (3, 1, 1.5, 1),
                        (5, 1, 1, 1),
                        (5, 1, 2, 1, 2, 1),
                        (2, 2, 3, 1.5),
                        (1, 2.5, 3, 1.2)]  #,
                        # (2, 2, 3, 1),
                        # (1, 1, 3, 1),
                        # (4, 3, 2, 1),
                        # (3, 1.5, 3.5, 2)]
            df_re['y'] = df_re['y'].astype('float')
            nb_colors = df_re['model'].nunique()
            palette = sns.color_palette("muted", nb_colors)
            ax = sns.lineplot(x='x', y='y', hue='model', style='model', data=df_re, ci="sd", palette=palette, dashes=dash_styles, sort=False)
            name = f'{ds}_re_best_min_val_loss'
            ax.set_title(name)  # 're_best_min_val_loss'  're_best_max_val_acc'  're_best_max_mean_test_acc'  're_best_max_10_test_acc'
            ax.set_xticklabels(x, rotation=15, horizontalalignment='right')
            ax.set_xlabel('Test Story Length')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Test Accuracy')
            plt.legend(loc=3)
            plt.savefig(os.path.join(img_dir, f'{name}.jpg'))
            plt.show()

    print('Analysis Complete!')


def draw_img(x, ys, model, img_dir, name):
    plt.figure()
    for y, m in zip(ys, model):
        plt.plot(x, y, label=m)
    plt.legend(loc=3)
    plt.xticks(rotation=15)
    plt.title(name)
    plt.savefig(os.path.join(img_dir, f'{name}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="obtain the dataset and model's name for anlaysis")
    parser.add_argument('--m', nargs='+', type=str, default="graph_lstm", help='model name')
    parser.add_argument('--ds', nargs='+', type=str, default="data_089907f8", help='dataset')
    parser.add_argument('--re', action='store_true', help='for error bar depict, repeat or not')
    parser.add_argument('--ty', type=str, default='1', help='select the type for analysis, [1, 2, 3, 4]='
                                               '[re_best_min_val_loss, re_best_max_val_acc, '
                                               're_best_max_mean_test_acc, re_best_max_10_test_acc]')
    args = parser.parse_args()

    model_name = args.m
    dataset = args.ds
    repe = args.re
    ty = args.ty

    analysis_draw(model_name, dataset, repe, ty)

# python codes/analysis_res.py --m gat gcn graph_lstm graph_rnn graph_gru graph_cnn graph_cnnh graph_boe --ds data_089907f8 data_db9b8f04 data_7c5b0e70 data_06b8f2a1 data_523348e6 data_d83ecc3e
