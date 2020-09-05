import glob, os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil


def get_file_name(dataset, model_name, ned, eed, hd, ep, fi, he, hi, hop):
    if model_name == 'gat':
        return f'{dataset}_{model_name}_ned_{ned}_eed_{eed}_he_{he}_ep_{ep}.csv'
    elif model_name in {'gcn', 'rgcn', 'sgcn', 'agnn'}:
        return f'{dataset}_{model_name}_ned_{ned}_eed_{eed}_ep_{ep}.csv'
    elif model_name == 'graph_cnn':
        return f'{dataset}_{model_name}_ed_{ned}_fi_{fi}_ep_{ep}.csv'
    elif model_name == 'graph_cnnh':
        return f'{dataset}_{model_name}_ed_{ned}_hi_{hi}_ep_{ep}.csv'
    elif model_name == 'graph_boe':
        return f'{dataset}_{model_name}_ed_{ned}_ep_{ep}.csv'
    elif model_name in {'graph_rnn', 'graph_birnn', 'graph_lstm', 'graph_bilstm', 'graph_gru', 'graph_bigru', 'graph_intra'}:
        return f'{dataset}_{model_name}_ed_{ned}_hd_{hd}_ep_{ep}.csv'
    elif model_name in {'graph_stack', 'graph_multihead'}:
        return f'{dataset}_{model_name}_ed_{ned}_hd_{hd}_he_{he}_ep_{ep}.csv'
    elif model_name in {'ctp_s', 'ctp_l', 'ctp_a', 'ntp'}:
        return f'{dataset}_{model_name}_ed_{ned}_hop_{"".join(hop)}_ep_{ep}.csv'
    elif model_name == 'ctp_m':
        return f'{dataset}_{model_name}_ed_{ned}_rules_{hd}_hop_{"".join(hop)}_ep_{ep}.csv'
    else:
        return f'{dataset}_{model_name}_PROCESSING.csv'


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
    mt = config.model.metric
    hop = config.model.encoder.hops_str

    base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines'
    file_dir = os.path.join(base_path, 'logs', model_name, dataset)
    if mt:  # for repeat analysis
        file_dir = os.path.join(base_path, 'tmp', dataset, mt, model_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if mt:
        file_path = os.path.join(file_dir,
                                 f'{se}=' + get_file_name(dataset, model_name, ned, eed, hd, ep, fi, he, hi, hop))
    else:
        file_path = os.path.join(file_dir, get_file_name(dataset, model_name, ned, eed, hd, ep, fi, he, hi, hop))
        # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines/logs/graph_lstm/data_089907f8/data_089907f8_graph_lstm_ed_512_hd_32.csv'

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
    print(f'save csv to: {file_path}')

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
    ds, ned, eed, hd, ep, fi, he, hi, hop, se, mt = hyperparas
    if model_name not in {'gcn', 'gat', 'rgcn', 'agnn', 'sgcn'}:
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
    config.model.metric = mt
    config.model.encoder.hops_str = hop

    print('*' * 100)
    print(get_file_name(ds, model_name, ned, eed, hd, ep, fi, he, hi, hop))
    print(f'seed={se}, metric={mt}')
    print('*' * 100)

    return config


def analysis_draw(model_name, dataset, mt):
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
        df_mt1 = df_mt2 = pd.DataFrame()  # log in table, min_val_loss, max_val_acc
        df_paras = []  # record paras for each model

        if mt:
            df_re = pd.DataFrame(columns=['x', 'y', 'model'])
            log_dir = os.path.join(path, 'tmp', ds, mt)   # log all models' result for each ds
            # if not os.path.exists(log_dir):
            #     os.makedirs(log_dir)
            metircs = {'1': 're_best_min_val_loss', '2': 're_best_max_val_acc'}
                     # '3': 're_best_max_mean_test_acc', '4': 're_best_max_10_test_acc'}
            res_df = []  # log in table, repeat

        for model in model_name:
            file_dir = os.path.join(path, 'logs', model, ds)
            if mt:
                # 1,2
                #  're_best_min_val_loss'  're_best_max_val_acc'
                file_dir = os.path.join(path, 'tmp', ds, mt, model)

            file_names = glob.glob(os.path.join(file_dir, "*.csv"))  # find all csv files.  os.listdir(file_dir)
            assert len(file_names)!=0, f'No file under {file_dir}'
            csv_datas = []
            for cf in file_names:
                assert os.stat(cf).st_size != 0, f'{cf.split("/")[-1]}'
                csv_datas.append(pd.read_csv(cf))  # all df under file_dir

            if mt:
                col_name = list(csv_datas[0].columns)
                df = pd.DataFrame(columns=col_name)
                col = int(mt)
                for csv_data_ in csv_datas: # based on type to choose and put them together
                    if mt == '1':
                        opt_idx = csv_data_[col_name[col]].idxmin()
                    else:
                        opt_idx = csv_data_[col_name[col]].idxmax()
                    df = df.append(csv_data_.iloc[opt_idx], ignore_index=True)

                # record the mean and std for each model
                mean_df = df.agg('mean').values[1:]
                std_df = df.agg('std').values[1:]
                log_dir_ = os.path.join(log_dir, f'{ds}_{metircs[mt]}.txt')
                with open(log_dir_, 'a') as fr:
                    print(f'Model: {file_names[0].split("=")[-1].split(".")[0]}', file=fr)
                    print(", ".join([f"{n}: {v}(\u00B1{s})" for n, v, s in zip(col_name[1:], mean_df, std_df)]), file=fr)
                    print(file=fr)

                # record in table
                for i in range(len(mean_df)):
                    res_df.append(f'{round(mean_df[i], 4)} (\u00B1{round(std_df[i], 4)})')


                x = df.columns[4:].values  # [2_test_acc, 3_test_acc, ...]
                y = df.iloc[:, 4:]
                for x_ in x:  # transfer into suitable df form for lineplot
                    df_re = pd.concat([df_re, pd.DataFrame({'x': x_, 'y': y[x_], 'model': model})], ignore_index=True)
                    # [x, y, model]  e.g. [1_test, 10, lstm]

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
                    # # max_mean_test_acc
                    # opt_idx_mean = csv_data_[col_name[3]].idxmax()
                    # df_mean = df_mean.append(csv_data_.iloc[opt_idx_mean], ignore_index=True)
                    # # max_10_test_acc
                    # opt_idx_10 = csv_data_[col_name[-1]].idxmax()
                    # # assert isinstance(opt_idx_10, int), f'{ds}_{model}_{file_names[i]}'
                    # df_10 = df_10.append(csv_data_.iloc[opt_idx_10], ignore_index=True)

                # min_val_loss
                min_idx = df_min[col_name[1]].astype(float).idxmin()  # corresponding to particular hyperparas
                min_val_loss = df_min.iloc[min_idx, 1]
                test_acc_min = df_min.iloc[min_idx, 4:].values
                # max_val_acc
                max_idx = df_max[col_name[2]].astype(float).idxmax()
                max_val_acc = df_max.iloc[max_idx, 2]
                test_acc_max = df_max.iloc[max_idx, 4:].values
                # # max_mean_test_acc
                # max_idx_mean = df_mean[col_name[3]].astype(float).idxmax()
                # max_mean_test_acc = df_mean.iloc[max_idx_mean, 3]
                # test_acc_max_mean = df_mean.iloc[max_idx_mean, 4:].values
                # # max_10_test_acc
                # max_idx_10 = df_10[col_name[-1]].astype(float).idxmax()
                # max_10_test_acc = df_10.iloc[max_idx_10, -1]
                # test_acc_best_10 = df_10.iloc[max_idx_10, 4:].values

                log_dir = os.path.join(path, 'logs', model, f'{ds}_{model}.txt')
                with open(log_dir, 'w') as fl:
                    print(f'Based on minimum val_loss={min_val_loss}: '
                          f'{file_names[min_idx].split("/")[-1].split(".")[0]}', file=fl)
                    print(", ".join([f"{n}: {v}" for n, v in zip(col_name, df_min.iloc[min_idx].values)]), file=fl)
                    print(file=fl)
                    print(f'Based on maximum val_acc={max_val_acc}: '
                          f'{file_names[max_idx].split("/")[-1].split(".")[0]}', file=fl)
                    print(", ".join([f"{n}: {v}" for n, v in zip(col_name, df_max.iloc[max_idx].values)]), file=fl)
                    # print(file=fl)
                    # print(f'Based on maximum mean_test_acc={max_mean_test_acc}: '
                    #       f'{file_names[max_idx_mean].split("/")[-1].split(".")[0]}',file=fl)
                    # print(", ".join([f"{n}: {v}" for n, v in zip(col_name, df_mean.iloc[max_idx_mean].values)]), file=fl)
                    # print(file=fl)
                    # print(f'Based on maximum 10_test_acc={max_10_test_acc}: '
                    #       f'{file_names[max_idx_10].split("/")[-1].split(".")[0]}', file=fl)
                    # print(", ".join([f"{n}: {v}" for n, v in zip(col_name, df_10.iloc[max_idx_10].values)]), file=fl)

                # record in table
                df_mt1 = df_mt1.append(df_min.iloc[min_idx, 1:], ignore_index=True)  # for min val loss
                df_mt2 = df_mt2.append(df_min.iloc[max_idx, 1:], ignore_index=True)  # for max val acc
                df_paras.append(file_names[min_idx].split("/")[-1].split(".")[0])
                df_paras.append(file_names[max_idx].split("/")[-1].split(".")[0])

                best_min_val_loss.append(test_acc_min)
                best_max_val_acc.append(test_acc_max)
                # best_max_mean_test_acc.append(test_acc_max_mean)
                # best_max_10_test_acc.append(test_acc_best_10)

        # plot img
        img_dir = os.path.join(path, 'plots', ds)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # else:
        #     shutil.rmtree(img_dir)
        #     os.makedirs(img_dir)

        if not mt:
            # record in table
            table_dir = os.path.join(path, 'logs', 'table')
            if not os.path.exists(table_dir):
                os.makedirs(table_dir)
            df_mt1 = df_mt1[col_name[1:]]
            df_mt1.index = model_name
            df_mt1.to_csv(os.path.join(table_dir, f'{ds}_min_val_loss.csv'))
            df_mt2 = df_mt2[col_name[1:]]
            df_mt2.index = model_name
            df_mt2.to_csv(os.path.join(table_dir, f'{ds}_max_val_acc.csv'))
            df_paras = pd.DataFrame(np.array(df_paras).reshape(-1,2), index=model_name, columns=['min_val_loss', 'max_val_acc'])
            df_paras.to_csv(os.path.join(table_dir, f'{ds}_paras.csv'))

            x = col_name[4:]
            draw_img(x, best_min_val_loss, model_name, img_dir, 'best_min_val_loss')
            draw_img(x, best_max_val_acc, model_name, img_dir, 'best_max_val_acc')
            # draw_img(x, best_max_mean_test_acc, model_name, img_dir, 'best_max_mean_test_acc')
            # draw_img(x, best_max_10_test_acc, model_name, img_dir, 'best_max_10_test_acc')
            # plt.show()

        else:
            # log table
            df_table = pd.DataFrame(np.array(res_df).reshape(len(model_name), -1), index=model_name, columns=col_name[1:])
            table_path = os.path.join(log_dir, f'{ds}_{metircs[mt]}.csv')
            df_table.to_csv(table_path)

            # draw
            plt.figure(figsize=(8, 7))
            sns.set_style("white")
            sns.set_style("ticks")
            plt.grid()
            # dash_styles = ["",
            #             (4, 1.5),
            #             (1, 1),
            #             (3, 1, 1.5, 1),
            #             (5, 1, 1, 1),
            #             (5, 1, 2, 1, 2, 1),
            #             (2, 2, 3, 1.5),
            #             (1, 2.5, 3, 1.2)]  #,
            #             # (2, 2, 3, 1),
            #             # (1, 1, 3, 1),
            #             # (4, 3, 2, 1),
            #             # (3, 1.5, 3.5, 2)]  # 11 line styles
            df_re['y'] = df_re['y'].astype('float')
            # nb_colors = df_re['model'].nunique()
            # palette = sns.color_palette("muted", nb_colors)
            ax = sns.lineplot(x='x', y='y', hue='model', style='model', data=df_re, ci="sd",
                              dashes=False, sort=False)  # palette=palette,
            name = f'{ds}_{metircs[mt]}'
            ax.set_title(name)
            # 're_best_min_val_loss'  're_best_max_val_acc'  're_best_max_mean_test_acc'  're_best_max_10_test_acc'
            ax.set_xlim(x[0], x[-1])
            ax.set_xticklabels(x, rotation=15)  # , horizontalalignment='right'
            ax.set_xlabel('Test Story Length')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Test Accuracy')
            plt.legend(loc=3)
            plt.savefig(os.path.join(img_dir, f'{name}.jpg'))
            # plt.show()

    print('Analysis Complete!')


def draw_img(x, ys, model, img_dir, name):
    dash_styles = [(4, 1.5),
                   (1, 1),
                   (3, 1, 1.5, 1),
                   (5, 1, 1, 1),
                   (5, 1, 2, 1, 2, 1),
                   (2, 2, 3, 1.5),
                   (1, 2.5, 3, 1.2),
                   (2, 2, 3, 1),
                   (1, 1, 3, 1),
                   (4, 3, 2, 1),
                   (3, 1.5, 3.5, 2)]   # 11 line styles
    plt.figure()
    for y, m, d in zip(ys, model, dash_styles):
        plt.plot(x, y, label=m, dashes=d)
    plt.legend(loc=3)
    plt.xticks(rotation=15)
    plt.title(name)
    plt.savefig(os.path.join(img_dir, f'{name}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="obtain the dataset and model's name for anlaysis")
    parser.add_argument('--m', nargs='+', type=str, default="graph_lstm", help='model name')
    parser.add_argument('--ds', nargs='+', type=str, default="data_089907f8", help='dataset')
    # parser.add_argument('--re', action='store_true', help='for error bar depict, repeat or not')
    parser.add_argument('--mt', type=str, default=None, choices=['1', '2', None],
                        help='select the metric for analysis, [1, 2]=[re_best_min_val_loss, re_best_max_val_acc]')
    args = parser.parse_args()

    model_name = args.m
    dataset = args.ds
    mt = args.mt

    analysis_draw(model_name, dataset, mt)


# model: gat gcn rgcn agcn sgcn
#        graph_boe graph_cnn graph_cnnh graph_rnn graph_lstm graph_gru graph_birnn graph_bilstm graph_bigru
#        graph_intra graph_multihead
#        ctp_s ctp_l ctp_a ctp_m ntp

# normal analysis
# python codes/analysis_res.py
# --m gat gcn graph_bilstm graph_birnn graph_bigru graph_cnn graph_cnnh graph_boe
# --ds data_089907f8 data_db9b8f04 data_7c5b0e70 data_06b8f2a1 data_523348e6 data_d83ecc3e

# for repeat analysis
# python codes/analysis_res.py
# --m gat gcn graph_bilstm graph_birnn graph_bigru graph_cnn graph_cnnh graph_boe
# --ds data_089907f8 data_db9b8f04 data_7c5b0e70 data_06b8f2a1 data_523348e6 data_d83ecc3e --mt 1