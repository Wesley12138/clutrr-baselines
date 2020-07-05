import os
import yaml
import pandas as pd


def create_csv(model_name, dataset, train_file, test_file, embedding_dim, hidden_dim):
    """

    :param dir:
    :param train_file:
    :param test_file:
    :param embedding_dim:
    :param hidden_dim:
    :return:
    """
    base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines'
    # logs_folder = os.path.join(str(base_path), 'logs', str(model_name))
    # file_name = os.path.join(logs_folder, f'{model_name}_ed_{embedding_dim}_hd_{hidden_dim}.csv')
    # model_name = dir.split('/')[-1]
    file_path = os.path.join(base_path, f'logs/{model_name}',
                             f'{dataset}_{model_name}_ed_{embedding_dim}_hd_{hidden_dim}.csv')
    # '/home/wesley/Documents/pycharm_workspace/clutrr-baselines/logs/graph_lstm/data_089907f8_graph_lstm_ed_512_hd_32.csv'
    train_name = train_file.split('/')[-1].split('.csv')[0]  # i.e. 1.2,1.3_train
    test_name = [eval(f.split('.')[1].split('_')[0]) for f in test_file]  # i.e. [10, 2, 3,...]
    attr = ['Epoch', 'Mean_test_accuracy', train_name] + [f'{t}_test' for t in sorted(test_name)]
    df = pd.DataFrame(columns=attr)
    df.set_index(['Epoch'], inplace=True)
    df.to_csv(file_path)

    return file_path


def save_to_csv(file_path, epoch, data_file, loss, accuracy, mode):
    """
    save each epoch results under each models with different hyper-parameters for analysis and graph drawing
    :param file_path:
    :param epoch:
    :param data_file:
    :param loss:
    :param accuracy:
    :param mode:
    :return:
    """
    df = pd.read_csv(file_path, index_col=0, dtype=object)

    if mode == 'mean':
        df.at[epoch, 'Mean_test_accuracy'] = accuracy
        df.to_csv(file_path)
        return
    elif mode == 'train':
        return
    elif mode == 'val':
        file_name = data_file.split('/')[-1].split('.csv')[0]  # i.e. 1.2,1.3_train
    elif mode == 'test':
        file_name = data_file.split('.')[1].split('_')[0] + '_test'  # i.e. 10_test

    df.at[epoch, file_name] = (loss, accuracy)
    df.to_csv(file_path)


# def run_all(config_id='gcn'):
    # base_config = get_config(config_id=config_id)
config_id=['graph_lstm']
# datasets = ['data_089907f8', 'data_db9b8f04']
# embedding_dim = [10, 50, 100, 200, 500, 1000]
# hidden_dim = [32, 64, 128, 256, 512, 1024]
datasets = ['data_089907f8']
embedding_dim = [10]
hidden_dim = [128]
print(os.path.dirname(os.path.realpath(__file__)))
path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]


for id_ in config_id:
    config_file_path = os.path.join(path, 'config', f'{id_}.yaml')
    with open(config_file_path, "r") as f:
        result = f.read()
        new_config = yaml.load(result, Loader=yaml.FullLoader)

        for d in datasets:
            for e in embedding_dim:
                for h in hidden_dim:
                    # new_config = deepcopy(base_config)

                    # modify value
                    new_config['dataset']['data_path'] = d
                    new_config['model']['embedding']['dim'] = e
                    new_config['model']['graph']['edge_dim'] = e
                    new_config['model']['encoder']['hidden_dim'] = h

                    with open(config_file_path, "w") as w_f:
                        yaml.dump(new_config, w_f)

                    # cmd = f'PYTHONPATH=. python codes/app/main.py --config_id {id_}'
                    # tmp = os.system(cmd)
                    # print(f'cmd result: {tmp}')  # success return 0, failure return 1



