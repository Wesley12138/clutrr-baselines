import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Argument parser to obtain the name of the config file")
    parser.add_argument('--config_id', default="graph_rel", help='config id to use')
    parser.add_argument('--exp_id', default='', help='optional experiment id to resume')

    parser.add_argument('--ds', type=str, default="data_089907f8", help='dataset')
    parser.add_argument('--ed', nargs='+', type=int, default=100, help='embedding dim')
    parser.add_argument('--hd', nargs='+', type=int, default=128, help='hidden dim')
    args = parser.parse_args()
    return args.config_id, args.exp_id, args.ds, args.ed, args.hd