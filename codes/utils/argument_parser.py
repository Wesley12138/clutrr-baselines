import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Argument parser to obtain the name of the config file")
    parser.add_argument('--config_id', default="graph_nn", help='config id to use')
    parser.add_argument('--exp_id', default='', help='optional experiment id to resume')

    parser.add_argument('--ds', type=str, default="data_089907f8", help='dataset')
    parser.add_argument('--ned', type=int, default=100, help='node embedding dim')
    parser.add_argument('--eed', type=int, default=100, help='edge embedding dim')
    parser.add_argument('--hd', type=int, default=128, help='hidden dim')
    parser.add_argument('--ep', type=int, default=50, help='num of epochs')
    parser.add_argument('--fi', type=int, default=1, help='num of filters')
    parser.add_argument('--he', type=int, default=3, help='num of heads')
    parser.add_argument('--hi', type=int, default=2, help='num of highway')
    parser.add_argument('--hop', nargs='+', type=str, default=['2', '2', '2', '2', '2', '2', '2', '2', '2', '2'])
    parser.add_argument('--se', type=int, default=42, help='seed')
    parser.add_argument('--mt', type=str, default=None, help='metric types: 1, 2')

    args = parser.parse_args()

    return args.config_id, args.exp_id, (args.ds, args.ned, args.eed, args.hd, args.ep,
                                         args.fi, args.he, args.hi, args.hop, args.se, args.mt)
