import argparse
import utils.parser_utils as parser_utils
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')

    parser_utils.parse_optimizer(parser)

    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--ckptdir', dest='ckptdir',
            help='Model checkpoint directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--cpu', dest='gpu', action='store_const',
            const=False, default=True,
            help='whether to use GPU.')
    parser.add_argument('--max_nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num_classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--bn', dest='bn', action='store_const',
            const=True, default=False,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
            help='Weight decay regularization constant.')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, ')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.add_argument('--p', dest='p', type=float,
            help='Binomial probability for modified syn4.')
    parser.add_argument('--seed', dest='seed', type=int,
            help='Binomial seed for modified syn4.')
    parser.add_argument('--graph-only', dest='graph_only', action='store_const',
            const=True, default=False, help='whether to only generate a graph.')
    parser.add_argument('--feat-gen', dest='feat_gen', default='Binomial',
                        help="Options:'Binomial', 'Correlated', 'CorrelatedXOR', or 'NormalNoise'")
    parser.add_argument('--syn-type', dest='syn_type', default='tree-cycle', type=str,
                        help="Options:'tree-cycle', 'tree-grid', 'tree-house', 'ba-cycle', 'ba-grid', 'ba-house'")

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        dataset="syn4",
        opt="adam",  # opt_parser
        opt_scheduler="none",
        # opt_scheduler="cos",
        # opt_restart=40,
        gpu=True,
        max_nodes=10000,
        cuda="0,1,2,3,4,5,6,7",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=64,
        num_epochs=2000,
        train_ratio=0.8,
        test_ratio=0.1,
        num_workers=8,
        input_dim=10,
        hidden_dim=32,
        output_dim=64,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        name_suffix="",
        assign_ratio=0.1,
        p=0.0,
        seed=998244353,
    )
    return parser.parse_args()

