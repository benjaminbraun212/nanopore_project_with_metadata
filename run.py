import argparse
from models_trainer import NanoporeTrainer
import datetime
import logging
import os
import cProfile
from config import cont_names, cat_names
def train(args): 
    with open(args.lock_file_name, 'w') as f:
        current_datetime = datetime.datetime.now()
        date_suffix = current_datetime.strftime("%Y-%m-%d_%H-%M")
        log_filename = f'logs/logs_{date_suffix}.log'
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        logger = logging.getLogger('my_logger')
        for arg_name, arg_value in vars(args).items():
            print(f"{arg_name}: {arg_value}")
            logger.info(f'Parameter {arg_name}:{arg_value}')

    trainer = NanoporeTrainer(model_name = args.model_name,
                fold = args.fold,
                hidden_size = args.hidden_size,
                batch_size = args.batch_size,
                optim_lr = args.optimLR,
                max_iter = args.max_iter,
                d_model = args.d_model,
                d_k = args.d_k,
                d_v = args.d_v,
                d_ff = args.d_ff,
                dropout = args.dropout,
                fc_dropout = args.fc_dropout,
                stride = args.stride,
                win_len = args.win_len,
                seq_len = args.seq_len,
                num_workers = args.num_workers,
                logger = logger,
                log_filename = log_filename,
                n_layers = args.n_layers,
                multi_gpu_flag = args.multi_gpu_flag,
                nf = args.nf,
                nhead = args.nhead,
                gpu_index = args.gpu_index,
                cat_names = args.cat_names,
                cont_names = args.cont_names,
                multimodel_flag = args.multimodel_flag,
                proj_size = args.proj_size
                )

    trainer._train()

if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser('Train the model.')
    parser.add_argument('--lock_file_name', default="run", type=str)
    parser.add_argument('--model_name', default='TST', type=str, help='Model name')
    parser.add_argument('--fold', default=1, type=int, help='cvfold')
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--batch-size', default=512, type=int,
                        help='The size of each batch')
    parser.add_argument('--optimLR', default=0.0001, type=float, help='LR; Default value coming from hyperparameters tuning')
    parser.add_argument('--max-iter', default=200, type=int,
                        help='The maximum iteration count')
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--d_k', default=128, type=int)
    parser.add_argument('--d_v', default=128, type=int)
    parser.add_argument('--d_ff', default=1024, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--fc_dropout', default=0.5, type=float)
    parser.add_argument('--stride', default=10, type=int)
    parser.add_argument('--win_len', default=32, type=int)
    parser.add_argument('--seq_len', default=200, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--transpose_flag', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--n_layers', default=16, type=int)
    parser.add_argument('--multi_gpu_flag', default=True, type=bool)
    parser.add_argument('--bn', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--nf', default=16, type=int, help='number of filters')
    parser.add_argument('--nhead', default=16, type=int, help='number of heads')
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--cat_names', nargs='+', type=str, default=cat_names,
    help='List of cat variables for the tab model')
    parser.add_argument('--cont_names', nargs='+', type=str, default=cont_names
    ,help='List of cat variables for the tab model')
    parser.add_argument('--multimodel_flag', type=str, default= "TS_only", help='indicates if its multi, only transformers or only tabular')
    parser.add_argument('--proj_size', type=int, default= 100, help='Dimension of the intermediaite projected size, for the tab and ts model')
    args = parser.parse_args()
    train(args=args)
