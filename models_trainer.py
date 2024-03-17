from config import *
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import *
from tsai.all import *
import logging
from fastai.callback.fp16 import *
from fastai.metrics import *
from fastai.callback.core import Callback
from fastai.callback.tracker import SaveModelCallback, ReduceLROnPlateau
from fastai.tabular.core import *
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
from fastai.vision.all import *
from mixedDL import *
from sklearn.preprocessing import StandardScaler

class NanoporeTrainer(object):
    def __init__(self,
                model_name,
                fold,
                hidden_size,
                batch_size,
                optim_lr,
                max_iter,
                d_model,
                d_k,
                d_v,
                d_ff,
                dropout,
                fc_dropout,
                stride,
                win_len,
                seq_len,
                num_workers,
                logger,
                log_filename,
                n_layers,
                multi_gpu_flag,
                nf,
                nhead,
                gpu_index,
                cat_names,
                cont_names,
                multimodel_flag,
                proj_size
                ):
        
        self.model_name = model_name
        self.fold = fold
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self._stride = stride
        self._win_len = win_len
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._max_iter = max_iter
        self.optim_lr = optim_lr
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.num_workers = num_workers
        self.logger = logger
        self.log_filename = log_filename
        self.n_layers = n_layers
        self.multi_gpu_flag = multi_gpu_flag
        self.nf = nf
        self.nhead = nhead
        self.gpu_index = gpu_index
        self.cont_names = cont_names
        self.cat_names = cat_names
        self.multimodel_flag = multimodel_flag
        self.proj_size = proj_size

    def _load_preprocess_df(self):
        self.df = pd.read_pickle(f'mixed_datasets/mixed_df_fold_{self.fold}.pkl')
        for cont in self.cont_names:
            med = self.df[cont].median()
            self.df[cont] = np.where(self.df[cont] >= 0, self.df[cont], med)
        for cat in self.cat_names:
            self.df[cat] = self.df[cat].astype(str)
            self.df[cat].fillna(f'missing_{cat}', inplace=True)
        
        scaler = StandardScaler()
        self.df[self.cont_names] = scaler.fit_transform(self.df[self.cont_names])

    def _create_transform(self):
        self._transform = torch.nn.Sequential(
            startMove_transform(),
            differences_transform(),
            cutToWindows_transform( self._seq_len, self._stride, self._win_len),
            noise_transform(noise_range = 2)
            )

    def _create_mixed_dls(self):
        if self.multimodel_flag != 'TS_only':
            # First, the tabular dls
            self.to = TabularPandas(self.df,
                    cat_names=self.cat_names, 
                    cont_names=self.cont_names,
                    procs = [Categorify],
                    y_names=["label"],
                    splits=ColSplitter('train_test_flag')(self.df),
                    y_block=CategoryBlock(),
                    reduce_memory=False)
            #tab_dls = self.to.dataloaders(bs=self._batch_size, drop_last=True, pin_memory=True, device=torch.device(f'cuda:{self.gpu_index}'),num_workers=self.num_workers)
            tab_dls = self.to.dataloaders(bs=self._batch_size, drop_last=True, pin_memory=True,num_workers=self.num_workers)
            #tab_dls = tab_dls.cuda()
        
        if self.multimodel_flag != 'tab_only':
            # Now, the TS dls
            def get_x(r):                    
                signal_array = get_signal_array(r['path'], r['read_id'])
                start_point = get_signal_start_index(signal_array)
                signal_array = signal_array[start_point:start_point + 5000]
                signal = self._transform(signal_array)
                return torch.from_numpy(signal).type(torch.FloatTensor)

            def get_y(r):
                return r['label']

            dblock = DataBlock(get_x=get_x, get_y=get_y, splitter=ColSplitter('train_test_flag'))
            #ts_dls = dblock.dataloaders(self.df, bs=self._batch_size, drop_last=True, pin_memory=True, device=torch.device(f'cuda:{self.gpu_index}'),num_workers=self.num_workers)
            ts_dls = dblock.dataloaders(self.df, bs=self._batch_size, drop_last=True, pin_memory=True,num_workers=self.num_workers)
            #ts_dls = ts_dls.cuda()
        
        if self.multimodel_flag == "TS_only":
            self.dls = ts_dls
        elif self.multimodel_flag == "tab_only":
            self.dls = tab_dls
        else:
            #mixed_train = MixedDL(tab_dls[0], ts_dls[0], device = f'cuda:{self.gpu_index}')
            #mixed_valid = MixedDL(tab_dls[1], ts_dls[1], device = f'cuda:{self.gpu_index}')
            mixed_train = MixedDL(tab_dls[0], ts_dls[0])
            mixed_valid = MixedDL(tab_dls[1], ts_dls[1])
            self.dls = DataLoaders(mixed_train, mixed_valid)
            #self.dls = self.dls.cuda()

    def _create_model(self):
        if self.multimodel_flag == "TS_only":
            self._model = TST(c_in = self._seq_len, seq_len=self._win_len,fc_dropout=self.fc_dropout,dropout=self.dropout,d_model=self.d_model, d_k=self.d_k,d_ff=self.d_ff,n_heads=self.nhead,n_layers=self.n_layers, c_out = 2)
        elif self.multimodel_flag == "tab_only":
            self._model = get_tabular_model(self.to, self.cont_names, out_sz=2, embed_p=0.5, layers=[50,80], ps=[0.4,0.5])
        elif self.multimodel_flag == "multimodel":
            self.ts_model = TST(c_in = self._seq_len, seq_len=self._win_len,fc_dropout=self.fc_dropout,dropout=self.dropout,d_model=self.d_model, d_k=self.d_k,d_ff=self.d_ff,n_heads=self.nhead,n_layers=self.n_layers, c_out = self.proj_size)
            self.tab_model = get_tabular_model(self.to, self.cont_names, out_sz=self.proj_size, embed_p=0.4, layers=[50,100], ps=[0.2,0.4])
            self._model = MelModel(self.proj_size, self.tab_model, self.ts_model)
        
        if self.multi_gpu_flag:
            self._model=torch.nn.DataParallel(self._model)

    def _train(self):
        self._create_transform()
        self._load_preprocess_df()
        self._create_mixed_dls()
        self._create_model()

        save_last_model = SaveModelCallback(at_end=True, fname=f'last_model_{self.multimodel_flag}_fold_{self.fold}')
        save_best_model = SaveModelCallback(monitor='valid_loss', fname=f'best_model_{self.multimodel_flag}_fold_{self.fold}')

        learn = Learner(self.dls,
                        self._model,
                        loss_func=LabelSmoothingCrossEntropyFlat(),
                        metrics=[accuracy],
                        cbs=[TrainEvalCallback(),Recorder(train_metrics=True), 
                        ReduceLROnPlateau(), 
                        save_best_model, 
                        save_last_model]
                        )
        learn.to(f'cuda')
        
        self.logger.info('Training started...')
        sys.stdout = open(self.log_filename, 'a')
        #sys.stderr = sys.stdout
        with learn.no_bar():    
            learn.fit_one_cycle(self._max_iter, lr_max=self.optim_lr)
        #sys.stdout = sys.__stdout__
        #sys.stderr = sys.__stderr__
        self.logger.info('Training completed.')