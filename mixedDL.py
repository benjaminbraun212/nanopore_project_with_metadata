from fastai.tabular.core import *
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
from fastai.vision.all import *
from utils import * 
from tsai.all import *

class MixedDL():
    def __init__(self, *dls, device='cpu'):
        "Accepts any number of `DataLoaders` and a device"
        self.device = device
        for dl in dls: dl.shuffle_fn = self.shuffle_fn
        self.dls = dls
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0, 0, pin_memory_device=self.device)
        self._get_idxs()
        self.n_inp = 1
        
    def __len__(self): return len(self.dls[0])
    
    def _get_vals(self, x):
        "Checks for duplicates in batches"
        idxs, new_x = [], []
        for i, o in enumerate(x): x[i] = o.cpu().numpy().flatten()
        for idx, o in enumerate(x):
            if not _arrayisin(o, new_x):
                idxs.append(idx)
                new_x.append(o)
        return idxs
    
    def _get_idxs(self):
        "Get `x` and `y` indicies for batches of data"
        dl_dict = dict(zip(range(0,len(self.dls)), [dl.n_inp for dl in self.dls]))
        inps = L([])
        outs = L([])
        for key, n_inp in dl_dict.items():
            b = next(iter(self.dls[key]))
            inps += L(b[:n_inp])
            outs += L(b[n_inp:])
        self.x_idxs = self._get_vals(inps)
        self.y_idxs = self._get_vals(outs)
    
    def __iter__(self):
        z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in self.dls])
        for b in z:   
            inps = []
            outs = []
            if self.device is not None: 
                b = to_device(b, self.device)
            for batch, dl in zip(b, self.dls):
                batch = dl.after_batch(batch)
                inps += batch[:dl.n_inp]
                outs += batch[dl.n_inp:]
            inps = L(inps)[self.x_idxs]
            outs = L(outs)[self.y_idxs]
            yield (inps, outs[0])
                
    def one_batch(self):
        "Grab one batch of data"
        with self.fake_l.no_multiproc(): res = first(self)
        if hasattr(self, 'it'): delattr(self, 'it')
        return res
    
    def shuffle_fn(self, idxs):
        "Shuffle the internal `DataLoaders`"
        if self.count == 0:
            self.rng = self.dls[0].rng.sample(idxs, len(idxs))
            self.count += 1
            return self.rng
        if self.count == 1:
            self.count = 0
            return self.rng

        
    def show_batch(self):
        "Show a batch of data"
        for dl in self.dls:
            dl.show_batch()
            
    def to(self, device): self.device = device
        

def _arrayisin(arr, arr_list):
    "Checks if `arr` is in `arr_list`"
    for a in arr_list:
        if np.array_equal(arr, a):
            return True
    return False

class MelModel(Module):
    def __init__(self, proj_size, tab_model, ts_model, *args, **kwargs):
        self.tab_model = tab_model
        self.ts_model = ts_model
        #self.final_head = nn.Linear(proj_size * 2, 2)
        self.final_head = nn.Sequential(nn.BatchNorm1d(proj_size * 2),
                            nn.Dropout(0.25),
                            nn.Linear(200, 100, bias=False),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(100),
                            nn.Dropout(0.5),
                            nn.Linear(100, 2)
                    )

    def forward(self, x):
        tab = self.tab_model(x[0], x[1])
        ts = self.ts_model(x[2])
        return self.final_head(torch.cat([tab, ts], dim=1))
