import json
import numpy as np
import progressbar as pb
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    filename='log')
logger = logging.getLogger('user')
streamhandler=logging.StreamHandler()
logger.addHandler(streamhandler)

lll_types = ["ewc","si","mas"]

def get_pbar(num, prefix=""):
    assert isinstance(prefix, str)
    pbar = pb.ProgressBar(widgets=[prefix, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num)
    return pbar


def json_pretty_dump(obj, fh):
    return json.dump(obj, fh, sort_keys=True, indent=2, separators=(',', ': '))


def collect_indices(data,index_size,embedding_size):
    indices_dict = {i:[] for i in range(index_size)}
    indices = data.indices
    values = data.values
    for i,v in zip(indices,values):
        #dim of v == embedding_size = hidden_size = 50
        indices_dict[i].append(v)
    for k,v in indices_dict.items():
        if len(v) == 0:
            indices_dict[k].append(np.zeros(embedding_size))
    return indices_dict

def get_indices(indices_dict,kind):
    if kind == "ave":
        d = map(lambda k: np.mean(np.array(k[1]),axis=0), indices_dict.items())
    elif kind == "last":
        d = map(lambda k: k[1][-1], indices_dict.items())
    elif kind == 'sum':
        d = map(lambda k: np.sum(np.array(k[1]),axis=0), indices_dict.items())
    return list(d) 

def get_indices_ave(indices_dict):
    d = map(lambda k: np.mean(np.array(k[1]),axis=0), indices_dict.items())
    return list(d) 

def get_last_indices(indices_dict):
    d = map(lambda k: k[1][-1], indices_dict.items())
    return list(d)

def ema(decay, prev_val, new_val):
    if decay == "sum":
        return prev_val + new_val
    return decay * prev_val + (1.0 - decay) * new_val
