# -*-- coding:utf-8 -*--
import numpy as np
import math
from collections import defaultdict
def point_dcg(arg):
    i, label = arg
    return (2 ** label - 1) / math.log(i + 2, 2)

def dcg(scores):
    return sum(map(point_dcg, enumerate(scores)))

def get_dict_by_qid(qids, pred_y):

    unique_qids = np.unique(qids)
    id_y = defaultdict(np.array)
    n_elems = pred_y.shape[0]
    for qid in unique_qids:
        ids_for_qid = np.arange(n_elems)[qids==qid]
        buf = np.zeros(n_elems)
        for idx in ids_for_qid:
            buf[idx] = pred_y[idx]
        id_y[qid] = np.argsort(np.argsort(buf)[::-1])
    return id_y


def ndcgl(pred_y, y, qids, k=10):
    id_y = get_dict_by_qid(qids, pred_y)
    buf = 0.
    for qid in np.unique(qids):
        dcgl = 0.
        idcgl = 0.
        #idx = np.argsort(y_pred)[::-1]
        idx = id_y[qid]
        #l = y[idx]
        for i in range(k):
            #if qids[i] == qid:
            l = y[idx == i][0]
            dcgl += point_dcg((i, l))
            # dcgl += (2. ** l - 1) / np.log2(i + 2)
            idcgl += 1 / np.log2(i + 2)
        buf += dcgl / idcgl
    return buf / len(np.unique(qids))