import numpy as np
import scipy.linalg as slin
import networkx as nx
import timeit
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append(os.path.abspath('../benchmark_data/'))
sys.path.append(os.path.abspath('../'))
import tensorflow as tf
import benchmark_data_reader
from BNGPU import NOBEARS
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


if __name__ == '__main__':
    benreader = benchmark_data_reader.BenchmarkReader()
    res = []

    for i in benreader.get_dataset_name():
        data = benreader.read_data(i)
        W_true = data.W

        tic = time()
        
        ## training no-bears 
        W_init = NOBEARS.W_reg_init(data.data).astype('float32')
        X = data.data.copy()
        X = benchmark_data_reader.rank_transform(X)
        X = benchmark_data_reader.mean_var_normalize(X)
        X = np.vstack([X] * 2)
        d, n = data.num_gene, data.num_sample


        with tf.device("/gpu:0"):
            tf.reset_default_graph()

            clf = NOBEARS.NoBearsTF(poly_degree=3, rho_init = 10.)
            clf.construct_graph(X, W_init)

        sess = tf.Session()
        sess.run(clf.graph_nodes['init_vars'])
        clf.model_init_train(sess)
        W_est_init = sess.run(clf.graph_nodes['weight'])
        
        clf.model_train(sess)
        
        W_est = sess.run(clf.graph_nodes['weight_ema'])
        ## END of training no-bears 
        
        time_span = time() - tic

        y_pred = np.abs(W_est.ravel())
        y_true = np.abs(W_true.ravel()) > 1e-5
        
        s0 = average_precision_score(y_true, y_pred)
        s1 = roc_auc_score(y_true, y_pred)
        
        y_pred = np.abs(W_est_init.ravel())        
        s2 = average_precision_score(y_true, y_pred)
        s3 = roc_auc_score(y_true, y_pred)
        res.append((i, d, n, time() - tic, s0, s1, s2, s3))
        
        df_res = pd.DataFrame(res)
        df_res.columns = ['experiment', 'num_gene', 'num_sample', 'runtime', 'aupr', 'roc', 'aupr_init', 'roc_init']
        df_res.to_csv('results/NOBEARS_benchmark_result.csv', index = False)

        np.savez('W_est/NOBEARS_%s.npz' % i.replace('/', '_'), W_est=W_est, W_est_init = W_est_init)