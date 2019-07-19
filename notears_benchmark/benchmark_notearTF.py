import numpy as np
import scipy.linalg as slin
import timeit
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
sys.path.append(os.path.abspath('../benchmark_data/'))
sys.path.append(os.path.abspath('../'))
import tensorflow as tf
import benchmark_data_reader
from BNGPU import NOTEARS
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


if __name__ == "__main__":
    benreader = benchmark_data_reader.BenchmarkReader()

    res = []

    for i in benreader.get_dataset_name():
        data = benreader.read_data(i)
        W_true = data.W
        X = data.data.copy()
        X = benchmark_data_reader.rank_transform(X)
        X = benchmark_data_reader.mean_var_normalize(X)
        d, n = data.num_gene, data.num_sample
        tic = time()

        with tf.device("/gpu:0"):
            tf.reset_default_graph()
            clf = NOTEARS.NoTearTF()
            clf.construct_graph(X)

        sess = tf.Session()
        sess.run(clf.graph_nodes['init_vars'])
        clf.model_train(sess)

        W_est = sess.run(clf.graph_nodes['weight'])
        time_span = time() - tic

        y_pred = np.abs(W_est.ravel())
        y_true = np.abs(W_true.ravel()) > 1e-5

        res.append((i, d, n, time() - tic, average_precision_score(y_true, y_pred), roc_auc_score(y_true, y_pred)))
        print(i, d, n, time() - tic, average_precision_score(y_true, y_pred), roc_auc_score(y_true, y_pred))

        df_res = pd.DataFrame(res)
        df_res.columns = ['experiment', 'num_gene', 'num_sample', 'runtime', 'aupr', 'roc']
        df_res.to_csv('results/NOTEARS_benchmark_result.csv', index = False)

        np.savez('W_est/%s.npz' % (i.replace('/', '_')), W_est=W_est)