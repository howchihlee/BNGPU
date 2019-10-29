import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


## configure path to package location
import sys
sys.path.append(os.path.abspath('../benchmark_data/'))
sys.path.append(os.path.abspath('../'))

import tensorflow as tf
from BNGPU import NOBEARS
import pandas as pd
import numpy as np


import argparse



def read_file(fn, read_index_col = True, read_header = True):
    print(fn)
    if read_index_col and read_header:
        df_in = pd.read_csv(fn, index_col = 0)
    if read_index_col and (not read_header):
        df_in = pd.read_csv(fn, header = None, index_col = 0)
    if (not read_index_col) and read_header:
        df_in = pd.read_csv(fn)
    if (not read_index_col) and (not read_header):
        df_in = pd.read_csv(fn, header = None)

    return df_in

if __name__ == '__main__':




    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_in', help='Filename input', required=True)
    parser.add_argument('--fn_out', help='Fileoutput', required=True)

    parser.add_argument('--poly_degree', help='polynomial degree. Default 3', type=int, default=3 )
    parser.add_argument('--rho_init', help='init rho value. Default 10', type=float, default=10)

    parser.add_argument('--beta1', help='TBD', type=float, default=0.05)
    parser.add_argument('--beta2', help='TBD', type=float, default=0.001)

    parser.add_argument('--alpha_init', help='TBD', type=float, default=0.01)
    parser.add_argument('--l1_ratio', help='TBD', type=float, default=0.5)
    parser.add_argument('--init_iter', help='TBD', type=int, default=200 )
    parser.add_argument('--inner_iter', help='TBD', type=int, default=100 )
    parser.add_argument('--outer_iter', help='TBD', type=int, default=200 )
    parser.add_argument('--init_global_step', help='TBD', type=float, default=0.001)
    parser.add_argument('--noise_level', help='TBD', type=float, default=0.1)



    parser.add_argument('--reg_init', help='TBD', action='store_false', default=True)
    parser.add_argument('--read_header', help='TBD', action='store_false',default=True)
    parser.add_argument('--read_index_col', help='TBD', action='store_false', default=True)


    parser.add_argument('--gpu', help='Decide which GPU. Default 0', default="0")
    args = parser.parse_args()


    fn_in               = args.fn_in
    fn_out              = args.fn_out
    poly_degree         = args.poly_degree
    rho_init            = args.rho_init
    alpha_init          = args.alpha_init
    l1_ratio            = args.l1_ratio
    init_iter           = args.init_iter
    inner_iter          = args.inner_iter
    outer_iter          = args.outer_iter
    init_global_step    = args.init_global_step
    noise_level         = args.noise_level


    beta1               = args.beta1
    beta2               = args.beta2

    reg_init            = args.reg_init
    read_header         = args.read_header
    read_index_col      = args.read_index_col

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(args)


    df_in = read_file(fn_in, read_index_col , read_header )
    print(df_in)
    X = np.vstack([df_in.values] * 2)

    if reg_init:
        W_init = NOBEARS.W_reg_init(X).astype('float32')
    else:
        W_init = None

    # W_init = None
    with tf.device("/gpu:0"):
        tf.reset_default_graph()

        clf = NOBEARS.NoBearsTF(beta1 = beta1, beta2 = beta2, alpha_init = alpha_init,
                                rho_init = rho_init, poly_degree = poly_degree, l1_ratio = l1_ratio)


        # clf.construct_graph(X, W_init)

    # sess = tf.Session()
    # sess.run(clf.graph_nodes['init_vars'])
    # clf.model_init_train(sess, init_iter = init_iter, init_global_step = init_global_step, noise_level = noise_level)
    # W_est_init = sess.run(clf.graph_nodes['weight'])

    # clf.model_train(sess, outer_iter = outer_iter, inner_iter = inner_iter,
    #                 init_global_step = init_global_step, noise_level = noise_level)

    # W_est = sess.run(clf.graph_nodes['weight_ema'])

    # df_out = pd.DataFrame(W_est)
    # df_out.columns = df_in.columns
    # df_out.index = df_in.columns
    # df_out.to_csv(fn_out, header = True)
