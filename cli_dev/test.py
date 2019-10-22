import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## configure path to package location
import sys
sys.path.append(os.path.abspath('../benchmark_data/'))
sys.path.append(os.path.abspath('../'))

import tensorflow as tf
from BNGPU import NOBEARS
import pandas as pd
import numpy as np




## TODO 0 
## read configuration to set these parameters
## default configuration
reg_init = True
fn_in = ''
poly_degree=3, 
rho_init = 10.
outer_iter = 200 
inner_iter = 100 
init_global_step = 1e-2 
noise_level = 0.1
init_iter = 200
init_global_step = 1e-2
noise_level = 0.1
beta1 = 0.05
beta2 = 0.001
alpha_init = 0.01 
rho_init = 1.0
poly_degree = 3
l1_ratio = .5

init_iter = 200
init_global_step = 1e-2
noise_level = 0.1

outer_iter = 200
inner_iter = 100
init_global_step = 1e-2
noise_level = 0.1


fn_out = 'test_out.csv'

## testing cases

fn_in, read_header, read_index_col = 'test_data0.csv', True, True
#fn_in, read_header, read_index_col = 'test_data1.csv', False, True
#fn_in, read_header, read_index_col = 'test_data2.csv', True, False
#fn_in, read_header, read_index_col = 'test_data3.csv', False, False
        

## END TODO 0

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
    
    df_in = read_file(fn_in, read_index_col = True, read_header = True)
    X = np.vstack([df_in.values] * 2)
    
    if reg_init:
        W_init = NOBEARS.W_reg_init(X).astype('float32')
    else:
        W_init = None
        
    with tf.device("/gpu:0"):
        tf.reset_default_graph()

        clf = NOBEARS.NoBearsTF(beta1 = beta1, beta2 = beta2, alpha_init = alpha_init, 
                                rho_init = rho_init, poly_degree = poly_degree, l1_ratio = l1_ratio)
        clf.construct_graph(X, W_init)

    sess = tf.Session()
    sess.run(clf.graph_nodes['init_vars'])
    clf.model_init_train(sess, init_iter = init_iter, init_global_step = init_global_step, noise_level = noise_level)
    W_est_init = sess.run(clf.graph_nodes['weight'])

    clf.model_train(sess, outer_iter = outer_iter, inner_iter = inner_iter, 
                    init_global_step = init_global_step, noise_level = noise_level)

    W_est = sess.run(clf.graph_nodes['weight_ema'])
    
    df_out = pd.DataFrame(W_est)
    df_out.columns = df_in.columns
    df_out.index = df_in.columns
    df_out.to_csv(fn_out, header = True)
    