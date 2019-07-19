import tensorflow as tf
import numpy as np

class NoTearTF():
    
    def __init__(self, lamb = 0.05, alpha_init = 0.00, rho_init = 1.0):
        self.alpha = alpha_init
        self.rho = rho_init
        self.lamb = lamb
        
    def model_train(self, sess, outer_iter = 50, inner_iter = 100, init_global_step = 1e-2, h_tol = 1e-8, eval_fun = None):

        h, h_new = np.inf, np.inf
        for _ in range(outer_iter):
            sess.run(self.graph_nodes['reset_opt'])
                
            for t1 in range(inner_iter):
                feed_dict = {self.graph_nodes['alpha']: self.alpha, 
                             self.graph_nodes['rho']: self.rho, 
                             self.graph_nodes['opt_step'] : init_global_step / np.sqrt(1.+ t1),
                             self.graph_nodes['lambda']: self.lamb}

                sess.run(self.graph_nodes['train_op'], feed_dict = feed_dict)

            h_ = sess.run(self.graph_nodes['loss_penalty'], 
                                  feed_dict = feed_dict)

            if h <= h_tol:
                break
            self.alpha += self.rho * h_
            self.rho *= 1.25
        return
            
    def construct_graph(self, X):
        ## X: nxd numpy array
        n, d = X.shape

        X_tf = tf.constant(X.astype('float32'), name='sem_data')
        W_tf = tf.get_variable("W", shape = (d, d))
        W_tf = tf.linalg.set_diag(W_tf, tf.zeros(d))
        
        rho_tf = tf.placeholder(tf.float32, name = 'rho')
        lambda_tf = tf.placeholder(tf.float32, name = 'lambda')
        d_lrd = tf.placeholder(tf.float32, name = 'opt_step')
        alpha_tf = tf.placeholder(tf.float32, name = 'alpha')

        d_xw = X_tf - tf.matmul(X_tf, W_tf)

        loss_penalty = tf.linalg.trace(tf.linalg.expm(W_tf ** 2)) - float(d)

        loss_xw = tf.reduce_sum(d_xw ** 2) / (float(n) * 2.0)
        loss_obj = loss_xw + (rho_tf / 2.0) * (loss_penalty ** 2) + alpha_tf * loss_penalty
        loss_obj = loss_obj + lambda_tf * tf.reduce_sum(tf.abs(W_tf))
        
        optim = tf.train.AdamOptimizer(d_lrd, beta1=0.9)     
        train_op = optim.minimize(loss_obj)

        reset_optimizer_op = tf.variables_initializer(optim.variables())
        model_init = tf.global_variables_initializer()

        self.graph_nodes = {'sem_data': X_tf,
                'weight': W_tf,
                 'rho': rho_tf,
                 'loss_penalty': loss_penalty,
                 'lambda': lambda_tf,
                 'opt_step': d_lrd,
                 'alpha': alpha_tf,
                 'loss_regress': loss_xw,
                 'loss_obj': loss_obj,
                 'train_op': train_op,
                 'reset_opt': reset_optimizer_op,
                 'init_vars': model_init,
                }

        return self.graph_nodes
    