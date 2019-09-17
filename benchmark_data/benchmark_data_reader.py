import pandas as pd
import os
from scipy.io import loadmat
from constants import dataset_list
import numpy as np
from scipy.stats import rankdata

_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))         

def mean_var_normalize(X):
    ## X: [n_sample, n_feature] array
    ## normalize each feature to zero-mean, unit std    
    return (X - np.mean(X, axis = 0, keepdims=True)) / np.std(X, axis = 0, keepdims=True)
        
def rank_transform(X):
    ## X: [n_sample, n_feature] array
    ## apply rank transform to each feature independently
    n, d = X.shape
    for i in range(d):
        X[:, i] = rankdata(X[:, i]) / float(n)
    return X

def transform_data(X, transform):
    if transform == 'rank':
        return rank_transform(X)
    if transform == 'log2':
        return np.log2(X)
    return

class BenchmarkData():
    def __init__(self, X, W_true, id2gene = None):
        ## X: sample x gene numpy array
        ## W_true: gene x gene binary adjacent matrix, numpy array
        ## id2gene: list of gene names
        
        self.data = X.copy()
        self.W = W_true.copy()
        self.num_sample = X.shape[0]
        self.num_gene = X.shape[1]
        
        if id2gene is None:
            self.id2gene = [i for i in range(self.num_gene)]
        else:
            self.id2gene = id2gene
            
class BenchmarkReader():
    def __init__(self):
        self._dataset_list = dataset_list
        self.dataset_name = ['/'.join(s) for s in dataset_list]
        self._dataset_name2list = {'/'.join(s):s for s in dataset_list}
        
        return
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def read_data(self, dataset_name):
        try:
            id0, id1 = self._dataset_name2list[dataset_name]
        except:
            print('find no dataset: %s' % dataset_name)
            print('use get_dataset_name to see all available datasets')
            return
        
        if id0 == 'syntern':
            return self._read_sytern(id1)  
        if id0 == 'GNWeaver':
            return self._read_GNWeaver(id1) 
        
        
    def _read_sytern(self, folder):
        path = _PATH + ('/syntern/%s/' % folder)
        for f in os.listdir(path):
            if f[-4:] == '.sif':
                fn_graph = f
            if f[-20:] == 'maxExpr1_dataset.txt':
                fn_exp = f

        df_graph = pd.read_csv(path + fn_graph, sep = ' ', header = None)
        df_exp = pd.read_csv(path + fn_exp, sep = '\t', index_col = 'GENE')

        id2gene = [g for g in df_exp.index if not g.startswith('bgr_')]
        gene2id = {g:i for i, g in enumerate(id2gene)}
        
        X = df_exp.loc[id2gene].values.T

        y_true = np.zeros((len(id2gene), len(id2gene)))

        for g0, g1 in zip(df_graph[0], df_graph[2]):
            if g0.startswith('bgr_') or g1.startswith('bgr_'):
                continue    
            y_true[gene2id[g0], gene2id[g1]] = 1
                #break
        return BenchmarkData(X, y_true, id2gene)
    
    def _read_GNWeaver(self, folder):
        path = _PATH + '/GNWeaver/%s/' % folder
        X = loadmat(path + 'data_obs.mat')['data_obs']  
        y_true = loadmat(path + 'GS.mat')['GS']

        return BenchmarkData(X, y_true)