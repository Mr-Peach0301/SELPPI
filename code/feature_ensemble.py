import numpy as np
import lightgbm as lgb
import os 
import pandas as pd 
from molfeaturizer import MorganFPFeaturizer, PhysChemFeaturizer,HashAPFPFeaturizer, predefined_mordred
from sklearn.model_selection import KFold,train_test_split
from metrics import compute_cls_metrics, compute_reg_metrics
from prettytable import PrettyTable
from sklearn.linear_model import LogisticRegression,LinearRegression,ElasticNet
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dgllife.utils import EarlyStopping
from tqdm import tqdm 
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import xgboost as xgb

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from deepforest import CascadeForestClassifier, CascadeForestRegressor


task_name = 'regression' 
dataset_name = 'ledgf_in' 

def batchify(iterable, batch_size):
    for ndx in range(0, len(iterable), batch_size):
        batch = iterable[ndx: min(ndx + batch_size, len(iterable))]
        yield batch
 


data_path = './data/'
def ptable_to_csv(table, filename, headers=True):
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

def lgb_model(x_train, y_train,  x_test, task_name):
    if task_name == 'classification':
        objective = 'binary'
        metric = 'auc'
    else:
        objective = 'regression'
        metric = 'rmse'

    num_round = 1000
    param = {'num_leaves': 100, 'objective': objective, 'metric': metric, 'verbose': -1}
        
    train_data = lgb.Dataset(x_train, label=y_train)
    # validation_data = lgb.Dataset(x_val, label=y_val)

    bst = lgb.train(param, train_data, num_round)

    ypred_test = bst.predict(x_test)
    ypred_train = bst.predict(x_train)
    
    return ypred_test, ypred_train

def xgb_model(x_train, y_train,  x_test, task_name):
    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data =  xgb.DMatrix(x_test)
    if task_name == 'classification':
        objective = 'binary:logistic'
        metric = 'auc'
    else:
        objective = 'reg:linear'
        metric = 'rmse'
    param_grid = {
                'n_estimators': 1000,
                'max_depth':10,
                'objective': objective,
                'eval_metric': metric,
            }

    
    clf = xgb.train(param_grid, train_data)
    ypred_test = clf.predict(test_data)
    ypred_train = clf.predict(train_data)
    return  ypred_test, ypred_train

def extra_tree_model(x_train, y_train,  x_test, task_name):
    if task_name == 'classification':
        lm = ExtraTreesClassifier(n_estimators=1000, random_state=0) 
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict_proba(x_test)[:,1:]  
        ypred_train = lm.predict_proba(x_train)[:,1:]  
    else:
        lm = ExtraTreesRegressor(n_estimators=1000, random_state=0 )
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict(x_test)
        ypred_train = lm.predict(x_train)
    print(ypred_test.shape, ypred_train.shape)
    return ypred_test, ypred_train

def rf_model(x_train, y_train,  x_test, task_name ):
    if task_name == 'classification':
        lm = RandomForestClassifier(max_depth=10,n_estimators=1000,  random_state=0) 
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict_proba(x_test)[:,1:] 
        ypred_train = lm.predict_proba(x_train)[:,1:]   
    else:
        lm = RandomForestRegressor(max_depth=10,n_estimators=1000, random_state=0 )
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict(x_test)
        ypred_train = lm.predict(x_train)
    print(ypred_test.shape, ypred_train.shape) 
    return ypred_test, ypred_train

def df_model(x_train, y_train,  x_test, task_name ):
    if task_name == 'classification':
        model = CascadeForestClassifier(random_state=1)
        model.fit(x_train, y_train)
        
        ypred_test = model.predict_proba(x_test)[:,1:] 
        ypred_train = model.predict_proba(x_train)[:,1:]   
    else:
        model = CascadeForestRegressor(random_state=1)
        model.fit(x_train, y_train)
          
        ypred_test = model.predict(x_test).flatten().astype(float)
        ypred_train = model.predict(x_train).flatten().astype(float)
    
    print(ypred_test.shape, ypred_train.shape)
    return ypred_test, ypred_train

 
def gb_model(x_train, y_train,   x_test, task_name ):
    if task_name == 'classification':
        lm = GradientBoostingClassifier(max_depth=10,n_estimators=1000,  random_state=0) 
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict_proba(x_test)[:,1:] 
        ypred_train = lm.predict_proba(x_train)[:,1:]   
    else:
        lm = GradientBoostingRegressor(max_depth=10,n_estimators=1000, random_state=0 )
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict(x_test)
        ypred_train = lm.predict(x_train)
 
    return ypred_test, ypred_train

def ada_model(x_train, y_train, x_test, task_name ):
    if task_name == 'classification':
        lm = AdaBoostClassifier( n_estimators=1000,  random_state=0) 
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict_proba(x_test)[:,1:]  
        ypred_train = lm.predict_proba(x_train)[:,1:]  
    else:
        lm = AdaBoostRegressor( n_estimators=1000, random_state=0 )
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict(x_test)
        ypred_train = lm.predict(x_train)
    return ypred_test, ypred_train

def mlp_model(x_train, y_train, x_test, task_name ):
    if task_name == 'classification':
        lm = MLPClassifier(random_state=1, max_iter=300,   learning_rate_init=1e-4)
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict_proba(x_test)[:,1:]  
        ypred_train = lm.predict_proba(x_train)[:,1:]  
    else:
        lm = MLPRegressor(random_state=1, max_iter=300,   learning_rate_init=1e-4)
        lm.fit(x_train, y_train)   
        ypred_test = lm.predict(x_test)
        ypred_train = lm.predict(x_train)
    return ypred_test, ypred_train
 
def compute_metrics(y_true, y_pred, task_name, method_name):
    if task_name == 'classification':
        F1, roc_auc, mcc,  tn, fp, fn, tp = compute_cls_metrics(y_true,y_pred)
        row = [method_name,  F1, roc_auc, mcc,  tp, tn, fp, fn]
    else:
        tau, rho, r, rmse, mae =  compute_reg_metrics(y_true,y_pred)
        row = [method_name, r, tau, rho, rmse, mae] 
    return row


def train_test(dataset_name, data_path):
    descriptor_sets = ['fragment', 'surface','druglikeness',   'refractivity', 'estate', 'charge', 'atom-bond-ring']
    rdkit_norm_fns = []
     
    rdkit_norm = PhysChemFeaturizer(normalise=  True, named_descriptor_set='surface-fragment')
    rdkit_norm_fn = lambda smiles: rdkit_norm.transform(smiles)[0]
 
    rdkit_norm1 = PhysChemFeaturizer(normalise=  True, named_descriptor_set='fragment')
    rdkit_norm_fn_fragement  = lambda smiles: rdkit_norm1.transform(smiles)[0]
    rdkit_norm_fns.append(rdkit_norm_fn_fragement)
    rdkit_norm2 = PhysChemFeaturizer(normalise=  True, named_descriptor_set='surface')
    rdkit_norm_fn_surface  = lambda smiles: rdkit_norm2.transform(smiles)[0]
    rdkit_norm_fns.append(rdkit_norm_fn_surface)
    rdkit_norm4 = PhysChemFeaturizer(normalise=  True, named_descriptor_set='druglikeness')
    rdkit_norm_fn_drug  = lambda smiles: rdkit_norm4.transform(smiles)[0]
    rdkit_norm_fns.append(rdkit_norm_fn_drug)
    rdkit_norm6 = PhysChemFeaturizer(normalise=  True, named_descriptor_set='refractivity')
    rdkit_norm_fn_refractivity  = lambda smiles: rdkit_norm6.transform(smiles)[0]
    rdkit_norm_fns.append(rdkit_norm_fn_refractivity)

    rdkit_norm7 = PhysChemFeaturizer(normalise=  True, named_descriptor_set='estate')
    rdkit_norm_fn_estate  = lambda smiles: rdkit_norm7.transform(smiles)[0]
    rdkit_norm_fns.append(rdkit_norm_fn_estate)

    rdkit_norm8 = PhysChemFeaturizer(normalise=  True, named_descriptor_set='charge')
    rdkit_norm_fn_charge = lambda smiles: rdkit_norm8.transform(smiles)[0]
    rdkit_norm_fns.append(rdkit_norm_fn_charge)

     
    rdkit_norm_fn = lambda smiles: predefined_mordred( smiles, 'atom-bond-ring')
    rdkit_norm_fns.append(rdkit_norm_fn)
     


    file_path = os.path.join(data_path, task_name )
    
    train_df = pd.read_csv(os.path.join(file_path, dataset_name+'_train.csv'))
    test_df = pd.read_csv(os.path.join(file_path, dataset_name+'_test.csv'))

    if task_name == 'classification':
        train_smiles = train_df[train_df.columns[0]].values
        train_labels = train_df[train_df.columns[-2]].replace('Inhibitor',1).replace('Non-inhibitor',0)
        test_smiles = test_df[test_df.columns[0]].values
        test_labels = test_df[test_df.columns[-2]].replace('Inhibitor',1).replace('Non-inhibitor',0)
    else:
        train_smiles = train_df[train_df.columns[0]].values
        train_labels = train_df[train_df.columns[-2]].values
        test_smiles = test_df[test_df.columns[0]].values
        test_labels = test_df[test_df.columns[-2]].values
    
    test_features =[]
     
    for rdkit_norm_fn in  rdkit_norm_fns:
        test_features.append(np.vstack([rdkit_norm_fn(batch) for batch in batchify(test_smiles, 32)]))
    
     
    if task_name == 'classification':
        t_tables = PrettyTable(['method', 'F1', 'AUC', 'MCC', 'TP', 'TN', 'FP', 'FN'  ])
    else:
        t_tables = PrettyTable(['method','R', 'Kendall', 'Spearman', 'RMSE', 'MAE' ])

    t_tables.float_format = '.3'  
    train_smiles_cv = train_smiles 
    train_Y_cv = np.array(train_labels)
    train_features =[]

    for rdkit_norm_fn in  rdkit_norm_fns:   
        train_features.append(np.vstack([rdkit_norm_fn(batch) for batch in batchify(train_smiles_cv, 32)]))
    
    tt=np.array(test_features)
    print(np.shape(tt))
    for split in range(1): 
        pred_test = []
        pred_train = []
        for i in range(len(train_features)):
            et_pred_test, et_pred_train  = extra_tree_model(train_features[i], train_Y_cv,   test_features[i], task_name )
            
            pred_test.append( et_pred_test.reshape( test_labels.shape[0], 1) )
            pred_train.append( et_pred_train.reshape( train_Y_cv.shape[0], 1) )
        for i in range(len(train_features)):
            et_pred_test, et_pred_train  = ada_model(train_features[i], train_Y_cv,   test_features[i], task_name )
            pred_test.append( et_pred_test.reshape( test_labels.shape[0], 1) )
            pred_train.append( et_pred_train.reshape( train_Y_cv.shape[0], 1) )
        for i in range(len(train_features)):
            et_pred_test, et_pred_train  = rf_model(train_features[i], train_Y_cv,   test_features[i], task_name )
            pred_test.append( et_pred_test.reshape( test_labels.shape[0], 1) )
            pred_train.append( et_pred_train.reshape( train_Y_cv.shape[0], 1) )
        for i in range(len(train_features)):
            et_pred_test, et_pred_train  = df_model(train_features[i], train_Y_cv,   test_features[i], task_name )
            pred_test.append( et_pred_test.reshape( test_labels.shape[0], 1) )
            pred_train.append( et_pred_train.reshape( train_Y_cv.shape[0], 1) )
        for i in range(len(train_features)):
            et_pred_test, et_pred_train  = lgb_model(train_features[i], train_Y_cv,   test_features[i], task_name )
            pred_test.append( et_pred_test.reshape( test_labels.shape[0], 1) )
            pred_train.append( et_pred_train.reshape( train_Y_cv.shape[0], 1) )
        for i in range(len(train_features)): 
            et_pred_test, et_pred_train  = xgb_model(train_features[i], train_Y_cv,   test_features[i], task_name )
            pred_test.append( et_pred_test.reshape( test_labels.shape[0], 1) )
            pred_train.append( et_pred_train.reshape( train_Y_cv.shape[0], 1) )
        train_features_mixture = np.concatenate(  pred_train, axis=1 ) 
        test_features_mixture = np.concatenate( pred_test, axis=1 )
        print("ha")
    

        if task_name == 'classification':
            lm = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5) 
            lm.fit(train_features_mixture, train_Y_cv)   
            ypred = lm.predict_proba(test_features_mixture)[:,1:]  
        else:
            lm = LinearRegression() 
            lm.fit(train_features_mixture, train_Y_cv)   
            ypred = lm.predict(test_features_mixture) 
        
        
        row = compute_metrics(test_labels,ypred, task_name, 'stackPPIM-LR' )
         
        t_tables.add_row(row)
        
        
        df_pred_test, df_pred_train  = df_model(train_features_mixture, train_Y_cv,  test_features_mixture, task_name )
            
        row = compute_metrics(test_labels, df_pred_test, task_name, 'stackPPIM-DF' )
         
        t_tables.add_row(row)
        
        et_pred_test, et_pred_train  = extra_tree_model(train_features_mixture, train_Y_cv,  test_features_mixture, task_name )
        
        row = compute_metrics(test_labels, et_pred_test, task_name, 'stackPPIM-ET' )
         
        t_tables.add_row(row)

        ada_pred_test, ada_pred_train  = ada_model(train_features_mixture, train_Y_cv,  test_features_mixture, task_name )

        row = compute_metrics(test_labels, ada_pred_test, task_name, 'stackPPIM-ADA' )
         
        t_tables.add_row(row)
            
        

        rf_pred_test, rf_pred_train  = rf_model(train_features_mixture, train_Y_cv,  test_features_mixture, task_name )
        row = compute_metrics(test_labels, rf_pred_test, task_name, 'stackPPIM-RF' )
         
        t_tables.add_row(row)

        lgb_pred_test, lgb_pred_train  = lgb_model(train_features_mixture, train_Y_cv,   test_features_mixture, task_name )
        row = compute_metrics(test_labels, lgb_pred_test, task_name, 'stackPPIM-LGB' )
         
        t_tables.add_row(row)
         

        mlp_pred_test, mlp_pred_train  = mlp_model(train_features_mixture, train_Y_cv,   test_features_mixture, task_name )
        
        row = compute_metrics(test_labels, mlp_pred_test, task_name, 'stackPPIM-MLP' )
         
        t_tables.add_row(row)


        xgb_pred_test, xgb_pred_train  = xgb_model(train_features_mixture, train_Y_cv,   test_features_mixture, task_name )
        
        row = compute_metrics(test_labels, xgb_pred_test, task_name, 'stackPPIM-XGB' )
         
        t_tables.add_row(row)

        print(t_tables)
        
        


        if task_name == 'classification':
            n_features = len(descriptor_sets)
            methods = ['ET', 'ADA', 'RF', 'DF', 'LGB', 'XGB']
            for j, method in enumerate(methods):
                for i in range(n_features):
                    F1, roc_auc, mcc,  tn, fp, fn, tp = compute_cls_metrics(test_labels, pred_test[i + j * len(methods)])
                    row = [ descriptor_sets[i]+ '-' + method, F1, roc_auc, mcc,  tp, tn, fp, fn]
                    t_tables.add_row(row)
             

        else:
            n_features = len(descriptor_sets)
            methods = ['ET', 'ADA', 'RF', 'DF', 'LGB', 'XGB']
            for j, method in enumerate(methods):
                for i in range(n_features):
                    tau, rho, r, rmse, mae = compute_reg_metrics(test_labels , pred_test[i+ j * len(methods)] )
                    row = [ descriptor_sets[i]+ '-'+ method,r, tau, rho, rmse, mae]
                    t_tables.add_row(row)
                    print(test_labels)
                    print(pred_test[i+ j * len(methods)])
        print(t_tables) 

    results_filename = 'results/feature_enesmble_PPIM-' + dataset_name + '-' + task_name + '.csv'
    ptable_to_csv(t_tables, results_filename)

    df = pd.read_csv(results_filename)


    df2 = df.groupby('method').mean()
    print(df2)

 

         
        

train_test(dataset_name, data_path)
 
