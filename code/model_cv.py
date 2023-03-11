import datetime
from time import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
import xgboost as xgb

class CrossLgbModel(object):
    def __init__(self, metric, objective):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.metric = metric
        self.object = objective
        

             
        self.params_ = {
            'objective':  objective,
            'boosting': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 2 ** 5,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 66,
            'feature_fraction': 0.7,
            'feature_fraction_seed': 66,
            'max_bin': 100,
            'max_depth': 5,
            'verbose': -1
        }
       
        self.params_['metric'] = self.metric
        self.Early_Stopping_Rounds = 150
        self.N_round = 8000
        self.Verbose = 100
         

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X_train, X_valid, y_train, y_valid, metric, Debug=False):
        def objective(trial):
            param_grid = {
                'num_leaves': trial.suggest_int('num_leaves', 2 ** 3, 2**9),
                'num_boost_round': trial.suggest_int('num_boost_round', 100, 8000),
                'max_depth': trial.suggest_int('max_depth', 1, 9),
                'objective': self.object,
                'boosting': 'gbdt',
                'learning_rate': trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
                'bagging_fraction': trial.suggest_discrete_uniform('bagging_fraction', 0.5, 0.95, 0.05),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'bagging_seed': 66,
                'feature_fraction': trial.suggest_discrete_uniform('feature_fraction', 0.5, 0.95, 0.05),
                'feature_fraction_seed': 66,
                'max_bin': 100,
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1
            }
            param_grid['metric'] = metric
            trn_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_valid, label=y_valid)
            clf = lgb.train(param_grid, trn_data, valid_sets=[trn_data, val_data], verbose_eval=False,
                            early_stopping_rounds=self.Early_Stopping_Rounds)
            pred_val = clf.predict(X_valid)
            if self.object == 'regression':
                object_val = mean_squared_error(y_valid, pred_val)
            else:
                object_val = roc_auc_score(y_valid, pred_val)
              

            return object_val

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        if self.object == 'regression':    
            study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='LgbModel')
        else:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name='LgbModel')
        study.optimize(objective, timeout=train_time)

        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trial:')
        trial = study.best_trial

        print(f'\tValue: {trial.value}')
        print('\tParams: ')
        for key, value in trial.params.items():
            print('\t\t{}: {}'.format(key, value))

        self.params_ = trial.params
        self.N_round = trial.params['num_boost_round']

    def fit(self, X_train, X_valid, y_train, y_valid, test,  Early_Stopping_Rounds=None, N_round=None, Verbose=None, tuning=True, Debug=False):
         
        

        if tuning:
            print("[+]tuning params")
            self.optuna_tuning(X_train, X_valid, y_train, y_valid, metric=self.metric, Debug=Debug)

        if Early_Stopping_Rounds is not None:
            self.Early_Stopping_Rounds = Early_Stopping_Rounds
        if N_round is not None:
            self.N_round = N_round
        if Verbose is not None:
            self.Verbose = Verbose

         

        trn_data = lgb.Dataset(X_train,label=y_train)
        val_data = lgb.Dataset(X_valid,label=y_valid)

        model = lgb.train(self.params_, trn_data, num_boost_round=self.N_round, valid_sets=[trn_data, val_data],
                        verbose_eval=self.Verbose,
                        early_stopping_rounds=self.Early_Stopping_Rounds)
                
        self.feature_importances_  = model.feature_importance()
        self.bst = model
        result = model.predict(test,num_iteration=model.best_iteration)
        return result
             

class CrossXGBModel(object):
    def __init__(self, metric, objective):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.metric = metric
        if objective == 'binary':
            self.object = 'binary:logistic'
        else:
            self.object = 'reg:linear'

        self.params_ = {
            'objective':  objective,
            'boosting': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 2 ** 5,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 66,
            'feature_fraction': 0.7,
            'feature_fraction_seed': 66,
            'max_bin': 100,
            'max_depth': 5,
            'verbose': -1
        }
       
        self.params_['metric'] = self.metric
        self.Early_Stopping_Rounds = 150
        self.N_round = 8000
        self.Verbose = 100
         

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X_train, X_valid, y_train, y_valid, metric, Debug=False):
        def objective(trial):
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 2 ** 3, 2**9),
                'num_boost_round': trial.suggest_int('num_boost_round', 100, 8000),
                'max_depth': trial.suggest_int('max_depth', 1, 9),
                'gamma': trial.suggest_int('gamma', 0, 5),
                'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
                'objective': self.object,
                'eval_metric': self.metric,
                'learning_rate': trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
                'max_bin': 100,
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9)

            }
            param_grid['metric'] = metric
            trn_data =  xgb.DMatrix(X_train, label=y_train)
            val_data =  xgb.DMatrix(X_valid, label=y_valid)
            clf = xgb.train(param_grid, trn_data, evals= [(val_data,'eval'), (trn_data,'train')], 
                            early_stopping_rounds=self.Early_Stopping_Rounds)
            pred_val = clf.predict(val_data)
            if self.object == 'regression':
                object_val = mean_squared_error(y_valid, pred_val)
            else:
                object_val = roc_auc_score(y_valid, pred_val)
              

            return object_val

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        if self.object == 'regression':    
            study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XGBModel')
        else:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name='XGBModel')
        study.optimize(objective, timeout=train_time)

        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trial:')
        trial = study.best_trial

        print(f'\tValue: {trial.value}')
        print('\tParams: ')
        for key, value in trial.params.items():
            print('\t\t{}: {}'.format(key, value))
        self.params_ = trial.params
        self.N_round = trial.params['num_boost_round']
        

    def fit(self, X_train, X_valid, y_train, y_valid, test,  Early_Stopping_Rounds=None, N_round=None, Verbose=None, tuning=True, Debug=False):
         
        

        if tuning:
            print("[+]tuning params")
            self.optuna_tuning(X_train, X_valid, y_train, y_valid, metric=self.metric, Debug=Debug)

        if Early_Stopping_Rounds is not None:
            self.Early_Stopping_Rounds = Early_Stopping_Rounds
        if N_round is not None:
            self.N_round = N_round
        if Verbose is not None:
            self.Verbose = Verbose

         

        trn_data = xgb.DMatrix(X_train,label=y_train)
        val_data = xgb.DMatrix(X_valid,label=y_valid)

        

        model = xgb.train(self.params_, trn_data,   evals= [(val_data,'eval'), (trn_data,'train')],
                        early_stopping_rounds=self.Early_Stopping_Rounds)
                
        self.bst = model
        test_data =  xgb.DMatrix(test)
        result = model.predict(test_data)
        return result
