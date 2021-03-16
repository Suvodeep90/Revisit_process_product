import pandas as pd
import numpy as np
import math
import pickle

from scipy import stats
import scipy.io
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky
from scipy.io import loadmat

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from newabcd import ABCD
import SMOTE
import CFS
import metrices_V2 as metrices
import tuner, learners
from learners import SK_SVM, SK_LSR, SK_NB
from tuner import DE_Tune_ML

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
from pathlib import Path
import time



from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def apply_smote(df):
    df.reset_index(drop=True,inplace=True)
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df

def apply_cfs(df):
    y = df.Bugs.values
    X = df.drop(labels = ['Bugs'],axis = 1)
    X = X.values
    selected_cols = CFS.cfs(X,y)
    cols = df.columns[[selected_cols]].tolist()
    cols.append('Bugs')
    return df[cols],cols


def load_both_data(project,metric):
    understand_path = 'data/understand_files_all/' + project + '_understand.csv'
    understand_df = pd.read_csv(understand_path)
    understand_df = understand_df.dropna(axis = 1,how='all')
    cols_list = understand_df.columns.values.tolist()
    for item in ['Kind', 'Name','commit_hash', 'Bugs']:
        if item in cols_list:
            cols_list.remove(item)
            cols_list.insert(0,item)
    understand_df = understand_df[cols_list]
    cols = understand_df.columns.tolist()
    understand_df = understand_df.drop_duplicates(cols[4:len(cols)])
    understand_df['Name'] = understand_df.Name.str.rsplit('.',1).str[1]
    
    commit_guru_file_level_path = 'data/commit_guru_file/' + project + '.csv'
    commit_guru_file_level_df = pd.read_csv(commit_guru_file_level_path)
    commit_guru_file_level_df['commit_hash'] = commit_guru_file_level_df.commit_hash.str.strip('"')
    commit_guru_file_level_df = commit_guru_file_level_df[commit_guru_file_level_df['file_name'].str.contains('.java')]
    commit_guru_file_level_df['Name'] = commit_guru_file_level_df.file_name.str.rsplit('/',1).str[1].str.split('.').str[0].str.replace('/','.')
    commit_guru_file_level_df = commit_guru_file_level_df.drop('file_name',axis = 1)
    
    df = understand_df.merge(commit_guru_file_level_df,how='left',on=['commit_hash','Name'])
      
    cols = df.columns.tolist()
    cols.remove('Bugs')
    cols.append('Bugs')
    df = df[cols]
    
    for item in ['Kind', 'Name','commit_hash']:
        if item in cols:
            df = df.drop(labels = [item],axis=1)
#     df.dropna(inplace=True)
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    
    y = df.Bugs
    X = df.drop('Bugs',axis = 1)
    cols = X.columns
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    imp_mean = IterativeImputer(random_state=0)
    X = imp_mean.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    
    if metric == 'process':
        X = X[['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
       'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
       'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
       'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr']]
    elif metric == 'product':
        X = X.drop(['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
       'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
       'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
       'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr'],axis = 1)
    else:
        X = X
    return X,y

def tune_learner(learner, train_X, train_Y, tune_X, tune_Y, goal,
                 target_class=None):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :param target_class:
    :return:
    """
    if not target_class:
        target_class = goal
    clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
    tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
    return tuner.Tune()


def learn(learner,clf,X_train, y_train, X_val, y_val, X_test, y_test, goal, tuning, loc):
    params, evaluation = tune_learner(learner, X_train, y_train, X_val,
                                    y_val, goal) if tuning else ({}, 0)
    clf.set_params(**params)
    clf.fit(X_train,y_train)
    try:
        importance = clf.feature_importances_ 
    except:
        importance = 0
    predicted = clf.predict(X_test)
    abcd = metrices.measures(y_test,predicted,loc)
    pf = abcd.get_pf()
    recall = abcd.calculate_recall()
    precision = abcd.calculate_precision()
    f1 = abcd.calculate_f1_score()
    g_score = abcd.get_g_score()
    pci_20 = abcd.get_pci_20()
    ifa = abcd.get_ifa()
    try:
        auc = roc_auc_score(y_test, predicted)
    except:
        auc = 0
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance

def run_self_k(project,metric):
    tuning=True
    precision = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    recall = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    pf = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    f1 = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    g_score = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    auc = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    pci_20 = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    ifa = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    importance = {'RF':[], 'LR':[], 'NB':[], 'SVM':[]}
    X,y = load_both_data(project,metric)
    for _ in range(1):
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X, y):
            try:
                print(project)
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]
                if metric == 'process':
                    loc = X_test['file_la'] + X_test['file_lt']
                elif metric == 'product':
                    loc = X_test.CountLineCode
                else:
                    loc = X_test['file_la'] + X_test['file_lt']
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
                df_smote = pd.concat([X_train,y_train],axis = 1)
                df_smote = apply_smote(df_smote)
                y_train = df_smote.Bugs
                X_train = df_smote.drop('Bugs',axis = 1)

                goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
                7: "Micro_F"}[6]

                # Random Forest
                clf = RandomForestClassifier()
                learner = RandomForestClassifier()
                tuning = False
                _recall, _precision, _pf, _f1, _g_score, _auc, _pci_20, _ifa, _importance = learn(learner,
                                                                                            clf,X_train, y_train, 
                                                                                            X_val, y_val, X_test, 
                                                                                            y_test, goal, 
                                                                                            tuning, loc)

                
                pf['RF'].append(_pf)
                recall['RF'].append(_recall)
                precision['RF'].append(_precision)
                f1['RF'].append(_f1)
                g_score['RF'].append(_g_score)
                pci_20['RF'].append(_pci_20)
                ifa['RF'].append(_ifa)
                auc['RF'].append(_auc)
                importance['RF'].append(_importance)


                # Logistic Regression
                clf = LogisticRegression()
                learner = SK_LSR
                tuning = True
                _recall, _precision, _pf, _f1, _g_score, _auc, _pci_20, _ifa, _importance = learn(learner,
                                                                                            clf,X_train, y_train, 
                                                                                            X_val, y_val, X_test, 
                                                                                            y_test, goal, 
                                                                                            tuning, loc)

                
                pf['LR'].append(_pf)
                recall['LR'].append(_recall)
                precision['LR'].append(_precision)
                f1['LR'].append(_f1)
                g_score['LR'].append(_g_score)
                pci_20['LR'].append(_pci_20)
                ifa['LR'].append(_ifa)
                auc['LR'].append(_auc)
                importance['LR'].append(_importance)

                # Naive Bayes
                clf = GaussianNB()
                learner = SK_NB
                tuning = True
                _recall, _precision, _pf, _f1, _g_score, _auc, _pci_20, _ifa, _importance = learn(learner,
                                                                                            clf,X_train, y_train, 
                                                                                            X_val, y_val, X_test, 
                                                                                            y_test, goal, 
                                                                                            tuning, loc)

                
                pf['NB'].append(_pf)
                recall['NB'].append(_recall)
                precision['NB'].append(_precision)
                f1['NB'].append(_f1)
                g_score['NB'].append(_g_score)
                pci_20['NB'].append(_pci_20)
                ifa['NB'].append(_ifa)
                auc['NB'].append(_auc)
                importance['NB'].append(_importance)

                # Support Vector Machine
                clf = SVC()
                learner = SK_SVM
                tuning = True
                _recall, _precision, _pf, _f1, _g_score, _auc, _pci_20, _ifa, _importance = learn(learner,
                                                                                            clf,X_train, y_train, 
                                                                                            X_val, y_val, X_test, 
                                                                                            y_test, goal, 
                                                                                            tuning, loc)

                
                pf['SVM'].append(_pf)
                recall['SVM'].append(_recall)
                precision['SVM'].append(_precision)
                f1['SVM'].append(_f1)
                g_score['SVM'].append(_g_score)
                pci_20['SVM'].append(_pci_20)
                ifa['SVM'].append(_ifa)
                auc['SVM'].append(_auc)
                importance['SVM'].append(_importance)

            except ArithmeticError as e:
                print(e)
                continue
    # print(recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance)
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance

def run(projects, _type):
    precision_list = {}
    recall_list = {}
    pf_list = {}
    f1_list = {}
    g_list = {}
    auc_list = {}
    pci_20_list = {}
    ifa_list = {}
    featue_importance = {}
    for project in projects:
        try:
            if project == '.DS_Store':
                continue
            print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
            recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance = run_self_k(project,_type)
            recall_list[project] = recall
            precision_list[project] = precision
            pf_list[project] = pf
            f1_list[project] = f1
            g_list[project] = g_score
            auc_list[project] = auc
            pci_20_list[project] = pci_20
            ifa_list[project] = ifa
            featue_importance[project] = importance
        except ArithmeticError as e:
            print(e)
            continue
    final_result = {}
    final_result['precision'] = precision_list
    final_result['recall'] = recall_list
    final_result['pf'] = pf_list
    final_result['f1'] = f1_list
    final_result['g'] = g_list
    final_result['auc'] = auc_list
    final_result['pci_20'] = pci_20_list
    final_result['ifa'] = ifa_list
    final_result['featue_importance'] = featue_importance
    return final_result

if __name__ == "__main__":
    div = 0
    start = time.time()
    proj_df = pd.read_csv('projects.csv')
    
    for _type in ['process','product','process+product']:
        projects = proj_df.repo_name.tolist()
        threads = []
        cores = cpu_count()
        final_result = {}
        split_projects = np.array_split(projects, cores)
        for i in range(cores):
            sub_projects = split_projects[i]
            print(sub_projects)
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = run, args = [sub_projects,_type])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            print(response)
            for key in response.keys():
                if key not in final_result.keys():
                    final_result[key] = {}
                final_result[key] = {**response[key], **final_result[key]}


        for model in ['RF', 'LR', 'NB', 'SVM']:
            model_results = {}
            for metric in final_result.keys():
                if metric not in model_results.keys():
                    model_results[metric] = {}
                model_results[metric] = final_result[metric][model]
                with open('../results/Performance/RQ_cross_' + _type + '_' + model + '_HPO.pkl', 'wb') as handle:
                    pickle.dump(model_results, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    end = time.time()
    print(end - start)