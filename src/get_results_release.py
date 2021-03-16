import pandas as pd
import numpy as np
import math
import pickle
from datetime import datetime

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import SMOTE
import CFS
import metrices_V2 as metrices

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

import matplotlib.pyplot as plt
import seaborn as sns

import random

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

def load_data(project,metric):
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
    
    release_df = pd.read_pickle('data/release/' + project + '_release.pkl')
    release_df = release_df.sort_values('created_at',ascending=False)
    release_df = release_df.reset_index(drop=True)
    release_df['created_at'] = pd.to_datetime(release_df.created_at)
    release_df['created_at'] = release_df.created_at.dt.date
    
    commit_guru_path = 'data/commit_guru/' + project + '.csv' 
    commit_guru_df = pd.read_csv(commit_guru_path)
    cols = understand_df.columns.tolist()
    commit_guru_df['created_at'] = pd.to_datetime(commit_guru_df.author_date_unix_timestamp,unit='s')
    commit_guru_df['created_at'] = commit_guru_df.created_at.dt.date
    
    commit_guru_df = commit_guru_df[['commit_hash','created_at']]
    
    df = understand_df.merge(commit_guru_file_level_df,how='left',on=['commit_hash','Name'])
    df = df.merge(commit_guru_df,how='left',on=['commit_hash'])


    
    cols = df.columns.tolist()
    cols.remove('Bugs')
    cols.append('Bugs')
    df = df[cols]
    commit_hash = df.commit_hash
    
    file_names = df.Name
    
    for item in ['Kind', 'Name','commit_hash']:
        if item in cols:
            df = df.drop(labels = [item],axis=1)
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    
    created_at = df.created_at
    df = df.drop('created_at',axis = 1)
    y = df.Bugs
    X = df.drop('Bugs',axis = 1)
    cols = X.columns
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    imp_mean = IterativeImputer(random_state=0)
    X = imp_mean.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    X['created_at'] = created_at
    
    if metric == 'process':
        X = X[['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
       'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
       'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
       'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr','created_at']]
    elif metric == 'product':
        X = X.drop(['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
       'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
       'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
       'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr'],axis = 1)
    else:
        X = X
    
    df = X
    df['Bugs'] = y
    df['commit_hash'] = commit_hash
    df['Name'] = file_names
    unique_commits = df.commit_hash.unique()
    count = 0
    last_train_date = None
    test_size = 0
    test_releases = []
    for release_date in release_df.created_at.unique()[:-1]:
        test_df =  df[(df.created_at >= release_date)]
        if (test_df.shape[0]) > 2 and (test_df.shape[0] > test_size):
            count += 1
            last_train_date = release_date
            test_size = test_df.shape[0]
            test_releases.append(release_date)
        if count == 4:
            # print("breaking")
            break 
    if count < 4:
        # print('not enough releases')
        return df,df,0
    train_df =  df[df.created_at < last_train_date]
    test_df =  df[df.created_at >= last_train_date]
    test_df = test_df.reset_index(drop= True)
    test_df['release'] = [0]*test_df.shape[0]
    i = 0
    for release_date in test_releases:
        test_df.loc[test_df['created_at'] < release_date,'release'] = i
        i += 1
    
    train_df = train_df.drop('created_at',axis = 1)
    test_df = test_df.drop('created_at',axis = 1)
    
    if train_df.shape[0] == 0:
        return df,df,0
    
    if test_df.shape[0] == 0:
        return df,df,0
    return train_df,test_df,1

def run_self_release(project,metric):
    precision = []
    recall = []
    pf = []
    f1 = []
    g_score = []
    auc = []
    pci_20 = []
    ifa = []
    train_probability = []
    predicted_probability = []
    test_probability = []
    train_df,test_df,success = load_data(project,metric)
    if success == 0:
        return 0,0,0,0,0,0,0,0,0,0,0,0
    train_df = train_df.drop('commit_hash',axis = 1)
    previous_X = pd.DataFrame()
    previous_y = []
    for release in test_df.release.unique():
        if len(previous_y) == 0:
            y = train_df.Bugs.values.tolist()
            X = train_df.drop('Bugs',axis = 1)
            train_file_names = X.Name.values.tolist()
            X = X.drop('Name', axis = 1)
            df_smote = X
            df_smote['Bugs'] = y
            df_smote = apply_smote(df_smote)
            y_train = df_smote.Bugs
            X_train = df_smote.drop('Bugs',axis = 1)
            clf =  RandomForestClassifier()
            clf.fit(X_train,y_train)
            importance = clf.feature_importances_
        else:
            y = train_df.Bugs.values.tolist()
            X = train_df.drop('Bugs',axis = 1)
            y = y + previous_y
            X = pd.concat([X,previous_X], axis = 0)
            new_train_file_names = X.Name.values.tolist()
            train_file_names = train_file_names + new_train_file_names
            X = X.drop('Name', axis = 1)
            df_smote = X
            df_smote['Bugs'] = y
            df_smote = apply_smote(df_smote)
            y_train = df_smote.Bugs
            X_train = df_smote.drop('Bugs',axis = 1)
            clf =  RandomForestClassifier()
            clf.fit(X_train,y_train)
            importance = clf.feature_importances_
        test_df_subset = test_df[test_df['release'] == release]
        test_df_subset = test_df_subset.drop('release',axis = 1)
        
        y_test = test_df_subset.Bugs
        test_file_names = test_df_subset.Name.values.tolist()
        X_test = test_df_subset.drop(['Bugs','commit_hash','Name'],axis = 1)
        
        previous_X = X_test
        previous_y = y_test.values.tolist()
        
        if metric == 'process':
            loc = X_test['file_la'] + X_test['file_lt']
        elif metric == 'product':
            loc = X_test.CountLineCode
        else:
            loc = X_test['file_la'] + X_test['file_lt']                 
        predicted = clf.predict(X_test)
        abcd = metrices.measures(y_test,predicted,loc)
        pf.append(abcd.get_pf())
        recall.append(abcd.calculate_recall())
        precision.append(abcd.calculate_precision())
        f1.append(abcd.calculate_f1_score())
        g_score.append(abcd.get_g_score())
        pci_20.append(abcd.get_pci_20())
        ifa.append(abcd.get_ifa())
        try:
            auc.append(roc_auc_score(y_test, predicted))
        except:
            auc.append(0)
        
        
        y_test = y_test.values.tolist()
        predicted = list(predicted)
        
        learned_proba = clf.predict_proba(X_train)
        learned_proba = list(zip(train_file_names,learned_proba[:,np.where(clf.classes_)[0]]))
        
        predict_proba = clf.predict_proba(X_test)
        predict_proba = list(zip(test_file_names,predict_proba[:,np.where(clf.classes_)[0]]))
        
        train_probability.append(learned_proba)
        predicted_probability.append(predict_proba)
        test_probability.append(y_test.count(1)/len(y_test))
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance,train_probability,predicted_probability,test_probability


if __name__ == "__main__":
    proj_df = pd.read_csv('projects.csv')
    projects = proj_df.repo_name.tolist()

    for _type in ['process','product','process+product']:
        precision_list = {}
        recall_list = {}
        pf_list = {}
        f1_list = {}
        g_list = {}
        auc_list = {}
        pci_20_list = {}
        ifa_list = {}
        featue_importance = {}
        train_probabilities = {}
        predicted_probabilities = {}
        test_probabilities = {}
        for project in projects:
            try:
                if project == '.DS_Store':
                    continue
                print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
                recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance,train_probability,predicted_probability,test_probability = run_self_release(project,_type)
                if recall == 0 and precision == 0 and pf == 0:
                    continue
                recall_list[project] = recall
                precision_list[project] = precision
                pf_list[project] = pf
                f1_list[project] = f1
                g_list[project] = g_score
                auc_list[project] = auc
                pci_20_list[project] = pci_20
                ifa_list[project] = ifa
                featue_importance[project] = importance
                train_probabilities[project] = train_probability
                predicted_probabilities[project] = predicted_probability
                test_probabilities[project] = test_probability
            except Exception as e:
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
        final_result['train_probability'] = train_probabilities
        final_result['predicted_probability'] = predicted_probabilities
        final_result['test_probability'] = test_probabilities

        with open('results/Performance/RQ_release_' + _type + '_RF.pkl', 'wb') as handle:
            pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('results/Performance/RQ6_' + _type + '.pkl', 'wb') as handle:
            pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)