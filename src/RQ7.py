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

import SMOTE
import CFS
import metrices_V2 as metrices

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
from os import path
import sys
import os
import copy
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


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

def load_data_commit_level(project,metric):
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
    commit_hash = df.commit_hash
    Name = df.Name
    for item in ['Kind', 'Name','commit_hash']:
        if item in cols:
            df = df.drop(labels = [item],axis=1)
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

    X['Name'] = Name
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=18)
    
    test_df = X_test
    test_df['Bugs'] = y_test
    
    train_df = X_train
    train_df['Bugs'] = y_train
    
    
    defective_train_files = set(train_df[train_df['Bugs'] == 1].Name.values.tolist())
    non_defective_train_files = set(train_df[train_df['Bugs'] == 0].Name.values.tolist())
    
    defective_non_defective_train_files = defective_train_files.intersection(non_defective_train_files)
    
    only_defective_train_files = defective_train_files - defective_non_defective_train_files
    
    only_non_defective_train_files = non_defective_train_files - defective_non_defective_train_files

    
    test_df_recurruing = test_df[test_df['Bugs'] == 1]
    test_df_recurruing = test_df_recurruing[test_df_recurruing.Name.isin(defective_train_files)]
    
    test_df_test_only = test_df[test_df['Bugs'] == 1]
    test_df_test_only = test_df_test_only[test_df_test_only.Name.isin(only_non_defective_train_files)]
    
    test_df_train_only = test_df[test_df['Bugs'] == 0]
    test_df_train_only = test_df_train_only[test_df_train_only.Name.isin(only_defective_train_files)]
    
    y_train = train_df.Bugs
    X_train = train_df.drop(['Bugs','Name'],axis = 1)
    
    
    test_non_defective = test_df[test_df['Bugs'] == 0]
    test_defective = test_df[test_df['Bugs'] == 1]
    
    
    test_df_recurruing = test_df_recurruing.drop(['Name'],axis = 1)
    test_df_test_only = test_df_test_only.drop(['Name'],axis = 1)
    test_df_train_only = test_df_train_only.drop(['Name'],axis = 1)
    

    return X_train,y_train,test_df_recurruing, test_df_train_only, test_df_test_only



def run_self(project,metric):
    
    pf = []
    recall = []
    precision = []
    f1 = []
    g_score = []
    pci_20 = []
    ifa = []
    auc = []
    
    X_train,y_train,test_df_recurruing, test_df_train_only, test_df_test_only = load_data_commit_level(project,metric)
    
    df_smote = pd.concat([X_train,y_train],axis = 1)
    df_smote = apply_smote(df_smote)
    y_train = df_smote.Bugs
    X_train = df_smote.drop('Bugs',axis = 1)
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    importance = clf.feature_importances_
    
    # recurrence only
    try:
        y_test = test_df_recurruing.Bugs
        X_test = test_df_recurruing.drop('Bugs',axis=1)
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
        print(classification_report(y_test, predicted))
    except:
        print(test_df_recurruing.shape)
        pf.append(-1)
        recall.append(-1)
        precision.append(1)
        f1.append(-1)
        g_score.append(-1)
        pci_20.append(-1)
        ifa.append(-1)
        auc.append(-1)
    
    # train only
    try:
        y_test = test_df_train_only.Bugs
        X_test = test_df_train_only.drop('Bugs',axis=1)
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
        print(classification_report(y_test, predicted))
    except:
        print(test_df_train_only.shape)
        pf.append(-1)
        recall.append(-1)
        precision.append(1)
        f1.append(-1)
        g_score.append(-1)
        pci_20.append(-1)
        ifa.append(-1)
        auc.append(-1)
    
    # test only
    try:
        y_test = test_df_test_only.Bugs
        X_test = test_df_test_only.drop('Bugs',axis=1)
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
    except:
        pf.append(-1)
        recall.append(-1)
        precision.append(1)
        f1.append(-1)
        g_score.append(-1)
        pci_20.append(-1)
        ifa.append(-1)
        auc.append(-1)
    
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance


def create_stagnation_model_graph():
    final_df = pd.DataFrame()
    order = ['recurrent','train only','test only']
    for metric_type in ['process','product']:
        file_df = pd.DataFrame()
        with open('results/Performance/RQ7_' + metric_type + '.pkl', 'rb') as handle:
            final_result = pickle.load(handle)
            for goal in final_result.keys():
                score_r = []
                score_train = []
                score_test = []
                sub_df = pd.DataFrame()
                for project in final_result[goal].keys():

                    if np.isnan(final_result[goal][project][0]):
                        score_r.append(0)
                    elif final_result[goal][project][0] != -1:
                        score_r.append(np.nanmedian(final_result[goal][project][0]))

                    if np.isnan(final_result[goal][project][1]):
                        score_train.append(0)
                    elif final_result[goal][project][1] != -1:
                        score_train.append(np.nanmedian(final_result[goal][project][1]))

                    if np.isnan(final_result[goal][project][2]):
                        score_test.append(0)    
                    elif final_result[goal][project][2] != -1:
                        score_test.append(np.nanmedian(final_result[goal][project][2]))

                all_scores = score_r + score_train + score_test
                all_order = [order[0]]*len(score_r) + [order[1]]*len(score_train) + [order[2]]*len(score_test)
                df = pd.DataFrame(zip(all_scores,all_order), columns = ['score','test_type'])

                sub_df = pd.concat([sub_df,df], axis = 0)
                sub_df['metric'] = [goal]*sub_df.shape[0]
                file_df = pd.concat([file_df,sub_df])
            file_df['metric type'] = [metric_type]*file_df.shape[0]
            final_df = pd.concat([final_df,file_df], axis = 0)
        

    final_df = final_df[final_df.metric.isin(['recall','pf'])]    
    final_df = final_df.replace({'recall':'Recall','pf':'Pf','process':'P','product':'C'})
    sns.set(style='whitegrid',font_scale=1.4)
    order = ["P", "C"]
    g = sns.catplot(x="metric type", y="score", col="test_type",row="metric",height=4,aspect=0.6,margin_titles=True,kind="box", 
                    order=order, data=final_df)
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    g.savefig('results/image/RQ7.pdf')

if __name__ == "__main__":
    proj_df = pd.read_csv('projects.csv')
    projects = proj_df.repo_name.tolist()
    projects = projects

    if not path.exists("results/Performance/RQ7_process.pkl"):
        for _type in ['process','product']:
            precision_list = {}
            recall_list = {}
            pf_list = {}
            f1_list = {}
            g_list = {}
            auc_list = {}
            pci_20_list = {}
            ifa_list = {}
            featue_importance = {}
            for project in projects[151:]:
                try:
                    if project == '.DS_Store':
                        continue
                    print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
                    recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance = run_self(project,_type)
                    recall_list[project] = recall
                    precision_list[project] = precision
                    pf_list[project] = pf
                    f1_list[project] = f1
                    g_list[project] = g_score
                    auc_list[project] = auc
                    pci_20_list[project] = pci_20
                    ifa_list[project] = ifa
                    featue_importance[project] = importance
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

            with open('results/Performance/commit_guru_file_specific/RQ7_' + _type + '.pkl', 'wb') as handle:
                pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    create_stagnation_model_graph()