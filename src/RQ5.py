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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

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

# Venn diag
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

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
    file_names = df.Name
    
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
    X['Name'] = file_names
    X['Bugs'] = y
    
    return X

def load_data_release_level(project,metric):
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
    file_names = df.Name
    commit_hash = df.commit_hash
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
    df['Name'] = file_names
    df['Bugs'] = y
    
    accepted_commit_dates = []
    all_data = pd.DataFrame()
    for i in range(release_df.shape[0]-1):
        sub_df = df[df['created_at'] <= release_df.loc[i,'created_at']]
        sub_df = sub_df[sub_df['created_at'] > release_df.loc[i+1,'created_at']]
        sub_df.sort_values(by = ['created_at'], inplace = True, ascending = False)
        sub_df.drop_duplicates(['Name'], inplace = True)
        
        all_data =pd.concat([all_data,sub_df], axis = 0)
    
    all_data = all_data.drop('created_at',axis = 1)
    
    return all_data

def load_package_data(project,metric):
    understand_path = 'data/package_level/understand_files_all/' + project + '_understand.csv'
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
    
    commit_guru_file_level_path = 'data/package_level/commit_guru_file/' + project + '.csv'
    commit_guru_file_level_df = pd.read_csv(commit_guru_file_level_path)
    commit_guru_file_level_df['commit_hash'] = commit_guru_file_level_df.commit_hash.str.strip('"')
    
    
    df = understand_df.merge(commit_guru_file_level_df,how='left',on=['commit_hash','Name'])
    
    
    cols = df.columns.tolist()
    cols.remove('Bugs')
    cols.append('Bugs')
    df = df[cols]
    
    file_names = df.Name
    
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
    
    df = X
    df['Name'] = file_names
    df['Bugs'] = y
    
    return df

def get_spearmanr_metrics(df):
    df = df.drop('Bugs',axis = 1)
    file_corr = {}
    names = df.Name.unique()
    for name in names:
        df_1 = copy.deepcopy(df)
        sub_df = df_1[df_1['Name'] == name]
        if sub_df.shape[0] < 2:
            continue
        sub_df = sub_df.drop('Name',axis = 1)
        sub_df = sub_df.reset_index(drop = 'True')
        for i in range(sub_df.shape[0]-1):
            before = sub_df.loc[i]
            after = sub_df.loc[i+1]
            rho, pval = stats.spearmanr(before,after,axis = 1)
            if name not in file_corr.keys():
                file_corr[name] = []
            file_corr[name].append(rho)
    return file_corr

def corr_graph():
    metrics = ['process_corr_release','product_corr_release','process_corr_JIT','product_corr_JIT','process_package_corr_JIT']
    # metrics = ['process','product']
    final_df = pd.DataFrame()
    for metric in metrics:
        project_corr = pd.read_pickle('results/Correlations/' + metric + '.pkl')
        flat_list = [np.nanmedian(sublist) for sublist in list(project_corr.values())]
        project_corr_df = pd.DataFrame(flat_list, columns = ['scores'])
        project_corr_df['metrics'] = [metric]*project_corr_df.shape[0]
        final_df = pd.concat([final_df,project_corr_df], axis = 0)
    final_df = final_df.dropna()

    final_df = final_df[final_df['metrics'] != 'process+product']
    metrics_ticks = ['P_R','C_R','P_J','C_J','P_P_J']
    sns.set(style='whitegrid',font_scale=1.2)
    fig_dims = (8, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    order = metrics
    g = sns.violinplot(x="metrics", y="scores",margin_titles=True,kind="box", scale = 'area',
                    inner = 'box',
                    order=order, data=final_df)
    plt.xlabel('Metric Types', fontsize=14)
    plt.ylabel('Correlation Scores', fontsize=14)
    ax.set_xticklabels(metrics_ticks, rotation=0, fontsize=14)
    plt.savefig('results/image/RQ5.pdf')


if __name__ == "__main__":
    proj_df = pd.read_csv('projects.csv')
    projects = proj_df.repo_name.tolist()
    projects = projects

    if path.exists("results/Correlations/process_corr_JIT.pkl"):
        corr_graph()
    else:
        project_corr = {}
        for metric in ['process','product','process+product']:
            for project in projects:
                try:
                    if project == '.DS_Store':
                        continue
                    print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
                    df = load_data_release_level(project,metric)
                    corr = get_spearmanr_metrics(df)
                    project_corr[project] = corr
                except Exception as e:
                    print('error',e)
                    continue
            with open('results/Correlations/' + metric +'_corr_release.pkl', 'wb') as handle:
                pickle.dump(project_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        project_corr = {}
        for metric in ['process','product','process+product']:
            for project in projects:
                try:
                    if project == '.DS_Store':
                        continue
                    print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
                    df = load_both_data(project,metric)
                    corr = get_spearmanr_metrics(df)
                    project_corr[project] = corr
                except Exception as e:
                    print('error',e)
                    continue
            with open('results/Correlations/' + metric +'_corr_JIT.pkl', 'wb') as handle:
                pickle.dump(project_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        project_corr = {}
        metric = 'process'
        for project in projects:
            try:
                if project == '.DS_Store':
                    continue
                print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
                df = load_package_data(project,metric)
                corr = get_spearmanr_metrics(df)
                project_corr[project] = corr
            except Exception as e:
                print('error',e)
                continue
        with open('results/Correlations/' + metric +'_package_corr_JIT.pkl', 'wb') as handle:
            pickle.dump(project_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)