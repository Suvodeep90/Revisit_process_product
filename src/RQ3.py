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
import sys
import os
import copy
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def package_vs_file(_type):
    dfs = ['RQ_cross_' + _type + '_RF_HPO.pkl', 'RQ_package_' + _type + "_RF_HPO.pkl"]
    final_df = pd.DataFrame()
    metrics = ['precision', 'recall', 'pf', 'auc', 'pci_20','ifa']
    i = 0
    for metric in metrics:
        data = []
        for df in dfs:
            file = pd.read_pickle('results/Performance/' + df)
            if metric == 'ifa':
                l = [np.nanmedian(sublist)/100 for sublist in list(file[metric].values())]
            else:
                l = [np.nanmedian(sublist) for sublist in list(file[metric].values())]
            data.append(l)
        data_df = pd.DataFrame(data)
        data_df.index = [['P_file','P_package']]
        x = pd.melt(data_df.T)
        x.columns = ['Metric Type','Score']
        if metric == 'pci_20':
            metric = 'popt_20'
        x['Evaluation Criteria'] = [metric]*x.shape[0]
        final_df = pd.concat([final_df,x])
    final_df.columns = x.columns
    final_df = final_df.replace({'P_file':'file',
                  'P_package':'package',
                  'precision':'Precision', 
                  'recall':'Recall', 'pf':'Pf',
                  'auc':'AUC', 
                  'pci_20':'Popt_20',
                  'ifa':'IFA'})
    
    sns.set(style='whitegrid',font_scale=1)
    order = ['file','package']
    g = sns.catplot(x="Metric Type", y="Score", col="Evaluation Criteria",height=4,aspect=0.6,margin_titles=True,kind="box", 
                    order=order, data=final_df)
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    g.savefig('results/image/RQ3.pdf')

if __name__ == "__main__":
    package_vs_file('process')