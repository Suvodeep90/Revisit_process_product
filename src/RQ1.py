import pandas as pd
import numpy as np
import math
import pickle

from scipy import stats
import scipy.io
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky
from scipy.io import loadmat


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

# Venn diag
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns


def between_learners():
    dfs = ['process','product','process+product']
    final_df = pd.DataFrame()
    learner_df = pd.DataFrame()
    metrics = ['recall','pf','auc','pci_20','precision','ifa']
    learners = ['RF','LR','NB','SVM']
    i = 0
    for learner in learners:
        for metric in metrics:
            data = []
            for df in dfs:
                _file = pd.read_pickle('results/Performance/RQ_cross_' + df + '_' + learner + '_HPO.pkl')
                if metric == 'ifa':
                    l = [np.median(sublist)/100 for sublist in list(_file[metric].values())]
                else:
                    l = [np.median(sublist) for sublist in list(_file[metric].values())]
                data.append(l)
            data_df = pd.DataFrame(data)
            data_df.index = [['P','C','P+C']]
            data_df = data_df.T
            data_df = data_df.dropna()
            x = pd.melt(data_df)
            x.columns = ['Metric Type','Score']
            x['Evaluation Criteria'] = [metric]*x.shape[0]
            learner_df = pd.concat([learner_df,x])
        learner_df['learner'] = [learner]*learner_df.shape[0]
        final_df = pd.concat([final_df,learner_df])

    sns.set(style='whitegrid',font_scale=1)
    order = ["P", "C", "P+C"]
    g = sns.catplot(x="Metric Type", y="Score",row="Evaluation Criteria" ,col='learner' ,height=4,aspect=0.5,margin_titles=True,kind="box", 
                    order=order, data=final_df)
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    g.savefig('results/image/RQ1_cross.pdf')


def release_based():
    dfs = ['process','product','process+product']
    final_df = pd.DataFrame()
    metrics = ['precision', 'recall', 'pf', 'auc', 'pci_20','ifa']
    i = 0
    for metric in metrics:
        data = []
        for df in dfs:
            file = pd.read_pickle('results/Performance/RQ_release_' + df + '_RF.pkl')
            if metric == 'ifa':
                l = [np.median(sublist)/100 for sublist in list(file[metric].values())]
            else:
                l = [np.median(sublist) for sublist in list(file[metric].values())]
            data.append(l)
        data_df = pd.DataFrame(data)
        data_df.index = [['P','C','P+C']]
        x = pd.melt(data_df.T)
        x.columns = ['Metric Type','Score']
        if metric == 'pci_20':
            metric = 'popt_20'
        x['Evaluation Criteria'] = [metric]*x.shape[0]
        final_df = pd.concat([final_df,x])
    final_df.columns = x.columns
    sns.set(style='whitegrid',font_scale=1)
    order = ["P", "C", "P+C"]
    g = sns.catplot(x="Metric Type", y="Score", col="Evaluation Criteria",height=4,aspect=0.6,margin_titles=True,kind="box", 
                    order=order, data=final_df)
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    g.savefig('results/image/RQ1_release.pdf')

if __name__ == "__main__":
    between_learners()
    release_based()

