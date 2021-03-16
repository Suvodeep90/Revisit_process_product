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

def release_stability():
    count = 0
    dfs = ['process','product','process+product']
    orders = ["P", "C", "P+C"]
    all_results = pd.DataFrame()
    for k in range(len(dfs)):
        df = dfs[k]
        order = orders[k]
        result_df = pd.DataFrame()
        final_result = pd.read_pickle('results/Performance/RQ_release_' + df +'_RF.pkl')
        for metric in final_result.keys():
            final_df = pd.DataFrame()
            release = [[],[],[]]
            for projects in final_result[metric].keys():
                if len(final_result[metric][projects]) < 3:
                    continue
                count += 1
                i = 0
                for value in final_result[metric][projects]:
                    if metric == 'ifa':
                        value = value/100
                    release[i].append(value)
                    i += 1
                    if i == 3:
                        break
            for j in range(3):
                score_df = pd.DataFrame(release[j], columns = ['scores'])
                score_df['release'] = [j+1]*score_df.shape[0]
                final_df = pd.concat([final_df,score_df], axis = 0)
            final_df['metrics'] = [metric]*final_df.shape[0]
            result_df = pd.concat([result_df,final_df], axis = 0)
        result_df['Metric Type'] = [order]*result_df.shape[0]
        all_results = pd.concat([all_results,result_df])
    all_results = all_results[all_results['metrics'] != 'featue_importance']

    all_results = all_results[all_results['Metric Type'] != 'P+C']
    all_results = all_results[all_results['metrics'] != 'g']
    all_results = all_results[all_results['metrics'] != 'f1']

    g = sns.set(style='whitegrid',font_scale=1.8)
    order = [1,2,3]
    g = sns.catplot(x="release", y="scores", col="metrics",row="Metric Type" ,height=4,aspect=0.6,margin_titles=True,kind="box", 
                    order=order, data=all_results)
    g.fig.set_figwidth(24)
    g.fig.set_figheight(14)
    g.savefig('results/image/RQ4.pdf')



if __name__ == "__main__":
    release_stability()
