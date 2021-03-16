import pandas as pd
import numpy as np
import math
import pickle
import random

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

def graph1():
    df = pd.read_pickle('results/Performance/RQ_cross_process+product_RF_HPO.pkl')
    df = df['featue_importance']
    df = pd.DataFrame.from_dict(df,orient='index')
    df.columns = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict',
        'AvgEssential', 'AvgLine', 'AvgLineBlank', 'AvgLineCode',
        'AvgLineComment', 'CountClassBase', 'CountClassCoupled',
        'CountClassCoupledModified', 'CountClassDerived',
        'CountDeclClassMethod', 'CountDeclClassVariable',
        'CountDeclInstanceMethod', 'CountDeclInstanceVariable',
        'CountDeclMethod', 'CountDeclMethodAll', 'CountDeclMethodDefault',
        'CountDeclMethodPrivate', 'CountDeclMethodProtected',
        'CountDeclMethodPublic', 'CountLine', 'CountLineBlank', 'CountLineCode',
        'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment',
        'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
        'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict',
        'MaxEssential', 'MaxInheritanceTree', 'MaxNesting',
        'PercentLackOfCohesion', 'PercentLackOfCohesionModified',
        'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified',
        'SumCyclomaticStrict', 'SumEssential', 'la', 'ld', 'lt',
        'age', 'ddev', 'nuc', 'own', 'minor', 'ndev',
        'ncomm', 'adev', 'nadev', 'avg_nddev',
        'avg_nadev', 'avg_ncomm', 'ns', 'exp', 'sexp',
        'rexp', 'nd', 'sctr']
    df_stats = pd.DataFrame(df.quantile([.25, .5, .75]))
    df_stats = df_stats.T
    df_stats.columns = ['25th','50th','75th']
    df_stats = df_stats.sort_values(by = ['50th'],ascending = True)
    x = range(0,66)
    y1 = np.array(df_stats['25th'].values.tolist())
    y2 = np.array(df_stats['50th'].values.tolist())
    y3 = np.array(df_stats['75th'].values.tolist())
    fig, ax = plt.subplots(figsize=(10,15))
    ax.set_yticks(x)
    plt.grid(b=None)
    ax.plot(y2,x,linestyle='-', color='b', linewidth='2')

    _x_tick = ['**nd', '**ns', 'CountDeclMethodDefault', 'CountClassDerived', 'AvgEssential', 
    'CountClassBase', 'CountDeclMethodProtected', 'MaxInheritanceTree', 'AvgLineBlank', 
    'CountDeclClassMethod', 'MaxEssential', 'AvgCyclomaticModified', 'AvgLineComment', 
    'AvgCyclomatic', 'CountDeclMethodPrivate', 'AvgCyclomaticStrict', 'CountDeclClassVariable', 
    'MaxNesting', 'MaxCyclomaticModified', 'MaxCyclomatic', 'CountDeclInstanceVariable', 
    'MaxCyclomaticStrict', 'CountClassCoupled', 'CountClassCoupledModified', 
    'PercentLackOfCohesionModified', 'CountDeclMethodPublic', 'CountDeclInstanceMethod', 
    'PercentLackOfCohesion', 'SumEssential', 'AvgLineCode', 'SumCyclomaticModified', 
    'CountDeclMethod', 'AvgLine', 'SumCyclomatic', 'SumCyclomaticStrict', 
    'CountDeclMethodAll', 'RatioCommentToCode', 'CountLineComment', 
    'CountStmtExe', 'CountLineBlank', 'CountStmtDecl', 'CountSemicolon', 
    'CountLineCodeExe', 'CountLineCodeDecl', 'CountStmt', 'CountLineCode', 
    'CountLine', '**lt', '**la', '**ld', '**sctr', '**sexp', '**exp', '**avg_nddev', '**ddev', '**ndev', 
    '**own', '**age', '**nuc', '**adev', '**minor', '**ncomm', '**nadev', '**rexp', '**avg_ncomm', '**avg_nadev']

    ax.set_yticklabels(_x_tick)

    colors = []
    process = ['**la', '**ld', '**lt',
        '**age', '**ddev', '**nuc', '**own', '**minor', '**ndev',
        '**ncomm', '**adev', '**nadev', '**avg_nddev',
        '**avg_nadev', '**avg_ncomm', '**ns', '**exp', '**sexp',
        '**rexp', '**nd', '**sctr']

    for tick in _x_tick:
        if tick in process:
            colors.append('blue')
        else:
            colors.append('black')

    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.fill_betweenx(x, y2 - y1, y2 + y3, alpha=0.2,color='red')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(b=None)
    plt.savefig('results/image/RQ8_1.pdf')

def graph2():
    df = pd.read_pickle('results/Performance/RQ_cross_process_RF_HPO.pkl')
    df = df['featue_importance']
    df = pd.DataFrame.from_dict(df,orient='index')
    df.columns = ['la', 'ld', 'lt', 'age', 'ddev',
        'nuc', 'own', 'minor', 'ndev', 'ncomm', 'adev',
        'nadev', 'avg_nddev', 'avg_nadev', 'avg_ncomm',
        'ns', 'exp', 'sexp', 'rexp', 'nd', 'sctr']
    df_stats = pd.DataFrame(df.quantile([.25, .5, .75]))
    df_stats = df_stats.T
    df_stats.columns = ['25th','50th','75th']
    df_stats_rf = df_stats.sort_values(by = ['50th'])
    df_stats_rf['rank_rf'] = range(0,21)
    df_stats_rf = df_stats_rf.drop(['25th','50th','75th'],axis = 1)

    df = pd.read_pickle('results/Performance/RQ_cross_process_RF_HPO.pkl')
    df = df['featue_importance']
    df = pd.DataFrame.from_dict(df,orient='index')

    # indexs = []
    # for i in range(5):
    #     indexs.append(random.choice(df.index))
        
    # print(indexs)
    indexs = ['react-native-sqlite-storage', 'dashboard-demo', 'hppc', 'react-native-share', 'esb-connectors']

    df = df.loc[indexs]

    df.columns = ['la', 'ld', 'lt', 'age', 'ddev',
        'nuc', 'own', 'minor', 'ndev', 'ncomm', 'adev',
        'nadev', 'avg_nddev', 'avg_nadev', 'avg_ncomm',
        'ns', 'exp', 'sexp', 'rexp', 'nd', 'sctr']
    df_stats = pd.DataFrame(df.quantile([.25, .5, .75]))
    df_stats = df_stats.T
    df_stats.columns = ['25th','50th','75th']
    df_stats = df_stats.apply(np.vectorize(abs))
    df_stats = df_stats.apply(np.vectorize(math.exp))
    df_stats_lr = df_stats.sort_values(by = ['50th'])
    df_stats_lr['rank_lr'] = range(0,21)
    df_stats_lr = df_stats_lr.drop(['25th','50th','75th'],axis = 1)

    new_df = df_stats_rf.join(df_stats_lr)

    x = range(0,21)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xticks(x)
    ax.set_yticks(x)
    plt.plot(new_df.rank_rf,new_df.rank_lr,marker='o', color='b')
    ax.set_xticklabels(new_df.index.tolist())
    plt.xticks(fontsize=18,rotation=90)
    ax.set_yticklabels(df_stats_lr.index.tolist())
    plt.yticks(fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.grid(b=None)
    ax.set_xlabel('Feature ranks for Random Forest (in-the-large)',fontsize=18)
    ax.set_ylabel('Feature ranks for Random Forest (in-the-small)',fontsize=18)
    for x,y in zip(new_df.rank_rf,new_df.rank_lr):

        label = "{:d},{:d}".format(x,y)

        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center',fontsize=15)
    plt.savefig('results/image/RQ8_2.pdf',dpi=600,bbox_inches='tight', pad_inches=0.3)

if __name__ == "__main__":
    graph1()
    graph2()
