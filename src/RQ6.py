import pandas as pd
import numpy as np
import math
import pickle
from datetime import datetime

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

import warnings
warnings.filterwarnings("ignore")

def create_stagnation_graph():
    with open('results/Performance/RQ6_product.pkl', 'rb') as handle:
        final_result = pickle.load(handle)
    all_corr = []
    for file in final_result['predicted_probability'].keys():
        for i in range(len(final_result['predicted_probability'][file])):
            learned_prob_all = {}
            predicted_prob_all = {}
            learned = final_result['train_probability'][file][0]
            predicted = final_result['predicted_probability'][file][0]

            for i in range(len(learned)):
                file_name = learned[i][0]
                prob = learned[i][1]
                if file_name not in learned_prob_all.keys():
                    learned_prob_all[file_name] = {'learned':[],'predicted':[]}
                learned_prob_all[file_name]['learned'].append(prob)

            for i in range(len(predicted)):
                file_name = predicted[i][0]
                prob = predicted[i][1]
                if file_name not in learned_prob_all.keys():
                    learned_prob_all[file_name] = {'learned':[],'predicted':[]}
                learned_prob_all[file_name]['predicted'].append(prob) 

            for file_name in learned_prob_all.keys():
                try:
                    learned_prob_all[file_name]['learned'] = learned_prob_all[file_name]['learned'][len(learned_prob_all[file_name]['learned'])-1][0]
                    learned_prob_all[file_name]['predicted'] = learned_prob_all[file_name]['predicted'][len(learned_prob_all[file_name]['predicted'])-1][0]
                except:
                    learned_prob_all[file_name]['learned'] = None
                    learned_prob_all[file_name]['predicted'] = None
                    continue
        learned_prob_all_df = pd.DataFrame.from_dict(learned_prob_all, orient = 'index')
        learned_prob_all_df = learned_prob_all_df.dropna()
        rho = learned_prob_all_df.learned.corr(learned_prob_all_df.predicted, method = 'spearman')
        all_corr.append(rho)
    all_corr_product = all_corr

    with open('results/Performance/RQ6_process.pkl', 'rb') as handle:
        final_result = pickle.load(handle)
    all_corr = []
    for file in final_result['predicted_probability'].keys():
        for i in range(len(final_result['predicted_probability'][file])):
            learned_prob_all = {}
            predicted_prob_all = {}
            learned = final_result['train_probability'][file][0]
            predicted = final_result['predicted_probability'][file][0]

            for i in range(len(learned)):
                file_name = learned[i][0]
                prob = learned[i][1]
                if file_name not in learned_prob_all.keys():
                    learned_prob_all[file_name] = {'learned':[],'predicted':[]}
                learned_prob_all[file_name]['learned'].append(prob)

            for i in range(len(predicted)):
                file_name = predicted[i][0]
                prob = predicted[i][1]
                if file_name not in learned_prob_all.keys():
                    learned_prob_all[file_name] = {'learned':[],'predicted':[]}
                learned_prob_all[file_name]['predicted'].append(prob) 

            for file_name in learned_prob_all.keys():
                try:
                    learned_prob_all[file_name]['learned'] = learned_prob_all[file_name]['learned'][len(learned_prob_all[file_name]['learned'])-1][0]
                    learned_prob_all[file_name]['predicted'] = learned_prob_all[file_name]['predicted'][len(learned_prob_all[file_name]['predicted'])-1][0]
                except:
                    learned_prob_all[file_name]['learned'] = None
                    learned_prob_all[file_name]['predicted'] = None
                    continue
        learned_prob_all_df = pd.DataFrame.from_dict(learned_prob_all, orient = 'index')
        learned_prob_all_df = learned_prob_all_df.dropna()
        rho = learned_prob_all_df.learned.corr(learned_prob_all_df.predicted, method = 'spearman')
        all_corr.append(rho)
    all_corr_process = all_corr

    stagnation_df = pd.DataFrame(zip(all_corr_process,all_corr_product), columns = ['P','C'])
    stagnation_df = stagnation_df.dropna()
    stagnation_df = pd.melt(stagnation_df)
    stagnation_df.columns = ['Metric Type', 'Correlation Score']

    plt.figure(figsize=(10, 12))
    g = sns.set(style='whitegrid',font_scale=2)
    g = sns.boxplot(x="Metric Type", y="Correlation Score", data=stagnation_df, width=0.35)
    plt.savefig('results/image/RQ6.pdf')

if __name__ == "__main__":
    create_stagnation_graph()