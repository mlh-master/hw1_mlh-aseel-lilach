# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.drop(extra_feature, axis=1).apply(pd.to_numeric, errors='coerce').dropna()
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.drop(extra_feature, axis=1).apply(pd.to_numeric, errors='coerce')
    c_cdf = c_ctg.apply(lambda x: np.where(x.isnull(), np.random.choice(x.dropna(),size=len(x.isnull())), x), axis=0)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    summary = c_feat.describe().loc[['min','25%','50%','75%','max'],:].T
    summary.rename(columns={'25%': 'Q1','50%': 'median','75%': 'Q3'}, inplace=True)
    d_summary = summary.T.to_dict()
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
#     c_no_outlier = c_feat.apply(lambda x: np.where((np.abs(x - x.mean()) <= (3*(x.std()))),x, None), axis=0).astype('float64').dropna()
    
    c_no_outlier = c_feat.copy()
    for feature in d_summary:
        IQR = np.abs(d_summary[feature]['Q3'] - d_summary[feature]['Q1'])
        upper = d_summary[feature]['Q3'] + 1.5 * IQR
        lower = d_summary[feature]['Q1'] - 1.5 * IQR
        c_no_outlier.loc[((c_feat[feature]<lower) | (c_feat[feature]>upper)), feature] = None
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[c_cdf[feature] <= thresh]
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = {}
    if mode=='standard':
        nsd_res = CTG_features.apply(lambda x: (x-x.mean())/x.std()) # standard
    elif mode=='MinMax':
        nsd_res = CTG_features.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x))) # MinMax
    elif mode=='mean':    
        nsd_res = CTG_features.apply(lambda x: (x-x.mean())/(np.max(x)-np.min(x))) # mean
    else:
        nsd_res = CTG_features
    
    xlbl = {'LB':'beats/sec', 'AC':'accelerations per second', 'FM':'fetal movements per second', 'UC':'uterine contractions per second',
            'DL':'light decelerations per second', 'DS':'severe decelerations per second','DP':'severe decelerations per second', 
            'ASTV':'percentage of time with abnormal short term variability', 'MSTV':'mean value of short term variability',
            'ALTV':'percentage of time with abnormal long term variability', 'MLTV':'mean value of long term variability',
            'Width':'width of FHR histogram', 'Min':'minimum of FHR histogram', 'Max':'Maximum of FHR histogram', 
            'Nmax':'# of histogram peaks', 'Nzeros':'# of histogram zeros', 'Mode':'histogram mode', 'Mean':'histogram mean', 
            'Median':'histogram median', 'Variance':'histogram variance', 'Tendency':'histogram tendency'}
    if flag:
        axarr = nsd_res[[y,x]].hist(bins=50, figsize=(15, 5))  # histograms of dataframe variables
        for idx, ax in enumerate(axarr.flatten()):
            if mode == 'none':
                ax.set_xlabel(xlbl[ax.get_title()])
            elif mode == 'standard':
                ax.set_xlabel(f'{xlbl[ax.get_title()]} - standardized')
            else:
                ax.set_xlabel(f'{xlbl[ax.get_title()]} - normalized')    
            ax.set_ylabel("Count")

        plt.suptitle(f'normalized/standardized by series according to mode={mode}')
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)

