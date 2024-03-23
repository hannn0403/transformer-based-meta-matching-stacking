# initialization and random seed set

import os
import scipy
import scipy.io
import torch
import math
import random
import sklearn
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import Counter
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from CBIG_model_pytorch import dnn_4l, dnn_5l
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")




def get_upper_tri(directory_path, length):
    file_list = []
    idx = []
    for filename in os.listdir(directory_path):
        subject = scipy.io.loadmat(directory_path + filename) 
        a = subject['connectivity'][np.triu_indices(length, k=1)]
        file_list.append(a) 
        idx.append(filename[:6])
    result = np.vstack(file_list) 
    return idx, result



def get_common_elements(list1, list2): 
    counter1 = Counter(list1) 
    counter2 = Counter(list2) 
    
    # 교집합 
    intersection = counter1 & counter2 
    common_values = list(intersection.elements())
    
    return common_values


def get_additional_elements(small_list, big_list):
    small_set = set(small_list)
    big_set = set(big_list) 
    
    result_set = big_set - small_set 
    result = list(result_set)
    return result 


def set_random_seeds(seed): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def preprocess_data(df, pheno_with_age, test_size, k_num, seed): 
    '''
    Shuffling을 진행한 뒤에 Meta-Train set과 Meta-Test set으로 분리한다. 
    이후에 Meta-Train set을 각각 Train / Validation으로 나눠 Z-score normalization을 진행하고 
    Meta-Test set을 각각 Kshot / Test으로 나눠 Z-Score normalization을 진행한다. 
    '''
    # Shuffling (이후에는 shuffling 하지 않음) 
    merged_df = pd.merge(df, pheno_with_age, on='Subject')
    shuffled_df = merged_df.sample(frac=1, random_state=seed) 
    
    pheno = shuffled_df[pheno_with_age.columns].to_numpy()
    df = shuffled_df[df.columns].to_numpy()
    
    
    # Meta-Train set (Train & Validation) / Meta-Test set (K-shot & Test)
    meta_train_df, meta_test_df, meta_train_pheno, meta_test_pheno = \
            train_test_split(df, pheno, test_size=test_size, random_state=seed, shuffle=False)

    # Train & Validation Split 
    train_df, val_df, train_pheno, val_pheno = \
            train_test_split(meta_train_df, meta_train_pheno, test_size=test_size, random_state=seed, shuffle=False)
    # K-Shot & Test split 
    kshot_df, test_df, kshot_pheno, test_pheno = \
            train_test_split(meta_test_df, meta_test_pheno,  train_size=k_num, random_state=seed, shuffle=False) 
    
    # Phenotype Scaling 
    meta_train_scaler = StandardScaler()
    meta_test_scaler = StandardScaler()
    
    meta_train_scaler.fit(train_pheno)
    train_pheno = meta_train_scaler.transform(train_pheno)
    val_pheno = meta_train_scaler.transform(val_pheno)
    
    meta_test_scaler.fit(kshot_pheno)
    kshot_pheno = meta_test_scaler.transform(kshot_pheno)
    test_pheno = meta_test_scaler.transform(test_pheno)
    
    return train_df, train_pheno, val_df, val_pheno, kshot_df, kshot_pheno, test_df, test_pheno

def get_dataloader(df, pheno, batch_size, generator, device): 
    df, pheno = (
        torch.Tensor(df).to(device), 
        torch.Tensor(pheno).to(device)
    )
    dataset= torch.utils.data.TensorDataset(df, pheno) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=generator)

    return dataloader

def get_cod_score(predict_and_gt):
    X = predict_and_gt['Age']
    y = predict_and_gt['prediction']
    # 절편(intercept)을 추가합니다.
    X = sm.add_constant(X)
    # OLS 모델을 만들고 fitting 합니다.
    model = sm.OLS(y, X).fit()
    # R-squared 값을 가져옵니다.
    r_squared = model.rsquared
    return r_squared

def get_corr_score(predict_and_gt):
    correlation = predict_and_gt['prediction'].corr(predict_and_gt['Age'], method='pearson')
    return correlation


def save_iteration_loss_plot(train_loss_list, val_loss_list, loss_img_pth, seed):
    
    epochs = range(1, len(train_loss_list)+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_list, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_loss_list, label='Validation Loss', marker='o', linestyle='-')

    # 그래프에 레이블, 제목, 범례 등 추가
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1.5)
    plt.title(f'{seed}th Iteration : Train and Validation Loss Over Epochs')
    plt.legend()
    
    plt.savefig(loss_img_pth, dpi=300)
    
def mean_and_std(lst):
    absolute_values = [abs(x) for x in lst]
    mean = sum(absolute_values) / len(absolute_values)
    variance = sum((x - mean) ** 2 for x in absolute_values) / len(absolute_values)
    std_dev = math.sqrt(variance)
    return mean, std_dev




def correlation_viz(dnn_corrs, adft_dnn_corrs, adst_dnn_corrs, data_file_name):
    fig = plt.figure(figsize =(10, 8))
    # dnn_corrs = [dnn_corrs_10, dnn_corrs_30, dnn_corrs_50, dnn_corrs_100]
    # adft_dnn_corrs = [adft_corrs_10, adft_corrs_30, adft_corrs_50, adft_corrs_100]
    # adst_dnn_corrs = [adst_age_corr_10, adst_age_corr_30, adst_age_corr_50, adst_age_corr_100]
    plt.boxplot(dnn_corrs, positions=[1, 5,9,13], patch_artist = True,boxprops = dict(facecolor = "lightsteelblue"))
    plt.boxplot(adft_dnn_corrs, positions=[2,6,10,14],patch_artist = True,boxprops = dict(facecolor = "cornflowerblue"))
    plt.boxplot(adst_dnn_corrs, positions=[3,7,11,15], patch_artist = True, boxprops = dict(facecolor = "royalblue"))

    # x축 라벨 설정
    plt.xticks([2,6,10,14], [10, 30, 50, 100], fontsize = 12)


    label_1 = mpatches.Patch(color='lightsteelblue', label='Basic DNN')
    label_2 = mpatches.Patch(color='cornflowerblue', label='Advanced Fine-tuning')
    label_3 = mpatches.Patch(color='royalblue', label='Advanced Stacking')

    plt.legend(handles=[label_1, label_2, label_3], fontsize = 12, frameon = False)
    
    title = f"{data_file_name.split('_')[0].upper()} : {data_file_name.split('_')[1].upper()}"
    plt.title(title, fontsize = 20)
    plt.xlabel('Number of participants(K-shot)',fontsize=14)
    plt.ylabel('Prediction performance(correlation)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig(f'D:/meta_matching_data/results/plot/corr/{data_file_name}.png')
    
    
def cod_viz(dnn_cods, adft_dnn_cods, adst_dnn_cods, data_file_name):
    fig = plt.figure(figsize =(10, 8))


    plt.boxplot(dnn_cods, positions=[1, 5,9,13], patch_artist = True,boxprops = dict(facecolor = "lightsteelblue"))
    plt.boxplot(adft_dnn_cods, positions=[2,6,10,14],patch_artist = True,boxprops = dict(facecolor = "cornflowerblue"))
    plt.boxplot(adst_dnn_cods, positions=[3,7,11,15], patch_artist = True, boxprops = dict(facecolor = "royalblue"))

    # x축 라벨 설정
    plt.xticks([2,6,10,14], [10, 30, 50, 100], fontsize = 12)


    label_1 = mpatches.Patch(color='lightsteelblue', label='Basic DNN')
    label_2 = mpatches.Patch(color='cornflowerblue', label='Advanced Fine-tuning')
    label_3 = mpatches.Patch(color='royalblue', label='Advanced Stacking')

    plt.legend(handles=[label_1, label_2, label_3], fontsize = 12, frameon = False)
    
    title = f"{data_file_name.split('_')[0].upper()} : {data_file_name.split('_')[1].upper()}"
    plt.title(title, fontsize = 20)
    plt.xlabel('Number of participants(K-shot)',fontsize=14)
    plt.ylabel('Prediction performance(COD)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig(f'D:/meta_matching_data/results/plot/cod/{data_file_name}.png')

    
def save_results(dnn_cods, dnn_corrs, adft_dnn_cods, adft_dnn_corrs, adst_dnn_cods, adst_dnn_corrs, data_file_name): 
    #dnn
    df1 = pd.DataFrame(dnn_cods).T
    df2 = pd.DataFrame(dnn_corrs).T

    columns = ['cods_k10', 'cods_k30', 'cods_k50', 'cods_k100','corrs_k10', 'corrs_k30', 'corrs_k50', 'corrs_k100']
    dnn_df = pd.concat([df1, df2], axis=1)
    dnn_df.columns = columns
    dnn_df.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_dnn.csv')


    #advanced finetuning
    df1 = pd.DataFrame(adft_dnn_cods).T
    df2 = pd.DataFrame(adft_dnn_corrs).T

    adft_df = pd.concat([df1, df2], axis=1)
    adft_df.columns = columns
    adft_df.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_adfn.csv')


    #advanced stacking
    df1 = pd.DataFrame(adst_dnn_cods).T
    df2 = pd.DataFrame(adst_dnn_corrs).T

    adst_df = pd.concat([df1, df2], axis=1)
    adst_df.columns = columns
    adst_df.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_adsn.csv')
    print("Save Results Complete!")


def save_best_nodes(adft_best_nodes, adst_best_nodes, data_file_name):
    # dnn best node나 adft best node 같아서 dnn best node에 관한 변수 추가 하지 않음
    df1 = pd.DataFrame(adft_best_nodes).T
    df2 = pd.DataFrame(adst_best_nodes).T

    columns = ['adft_k10','adft_k30','adft_k50','adft_k100', 'adst_k10','adst_k30','adst_k50','adst_k100']
    best_nodes = pd.concat([df1, df2], axis=1)
    best_nodes.columns = columns
    best_nodes.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_bestnodes.csv')
    print("Save Best Nodes Complete!")