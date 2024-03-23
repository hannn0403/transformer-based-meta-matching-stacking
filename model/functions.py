# initialization and random seed set

import os
import scipy
import scipy.io
import torch
import random
import math
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
    

    
    
### BASIC DNN 관련한 함수 START ### 
    
# 반복실험을 이 함수 안에서 도는게 아니라, 이 함수에 안에서는 오직 한번의 Training 및 Validation을 실행 
def basic_dnn_training(train_df, train_pheno, val_df, val_pheno, batch_size, generator, seed, data_file_name):
    
    device='cuda'
    best_loss = float('inf')
    
    # Data Loader 
    train_dataloader = get_dataloader(train_df, train_pheno, batch_size, generator, device)
    val_dataloader = get_dataloader(val_df, val_pheno, batch_size, generator, device)
    
    # Model Initialization
    # model = dnn_4l(train_df.shape[1], 87, 386, 313, 0.242, train_pheno.shape[1]).to(device)
    model = dnn_4l(train_df.shape[1], 128, 512, 64, 0.3, train_pheno.shape[1]).to(device)
    
    model.to(device) 
    loss_function = nn.MSELoss()
    epochs_to_decrease_lr = 200
    optimizer = optim.AdamW(model.parameters(), lr=1e-03, weight_decay=0.4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    # 모든 Epoch에 대한 train loss와 val loss를 저장할 리스트 
    train_losses = []
    val_losses = []
    
    model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth"
    folder_path = f"D:/meta_matching_data/model_pth/{data_file_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"{folder_path}가 생성되었습니다!")
        
    
    num_epochs = 200
    
    for epoch in range(num_epochs):
        # Training 
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_dataloader: 
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = loss_function(outputs, targets) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Validation 
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_dataloader: 
                outputs = model(inputs) 
                loss = loss_function(outputs, targets) 
                val_loss += loss.item()
                
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        if best_loss > val_loss: 
            best_loss = val_loss 
            torch.save(model, model_pth)
            print(f"Epoch : {epoch}   Best Model! \t : Train Loss - {train_loss:.4f} | Val Loss - {val_loss:.4f}")
        
        elif epoch % 50 == 0: 
            print(f"Epoch : {epoch}               \t : Train Loss - {train_loss:.4f} | Val Loss - {val_loss:.4f}")
    
    
    return train_losses, val_losses

# FINE TUNING 에서 사용한 get_kshot_idx (kshot_age를 함수 안에서 나눈다.)
def get_kshot_idx(model, kshot_df, kshot_pheno, k_num, generator, device, seed): 
    '''
    Trained 된 모델을 가져와서 K sample들을 가지고 먼저 어떤 노드가 가장 높은 값을 가지는 지 확인한다. 
    '''
    model = model.to('cuda')
    kshot_age = kshot_pheno[:, -1]
    kshot_pheno = kshot_pheno[:, :-1]
    # to('cuda') & DataLoader 
    kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, k_num, generator, device)
    
    # K Samples로 최대의 COD를 가지는 값
    model.eval()
    kshot_outputs = []
    
    with torch.no_grad():
        for inputs, targets in kshot_dataloader: 
            output = model(inputs)
            kshot_outputs.append(output)
            
    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy().T
    kshot_age_cods = []
    
    for i in range(len(kshot_outputs)):
        pred = pd.DataFrame({'prediction':kshot_outputs[i], 'Age': kshot_age})
        kshot_age_cods.append(get_cod_score(pred)) # 각 phenotype을 예측하여 이것을 Age와 COD를 계산
        # kshot_age_cods.append(get_corr_score(pred))
        
    max_cod_idx = kshot_age_cods.index(max(kshot_age_cods))
    
    return max_cod_idx



def test_model(model, test_df, test_pheno, test_age, max_cod_idx, batch_size, generator):
    device='cuda'
    # to('cuda') & DataLoader 
    test_df, test_pheno = (
        torch.Tensor(test_df).to(device),
        torch.Tensor(test_pheno).to(device)
    )
    test_dataset = torch.utils.data.TensorDataset(test_df, test_pheno)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    
    
    # Testing
    test_outputs = []
    with torch.no_grad():
        for inputs, targets in test_dataloader: 
            output = model(inputs)
            test_outputs.append(output)
            
    test_outputs = torch.cat(test_outputs).cpu().detach().numpy()
    test_outputs= test_outputs[:, max_cod_idx].reshape(-1, 1) 
    pred_df = pd.DataFrame({'prediction' : test_outputs.flatten(), 'Age':test_age.flatten()})
    
    age_cod = get_cod_score(pred_df)
    age_corr = get_corr_score(pred_df)
    
    return age_corr, age_cod


def basic_dnn(df, pheno_with_age, k_num_list, data_file_name, batch_size=128, iteration=10, only_test=False):
    device='cuda'
    # 기본 k_num
    k_num=10
    test_size=0.2
    if not only_test: 
        # Training
        for seed in range(1, iteration+1): 
            print(f'==========================================Iter{seed}==========================================')
            # Random Seed Setting 
            set_random_seeds(seed) 
            generator = torch.Generator()
            generator.manual_seed(seed) 

            train_df, train_pheno, val_df, val_pheno, _, _, _, _ = \
                                        preprocess_data(df, pheno_with_age, test_size, k_num, seed)

            train_age = train_pheno[:, -1]
            train_pheno = train_pheno[:, :-1]
            val_age = val_pheno[:, -1]
            val_pheno = val_pheno[:, :-1]

            # 해당 Iteration에 대한 DataLoader 생성 / Model Initialization / Training 및 Validation을 진행하고, Best Model을 저장한다. 
            train_losses, val_losses = \
                    basic_dnn_training(train_df, train_pheno, val_df, val_pheno, batch_size, generator, seed, data_file_name)

            # 해당 Iteration의 train_loss와 val_loss를 가지고 plot을 생성 
            loss_img_pth = f'D:meta_matching_data/model_pth/plot/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.png' 
            loss_img_folder_pth = f"d:/meta_matching_data/model_pth/plot/{data_file_name}"
            if not os.path.exists(loss_img_folder_pth):
                os.makedirs(loss_img_folder_pth)
                print(f"{loss_img_folder_pth}가 생성되었습니다!")
            save_iteration_loss_plot(train_losses, val_losses, loss_img_pth, seed)
        print('==========================================학습을 완료하였습니다.==========================================')
        print('\n\n')
    else: 
        print("학습을 건너뜁니다.")
        
        
        
    corrs_10 = []
    corrs_30 = []
    corrs_50 = []
    corrs_100 = []
    cods_10 = []
    cods_30 = []
    cods_50 = [] 
    cods_100 = []

    
    # K-shot / Test 성능 측정 
    for seed in range(1, iteration+1): 
        set_random_seeds(seed) 
        generator = torch.Generator()
        generator.manual_seed(seed) 
        
        # 학습 완료된 모델 Load 
        model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth"
        model = torch.load(model_pth)
        
        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, test_size, k_num, seed)
            # kshot_age = kshot_pheno[:, -1]
            # kshot_pheno = kshot_pheno[:, :-1]
            test_age = test_pheno[:, -1]
            test_pheno = test_pheno[:, :-1]
            
            # max_cod_idx를 K Samples로 계산 
            max_cod_idx = get_kshot_idx(model, kshot_df, kshot_pheno, k_num, generator, device, seed)
            
            # 계산된 max_cod_idx로 Test 성능 
            test_corr, test_cod = test_model(model, test_df, test_pheno, test_age, max_cod_idx, batch_size, generator)
            
            if k_num == 10 : 
                corrs_10.append(test_corr)
                cods_10.append(test_cod)
            elif k_num == 30: 
                corrs_30.append(test_corr)
                cods_30.append(test_cod)
            elif k_num == 50:
                corrs_50.append(test_corr)
                cods_50.append(test_cod)
            elif k_num == 100: 
                corrs_100.append(test_corr) 
                cods_100.append(test_cod)            
            print(f"Iteration {seed} | K = {k_num} : Corr - {test_corr:.4f} & R2 - {test_cod:.4f}")
            
        print('\n\n')
    
    print(f"K=10 : Average COD : {np.mean(cods_10)}")
    print(f"K=10 : STD     COD : {np.std(cods_10)}")
    print()
    print(f"K=10 : Average Corr : {np.mean(corrs_10)}")
    print(f"K=10 : STD     Corr : {np.std(corrs_10)}")
    print('\n\n')
    print(f"K=30 : Average COD : {np.mean(cods_30)}")
    print(f"K=30 : STD     COD : {np.std(cods_30)}")
    print()
    print(f"K=30 : Average Corr : {np.mean(corrs_30)}")
    print(f"K=30 : STD     Corr : {np.std(corrs_30)}")
    print('\n\n')
    print(f"K=50 : Average COD : {np.mean(cods_50)}")
    print(f"K=50 : STD     COD : {np.std(cods_50)}")
    print()
    print(f"K=50 : Average Corr : {np.mean(corrs_50)}")
    print(f"K=50 : STD     Corr : {np.std(corrs_50)}")
    print('\n\n')
    print(f"K=100 : Average COD : {np.mean(cods_100)}")
    print(f"K=100 : STD     COD : {np.std(cods_100)}")
    print()
    print(f"K=100 : Average Corr : {np.mean(corrs_100)}")
    print(f"K=100 : STD     Corr : {np.std(corrs_100)}")   
            
        
        
    return corrs_10, cods_10, corrs_30, cods_30, corrs_50, cods_50, corrs_100, cods_100
      
    
# corrs_10, cods_10, corrs_30, cods_30, corrs_50, cods_50, corrs_100, cods_100  = \
                # basic_dnn(df, pheno_with_age, k_num_list=[10, 30, 50, 100], batch_size=128, iteration=100, only_test=False)
    
### BASIC DNN 관련한 함수 END ### 





### BASIC KRR 관련한 함수 START ### 
def basic_krr(df, pheno_with_age, k_num, batch_size=128, iteration=10, only_test=False):
    test_size=0.2
    corrs = []
    cods = []
    # Training
    for seed in range(1, iteration+1):
        print(f'==========================================Iter{seed}==========================================')
        # Random Seed Setting
        set_random_seeds(seed)
        # Train / Val / Kshot/ Test
        train_df, train_pheno, val_df, val_pheno, _, _, _, _ = \
                                    preprocess_data(df, pheno_with_age, test_size, k_num, seed)
        _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, test_size, k_num, seed)
        train_df = np.concatenate((train_df, val_df), axis=0)
        train_pheno = np.concatenate((train_pheno, train_pheno), axis=0)
        train_age = train_pheno[:, -1]
        train_pheno = train_pheno[:, :-1]
        kshot_age = kshot_pheno[:, -1]#10,1
        kshot_pheno = kshot_pheno[:, :-1]#10,58
        test_age = test_pheno[:, -1]#140,58
        test_pheno = test_pheno[:, :-1]#
        kf = KFold(n_splits=5) # K-fold 설정
        r2_scores = []
        krr_model = KernelRidge()
        alphas = [0.1, 0.7, 1, 5, 10]
        param_grid = {'alpha':alphas}
        krr_models = []
        # K-fold 로 Iteration 돈다.
        for train_index, test_index in kf.split(train_df):
            train_df_fold, val_df_fold = train_df[train_index], train_df[test_index]
            train_pheno_fold, val_pheno_fold = train_pheno[train_index], train_pheno[test_index]
            for i in range(train_pheno_fold.shape[1]):
                krr_model = KernelRidge()
                grid_search = GridSearchCV(krr_model, param_grid, cv = kf)
                grid_search.fit(train_df_fold, train_pheno_fold[:, i])
                best_alpha = grid_search.best_params_['alpha']
                best_krr = KernelRidge(alpha = best_alpha)
                best_krr.fit(train_df_fold, train_pheno_fold[:, i])
                krr_models.append(best_krr) # 모델을 List에 넣는다...?
        # 58개의 phenotype에 대해서 5 fold했고
        # 5개의 fold 중 max cod를 갖는 fold의 krr model만 추출
        max_cod_model = []
        for i in range(58):
            subset = krr_models[i::58]
            r2_scores_kshot = []
            for i in range(len(subset)):
                pheno_pred = subset[i].predict(kshot_df)
                # 여기에서는 Age가 아니라 각 phenotype이지만, 일단은 age라고 해야 한다.
                pheno_df = pd.DataFrame({'prediction' : pheno_pred, 'Age' : kshot_pheno[:, i]})
                # r2 = get_cod_score(kshot_pheno[:, i], pheno_pred)
                r2 = get_cod_score(pheno_df)
                r2_scores_kshot.append(r2)
            max_cod_model.append(r2_scores_kshot.index(max(r2_scores_kshot)))
        max_cod_krr_models = []
        for i in range(len(max_cod_model)):
            max_cod_krr_models.append(krr_models[i*5+max_cod_model[i]])
        # best cod 추출된 krr model로 k shot age cod 계산 후 best krr node 추출
        kshot_cods = []
        for i in range(len(max_cod_krr_models)):
            k_pred = max_cod_krr_models[i].predict(kshot_df)
            k_pred_df = pd.DataFrame({'prediction' : k_pred, 'Age':kshot_age})
            k_age_cod = get_cod_score(k_pred_df)
            kshot_cods.append(k_age_cod)
        print('max cod index:', kshot_cods.index(max(kshot_cods)))
        max_kshot_cod_idx = kshot_cods.index(max(kshot_cods))
        # best krr model로 test
        test_pred = max_cod_krr_models[max_kshot_cod_idx].predict(test_df)
        test_pred_df = pd.DataFrame({'prediction' : test_pred, 'Age' : test_age})
        test_age_cod = get_cod_score(test_pred_df)
        test_age_corr = get_corr_score(test_pred_df)
        corrs.append(test_age_corr)
        cods.append(test_age_cod)
        print(f'cod: {test_age_cod:.4f}')
        print(f'corr: {test_age_corr:.4f}')
    print('==========================================학습을 완료하였습니다.==========================================')
    print('\n\n')
    print(f"Average COD : {np.mean(cods):.4f}")
    print(f"STD     COD : {np.std(cods):.4f}")
    print(f"Average Corr : {np.mean(corrs):.4f}")
    print(f"STD     Corr : {np.std(corrs):.4f}")
    return cods, corrs
   
    
# basic_krr(df, pheno_with_age, k_num=10, batch_size=128, iteration=10)




### BASIC KRR END ### 




### ADVANCED FINE_TUNING START ### 
def get_kshot_model(model_pth, max_cod_idx):
    # dnn4l이라는 전제 하에 함수가 짜여졌다. 
    before_model = torch.load(model_pth)
    after_model = torch.load(model_pth)
    
    # 변화시키려는 모델의 마지막 layer를 (64, 1)의 shape을 가지도록 만든다. (다른 조건들은 동일하게 유지)
    after_model.fc4 = nn.Sequential(nn.Dropout(p=0.3, inplace=False),
                          nn.Linear(in_features=64, out_features=1, bias=True))
    
    
    with torch.no_grad():
        after_model.fc4[1].weight = nn.Parameter(before_model.fc4[1].weight[max_cod_idx:max_cod_idx+1, :])
        after_model.fc4[1].bias = nn.Parameter(before_model.fc4[1].bias[max_cod_idx:max_cod_idx+1])
        
        
    for param in after_model.fc3.parameters():
        param.requires_grad = True 
    for param in after_model.fc4.parameters():
        param.requires_grad = True 
    
    after_model = after_model.to('cuda')
    
    return after_model


def kshot_fine_tuning(model, kshot_df, kshot_pheno, max_cod_idx, fine_tuned_model_pth, generator, device, seed, data_file_name): 
    
    # epochs_to_decrease_lr=30
    
    # last 2 layers update
    optimizer = torch.optim.AdamW([{'params': model.fc3.parameters()}, {'params': model.fc4.parameters()}], lr=1e-05, weight_decay=0.4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    loss_function = nn.MSELoss()
    
    
    kshot_train_df, kshot_val_df, kshot_train_pheno, kshot_val_pheno = \
                            train_test_split(kshot_df, kshot_pheno, test_size=0.2, shuffle=False, random_state=seed)
    
    kshot_train_age = kshot_train_pheno[:, -1]
    # kshot_train_pheno = kshot_train_pheno[:, :-1]
    kshot_train_pheno = kshot_train_pheno[:, max_cod_idx] # 모델의 구조가 바뀌었으므로 max_cod_idx에 해당하는 phenotype만 남긴다. 
    kshot_val_age = kshot_val_pheno[:, -1]
    # kshot_val_pheno = kshot_val_pheno[:, :-1]
    kshot_val_pheno = kshot_val_pheno[:, max_cod_idx] # 모델의 구조가 바뀌었으므로 max_cod_idx에 해당하는 phenotype만 남긴다. 
    
    # Load to GPU & DataLoader 
    kshot_train_dataloader = get_dataloader(kshot_train_df, kshot_train_pheno, kshot_train_df.shape[1], generator, device)
    kshot_val_dataloader = get_dataloader(kshot_val_df, kshot_val_pheno, kshot_val_df.shape[1], generator, device)
    
    # Fine Tuning 진행 
    best_val_loss = float('inf')
    best_model = None
    fine_tune_count = 0
    epoch = 1 
    train_loss_list = []
    val_loss_list = []
    while epoch <= 200: 
        total_train_loss = 0
        model.train()
        
        for inputs, targets in kshot_train_dataloader: 
            optimizer.zero_grad()
            outputs = model(inputs) 
            train_loss = loss_function(outputs, targets) 
            total_train_loss += train_loss 
            train_loss.backward()
            optimizer.step() 
        scheduler.step()
        
        total_train_loss = total_train_loss / len(kshot_train_dataloader.dataset) 
        total_train_loss = torch.tensor(total_train_loss)
        train_loss_list.append(total_train_loss.to('cpu'))
        
        
        # kshot_val로 Validation Loss 계산하여 Stopping criterion
        val_preds = []
        model.eval() 
        with torch.no_grad():
            val_loss = 0 
            for inputs, targets in kshot_val_dataloader: 
                outputs = model(inputs) 
                loss = loss_function(outputs, targets) 
                val_loss += loss.item() * inputs.size(0)
                val_preds.append(outputs) 

        val_loss = val_loss / len(kshot_val_dataloader.dataset) 
        val_loss = torch.tensor(val_loss)
        val_loss_list.append(val_loss.to('cpu'))
        val_preds = torch.cat(val_preds).cpu().detach().numpy() 

        val_preds = val_preds.T
        val_gt = kshot_val_pheno.T
        val_pheno_cods = []

        if epoch % 50 == 0: 
            print(f"Epoch : {epoch} | Fine Tune Train Loss : {total_train_loss:.4f} | Validation Loss : {val_loss:.4f}")
        
        if val_loss < best_val_loss : 
            fine_tune_count = 0 
            best_val_loss = val_loss 
            best_model = model.state_dict() 
            print(f"Best Model! : Epoch {epoch} | Train Loss : {total_train_loss:.4f} | Validation Loss : {best_val_loss:.4f}")

        epoch += 1
    loss_img_pth = f'D:/meta_matching_data/model_pth/plot/fine_tune_{data_file_name}/{seed}_K_{kshot_train_df.shape[0]}_fine_tuned_dnn4l_adamw_{data_file_name}.png'
    folder_pth = f"d:/meta_matching_data/model_pth/plot/fine_tune_{data_file_name}"
    if not os.path.exists(folder_pth):
        os.makedirs(folder_pth)
        print(f"{folder_pth}가 생성되었습니다!")
        
    save_iteration_loss_plot(train_loss_list, val_loss_list, loss_img_pth, seed)
    model.load_state_dict(best_model) 
    print(f"fine_tuned_model_pth : {fine_tuned_model_pth}")
    torch.save(model, fine_tuned_model_pth) 
    print(f"Saved Best Model!!!")
    
    
    
def test_finetuned_model(fine_tuned_model_pth, test_df, test_pheno, max_cod_idx, max_cod_list, generator, seed ): 
    # Basic Model과 Fine-tuned 모델을 모두 비교해야 하기에 두 모델을 모두 불러오려고 했으나, 이것은 그냥, 나중에 test에 대한 성능을 측정하고 
    # 비교하여 Basic DNN과 Fine-tuned 중 고르면 될 것 같다. 
    # basic_model = torch.load(basic_model_pth)
    device='cuda'
    ft_model = torch.load(fine_tuned_model_pth)
    loss_function = nn.MSELoss()
    
    test_age = test_pheno[:, -1]
    # test_pheno = test_pheno[:, :-1]
    test_pheno = test_pheno[:, max_cod_idx:max_cod_idx+1]
    test_dataloader = get_dataloader(test_df, test_pheno, test_df.shape[0], generator, device)
    
    # Testing 
    test_outputs= []
    ft_model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader: 
            outputs = ft_model(inputs)
            # loss = loss_function(outputs, targets)
            test_outputs.append(outputs)
            
    test_outputs = torch.cat(test_outputs).cpu().detach().numpy()
    # test_outputs = test_outputs[:, max_cod_list[seed]].reshape(-1, 1) 
    test_outputs = test_outputs[:, 0].reshape(-1, 1) 
    
    pred = pd.DataFrame({'prediction': test_outputs[:, 0], 'Age':test_age})
    
    age_corr = get_corr_score(pred)
    age_cod = get_cod_score(pred)
    
    
    return age_corr, age_cod
   
    
    
def advanced_finetuning(df, pheno_with_age, k_num_list, data_file_name, batch_size=128, iteration=10, only_test=False):
    '''
    Basic DNN에서 저장한 모델을 불러와서, K Sample로 어떤 Node가 가장 성능이 좋은지를 판단한다. 
    이후에, K sample을 4:1의 비율로 나눠 Training을 진행하고, Validation set으로 stopping criterion을 정한다. 
    '''
    device='cuda'
    patients = 30
    test_size=0.2
    
    # 1 ~ (iteration+1)번 모델까지 어떤 node가 가장 높은 성능을 보이는 지 저장할 리스트 
    ten_shot_idx = [-100]
    thirty_shot_idx = [-100]
    fifty_shot_idx = [-100]
    hund_shot_idx = [-100]

    
    test_age_corr_10 = []
    test_age_cod_10 = []
    test_age_corr_30 = []
    test_age_cod_30 = []
    test_age_corr_50 = []
    test_age_cod_50 = []
    test_age_corr_100 = []
    test_age_cod_100 = []
    
    
    if not only_test:
        
        # 학습은 Basic DNN에서 저장한 모델을 사용 
        for seed in range(1, iteration+1): 
            set_random_seeds(seed) 
            generator = torch.Generator()
            generator.manual_seed(seed)
            # 학습 완료된 모델 Load 
            model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth"
            basic_model = torch.load(model_pth)

            # Fine-Tuning 이전 K Sample을 가지고 각각 어떤 Node가 가장 높은 성능을 보이는 지 확인 및 저장
            for k_num in k_num_list: 
                print(f"==========================================K : {k_num}==========================================")
                _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, test_size, k_num, seed)

                # BASIC DNN으로부터 K Samples로 max_cod_idx를 계산 
                max_cod_idx = get_kshot_idx(basic_model, kshot_df, kshot_pheno, k_num, generator, device, seed)

                print(f"MAX R2 score Node index is : {max_cod_idx}")
                if k_num == 10: 
                    ten_shot_idx.append(max_cod_idx) 
                elif k_num == 30: 
                    thirty_shot_idx.append(max_cod_idx) 
                elif k_num == 50: 
                    fifty_shot_idx.append(max_cod_idx) 
                elif k_num == 100: 
                    hund_shot_idx.append(max_cod_idx) 

            print("Best Node Selection 끝")

            # FINE_TUNING
            print("Fine Tuning Start!")
            for k_num in k_num_list: 
                _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, test_size, k_num, seed)

                basic_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth'
                fine_tuned_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_k_{k_num}_fine_tuned_dnn4l_adamw_{data_file_name}.pth'
             
                if k_num == 10: 
                    max_cod_list = ten_shot_idx
                elif k_num == 30: 
                    max_cod_list = thirty_shot_idx
                elif k_num == 50: 
                    max_cod_list = fifty_shot_idx
                elif k_num == 100: 
                    max_cod_list = hund_shot_idx


                # model = torch.load(basic_model_pth)
                fine_tune_model = get_kshot_model(basic_model_pth, max_cod_idx) # Basic DNN에서 best node만 뽑아 구조를 변경한 모델
                kshot_fine_tuning(fine_tune_model, kshot_df, kshot_pheno, max_cod_idx, fine_tuned_model_pth, generator, device, seed, data_file_name)

                max_cod_idx = get_kshot_idx(fine_tune_model, kshot_df, kshot_pheno, k_num, generator, device, seed)
                
                age_corr, age_cod = test_finetuned_model(fine_tuned_model_pth, test_df, test_pheno, max_cod_idx, max_cod_list, generator, seed)

                print(f"Iteration {seed}, K : {k_num} - Correlation :{age_corr:.4f}".rjust(50))
                print(f"R2 Score :{age_cod:.4f}".rjust(50))
                print('\n\n\n')


                if k_num == 10: 
                    test_age_corr_10.append(age_corr) 
                    test_age_cod_10.append(age_cod)
                elif k_num == 30: 
                    test_age_corr_30.append(age_corr)
                    test_age_cod_30.append(age_cod)
                elif k_num ==50: 
                    test_age_corr_50.append(age_corr) 
                    test_age_cod_50.append(age_cod)
                elif k_num == 100: 
                    test_age_corr_100.append(age_corr) 
                    test_age_cod_100.append(age_cod) 
        print('\n\n\n\n')
        print("FINE TUNING 완료")
    else:
        print("FINE TUNING을 건너뜁니다.")    
    
    print(f"K=10 : Average COD : {np.mean(test_age_cod_10):.4f}")
    print(f"K=10 : STD     COD : {np.std(test_age_cod_10):.4f}")
    print()
    print(f"K=10 : Average Corr : {np.mean(test_age_corr_10):.4f}")
    print(f"K=10 : STD     Corr : {np.std(test_age_corr_10):.4f}")
    print('\n\n')
    print(f"K=30 : Average COD : {np.mean(test_age_cod_30):.4f}")
    print(f"K=30 : STD     COD : {np.std(test_age_cod_30):.4f}")
    print()
    print(f"K=30 : Average Corr : {np.mean(test_age_corr_30):.4f}")
    print(f"K=30 : STD     Corr : {np.std(test_age_corr_30):.4f}")
    print('\n\n')
    print(f"K=50 : Average COD : {np.mean(test_age_cod_50):.4f}")
    print(f"K=50 : STD     COD : {np.std(test_age_cod_50):.4f}")
    print()
    print(f"K=50 : Average Corr : {np.mean(test_age_corr_50):.4f}")
    print(f"K=50 : STD     Corr : {np.std(test_age_corr_50):.4f}")
    print('\n\n')
    print(f"K=100 : Average COD : {np.mean(test_age_cod_100):.4f}")
    print(f"K=100 : STD     COD : {np.std(test_age_cod_100):.4f}")
    print()
    print(f"K=100 : Average Corr : {np.mean(test_age_corr_100):.4f}")
    print(f"K=100 : STD     Corr : {np.std(test_age_corr_100):.4f}")   
            
    
    return test_age_corr_10, test_age_cod_10, test_age_corr_30, test_age_cod_30, test_age_corr_50, test_age_cod_50, test_age_corr_100, test_age_cod_100, ten_shot_idx[1:], thirty_shot_idx[1:], fifty_shot_idx[1:], hund_shot_idx[1:]
   
# corrs_10, cods_10, corrs_30, cods_30, corrs_50, cods_50, corrs_100, cods_100 = \
#           advanced_finetuning(df, pheno_with_age, k_num_list=[10,30,50, 100], batch_size=128, iteration=100)


### FINE_TUNING END ### 




### STACKING START ### 
def get_top_k_indices(kshot_age_cods, k_num):
    # kshot_age_cods의 각 요소와 인덱스를 묶어서 리스트로 만듭니다.
    elements_with_indices = list(enumerate(kshot_age_cods))    
    # 크기를 기준으로 내림차순으로 정렬합니다.
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1], reverse=True)
    # 상위 k_num개의 요소의 인덱스를 추출합니다.
    top_k_indices = [index for index, value in sorted_elements[:k_num]]
    
    return top_k_indices


def get_elements_by_indices(input_list, indices):
    return [input_list[i] for i in indices]

def get_predict_df(model, df, pheno_with_age, k_num, generator, seed):
    '''
    K Sample 들을 이용하여, M개의 가장 좋은 예측을 수행하는 Node를 선정하고, 이 노드에 대한 예측 값들만으로 구성한 
    데이터 프레임 반환 (pred_df)
    '''
    device = 'cuda'
    m = min(58, k_num)
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, 0.2, k_num, seed)
    model = model.to(device)
    kshot_age = kshot_pheno[:, -1]
    kshot_pheno = kshot_pheno[:, :-1]
    test_age = test_pheno[:, -1]
    test_pheno = test_pheno[:, :-1]
    
    kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, k_num, generator, device)
    test_dataloader = get_dataloader(test_df, test_pheno, test_df.shape[0], generator, device)
    
    
    model.eval()
    kshot_outputs = []
    with torch.no_grad(): 
        for inputs, targets in kshot_dataloader:
            output = model(inputs) 
            kshot_outputs.append(output)
    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy().T
    kshot_age_cods = []
    
    kshot_pred_df = pd.DataFrame()
    for i in range(len(kshot_outputs)):
        if i == 0:
            kshot_pred_df = pd.DataFrame({'Age':kshot_age, f'prediction_{i}':kshot_outputs[i]})
        else: 
            kshot_pred_df[f"prediction_{i}"]=kshot_outputs[i]
        pred = pd.DataFrame({'prediction':kshot_outputs[i], 'Age':kshot_age})
        kshot_age_cods.append(get_cod_score(pred)) # 각 phenotype을 예측하여, 예측 값과 COD를 계산
    
    top_k_indices = get_top_k_indices(kshot_age_cods, m)
    column_list = [f"prediction_{i}" for i in top_k_indices]
    column_list.append('Age')    
        
    
    
    model.eval()
    test_outputs = []
    with torch.no_grad(): 
        for inputs, targets in test_dataloader: 
            output = model(inputs)
            test_outputs.append(output) 
        test_outputs = torch.cat(test_outputs).cpu().detach().numpy().T
        test_age_cods = []
        
        
    test_pred_df = pd.DataFrame()
    for i in range(len(test_outputs)):
        if i == 0: 
            test_pred_df = pd.DataFrame({'Age': test_age, f"prediction_{i}":test_outputs[i]})
        else: 
            test_pred_df[f"prediction_{i}"]=test_outputs[i]
        
    
    
    kshot_pred_df = kshot_pred_df.loc[:, column_list]
    test_pred_df = test_pred_df.loc[:, column_list]
    
    return kshot_pred_df, test_pred_df, top_k_indices
    
    
def advanced_stacking(df, pheno_with_age, k_num_list, data_file_name, batch_size=128, iteration=10):
    device='cuda'
    
    age_corr_10 = []
    age_cod_10 = []
    age_corr_30 = []
    age_cod_30 = []
    age_corr_50 = []
    age_cod_50 = []
    age_corr_100 = []
    age_cod_100 = []
    best_node_10 = []
    best_node_30 = []
    best_node_50 = []
    best_node_100 = []

    for seed in range(1, iteration+1):
        basic_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth'
        set_random_seeds(seed)
        generator = torch.Generator()
        generator.manual_seed(seed) 

        basic_model = torch.load(basic_model_pth)

        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            # Secondary Model(KRR)에 들어갈 INPUT (K-shot & test) 
            kshot_pred_df, test_pred_df, top_k_indices = get_predict_df(basic_model, df, pheno_with_age, k_num, generator, seed) 
            # Hyperparam Tuning 
            kshot_x = kshot_pred_df.drop('Age', axis=1)
            kshot_y = kshot_pred_df['Age']
            krr = KernelRidge(kernel='rbf')# 하이퍼 파라미터 튜닝? 
            alphas = [0.1, 0.7, 1, 5, 10]
            param_grid = {'alpha':alphas}
            kf = KFold(n_splits=5)
            grid_search = GridSearchCV(krr, param_grid, cv=kf)
            grid_search.fit(kshot_x, kshot_y)

            best_alpha = grid_search.best_params_['alpha']
            best_krr = KernelRidge(kernel='rbf', alpha=best_alpha)  # 최적의 alpha를 가진 KRR 선언 
            best_krr.fit(kshot_x, kshot_y)

            # Testing 
            test_x = test_pred_df.drop('Age', axis=1)
            test_y = test_pred_df['Age']
            # prediction
            stacking_pred = best_krr.predict(test_x)

            final_pred_df = pd.DataFrame({'prediction':stacking_pred, 'Age':test_y})
            age_corr = get_corr_score(final_pred_df)
            age_cod = get_cod_score(final_pred_df)

            if k_num==10:
                age_corr_10.append(age_corr) 
                age_cod_10.append(age_cod)
                best_node_10.append(top_k_indices)
            elif k_num == 30: 
                age_corr_30.append(age_corr) 
                age_cod_30.append(age_cod)
                best_node_30.append(top_k_indices)
            elif k_num == 50: 
                age_corr_50.append(age_corr) 
                age_cod_50.append(age_cod)
                best_node_50.append(top_k_indices)
            elif k_num == 100: 
                age_corr_100.append(age_corr) 
                age_cod_100.append(age_cod)
                best_node_100.append(top_k_indices)
               
            print(f"Iteration {seed}, K : {k_num} - Correlation :{age_corr:.4f}".rjust(50))
            print(f"R2 Score :{age_cod:.4f}".rjust(50))
        print('\n\n\n')
    
    
    
    print(f"K=10 : Average COD : {np.mean(age_cod_10):.4f}")
    print(f"K=10 : STD     COD : {np.std(age_cod_10):.4f}")
    print()
    print(f"K=10 : Average Corr : {np.mean(age_corr_10):.4f}")
    print(f"K=10 : STD     Corr : {np.std(age_corr_10):.4f}")
    print('\n\n')
    print(f"K=30 : Average COD : {np.mean(age_cod_30):.4f}")
    print(f"K=30 : STD     COD : {np.std(age_cod_30):.4f}")
    print()
    print(f"K=30 : Average Corr : {np.mean(age_corr_30):.4f}")
    print(f"K=30 : STD     Corr : {np.std(age_corr_30):.4f}")
    print('\n\n')
    print(f"K=50 : Average COD : {np.mean(age_cod_50):.4f}")
    print(f"K=50 : STD     COD : {np.std(age_cod_50):.4f}")
    print()
    print(f"K=50 : Average Corr : {np.mean(age_corr_50):.4f}")
    print(f"K=50 : STD     Corr : {np.std(age_corr_50):.4f}")
    print('\n\n')
    print(f"K=100 : Average COD : {np.mean(age_cod_100):.4f}")
    print(f"K=100 : STD     COD : {np.std(age_cod_100):.4f}")
    print()
    print(f"K=100 : Average Corr : {np.mean(age_corr_100):.4f}")
    print(f"K=100 : STD     Corr : {np.std(age_corr_100):.4f}")               
    
    
    return age_cod_10, age_corr_10, age_cod_30, age_corr_30, age_cod_50, age_corr_50, age_cod_100, age_corr_100, best_node_10, best_node_30, best_node_50, best_node_100
    
    
# advanced_stacking(df, pheno_with_age, [10, 30, 50, 100], batch_size=128, iteration=100)


### STACKING END ### 