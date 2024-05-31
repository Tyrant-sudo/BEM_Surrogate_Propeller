import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.data import random_split
    
import os
import math
import pandas as pd
import numpy as np
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 99,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 30000,     # Number of epochs.            
    'batch_size': 128, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.   
    'stage_step': 10000,  
    'save_path': '/home/sh/WCY/auto_propeller/resource5/1_model/Noise_model1.ckpt', # Your model will be saved here.
    'loss_history_path': '/home/sh/WCY/auto_propeller/resource5/1_model/Loss_Noise_model1.json'
}


def train_valid_split(train_x, train_y, valid_ratio, seed):
    # 移除含有 NaN 值的行
    nan_mask_x = np.isnan(train_x).any(axis=(1, 2))
    nan_mask_y = np.isnan(train_y).any(axis=(1))
    nan_mask = nan_mask_x | nan_mask_y
    
    train_x_clean = train_x[~nan_mask]
    train_y_clean = train_y[~nan_mask]
    
    # 清洗 train_y 中大于三倍平均值的值
    # mean_y = np.mean(train_y_clean, axis=0)
    # outlier_mask_y = np.any(train_y_clean > 3 * mean_y, axis=1)

    # 应用逻辑非操作以保留非异常值
    # train_x_clean = train_x_clean[~outlier_mask_y]
    # train_y_clean = train_y_clean[~outlier_mask_y]
    
    # 计算分割后的训练集和验证集大小
    total_size = len(train_x_clean)
    valid_size = int(total_size * valid_ratio)
    train_size = total_size - valid_size

    # 随机生成索引并分割数据
    np.random.seed(seed)
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_x_split = train_x_clean[train_indices]
    train_y_split = train_y_clean[train_indices]
    vali_x_split = train_x_clean[valid_indices]
    vali_y_split = train_y_clean[valid_indices]

    return train_x_split, train_y_split, vali_x_split, vali_y_split

def trainer(train_loader, valid_loader, model, config, device):
    
    model = model.to(device)
    criterion = nn.MSELoss(reduction='mean') # Define your loss function

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9,weight_decay=0.001) 

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        for x, y in train_loader:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            # exit()

        mean_train_loss = sum(loss_record)/len(loss_record)
        train_loss_history.append(mean_train_loss) # 保存训练损失

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                
            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        valid_loss_history.append(mean_valid_loss) # 保存验证损失

        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            # print('\nModel is not improving, so we halt the training session.')
            1
            # return
            # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate']/10, momentum=0.9,weight_decay=0.001) 
    loss_history = {
        "train_loss": train_loss_history,
        "valid_loss": valid_loss_history
    }
    with open(config['loss_history_path'], 'w') as f:
        json.dump(loss_history, f)
  

import re
def parse_float_array_from_df_column(column):
    """
    解析 DataFrame 列中的字符串数组，每个字符串表示一个浮点数数组。
    返回一个 numpy 数组，形状为 (n_samples, n_features)。
    """
    parsed_data = []
    for item in column:
        # 去除不需要的字符并基于空格分割字符串
        clean_str = re.sub(r'[^\d.\s]', '', item)
        float_array = np.array([float(num) for num in clean_str.split()])
        parsed_data.append(float_array)
    
    return np.array(parsed_data)

if __name__ == "__main__":
    from scipy.interpolate import splprep, splev


    Noise_data = "../0_database/Noisedabase.csv"
    head_Noise = ("r/R(chord)","c/R","r/R(pitch)","twist (deg)","SPL1","SPL2","OASPL")
    df = pd.read_csv(Noise_data)
    # 解析输入列
    train_x_col1 = parse_float_array_from_df_column(df[head_Noise[1]])
    
    x0  = parse_float_array_from_df_column(df[head_Noise[0]])
    x,y = parse_float_array_from_df_column(df[head_Noise[2]]),parse_float_array_from_df_column(df[head_Noise[3]])*np.pi/180
    
    train_x_col2 = []
    for i in range(x0.shape[0]):
        tck, u = splprep([x[i], y[i]], k=4, s=2)
        x_new, y_new = splev(x0[i], tck)
        train_x_col2.append(y_new)
    
    train_x = np.stack((train_x_col1, np.array(train_x_col2)), axis=-1)
    
    # 选择输出列
    train_y = parse_float_array_from_df_column(df[head_Noise[6]])
    
    # train_y = 
    
    train_x, train_y, vali_x,vali_y = train_valid_split(train_x, train_y, config["valid_ratio"], config["seed"])
    
    from models import My_Dataset
    train_dataset, vali_dataset, test_dataset = My_Dataset(train_x,train_y), \
                                            My_Dataset(vali_x, vali_y), \
                                            My_Dataset(vali_x)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    vali_loader  = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    from models import Model_N0
    
    model = Model_N0()
    trainer(train_loader, vali_loader, model, config, device)
