import torch
import csv
import numpy as np

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in test_loader:
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

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

from scipy.interpolate import splprep, splev

def get_samedim(x0, x, y):
    # x0:tragetdim, x,y: original data

    train_x_col2 = []
    for i in range(x0.shape[0]):
        tck, u = splprep([x[i], y[i]], k=4, s=2)
        x_new, y_new = splev(x0[i], tck)
        train_x_col2.append(y_new)

    return np.array(train_x_col2)