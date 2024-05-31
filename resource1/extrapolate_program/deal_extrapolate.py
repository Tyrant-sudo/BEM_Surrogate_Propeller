from copyreg import add_extension
from itertools import count
from .airfoilprep import Polar, Airfoil
import numpy as np
import csv
import matplotlib.pyplot as plt
import ast
import os

file_path = './result_xfoil3/'
save_path = './result_extrapolate3/'

camber_list          = np.linspace(0,5,6,dtype=np.int0)
camber_position_list = np.linspace(0,5,6,dtype=np.int0)
thickness_list       = np.linspace(10,30,21,dtype=np.int0)

def form_filename(file_path,endwith:str):
    
    file_list = []
    file_name = os.listdir(file_path)
    if endwith and endwith[0]!='.':
        endwith = '.'+endwith

    for path in file_name:
        i = path
        j = os.path.splitext(path)
        if j[1] == endwith:
            file_list.append(file_path + i)
    
    return file_list

def extract_matrix(file_path,file_name:str):
    
    Re_list    = []
    alpha_list = []
    cl_list    = []
    cd_list    = []
    cm_list    = []

    file = file_path+file_name

    with open(file,'r') as f:
        reader = csv.DictReader(f)
        lenth  = 0
        try:
            pre    = (next(reader))
            Re_pre = ast.literal_eval(pre['Re'])

            alpha_pre = ast.literal_eval(pre['alpha'])
        except:
            return file_name,Re_list,alpha_list,cl_list,cd_list,cm_list
        
        alpha  = []
        cl     = []
        cd     = []
        cm     = []

        alpha.append(ast.literal_eval(pre['alpha']))
        try:
            cl   .append(ast.literal_eval(pre[' cl']))
        except:
            cl   .append(ast.literal_eval(pre['cl']))
        cd   .append(ast.literal_eval(pre['cd']))
        cm   .append(ast.literal_eval(pre['cm']))

        for i in reader:
            lenth+=1 
            Re_cur    = ast.literal_eval(i['Re'])

            alpha_cur = ast.literal_eval(i['alpha'])
            
            # if Re_cur != Re_pre:
            if alpha_cur < alpha_pre:
                
                Re_list   .append(Re_pre)
                alpha_list.append(alpha)
                cl_list   .append(cl)
                cd_list   .append(cd)
                cm_list   .append(cm)
                
                return file_name,Re_list,alpha_list,cl_list,cd_list,cm_list
                Re_pre    = Re_cur
                # alpha_pre = alpha_cur
                alpha,cl,cd,cm  = [],[],[],[]

            alpha.append(ast.literal_eval(i['alpha']))
            try:
                cl   .append(ast.literal_eval(i[' cl']))
            except:
                cl   .append(ast.literal_eval(i['cl']))
            cd   .append(ast.literal_eval(i['cd']))
            cm   .append(ast.literal_eval(i['cm']))
            
            alpha_pre = alpha_cur #重置判断
    
    try:
        Re_list   .append(Re_cur)
    except:
        return file_name,Re_list,alpha_list,cl_list,cd_list,cm_list
    alpha_list.append(alpha)
    cl_list   .append(cl)
    cd_list   .append(cd)
    cm_list   .append(cm)
    
    return file_name,Re_list,alpha_list,cl_list,cd_list,cm_list

def form_extrapolate(save_path,file_name,Re,alpha,cl,cd,cm):
    
    p2 = Polar(Re,alpha,cl,cd,cm)

    af = Airfoil([p2])

    cdmax = 1.3
    cdmin = 0.0138
    try:
        af_extrap = af.extrapolate(cdmax,cdmin=0.0138)
        af_extrap.writeToAerodynFile(save_path+file_name)
    except:
        1
    
    print('extrapolate result saved')
    # fig = af_extrap.plot()
    # plt.savefig('extrapolate_fig/'+file_name+'_'+str(Re)+'.jpg')

def is_number(str):
    try:
        # 因为使用float有一个例外是'NaN'
        if str=='NaN':
            return False
        float(str)
        return True
    except ValueError:
        return False

def form_plt(filename:str,stratline =0):
    # 从一个空格分割的文件提取矩阵
    file = open(filename,encoding='unicode_escape')

    a    = []
    while 1:
        lines = file.readlines()
        if not lines:
            break
        count = 1
        for line in lines:
            if count > stratline:
                a.append(line)
            count+=1
    
    for i in range(len(a)):

        tmp = a[i].split("\n")[0].split("\t")

       
        ttmp = []
        for j in tmp:
            if is_number(j):
                ttmp.append(float(j))
        
        if ttmp ==[]:
            tmp = a[i].split("\n")[0].split(" ")
            ttmp = []
            for j in tmp:
                if is_number(j):
                    ttmp.append(float(j))

        a[i] = ttmp
        
    
    file.close()
    return np.array(a)

def draw_curve(file_name,foil_name):

    f = file_name + foil_name 
    # print(f)
    m = form_plt(f,14)

    
    angle = m[:,0]
    cl    = m[:,1]
    cd    = m[:,2]

    plt.plot(angle,cl,label = 'cl')
    plt.plot(angle,cd,label = 'cd')
    plt.legend()
    plt.savefig('extrapolate_fig/'+foil_name+'_'+'.jpg')
    plt.show()
    plt.close()


if __name__ == '__main__':

    file_list = form_filename(file_path,'')

    for i in file_list:
        foil_name = i.split('/')[-1]
        
        file_name,Re_list,alpha_list,cl_list,cd_list,cm_list = extract_matrix(file_path,foil_name)
        print(i)
        for j in range(len(Re_list)):

            form_extrapolate(save_path,file_name,Re_list[j],alpha_list[j],cl_list[j],cd_list[j],cm_list[j])
    

        draw_curve(save_path,foil_name)