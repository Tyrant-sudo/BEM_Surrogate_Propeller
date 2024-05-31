from ast import Delete
from cmath import isnan
from email import header
import numpy as np
from deal_xfoil import oper_visc_alpha
import csv
import tqdm
import time
import os


header = ('Re', 'alpha','cl', 'cd','cm')
def form_xfoil(foil_name,file_path,Re,alpha,mach=0.1):
    

    if 'naca' in foil_name or 'NACA' in foil_name:
        a = foil_name
        gen_naca =True
    else:
        a = foil_name.split('/')[-1].split('.')[0]
        # foil_name = file_path + '/'+ foil_name
        gen_naca = False
    
        
    # Re_list      =  np.random.uniform(low = 10e4, high = 10e7, size=20)

    print(a,foil_name)

    file_name = file_path+'/xfoil_Re'+str(Re)+'_'+a

    with open(file_name,'w',errors='ignore') as f:
        write = csv.writer(f)
        write.writerow(header)
        


        tmp = oper_visc_alpha(foil_name,alpha,Re,mach,gen_naca=gen_naca,show_seconds=0)
        # print(tmp.shape)
        # print(tmp)
        try:
            
            tmp = np.delete(tmp,[3,5,6],axis=1)
        
            err = []
            ttmp = np.isnan(tmp)
            for i in range(len(ttmp)):
                if True in np.isnan(tmp[i]):
                    err.append(i)
            tmp = np.delete(tmp,err,axis=0)
            tmp = np.insert(tmp,0,values=Re,axis=1)
            
            pst_a = tmp[0][1]
            st = 0
            for i in range(len(tmp)):
                cur_a = tmp[i][1]
                if cur_a < pst_a:
                    st = i
                    break
                else:
                    pst_a = cur_a
            
            tmp = tmp[st:]

            write.writerows(tmp)
            
            print('xfoil       result saved')
            return a,tmp
        except:
            print('xfoil calculate failed!!!')

            return None

# def get_xfoil(coordinate):



            
if __name__ =="__main__":

    foil_name,file_path,Re,alpha,mach='ara_d_20.dat','./',1e5,[-15,15,1],0.1

    form_xfoil(foil_name,file_path,Re,alpha,mach)

