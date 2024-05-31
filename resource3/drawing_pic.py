import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def draw_block(r_R,c_R,beta,save_path,epoch,eta,T,st = True):
    
    if st == True:
        name_txt = save_path +'/' + str(epoch)+'_'+str(eta)[:5]+'_'+str(T)[:5] + '.txt'
        with open(name_txt,'w') as f:
            f.write('r_R  \n' + str(r_R.tolist()) +'\n')
            f.write('c_R  \n' + str(c_R.tolist()) +'\n')
            f.write('beta  \n' + str(beta.tolist()) +'\n')
        


    name = save_path +'/' +  str(epoch)
    r_R,c_R,beta
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111)
    ax.plot(r_R,beta,label='beta',color='r')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(r_R,c_R,label='chord')
    print('max c:',np.max(c_R),'max beta:',np.max(beta))
    ax2.legend(loc='upper right')

    plt.title("eta = {:.3f},T={:.3f}".format(eta,T))
  
    plt.savefig(name)
    plt.close()

def draw_process(process_eta,base_eta,process_T,base_T,fig_path):
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111)
    ax.plot(process_eta,label='eta',color = 'r')
    ax.axhline(y = base_eta,color = 'r')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(process_T,label = 'T',color = 'b')
    ax2.axhline(y = base_T,color = 'b')
    ax2.legend(loc='upper right')
    plt.savefig(fig_path+'/process')
    plt.close()


