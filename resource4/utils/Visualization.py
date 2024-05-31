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


def easy_draw(foil_data,save_name,elevation_angle=-15, azimuth_angle = 65 ):
    # -15, 65

    # p = foil_data
    foil_data = foil_data.reshape(-1,3)
    x = foil_data[:,1]
    y = foil_data[:,2]
    z = foil_data[:,0]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    max_range = np.array([x.max()-x.min(), 
                          y.max()-y.min(), 
                          z.max()-z.min()]).max() / 2.0

    # 找到中心点
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    # 设置统一的轴比例
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)

    ax.scatter3D(x,y,z,s=0.5)
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)

    # plt.axis('off')
    plt.show()
    plt.savefig(save_name)
    plt.close()

    return 
