import numpy as np
import pandas as pd

from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz


def get_cR_beta(coordinate_list,R=1):
    
    r_R       = []
    c_R_list  = []
    beta_list = []
    for i in coordinate_list:
        
        x = i[:,0]
        y = i[:,1]
        z = i[:,2]

        r_R.append(x[0])
        l_max = np.argmax(y)
        l_min = np.argmin(y)
        y_max = y[l_max]
        z_max = z[l_max]
        y_min = y[l_min]
        z_min = z[l_min]
        
        c_R  = np.sqrt((y_max-y_min)**2 + (z_max - z_min)**2)/R
        beta = np.arctan((z_max-z_min)/(y_max-y_min))/np.pi*180
        c_R_list.append(c_R)
        beta_list.append(beta)

    return r_R,c_R_list,beta_list

def interpolate_airfoil(Q, N, k=3, D=20, resolution=1000):
    '''Q.shape = (,2) Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res
    # # tck为B样条(权重，坐标值，阶数3），u为参数权重
    uu = np.linspace(u.min(), u.max(), resolution)
    # # 得到插值点坐标
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D

    # 得到1000个点曲率？
    cv_int = cumtrapz(cv, uu, initial=0)
    # # 求曲率+20的积分函数？
    fcv = interp1d(cv_int, uu)
    # # 积分函数连续
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    # 积分函数上的196个点
    x_new, y_new = splev(u_new, tck, der=0)

    xy_new = np.vstack((x_new,y_new)).T

    
    return xy_new

def find_furthest_points(slice_data):
    """
    找到截面上距离最远的两个点。

    :param slice_data: DataFrame或二维numpy数组，包含截面上的点坐标。
    :return: 一个元组，包含距离最远的两个点的坐标。
    """
    from scipy.spatial.distance import pdist, squareform

    if isinstance(slice_data, pd.DataFrame):
        points = slice_data.to_numpy()
    else:
        points = slice_data

    # 计算所有点对之间的距离
    distance_matrix = squareform(pdist(points, 'euclidean'))

    # 找到距离最远的两个点的索引
    furthest_points_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    
    if points[furthest_points_idx[0],0] < points[furthest_points_idx[1],0]:
        left  = furthest_points_idx[0]
        right = furthest_points_idx[1]
    else:
        right = furthest_points_idx[0] 
        left  = furthest_points_idx[1]

    return left,right ,np.max(distance_matrix)


def get_BEMsection(SectionPoints, arg0, arg1, chord, twist):
    # 调整点的位置
    SectionPoints[:, 0] -= SectionPoints[arg0, 0]
    SectionPoints[:, 1] -= SectionPoints[arg0, 1]

    # 缩放到弦长
    SectionPoints[:, 0] /= chord
    SectionPoints[:, 1] /= chord

    # 计算旋转矩阵
    cos_theta = np.cos(-twist)
    sin_theta = np.sin(-twist)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    # 对每个点应用旋转矩阵
    rotated_points = np.dot(SectionPoints, rotation_matrix.T)

    # 判断是否需要翻转
    # if rotated_points[:, 0].mean() < 0:
    #     rotated_points[:, 0] = -rotated_points[:, 0]
    #     rotated_points[:, 1] = -rotated_points[:, 1]

    # 设置特定点的坐标
    rotated_points[arg1, :] = [1.0, 0.0]

    # 分割为上下两部分，并按照x坐标进行排序
    points_below = rotated_points[rotated_points[:, 1] < 0]
    points_below = points_below[points_below[:, 0].argsort()[::-1]]

    points_above = rotated_points[rotated_points[:, 1] >= 0]
    points_above = points_above[points_above[:, 0].argsort()]

    # 合并并重新排序
    reordered_points = np.vstack([points_below, points_above])

    # 计算厚度
    t_c = reordered_points[:, 1].max() - reordered_points[:, 1].min()

    # 确保首尾相接
    reordered_points = np.vstack([reordered_points[-1], reordered_points])
    
    return reordered_points, t_c


def save_array_as_txt(array, filename):
    # 确保数组是二维的且有两列
    if array.shape[1] != 2:
        raise ValueError("Array must be of shape (n, 2)")

    with open(filename, 'w') as file:
        # 写入头部信息
        # file.write("Section 20% AIRFOIL\n  1.000000  0.000000\n")
        file.write("Section 20% AIRFOIL\n ")

        # 遍历数组中的每一行并写入文件
        for row in array:
            file.write(f"  {row[0]:.6f}  {row[1]:.6f}\n")
        
        # file.write("  1.000000  0.000000\n")
            
def build_BEMprop_from_PointsCloud(PointsCloud:np.array,direct = False, Diameter = 2.0 , root_rR = 0.15, Beta_34 = 20):
    # 16,68,3 for the poinscloud num_slices, section_coordinate, zxy

    PointsCloud = PointsCloud.copy()
    Num_Sections = np.shape(PointsCloud)[0]

    z_min = PointsCloud[:,0,0].min()
    z_max = PointsCloud[:,0,0].max()
    R     = Diameter / 2

    ratio  = (z_max - z_min)/((1 - root_rR) * R)
    root_x = PointsCloud[0,:,1].mean()
    root_y = PointsCloud[0,:,2].mean()
    
    clockwise = -1 if direct is True else 1
    PointsCloud[:,:,0] = (PointsCloud[:,:,0] - z_min)/ ratio + root_rR
    PointsCloud[:,:,1] = (PointsCloud[:,:,1] - root_x)/ratio * clockwise 
    PointsCloud[:,:,2] = (PointsCloud[:,:,2] - root_y)/ratio  

    Radius_list = []
    Chord_list  = []
    Twist_list  = []

    SectionCoordinate_list = []
    
    for section_coor in PointsCloud:
        
        Radius_list.append(section_coor[0,0]/R)
        
        arg0, arg1, chord = find_furthest_points(section_coor[:,[1,2]])
        Chord_list.append(chord/R)

        x0,y0 = section_coor[arg0,1], section_coor[arg0,2]
        x1,y1 = section_coor[arg1,1], section_coor[arg1,2]

        twist      =  - np.arctan((y1- y0)/ (x1 - x0 + 1e-10))
        Twist_list.append( twist  * 180/ np.pi)
    
        cos_theta = np.cos(twist)
        sin_theta = np.sin(twist)

        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        # 调整点的位置
        section_coor[:, 1] -= section_coor[arg0, 1]
        section_coor[:, 2] -= section_coor[arg0, 2]

        # 缩放到弦长
        # section_coor[:, 1] /= chord
        # section_coor[:, 2] /= chord
        SectionPoints  = np.dot( section_coor[::1,[1,2]] , rotation_matrix.T)/chord

        SectionCoordinate_list.append(np.array(interpolate_airfoil(SectionPoints, 200)))

        # SectionCoordinate_list.append(SectionPoints)

    NumSec_34        = int(np.floor(Num_Sections * 3/4))
    twist_drift      = Twist_list[NumSec_34] - Beta_34
    Pitch_list       = [_ - twist_drift for _ in Twist_list]

    return Radius_list, Chord_list, Pitch_list, SectionCoordinate_list

def build_PointsCloud_from_list(ChordList, PitchList, SectionCoordinateList, Diameter = 2.0 , root_rR = 0.15, Beta_34 = 20):
    
    1


def build_BEM_from_List(output_file,Radius_R_list, Section_list, Chord_R_list, Twist_list,direct = True, Beta_34 = 20, Num_Blade = 2, Diameter = 0.12):
    
    Title            = "...BEM Propeller..."   
    Num_Sections     = len(Radius_R_list) 
    Num_Blade        = Num_Blade
    Diameter         = Diameter * 100
    Beta_34          = Beta_34

    Feather          = 0
    Pre_Cone         = 0
    Center           = [0.0, 0.0 ,0.0]
    Normal           = [-1.0, 0.0, 0.0]


    NumSec_34        = int(np.floor(Num_Sections * 3/4))

    twist_drift      = Twist_list[NumSec_34] - Beta_34
    Twist_list       = [_ - twist_drift for _ in Twist_list]

    Rake_R_list      = [0 for _ in range(Num_Sections)]
    Sweep_list       = [0 for _ in range(Num_Sections)]
    t_c_list         = [0 for _ in range(Num_Sections)]
    Skew_R_list      = [0 for _ in range(Num_Sections)]
    Cli_list         = [0 for _ in range(Num_Sections)]
    Axial_list       = [0 for _ in range(Num_Sections)]
    Tangential_list  = [0 for _ in range(Num_Sections)]

    for _ in range(Num_Sections):
        Section_array = Section_list[_]
        t_c           = Section_array[:,1].max() - Section_array[:,1].min()
        print(Section_array[:,1].max(),Section_array[:,1].min())
        t_c_list[_] = t_c

    data = [[Radius_R_list[_],Chord_R_list[_],Twist_list[_],Rake_R_list[_],Skew_R_list[_],Sweep_list[_],t_c_list[_],Cli_list[_],Axial_list[_],Tangential_list[_]]\
            for _ in range(Num_Sections)]

    # 创建文件并写入数据
    with open(output_file, 'w') as file:
        file.write('...BEM Propeller...\n')
        file.write(f'Num_Sections: {Num_Sections}\n')
        file.write(f'Num_Blade: {Num_Blade}\n')
        file.write(f'Diameter: {Diameter:.8f}\n')
        file.write(f'Beta 3/4 (deg): {Beta_34:.8f}\n')
        file.write(f'Feather (deg): {Feather:.8f}\n')
        file.write(f'Pre_Cone (deg): {Pre_Cone:.8f}\n')
        file.write(f'Center: {Center[0]:.8f}, {Center[1]:.8f}, {Center[2]:.8f}\n')
        file.write(f'Normal: {Normal[0]:.8f}, {Normal[1]:.8f}, {Normal[2]:.8f}\n')
        file.write('\nRadius/R, Chord/R, Twist (deg), Rake/R, Skew/R, Sweep, t/c, CLi, Axial, Tangential\n')
        for row in data:
            file.write(', '.join(f'{val:.8f}' for val in row) + '\n')
        file.write('\n')
        for idx, section_data in enumerate(Section_list):
            file.write(f'Section {idx} X, Y\n')
            for point in section_data:
                file.write(', '.join(f'{val:.8f}' for val in point) + '\n')
            file.write('\n')
        
    return 



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    start_mesh = np.load('/home/sh/WCY/auto_propeller/resource4/0_data/start_mesh.npy')
    print(start_mesh.shape)

    Radius_list, Chord_list, Pitch_list, SectionCoordinate_list = \
    build_BEMprop_from_PointsCloud(start_mesh)
    
    SectionCoordinate_list = np.array(SectionCoordinate_list)
    print([_ *0.3 for _ in Radius_list], Chord_list, Pitch_list, SectionCoordinate_list.shape)
    
    section_path = '/home/sh/WCY/auto_propeller/resource4/0_data/sections/'
    for i,SectionCoor in enumerate(SectionCoordinate_list):
    
        plt.plot( SectionCoor[:,0], SectionCoor[:,1])
        plt.axis('equal')
        plt.savefig('{}Section{}.png'.format(section_path, i))
        plt.close()

        # np.save('{}Section{}'.format(section_path, i), SectionCoor)
        save_array_as_txt(SectionCoor,'{}Section{}.txt'.format(section_path, i) )

    from Visualization import easy_draw

    # easy_draw(start_mesh, 'test3.png', -15, 65)