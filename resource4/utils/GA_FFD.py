from pymoo.core.problem import Problem
import pymoo.gradient.toolbox as anp

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter
from bisect import bisect_left

from Visualization import draw_block,draw_process, easy_draw
from Configuration import Constraint,HiddenPrints
from Geometry  import build_BEMprop_from_PointsCloud

from ffd.deform import get_ffd
from BEMT_program.solver import Solver

eps = 0.01


def f3(x, t):
    i = bisect_left(x, t)
    if x[i] - t > 0.5:
        i-=1
    return i

def get_FFDPara(PointsCloud, out_dim = (3, 7, 1) ,z_dim = 0):

    Points = PointsCloud.reshape(-1, 3)
    dim = np.array(out_dim)

    b,p = get_ffd(Points, dim)
    
    return b,p 

def revert_FFDPoints(control_points, base_para):
    
    PointsCloud = np.dot(base_para, control_points)

    return PointsCloud

def getGroup_EtaT_from_Solver(displace_x, BasePoints, BasePara, Mesh_dim , s:Solver, min_eta, min_T):
    eta_list = []
    T_list   = []
    Chord_list_list = []
    Pitch_list_list = []
    PointsCloud_list = []

    BEM_path = s.path
    v_inf    = s.v_inf
    
    BasePara   = BasePara.copy()
    BasePoints = BasePoints.copy()

    # 读取s，提取每个的airfoil，输入得到airfoil参数
    for i in range(len(displace_x)):
        CurPoints = BasePoints.copy()
        CurPoints[:,[1,2]] = CurPoints[:,[1,2]] * (1 + displace_x[i].reshape(-1,2))

        PointsCloud = np.dot(BasePara, CurPoints)
        PointsCloud = PointsCloud.reshape(Mesh_dim)

        try:
            eta, T, Chord_list, Pitch_list = get_EtaT_from_PointsCloud(PointsCloud, v_inf, BEM_path)
        except:
            eta, T = min_eta, min_T
            _, Chord_list, Pitch_list, _= \
            build_BEMprop_from_PointsCloud(PointsCloud)
            
        eta_list.append(eta)
        T_list.append(T)
        Chord_list_list.append(Chord_list)
        Pitch_list_list.append(Pitch_list)
        PointsCloud_list.append(PointsCloud)
       
    return np.array(eta_list), np.array(T_list), np.array(Chord_list_list), np.array(Pitch_list_list), PointsCloud_list

def get_EtaT_from_Solver(s:Solver):

    v_inf    = s.v_inf
    with HiddenPrints():
        df, sections = s.run_sweep('v_inf', 1, v_inf, v_inf)
    if df['eta'][0]>1:
        df['eta'][0] = 10
    else:
        df['eta'][0] = df['eta'][0]*100
    
    if v_inf > eps:
        FM = df['eta'][0]
    else:
        t0 = df['T'][0]
        p0 = df['P'][0]
        R  = s.rotor.diameter/2
        rho = s.fluid.rho
        FM = t0/p0*np.sqrt(t0/2/np.pi/R/R/rho)

    T = df['T'][0]

    return FM, T    


def get_EtaT_from_PointsCloud(PointsCloud, v_inf, BEM_path):
    
    Radius_list, Chord_list, Pitch_list, SectionCoordinate_list = \
    build_BEMprop_from_PointsCloud(PointsCloud)

    s = Solver(BEM_path, s1 = SectionCoordinate_list, c1 = Chord_list, p1 = Pitch_list)
    
    with HiddenPrints():
        df, sections = s.run_sweep('v_inf', 1, v_inf, v_inf)
    if df['eta'][0]>1:
        df['eta'][0] = 10
    else:
        df['eta'][0] = df['eta'][0]*100
    
    if v_inf > eps:
        FM = df['eta'][0]
    else:
        t0 = df['T'][0]
        p0 = df['P'][0]
        R  = s.rotor.diameter/2
        rho = s.fluid.rho
        FM = t0/p0*np.sqrt(t0/2/np.pi/R/R/rho)

    T = df['T'][0]

    return FM, T , Chord_list, Pitch_list


def get_quadric(radius,c_R):
    posi_max = np.argmax(c_R)
    po_r     = radius[posi_max]
    c_R_left = c_R[:posi_max]
    radius_left = radius[:posi_max]
    c_R_right = c_R[posi_max:]
    radius_right = radius[posi_max:]

    coef1 = np.polyfit(radius_left,c_R_left,2)
    coef2 = np.polyfit(radius_right,c_R_right,2)

    return coef1,coef2,np.array([po_r])

def get_para(list):
    list = abs(list)
    try:
        lenth = len(list[0])
    except:
        lenth = 1 
    return np.max(list),lenth


def get_cons(local):
    if local[0]==0 and local[1] == 0:
        return 0
    else:
        return len(local)//2

class Optimize_BestEta_ConsPitchCl(Problem):

    def __init__(self, s:Solver,c:Constraint,save_path):
        self.process_eta = []
        self.process_T  = []
        self.l  = 0
        self.s  = s

        c.get_FFD_cons()
        self.n_var = c.FFD_n_var
        self.n_obj = c.FFD_n_obj
        self.n_ieq_constr = c.FFD_n_ieq_constr
        self.c  = c
        self.save_path = save_path
        
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_ieq_constr=self.n_ieq_constr, xl=- c.FFD_range,xu= c.FFD_range)

    def get_rotor(self):
        s = self.s
        c = self.c

        R = s.rotor.diameter
        v_inf    = s.v_inf
        rpm      = s.rpm
        
        self.PointsCloud = np.load(c.StartMesh_Path)


        radius, c_R, pitch, SectionCoordinate_list= \
        build_BEMprop_from_PointsCloud(self.PointsCloud)
        
        num_sec = len(SectionCoordinate_list)
        radius,pitch,c_R = np.array(radius),np.array(pitch),np.array(c_R)

        self.num_sec = num_sec
        self.radius  = radius
        self.pitch   = pitch
        self.c_R      = c_R
        self.Dim_Point  = c.Dim_ControlPoints

        self.v_inf   = v_inf
        self.rpm     = rpm
        return num_sec,radius,pitch,c_R
    
    def get_base(self):
        c = self.c
        s = self.s

        pop_size = int(c.pop_size)
        x = np.zeros((pop_size,1))
        
        self.StartMesh  = np.load(c.StartMesh_Path)
        easy_draw(self.StartMesh, self.c.result_path+ 'test_FFD/BaseMesh0.png', -15, 65)

        self.MeshDim    = self.StartMesh.shape
        
        self.BasePoints = np.load(c.BasePoints_Path)
        self.BasePara   = np.load(c.BasePara_Path)
        
        easy_draw(np.dot(self.BasePara, self.BasePoints).reshape(self.StartMesh.shape), self.c.result_path+ 'test_FFD/BaseMesh1.png', -15, 65)
        
        Dim_Point  = self.Dim_Point
        Dim_x      = Dim_Point[0] * Dim_Point[1] * Dim_Point[2]

        if Dim_x !=  self.BasePoints.shape[0]:
            print("Dim {} error for control points {}".format(Dim_Point, self.BasePoints.shape))
            exit()

        self.displace = np.zeros_like(self.BasePoints).reshape(-1)
        self.displace = np.expand_dims(self.displace,0).repeat(x.shape[0],axis=0)

        with HiddenPrints():
            FM, T = get_EtaT_from_Solver(s)
            self.base_T   = T
            self.base_eta = FM 
        self.min_T = self.c.lb * self.base_T
        self.min_eta = self.c.lb * self.base_eta

        draw_block(self.radius,self.c_R,self.pitch,self.save_path,'base',self.base_eta,self.base_T)

    def _evaluate(self, x, out, *args, **kwargs):
        process_eta = self.process_eta
        process_T   = self.process_T
        
        StartMesh  = self.StartMesh.copy()
        BasePoints = self.BasePoints.copy()
        BasePara   = self.BasePara.copy() 
        s          = self.s
        self.l += 1
        print('current loop:',self.l)
        
        eta_list,T_list, Chord_list, Pitch_list, PointsCloud_list = \
        getGroup_EtaT_from_Solver(x, BasePoints=BasePoints, BasePara= BasePara, Mesh_dim= self.MeshDim, s=s, min_eta=self.min_eta, min_T=self.min_T, BaseMesh = StartMesh )

        process_eta.append(np.max(eta_list))
        arg_best = np.argmax(eta_list)

        draw_block(self.radius,Chord_list[arg_best], Pitch_list[arg_best],self.save_path,'best',eta_list[arg_best],T_list[arg_best],False)

        easy_draw(PointsCloud_list[arg_best], self.c.result_path+ 'test_FFD/BestMesh{}.png'.format(self.l), -15, 65)
        
        process_T.append(np.max(T_list))

        draw_process(process_eta,self.min_eta,process_T,self.min_T,self.save_path)
        
        f1 = -eta_list
        
        np.save(self.c.result_path+'test_FFD/FFD_x',x)
        np.save(self.c.result_path+'test_FFD/FFD_process_eta', np.array(process_eta))
        np.save(self.c.result_path+'test_FFD/FFD_process_T', np.array(process_T))
        
        out['F'] = f1
    
    def draw_result(self,result_file,save_file):
        x = np.load(result_file)

        

if __name__ == '__main__':
    
    BEM_path  = '/home/sh/WCY/auto_propeller/resource4/0_data/start_prop.ini'
    Geom_path = '/home/sh/WCY/auto_propeller/resource4/0_data/start_mesh.npy'
    res_path  ='/home/sh/WCY/auto_propeller/resource4/utils/restrict.ini'
    result_path  = '/home/sh/WCY/auto_propeller/resource4/2_output/test_FFD/FFD_x.npy'
    pic_path  = '/home/sh/WCY/auto_propeller/resource4/2_output/test_FFD/'
    save_path = '/home/sh/WCY/auto_propeller/resource4/2_output/test_FFD/'
    
    
    PointsCloud = np.load(Geom_path)
    print(PointsCloud.shape)
    
    Radius_list, Chord_list, Pitch_list, SectionCoordinate_list = \
    build_BEMprop_from_PointsCloud(PointsCloud)

    s = Solver(BEM_path, s1 = SectionCoordinate_list, c1 = Chord_list, p1 = Pitch_list)
    r = Constraint(res_path)

    problem = Optimize_BestEta_ConsPitchCl(s,r, save_path)
    problem.get_rotor()
    problem.get_base()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import BinaryRandomSampling,FloatRandomSampling
    from pymoo.optimize import minimize

    algorithm = NSGA2(pop_size=int(r.pop_size),
                  sampling=FloatRandomSampling(),
                #   crossover=TwoPointCrossover(),
                #   mutation=BitflipMutation(),
                  eliminate_duplicates=True)

    res = minimize(problem,
                algorithm,
                ('n_gen', int(r.total)),
                seed=int(r.seed),
                #    save_history = True,
                #    verbose=True)
    )
    
    problem.draw_result(result_path,pic_path)