from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling,FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
import pymoo.gradient.toolbox as anp
from configparser import SafeConfigParser
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter
from bisect import bisect_left
from BEMT_program.solver import Solver
 
from drawing_pic import draw_block,draw_process

eps = 0.01
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_cons(local):
    if local[0]==0 and local[1] == 0:
        return 0
    else:
        return len(local)//2

class constraint:

    def __init__(self,config_path):
        cfg = SafeConfigParser()
        cfg.read(config_path,encoding='utf-8')
        
        self.range = cfg.getfloat('cacl','range')
        self.lb    = cfg.getfloat('cacl','lower_T')
        self.local_cl = [float(_) for _ in cfg.get('cacl','local_cl').split()]
        self.local_pitch = [float(_) for _ in cfg.get('cacl','local_pitch').split()]
        self.local_cR = [float(_) for _ in cfg.get('cacl','local_cR').split()]
        self.smooth  = [float(_) for _ in cfg.get('cacl','smooth').split()]

        self.pop_size = cfg.getfloat('GA','pop_size')
        self.seed     = cfg.getfloat('GA','seed')
        self.total    = cfg.getfloat('GA','total')

        self.process  = cfg.getboolean('save','process')
        self.process_path = cfg.get('save','process_path')
        self.result   = cfg.getboolean('save','result')
        self.result_path = cfg.get('save','result_path')


    def get_len(self,len_sec):

        self.n_var = 7 + len_sec
        self.n_obj = 1 #效率最大(悬停或者有前进速度）
        self.n_ieq_constr = 1 + get_cons(self.local_cl)+get_cons(self.local_cR)+get_cons(self.local_pitch)
        #三个截面限制+最小值限制
        return self.n_var,self.n_obj,self.n_ieq_constr

def f3(x, t):
    i = bisect_left(x, t)
    if x[i] - t > 0.5:
        i-=1
    return i

def get_Bspline(radius,pitch,k=4):
    t_pitch,c_pitch,_  = interpolate.splrep(radius,pitch,s=0,k=k)
    
    tail     = np.zeros(k+1)
    c_pitch  = c_pitch[:len(radius)]
    return t_pitch,c_pitch,tail


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

def reduce_B(radius,t,c,k=4):

    pitch_x = []
    for i in range(len(t)):
        t1 = t[i]
        c1 = c[i]

        pitch_x.append(interpolate.splev(radius,(t1,c1,k)))
    pitch_x = np.array(pitch_x)

    return pitch_x

def reduce_q(radius,po,p1,p2):
    
    c_Rx = []
    for i in range(len(po)):

        po1 = po[i].tolist()
        p11 = p1[i].tolist()
        p21 = p2[i].tolist()
        
        po1 = po1[0]

        po_r = f3(radius,po1)
        r_left  = radius[:po_r]
        r_right = radius[po_r:]

        c_left  = np.polyval(p11,r_left)
        c_right = np.polyval(p21,r_right)

        c_R1 = np.concatenate((c_left,c_right))
        c_Rx.append(c_R1)
    
    c_Rx = np.array(c_Rx)
    return c_Rx 

def get_eta(pitch_x,c_Rx,s:Solver,c:constraint):
    eta = []
    T   = []
    g_cl_list    = []
    g_pitch_list = []
    g_cR_list    = []    
    degree = c.smooth  
    BEM_path = s.path
    v_inf    = s.v_inf
    local_cl = np.array(c.local_cl).reshape(-1,2)
    local_pitch = np.array(c.local_pitch).reshape(-1,2)
    local_cR = np.array(c.local_cR).reshape(-1,2)
    R  = s.rotor.diameter/2
    for i in range(len(pitch_x)):
        
        c1 = c_Rx[i] * R
        p1 = pitch_x[i] * 180/np.pi
        
        window_lenth = int(len(c1)*degree[0])
        if window_lenth%2 ==0:
            window_lenth = window_lenth -1
        smooth_k = int(degree[1])

        c1 = savgol_filter(c1,window_lenth,smooth_k)
        p1 = savgol_filter(p1,window_lenth,smooth_k)

        s1       = Solver(BEM_path,c1=c1,p1=p1)
        with HiddenPrints():
            df, sections = s1.run_sweep('v_inf', 1, v_inf, v_inf)
        if df['eta'][0]>1:
            df['eta'][0] = 100
        else:
            df['eta'][0] = df['eta'][0]*100
        
        va     = sections[0].values
        radius = va[:,0]
        chord  = va[:,1]
        pitch  = va[:,2]
        cl     = va[:,3]

        po_cl_list    = []
        po_pitch_list = []
        po_cR_list    = []
        for _ in local_cl:
            po_cl_list.append(f3(radius,_[0]))
        for _ in local_pitch:
            po_pitch_list.append(f3(radius,_[0]))
        for _ in local_cR:
            po_cR_list.append(f3(radius,_[0]))
        
        a = []
        for _ in range(len(po_cl_list)):
            a.append((cl[po_cl_list[_]] - local_cl[_][1] )**2)
        g_cl_list.append(a)

        a = []
        for _ in range(len(po_pitch_list)):
            a.append((pitch[po_pitch_list[_]] - local_pitch[_][1] )**2)
        g_pitch_list.append(a)

        a = []
        for _ in range(len(po_cR_list)):
            a.append((chord[po_cR_list[_]] - local_cR[_][1])**2)
        g_cR_list.append(a)
        
        if v_inf > eps:
            eta.append(df['eta'][0])
        else:
            t0 = df['T'][0]
            p0 = df['P'][0]
            R  = s.rotor.diameter/2
            rho = s.fluid.rho
            FM = t0/p0*np.sqrt(t0/2/np.pi/R/R/rho)
            eta.append(FM)

        T.append(df['T'][0])
    eta = np.array(eta)
    T   = np.array(T)
    # g_cl_list    = np.array(g_cl_list)
    # g_pitch_list = np.array(g_pitch_list)
    # g_cR_list    = np.array(g_cR_list)

    return eta,T,g_cl_list,g_pitch_list,g_cR_list

process_eta = []
process_T  = []

class SphereWithConstraint(Problem):

    def __init__(self, s:Solver,c:constraint,save_path):
        self.l  = 0
        self.s  = s
        self.n_var = c.n_var
        self.n_obj = c.n_obj
        self.n_ieq_constr = c.n_ieq_constr
        self.c  = c
        self.save_path = save_path

        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_ieq_constr=self.n_ieq_constr, xl=- c.range,xu= c.range)

    def get_rotor(self):
        s = self.s
        R = s.rotor.diameter
        sections = s.rotor.sections
        foil_name = []
        radius   = []
        pitch    = []
        chord    = []

        v_inf    = s.v_inf
        rpm      = s.rpm
        for i in sections:
            foil_name.append(i.airfoil.name)
            radius.append(i.radius)
            pitch.append(i.pitch)
            chord.append(i.chord/R)
        
        num_sec = len(foil_name)
        radius,pitch,c_R = np.array(radius),np.array(pitch),np.array(chord)

        self.num_sec = num_sec
        self.radius  = radius
        self.pitch   = pitch
        self.c_R      = c_R
        
        self.v_inf   = v_inf
        self.rpm     = rpm
        return num_sec,radius,pitch,c_R
    
    def get_base(self):
        pop_size = int(self.c.pop_size)
        x = np.zeros((pop_size,1))
        t_pitch,c_pitch,tail = get_Bspline(self.radius,self.pitch,k=4)
        
        self.tail0          = np.expand_dims(tail,0).repeat(x.shape[0],axis=0)
        self.t_pitch   = np.expand_dims(t_pitch,0).repeat(x.shape[0],axis=0)
        self.c_pitch   = np.expand_dims(c_pitch,0).repeat(x.shape[0],axis=0)
        
        cR_p1,cR_p2,cR_posi = get_quadric(self.radius,self.c_R)
        self.cR_p1     = np.expand_dims(cR_p1,0).repeat(x.shape[0],axis=0)
        self.cR_p2     = np.expand_dims(cR_p2,0).repeat(x.shape[0],axis=0)
        self.cR_posi   = np.expand_dims(cR_posi,0).repeat(x.shape[0],axis=0)
        v_inf   = self.v_inf
        with HiddenPrints():
            df, sections = self.s.run_sweep('v_inf', 1, v_inf, v_inf)
        
        self.base_T       = df["T"][0]
        self.R  = self.s.rotor.diameter/2
        R = self.R
        if v_inf > eps:
            self.base_eta     = df['eta'][0]
        else:
            t0 = df['T'][0]
            p0 = df['P'][0]
            R  = self.s.rotor.diameter/2
            rho = self.s.fluid.rho
            FM = t0/p0*np.sqrt(t0/2/np.pi/R/R/rho)
            self.base_eta = FM
        
        self.min_T = self.c.lb * self.base_T
        draw_block(self.radius,self.c_R*R,self.pitch*180/np.pi,'OPTIMIZE_fig/','base',self.base_eta,self.base_T)

    def _evaluate(self, x, out, *args, **kwargs):
        self.l += 1
        
        print('current loop:',self.l)
        

        para0,len0 = get_para(self.c_pitch)
        para1,len1 = get_para(self.cR_p1)
        para2,len2 = get_para(self.cR_p2)
        para3,len3 = get_para(self.cR_posi)
        
        cur_p   = 0       
        c_pitch   = np.add(x[:,cur_p:cur_p+len0]*para0,self.c_pitch)
        c_pitch   = np.concatenate((c_pitch,self.tail0),1)
        cur_p    += len0
        cR_p1     = np.add(x[:,cur_p:cur_p+len1]*para1,self.cR_p1)
        cur_p    += len1
        cR_p2     = np.add(x[:,cur_p:cur_p+len2]*para2,self.cR_p2)
        cur_p    += len2
        cR_posi   = np.add(x[:,cur_p:cur_p+len3]*para3,self.cR_posi)
        
        cur_p    += len3
        pitch_x  = reduce_B(self.radius,self.t_pitch,c_pitch)

        cR_x     = reduce_q(self.radius,cR_posi,cR_p1,cR_p2)
        
        eta,T,g_cl_list,g_pitch_list,g_cR_list = \
        get_eta(pitch_x,cR_x,self.s,self.c)
        
        process_eta.append(np.max(eta)/100)
        process_T.append(np.max(T))

        draw_process(process_eta,self.base_eta,process_T,self.base_T,self.save_path)
        draw_block(self.radius,cR_x[0]*self.R,pitch_x[0]*180/np.pi,'OPTIMIZE_fig/',str(0),eta[0],T[0])
        f1 = -1*eta

        g1 = self.min_T - np.array(T)
        g2 = np.array( g_cl_list) - eps
        g3 = np.array(g_pitch_list) - eps
        g4 = np.array(g_cR_list) - eps
        
        np.save(self.c.result_path+'x',x)
        out['F'] = f1
        out['G'] = np.column_stack([g1,g2,g3,g4])
        



        

