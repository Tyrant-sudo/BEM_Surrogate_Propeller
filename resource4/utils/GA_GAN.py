import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
import time
import torch.optim as optim
import torch.autograd as autograd
import random

from pymoo.core.problem import Problem
import pymoo.gradient.toolbox as anp

import matplotlib.pyplot as plt
import itertools

from BEMT_program.solver import Solver

from Visualization import draw_block,draw_process, easy_draw
from Configuration import Constraint, HiddenPrints
from Geometry  import build_BEMprop_from_PointsCloud

eps = 0.01

class generator5(nn.Module):
    # initializers
    def __init__(self, noize = 16):
        super(generator5, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noize,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.LeakyReLU()
        )
        self.con = nn.Sequential(
            nn.ConvTranspose3d(256,128,(4,4,4),(2,2,2),(1,1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128,64,(4,4,4),(2,2,1),(2,2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64,32,(4,4,4),(1,2,1),(1,1,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32,2,(4,4,4),(1,2,2),(1,1,2)),
            # nn.LeakyReLU()
        )
        
    
    def forward(self, input, base_point, base_para):
    
        input = input.float()
        b  = input.size()[0]
        x = self.fc(input)
        x = x.view(-1,256,1,1,1)

        x = self.con(x)
        # print(torch.mean(x))

        x = x.view(b,2,64).transpose(1,2)  
        
        base_points  = base_point.expand(b,-1,-1).clone()
        base_paras   = base_para.expand(b,-1,-1).clone()
        
        base_points[:,:,[1,2]] += x

        x = torch.bmm(base_paras,base_points)
        
        # x2 = x[:,:,[0]]
        return x

def get_cons(local):
    if local[0]==0 and local[1] == 0:
        return 0
    else:
        return len(local)//2


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

def getGroup_EtaT_from_Solver(group_x, BasePoints, BasePara, Generator, MeshDim, s:Solver, min_eta, min_T):
    eta_list = []
    T_list   = []
    Chord_list_list = []
    Pitch_list_list = []
    PointsCloud_list = []

    BEM_path = s.path
    v_inf    = s.v_inf
    
    BasePoints = torch.tensor(BasePoints.copy())
    BasePara   = torch.tensor(BasePara.copy())
    group_z = torch.tensor(np.array(group_x))

    group_generatedpoints = Generator(group_z, BasePoints, BasePara).detach().numpy()

    for i in range(group_generatedpoints.shape[0]):

        PointsCloud = group_generatedpoints[i].reshape(MeshDim)
        
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



class Optimize_BestEta_ConsZ(Problem):

    def __init__(self, s:Solver, c:Constraint, save_path):

        self.process_eta = []
        self.process_T  = []
        self.l  = 0
        self.s  = s
        
        c.get_GAN_cons()
        self.n_var = c.GAN_n_var
        self.n_obj = c.GAN_n_obj
        self.n_ieq_constr = c.GAN_n_ieq_constr
        self.c  = c
        self.save_path = save_path
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_ieq_constr=self.n_ieq_constr, xl=- c.gan_range,xu= c.gan_range)

    def get_rotor(self):

        s = self.s
        c = self.c

        R = s.rotor.diameter
        v_inf    = s.v_inf
        rpm      = s.rpm
        
        seed = c.seed
        self.BatchSize = c.pop_size
        self.Dim_noize = c.gan_dim
        self.range     = c.gan_range
        
        G = generator5(self.Dim_noize)
        G.load_state_dict(torch.load(c.gan_model))
        self.G = G

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
        easy_draw(self.StartMesh, self.c.result_path+ 'test_GAN/BaseMesh0.png', -15, 65)

        self.MeshDim    = self.StartMesh.shape
        
        self.BasePoints = np.load(c.gan_basepoints)
        self.BasePara   = np.load(c.gan_basepara)
        
        easy_draw(np.dot(self.BasePara, self.BasePoints).reshape(self.StartMesh.shape), self.c.result_path+ 'test_GAN/BaseMesh1.png', -15, 65)
        

        self.z = np.zeros_like(c.gan_dim).reshape(-1)

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
        G          = self.G
        self.l += 1
        print('current loop:',self.l)

        eta_list,T_list, Chord_list, Pitch_list, PointsCloud_list = \
        getGroup_EtaT_from_Solver(x, BasePoints= BasePoints, BasePara= BasePara, Generator=G, MeshDim= self.MeshDim, s=s, min_eta= self.min_eta, min_T= self.min_T)

        process_eta.append(np.max(eta_list))
        arg_best = np.argmax(eta_list)

        draw_block(self.radius,Chord_list[arg_best], Pitch_list[arg_best],self.save_path,'best',eta_list[arg_best],T_list[arg_best],False)

        easy_draw(PointsCloud_list[arg_best], self.c.result_path+ 'test_GAN/BestMesh{}.png'.format(self.l), -15, 65)
        
        process_T.append(np.max(T_list))

        draw_process(process_eta,self.min_eta,process_T,self.min_T,self.save_path)
        
        f1 = -eta_list
        
        np.save(self.c.result_path+'test_GAN/GAN_x',x)
        np.save(self.c.result_path+'test_GAN/GAN_process_eta', np.array(process_eta))
        np.save(self.c.result_path+'test_GAN/GAN_process_T', np.array(process_T))
        
        out['F'] = f1   

if __name__ == "__main__":
    BEM_path  = '/home/sh/WCY/auto_propeller/resource4/0_data/start_prop.ini'
    Geom_path = '/home/sh/WCY/auto_propeller/resource4/0_data/start_mesh.npy'
    res_path  ='/home/sh/WCY/auto_propeller/resource4/utils/restrict.ini'
    result_path  = '/home/sh/WCY/auto_propeller/resource4/2_output/test_GAN/GAN_x.npy'
    pic_path  = '/home/sh/WCY/auto_propeller/resource4/2_output/test_GAN/'
    save_path = '/home/sh/WCY/auto_propeller/resource4/2_output/test_GAN/'


    PointsCloud = np.load(Geom_path)
    print(PointsCloud.shape)
    
    Radius_list, Chord_list, Pitch_list, SectionCoordinate_list = \
    build_BEMprop_from_PointsCloud(PointsCloud)

    s = Solver(BEM_path, s1 = SectionCoordinate_list, c1 = Chord_list, p1 = Pitch_list)
    r = Constraint(res_path)

    problem = Optimize_BestEta_ConsZ(s,r, save_path)
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
    