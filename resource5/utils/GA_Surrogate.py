import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os
from scipy.signal import savgol_filter
from bisect import bisect_left

from pymoo.core.problem import Problem

from BEMT_program.solver import Solver
from models import Model_N0, Model_T0

from visualization import draw_block,draw_process, easy_draw

config = {
    "range":0.08,
    "lower_T": 0.9,

    'seed': 99,      # Your seed number, you can pick your lucky number. :)
    "pop_size": 10,
    "total": 500,
    
    "result": True,
    "result_path": "/home/sh/WCY/auto_propeller/resource5/2_output/optimization_pic",

}


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def f3(x, t):
    i = bisect_left(x, t)
    if x[i] - t > 0.5:
        i-=1
    return i

def build_chord(delta_x, x0, y0, arg_points = [1,3], inter_range =[-0.1, 0.2] , internum = 26, num = 101, k = 4, s = 2):

    from scipy.interpolate import splprep, splev
    tck, u = splprep([x0, y0], k=k, s=s)
    new_control_points = np.copy(tck[1])

    for i,j in enumerate(arg_points):
        po = inter_range[0] + (inter_range[1] - inter_range[0])* delta_x[i]
        new_control_points[1][j] += po
    new_tck = (tck[0], new_control_points, tck[2])

    x_new, y_new = splev(np.linspace(0, 1, internum), new_tck)

    return x_new, y_new

def build_pitch(delta_x, x0, y0, arg_points = [2,3,4], inter_range =[-3, 5] , internum = 26, num = 101, k = 4, s = 2):
    
    y0 = np.array(y0)*180/np.pi
    from scipy.interpolate import splprep, splev
    tck, u = splprep([x0, y0], k=k, s=s)
    new_control_points = np.copy(tck[1])

   
    for i,j in enumerate(arg_points):
        po = inter_range[0] + (inter_range[1] - inter_range[0])* delta_x[i]
        new_control_points[1][j] += po
    new_tck = (tck[0], new_control_points, tck[2])

    x_new, y_new = splev(np.linspace(0, 1, internum), new_tck)
    
    y_new = y_new * np.pi/180
    return x_new, y_new

def get_eta_noise_fromModel(chord_x, pitch_x, model_T, model_N):
    # chord_x: [B,26]
    # pitch_x: [B,26]

    input = torch.tensor(np.stack((chord_x, pitch_x),axis=-1),dtype=torch.float32)

    model_T.eval()
    with torch.no_grad():
        pred_T = model_T(input)

    pred_T = pred_T.detach().cpu().numpy()
    
    model_N.eval()
    with torch.no_grad():
        pred_N = model_N(input)
    pred_N = pred_N.detach().cpu().numpy()
    pred_N = np.mean(pred_N,axis=-1)
    pred_N = np.where(pred_N < 55, 75, pred_N)

    pred_N = np.expand_dims(pred_N,axis=-1)

    pred = np.concatenate((pred_T, pred_N), axis=-1)
    return pred

def get_pop_EtaNoise(x_chord, x_pitch, chord_x, chord_y, pitch_x, pitch_y):

    pop_size = x_chord.shape[0]
    
    chord_list = []
    pitch_list = []

    for i in range(pop_size):

        delta_x0 = x_chord[i]
        delta_x1 = x_pitch[i]

        _, chord = build_chord(delta_x0, chord_x, chord_y)
        _, pitch = build_pitch(delta_x1, pitch_x, pitch_y)

        chord_list.append(chord)
        pitch_list.append(pitch)
    
    chord_list = np.array(chord_list)
    pitch_list = np.array(pitch_list)

    pred = get_eta_noise_fromModel(chord_list, pitch_list, model_T, model_N)
    

    return chord_list, pitch_list, pred[:,0], pred[:,1], pred[:,2]

def get_eta_noise(pitch_x,c_Rx,s:Solver, c):
    eps = 0.01
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
        
        c1 = c_Rx[i] 
        p1 = pitch_x[i] * 180/np.pi

        s1       = Solver(BEM_path,c1=c1,p1=p1)
        
        with HiddenPrints():
            df, sections = s1.run_sweep('v_inf', 1, v_inf, v_inf)
        if df['eta'][0]>1:
            df['eta'][0] = 10
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
            a.append(abs(cl[po_cl_list[_]] - local_cl[_][1] ))
        g_cl_list.append(a)

        a = []
        for _ in range(len(po_pitch_list)):
            a.append(abs(pitch[po_pitch_list[_]] - local_pitch[_][1] ))
        g_pitch_list.append(a)

        a = []
        for _ in range(len(po_cR_list)):
            a.append(abs(chord[po_cR_list[_]] - local_cR[_][1]))
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
    g_cl_list    = np.array(g_cl_list)
    g_pitch_list = np.array(g_pitch_list)
    g_cR_list    = np.array(g_cR_list)

    return eta,T,g_cl_list,g_pitch_list,g_cR_list


def get_cons(local):
    if local[0]==0 and local[1] == 0:
        return 0
    else:
        return len(local)//2
    
class Optimize_BestEtaN_test(Problem):

    def __init__(self):

        self.c = config
        self.l = 0

        self.n_var  = 5
        self.dim1   = 2
        self.dim2   = 3

        self.n_obj  = 2
        self.n_cons = 1

        # self.xl = np.array([0.1,0.1,0.2,0.2,0.2])
        # self.xu = np.array([0.6,0.6,0.5,0.5,0.5])

        self.xl = np.array([0,0,0,0,0])
        self.xu = np.array([1,1,1,1,1])
        
        self.save_path = config["result_path"]
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr = self.n_cons , xl = self.xl,xu= self.xu)
    
    def get_base(self, path_chord, path_pitch, model_T, model_N):
        pop_size = 10
        x = np.zeros((pop_size, 1))
        self.process_FM  = []
        self.process_T   = []
        self.process_N   = []

        df_chord = pd.read_csv(path_chord)
        df_pitch = pd.read_csv(path_pitch)

        self.chord_x    = df_chord["r/R"].values
        self.chord_list = df_chord["c/R"].values
        self.pitch_x    = df_pitch["r/R"].values
        self.pitch_list = df_pitch['twist (deg)'].values * np.pi / 180
        
        self.r_R,self.base_c = build_chord([0.3,0.3], self.chord_x, self.chord_list)
        _,self.base_p = build_pitch([0.4,0.4,0.4], self.pitch_x, self.pitch_list)
        base_c = np.expand_dims(self.base_c,0).repeat(pop_size, 0)
        base_p = np.expand_dims(self.base_p,0).repeat(pop_size, 0)

        pred = get_eta_noise_fromModel(base_c, base_p, model_T, model_N)
        
        self.base_T, self.base_FM, self.base_N = pred[0,0], pred[0,1], pred[0,2]
        
        self.min_T   = self.c["lower_T"] * self.base_T
        self.min_FM  = self.c["lower_T"] * self.base_FM
        self.min_N   = self.base_N / self.c["lower_T"] 
        
        draw_block(self.r_R,self.base_c,self.base_p*180/np.pi,self.save_path,'base',self.base_FM,self.base_T, self.base_N)

    def _evaluate(self, x, out, *args, **kwargs):
        process_FM  = self.process_FM
        process_T   = self.process_T
        process_N   = self.process_N

        self.l += 1
        
        print('current loop:',self.l)
        
        x_chord = x[:, :self.dim1]
        x_pitch = x[:, self.dim1:]

        chord_list, pitch_list, T, FM, N = get_pop_EtaNoise(x_chord, x_pitch, self.chord_x, self.chord_list, self.pitch_x, self.pitch_list)
        
        process_T.append(T)
        process_FM.append(FM)
        process_N.append(N)
        arg_best  = np.argmin(N)

        draw_block(self.r_R,chord_list[arg_best],pitch_list[arg_best]*180/np.pi,self.save_path,'best',FM[arg_best],T[arg_best],N[arg_best],False)
        draw_process(process_FM,self.base_FM,process_T,self.base_T,self.process_N, self.base_N, self.save_path)

        f1 = -100* FM
        f2 = N

        g1 = self.min_T - T

        out["F"] = np.column_stack([f2,f1])
        out['G'] = g1

        np.save(self.save_path + "/process",x)
        np.save(self.save_path +'/process_FM', np.array(process_FM))
        np.save(self.save_path +'/process_T', np.array(process_T))
        np.save(self.save_path +'/process_N', np.array(process_N))


if __name__ == "__main__":
    import pandas as pd

    path_modelT = "/home/sh/WCY/auto_propeller/resource5/1_model/Thrust_model.ckpt"
    path_modelN = "/home/sh/WCY/auto_propeller/resource5/1_model/Noise_model.ckpt"

    path_chord = "/home/sh/WCY/auto_propeller/resource5/0_database/DJI9443_chorddist.csv"
    path_pitch = "/home/sh/WCY/auto_propeller/resource5/0_database/DJI9443_pitchdist.csv"

    model_T = Model_T0()
    state_dict = torch.load(path_modelT)
    model_T.load_state_dict(state_dict, strict= False)

    model_N = Model_N0()
    state_dict = torch.load(path_modelN)
    model_N.load_state_dict(state_dict, strict= False)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import BinaryRandomSampling,FloatRandomSampling
    from pymoo.optimize import minimize
    problem = Optimize_BestEtaN_test()
    problem.get_base(path_chord, path_pitch, model_T, model_N)
 
    algorithm = NSGA2(pop_size=int(config["pop_size"]),
                  sampling=FloatRandomSampling(),
                #   crossover=TwoPointCrossover(),
                #   mutation=BitflipMutation(),
                  eliminate_duplicates=True)

    res = minimize(problem,
                algorithm,
                ('n_gen', int(config["total"])),
                seed=int(config["seed"]),
                #    save_history = True,
                #    verbose=True)
    )
    
    