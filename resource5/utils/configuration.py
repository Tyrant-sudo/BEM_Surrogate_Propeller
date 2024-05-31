from configparser import SafeConfigParser
import sys
import os

def get_cons(local):
    if local[0]==0 and local[1] == 0:
        return 0
    else:
        return len(local)//2

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Constraint:

    def __init__(self,config_path):
        cfg = SafeConfigParser()
        cfg.read(config_path,encoding='utf-8')
        
        self.range = cfg.getfloat('Bspline','range')
        self.lb    = cfg.getfloat('Bspline','lower_b')
        self.local_cl = [float(_) for _ in cfg.get('Bspline','local_cl').split()]
        self.local_pitch = [float(_) for _ in cfg.get('Bspline','local_pitch').split()]
        self.local_cR = [float(_) for _ in cfg.get('Bspline','local_cR').split()]
        self.smooth  = [float(_) for _ in cfg.get('Bspline','smooth').split()]

        self.StartMesh_Path      = cfg.get('FFD','StartMesh_Path')
        self.BasePoints_Path     = cfg.get('FFD','BasePoints_Path')
        self.BasePara_Path       = cfg.get('FFD','BasePara_Path')
        self.BEM_path            = cfg.get('FFD','BEM_path')
        self.Dim_ControlPoints   = tuple([int(_) for _ in cfg.get('FFD','Dim_ControlPoints').split()])
        self.FFD_range           = cfg.getfloat('FFD', 'range')

        self.gan_model      = cfg.get('GAN', 'model_path')
        self.gan_basepoints = cfg.get('GAN', 'basepoint_path')
        self.gan_basepara   = cfg.get('GAN', 'basepara_path')
        self.gan_range      = cfg.getfloat('GAN', 'range')
        self.gan_dim        = cfg.getint('GAN', 'noize_dim')

        self.pop_size = cfg.getfloat('GA','pop_size')
        self.seed     = cfg.getfloat('GA','seed')
        self.total    = cfg.getfloat('GA','total')
        
        self.result   = cfg.getboolean('save','result')
        self.result_path = cfg.get('save','result_path')
        

    def get_Bspline_cons(self,len_sec):

        self.Bspline_n_var = len_sec + len_sec
        self.Bspline_n_obj = 1 #效率最大(悬停或者有前进速度)

        self.Bspline_n_ieq_constr = get_cons(self.local_cR)+get_cons(self.local_pitch)
        # self.Bspline_n_ieq_constr =2 + get_cons(self.local_cl)+get_cons(self.local_cR)+get_cons(self.local_pitch)
        #三个截面限制+最小值限制（两个）

    def get_FFD_cons(self):

        self.FFD_n_var = 2 * self.Dim_ControlPoints[0] * self.Dim_ControlPoints[1] * self.Dim_ControlPoints[2]
        self.FFD_n_obj = 1
        self.FFD_n_ieq_constr = 0

    def get_GAN_cons(self):

        self.GAN_n_var = self.gan_dim
        self.GAN_n_obj = 1
        self.GAN_n_ieq_constr = 0
