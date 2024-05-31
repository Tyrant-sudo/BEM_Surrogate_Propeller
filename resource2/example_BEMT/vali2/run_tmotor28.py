 # -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import pandas as pd
from math import pi
import sys
import time

sys.path.insert(0,"../../")

from BEMT_program.solver import Solver
import os,sys
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


s = Solver('tmotor28.ini')


with HiddenPrints():
    df, sections = s.run_sweep('rpm', 20, 1000.0,3200.0)

df_exp = pd.read_csv("tmotor28_data.csv", delimiter=';')

eta_list = []
ct_list  = []
cq_list  = []
cp_list  = []
rpm_list = []
D   = 0.7112
rho = 1.225

for i in range(len(df_exp['T(N)'])):
    
    n   = df_exp['RPM'][i]/60
    J   = 1/(n*D)
    
    CT = df_exp['T(N)'][i]/(rho*n**2*D**4)
    CQ = df_exp['Q(Nm)'][i]/(rho*n**2*D**5)
    CP = 2*pi*CQ
    eta = (CT/CP)*J

    ct_list.append(CT)
    cq_list.append(CQ)
    cp_list.append(CP)
    eta_list.append(eta)
    rpm_list.append(df_exp['RPM'][i])

cols = ['rpm','CT', 'CQ', 'CP', 'eta']
k_l  = [[rpm_list[_],ct_list[_],cq_list[_],cp_list[_],eta_list[_]] for _ in range(20)]

k = pd.DataFrame(k_l,index= range(20),columns=cols)

# for i,p in enumerate(rpm_list):

#     k.iloc[i] = rpm_list[i],ct_list[i],cq_list[i],cp_list[i],eta_list[i]

# print(k['rpm'])
# print(cp_list)
# exit()


ax = df.plot(x='rpm', y='eta', legend=None) 
k.plot(x='rpm',y='eta',style='C0o',ax=ax, legend=None)
pl.legend(('BEMT, $\eta$','Exp, $\eta$'),loc='center left')

pl.ylabel('$\eta$')
ax2 = ax.twinx()
pl.ylabel('$C_P, C_T$')

df.plot(x='rpm', y='CP', style='C1-',ax=ax2, legend=None) 
k.plot(x='rpm',y='CP',style='C1o',ax=ax2, legend=None)

df.plot(x='rpm', y='CT', style='C2-',ax=ax2, legend=None) 
k.plot(x='rpm',y='CT',style='C2o',ax=ax2, legend=None)


pl.legend(('BEMT, $C_P$','Exp, $C_P$',
    'BEMT, $C_T$','Exp, $C_T$'),loc='center right')

pl.show()



