 # -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import pandas as pd
from math import pi
import sys

sys.path.insert(0,"../../")
from BEMT_program.solver import Solver
import os,sys
import numpy as np
# import form_all

from scipy import interpolate
import matplotlib.pyplot as plt
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


s = Solver('prop.ini')

with HiddenPrints():
    df, sections = s.run_sweep('v_inf', 20, 0.1, 20)


print(df['J'],df['eta'],df['CT'],df['CP'])


df_exp = pd.read_csv("prop.csv", delimiter=';')

# print(df_exp['eta'])
# exit()
ax = df.plot(x='J', y='eta', legend=None) 
df_exp.plot(x='J',y='eta',style='C0o',ax=ax, legend=None)
pl.legend(('BEMT, $\eta$','Exp, $\eta$'),loc='center left')

pl.ylabel('$\eta$')
ax2 = ax.twinx()
pl.ylabel('$C_P, C_T$')

df.plot(x='J', y='CP', style='C1-',ax=ax2, legend=None) 
df_exp.plot(x='J',y='CP',style='C1o',ax=ax2, legend=None)

df.plot(x='J', y='CT', style='C2-',ax=ax2, legend=None) 
df_exp.plot(x='J',y='CT',style='C2o',ax=ax2, legend=None)

pl.legend(('BEMT, $C_P$','Exp, $C_P$',
    'BEMT, $C_T$','Exp, $C_T$'),loc='center right')

pl.savefig("test.png")

J1 = df_exp['J'].tolist()
CP1 = df_exp['CP'].tolist()
CT1 = df_exp['CT'].tolist()
eta1 = df_exp['eta'].tolist()

J = df['J'].tolist()
CP = df['CP'].tolist()
CT = df['CT'].tolist()
eta = df['eta'].tolist()

# a = J
# print(a)
print(np.array(J).shape, np.array(CP).shape)
tck  = interpolate.splrep(J,CP)

CP  = interpolate.splev(J1,tck)
tck  = interpolate.splrep(J,CT)
CT  = interpolate.splev(J1,tck)
tck  = interpolate.splrep(J,eta)
eta = interpolate.splev(J1,tck)

l = len(J1)
p = []
t = []
e = []

for i in range(l):
    p.append(np.abs(CP1[i]-CP[i]))
    t.append(np.abs(CT1[i]-CT[i]))
    e.append(np.abs(eta1[i]-eta[i]))

p = np.mean(p)/l/np.mean(CP)
e = np.mean(e)/l/np.mean(eta)
t = np.mean(t)/l/np.mean(CT)
print(p,t,e)
print(np.max(p)/np.mean(CP),np.max(t)/np.mean(CT),np.max(e)/np.mean(eta))