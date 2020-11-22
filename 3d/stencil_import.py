# -*- coding: utf-8 -*-
from scipy.integrate import solve_ivp
import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from matplotlib import colors as mcolors
import matplotlib.animation as animation
import matplotlib.cm as cm


from mpl_toolkits.mplot3d import Axes3D
import csv


import paras_dorsoventral as dors
import paras_rostrocaudal as ros
import testround_difftest_set as r

stencpath = "cooltube_0.5"
wntsecrpath = "cooltube_0.5_WNT"
wntsecrpath2 = "wntbla2"
shhsecrpath = "cooltube_0.5_SHH"

Wnt0 = 2.0 #Wnt concentration at secretion (from dorso, rostro: 1.9)
Shh0 = 1.0 #Shh conc at secretion


# stencpath = "bvec"
# wntsecrpath = "bvec_WNT"
# wntsecrpath2 = "wntbla2"
# shhsecrpath = "bvec_SHH"

# stencpath = "smallsphere"
# wntsecrpath = "smallsphere_WNT"
# wntsecrpath2 = "wntbla2"
# shhsecrpath = "smallsphere_SHH"


with open(stencpath+'.txt') as inf:
    reader = csv.reader(inf, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
    readlist=list(zip(*reader))
    xval = readlist[0]
    yval = readlist[1]
    zval = readlist[2]
    
    #xmax = int(np.max(xval)*10)+1
    #ymax = int(np.max(yval)*10)+1
    #zmax = int(np.max(zval)*10)+1
    xmax = int(np.max(xval))+2#1
    ymax = int(np.max(yval))+2#1
    zmax = int(np.max(zval))+2#1
    #print(xmax,ymax,zmax)

tube = (np.transpose(np.asarray([xval,yval,zval]))).astype(int)

with open(wntsecrpath+'.txt') as wnf:
    readerw = csv.reader(wnf, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
    readlistw=list(zip(*readerw))
    if not readlistw:
        xvalw = []
        yvalw = []
        zvalw= []
        svalw = []
    
    else:
        xvalw = readlistw[0]
        yvalw = readlistw[1]
        zvalw= readlistw[2]
        svalw = readlistw[3]
    
wntsecr = (np.transpose(np.asarray([xvalw,yvalw,zvalw,svalw]))).astype(int)



with open(shhsecrpath+'.txt') as snf:
    readers = csv.reader(snf, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
    readlists=list(zip(*readers))
    if not readlists:
        xvals = []
        yvals = []
        zvals= []
        svals = []
    else:    
        xvals = readlists[0]
        yvals = readlists[1]
        zvals= readlists[2]
        svals = readlists[3]
        
shhsecr = (np.transpose(np.asarray([xvals,yvals,zvals,svals]))).astype(int)

stenc = r.stencil(xmax,ymax,zmax,"stenc",tube) 
Wstenc = r.secrStencil(stenc, "WntSecr", wntsecr,Wnt0)
Sstenc = r.secrStencil(stenc, "ShhSecr", shhsecr,Shh0)
#print(Wstenc.secretion_levels)
#print(Wstenc.grid_within_bounds)
# Amatr = r.Amatrix(stenc)
# seeds = np.sum(stenc.grid)

"""
fug = plt.figure()
axStenc = fug.add_subplot(1, 1, 1, projection='3d')
r.plotSecrStenc(Sstenc,axStenc)
#print(np.transpose(np.nonzero(Wstenc.grid)))
plt.show()
"""
#fug.savefig("test.png")
