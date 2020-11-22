# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:40:26 2019

@author: Ariane
"""

#dorsoventral coefficients



alpha = 3.0
beta = 5.0
gamma = 5.0

h1 = 6.0
h2 = 2.0
h3 = 5.0
h4 = 1.0
h5 = 1.0

k1= 1.0
k2 = 1.0
k3 = 1.0
OcritP = 1.0
NcritP = 1.0
OcritN = 1.0
NcritO = 1.0
PcritN = 1.0
para_n = 1.0
para_m = 1.0



"""Debatable Parameters!"""
"""TRANSFORM INTO CELL!"
D_Shh=114.6 mum**2/s #Diffusion Coefficients
D_Wnt=123.6  mum**2/s
"""




D_G = 0.0 #madeup!!!


D_O = 0.0 #madeup!!!
D_P = 0.0 #madeup!!!
D_N = 0.0 #madeup!!!





#not there
O0 = 0.0    #madeup!!!
N0 = 0.0    #madeup!!!

#everywhere:
Gli0 = 3.0 #Gli expression without Shh/Wnt
P0 = 0.0   #3.0  #madeup!!!

delta=5.0
k4 =0.15
h6=1.0
WcritG=1.0

"""...."""
"""
N_cells = 50
N_genes = 6 #Pax6,Olig2,Nkx2.2,Gli,Shh,Wnt
timestep = 0.1
maxtime = 70.0
cell0 = [3,0,0,Gli0,Shh0,0]
cell1 = [3,0,0,Gli0,0,0]
cellend = [3,0,0,Gli0,0,Wnt0]
"""
#colours in order P,O,N,G = forestgreen,goldenrod,firebrick
#colours =[[0.13,0.55,0.13],[0.85,0.65,0.13],[0.70,0.13,0.13]]
#colours in order P,O,N = blue, red, green
colours = [[0.15686275, 0.12156863, 0.48627451], [0.7254902,  0.12156863, 0.14901961], [0.29803922, 0.59607843, 0.25882353]]