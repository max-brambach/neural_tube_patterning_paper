# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:39:10 2019

@author: Ariane
"""

#coefficients
#node V
V0 =0
c16 = 25.0 #suppress U by U
c17 =5.0 #suppress U
n16 = 4
n17 = 4.0
d6 = 0.1

#node U
U0 = 0
c14 = 50.0#fill by wnt #how fast
n14 = 2.5 #controls how broad
c15 = 0.14#0.1 #self activation
n15 = 1.0
d5 = 0.1




#rostrocaudal
c1 = 0.015
c2 = 8.00
c3 = 1000
c4 = 0.201
c5 = 0.5
c6 = 0.201
c7 = 0.005
c8 = 0.205
c9 = 2.00
c10 = 0.050
c11 = 0.051
c12 = 0.248
c13 = 0.152

n1 = 4.0
n2 = 4.0
n3 = 4.0
n4 = 1.0
n5 = 2.0
n6 = 1.0
n7 = 3.0
n8 = 1.0
n9 = 3.0
n10 = 1.0
n11 = 3.0
n12 = 1.0
n13 = 1.0

d1 = 0.169
d2 = 0.169
d3 = 0.170
d4 = 0.171


FB0=1
MB0 =0.001
HB0 =0.001
CT0 = 1
GSK30=1#1.9

#colours in order FB,MB,HB  blue,orange,green
colours=[[0.12156863, 0.46666667,0.70588235],[1.,0.49803922,0.05490196],[0.17254902,0.62745098,0.17254902]]