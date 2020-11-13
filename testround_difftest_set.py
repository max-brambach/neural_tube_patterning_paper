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

import paras_dorsoventral as dors
import paras_rostrocaudal as ros


xlen =4
ylen =20
zlen =50
seeds =3
spheresize=30
class stencil:
    def __init__(self,xdim,ydim,zdim, Name, coordlist):
        self.grid = np.zeros((xdim,ydim,zdim))  
        self.name=Name
        self.xlen=xdim
        self.ylen=ydim
        self.zlen=zdim
        
        for coord in coordlist:
            
            self.grid[coord[0]][coord[1]][coord[2]] = 1

class secrStencil:
    def __init__(self,stencil, Name, coordlist,Base):
        self.grid = np.zeros_like(stencil.grid)
        self.grid_within_bounds =  np.zeros_like(stencil.grid)
        self.name=Name
        self.xlen=stencil.xlen
        self.ylen=stencil.ylen
        self.zlen=stencil.zlen
        self.stencil=stencil
        self.base = Base
        
        for coord in coordlist:            
            self.grid[coord[0]][coord[1]][coord[2]] = coord[3]
            if stencil.grid[coord[0]][coord[1]][coord[2]] ==1:
                self.grid_within_bounds[coord[0]][coord[1]][coord[2]] = coord[3]
        
        
        self.secretion_levels, self.secretion_coords, = grid_to_vector(self.grid_within_bounds, justgrid = True)
        self.secretion_levels = self.secretion_levels/255.0

def plotstenc(stenc,ax,r=0.47,g=0.0,b=1.0,color='red',alpha=0.8):    
    tubeindices = np.where(stenc.grid)    
    
    #restindices =  np.where((np.ones_like(stenc.grid)-stenc.grid))
    ax.scatter(tubeindices[0],tubeindices[1],tubeindices[2],marker = 'o',c=color,linewidth=0,vmin=0,vmax=1,depthshade=False,s=spheresize,alpha=alpha)      
    #if np.any(restindices) ==True:
        #ax.scatter(restindices[0],restindices[1],restindices[2],marker = 'o',c='blue',linewidth=0,vmin=0,vmax=1,depthshade=False,s=spheresize )


def plotSecrStenc(secrStenc,ax):
     gridvector, coordvector = grid_to_vector(secrStenc)
     plotstenc(secrStenc.stencil,ax,color='blue',alpha=0.01)
     for i in range(len(gridvector)):
         x,y,z=coordvector[i][0],coordvector[i][1],coordvector[i][2]
         colour = np.asarray([gridvector[i],0,0])
         ax.scatter(x,y,z,marker = 'o',c=colour/255.0,linewidth=0,depthshade=False,s=spheresize,alpha=0.7)

def tubecoords(x,y,z,borders=True,bordersize=3):
    grid= np.ones((x,y,z))
    
    grid[0,:,: ]=0
    if borders == True:     
        for i in range(bordersize):
            print(i)
            grid[:,:,i] =0
            grid[:,i,:] =0
            
            grid[:,:,-i-1] =0
            grid[:,-i-1,:] =0
           
    
    
    #for i in range(x):
        
        
        #for j in np.arange(borders,int(y/2)+1):
            #for k in np.arange(borders,int(z/2)+1):
                
                #if j**(1.6) +borders >= k:
                    #print(j,k)
                    #grid[i][int(y/2)+1-(j+borders)][k]=0
                    #grid[i][int(y/2)+1-(j+borders)][-(k-1)]=0
                    #grid[i][-(int(y/2)+1-(j+borders))-1][k]=0
                    #grid[i][-(int(y/2)+1-(j+borders))-1][-(k-1)]=0
        
        
        #for j in np.arange(borders,int(y/2)+1):
            #for k in np.arange(borders,int(z/2)+1):
                
                #if j**(3.5) +borders +15 <= k:
                    #print(j,k)
                    #grid[i][int(y/2)+1-(j+borders)][k]=0
                    #grid[i][int(y/2)+1-(j+borders)][-(k-1)]=0
                    #grid[i][-(int(y/2)+1-(j+borders))-1][k]=0
                    #grid[i][-(int(y/2)+1-(j+borders))-1][-(k-1)]=0
        
        # grid[i][int(y/2)][int(z/2)]=0
        # grid[i][int(y/2)+1][int(z/2)]=0
        # grid[i][int(y/2)][int(z/2)+1]=0
        # grid[i][int(y/2)-1][int(z/2)]=0
        # grid[i][int(y/2)][int(z/2)-1]=0
    
    return np.transpose(np.where(grid))

def grid_to_vector(gridname, justgrid =False):
    if justgrid ==False:
        grid = gridname.grid 
    else:
        grid = gridname

    coordvector = np.transpose(np.nonzero(grid))
    gridvector = grid[grid!=0]
    #print(coordvector,len(gridvector))
    for i in range(len(coordvector)):
        c = coordvector[i]
        if grid[c[0]][c[1]][c[2]] != gridvector[i]:
            print("False!")
    #print("grid_to_vector:",gridvector)
    return gridvector,coordvector        

def vector_to_grid(u,gridname,coords):
    newgrid = np.zeros_like(gridname.grid)
    #print(len(gridname.grid),len(gridname.grid[0]),len(gridname.grid[0][0]))
    c1,c2,c3 = np.transpose(coords)
    newgrid[c1,c2,c3] = u
    return newgrid
    
def Amatrix(stencil):
    xdim = stencil.xlen
    ydim = stencil.ylen
    zdim = stencil.zlen
    
    u,coord = grid_to_vector(stencil)
    coorddict = {}
    for i in range(len(coord)):
        c = coord[i]
        coorddict[str(c)] = i


    grid = stencil.grid
    dim=len(u)
    print("dim:",dim)
    print("xdimydimzdim",xdim*ydim*zdim)
    A = []
    for i in range(dim):   
        if i % 10000 ==0:
            print(i)
        
        x = coord[i][0]
        y = coord[i][1]
        z = coord[i][2]
        
        frontcoords = [x,y+1,z]
        backcoords = [x,y-1,z]
        leftcoords = [x-1,y,z]
        rightcoords = [x+1,y,z]
        upcoords = [x,y,z+1]
        downcoords = [x,y,z-1]
        
        #nblist = [0,0,0,0,0,0]#[up,down,left,right,front,back] - zero if next to inner part, 1 if next to boundary
        nbcoords = np.asarray([upcoords,downcoords,leftcoords,rightcoords,frontcoords,backcoords])
        
        nb = 0
        for c in range(len(nbcoords)):
            coordStr = str(nbcoords[c])
            j = coorddict.get(coordStr)
            
            if j != None:
                # nblist[c] = 0     
                A.append([i,j,1])                
            else:
                nb += 1
                #nblist[c] = 1
        
        A.append([i,i, -(6-nb)])

    return np.asarray(A)
    
def Amatrix_bs(stencil, secrStencil):
    #stencil
    xdim = stencil.xlen
    ydim = stencil.ylen
    zdim = stencil.zlen
    
    u,coord = grid_to_vector(stencil)
    coorddict = {}
    for i in range(len(coord)):
        c = coord[i]
        coorddict[str(c)] = i


    grid = stencil.grid
    dim=len(u)
    print("dim:",dim)
    print("xdimydimzdim",xdim*ydim*zdim)
    
    #secretion stencil
    usecr,coordsecr = grid_to_vector(secrStencil)
    secrcoorddict = {}
    for i in range(len(coordsecr)):
        c = coordsecr[i]
        secrcoorddict[str(c)] = i

    
    
    
    A = []
    b=np.zeros(len(u))
    for i in range(dim):   
        if i % 10000 ==0:
            print(i)
        
        x = coord[i][0]
        y = coord[i][1]
        z = coord[i][2]
        
        frontcoords = [x,y+1,z]
        backcoords = [x,y-1,z]
        leftcoords = [x-1,y,z]
        rightcoords = [x+1,y,z]
        upcoords = [x,y,z+1]
        downcoords = [x,y,z-1]
        
        #nblist = [0,0,0,0,0,0]#[up,down,left,right,front,back] - zero if next to inner part, 1 if next to boundary
        nbcoords = np.asarray([upcoords,downcoords,leftcoords,rightcoords,frontcoords,backcoords])
        
        nb = 0
        for c in range(len(nbcoords)):
            coordStr = str(nbcoords[c])
            j = coorddict.get(coordStr)
            
            if j != None:
                # nblist[c] = 0     
                A.append([i,j,1])                
            else:
                k = secrcoorddict.get(coordStr)
                if k == None:
                    nb += 1
                else:
                    print("coords",coordStr,"k",usecr[k])
                    b[i] += (usecr[k]/255.0) *secrStencil.base 
                    
                #nblist[c] = 1
        
        A.append([i,i, -(6-nb)])

    
    return np.asarray(A),b
