# -*- coding: utf-8 -*-
from scipy.integrate import solve_ivp
import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from matplotlib import colors as mcolors

from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D

import paras_dorsoventral as dors
import paras_rostrocaudal as ros
import testround_difftest_set as r #for sparse matrix stuff
#import testround_difftest_backup as r
import stencil_import as tubemodel
import slicemovie_iii as slicemov
import colourmix as colmix
import ast
import os

matplotlib.rcParams.update({'font.size': 22})

dataPath = tubemodel.stencpath +'_1'
spheresize=60
figsize = (19,19)
edgecolor='black'
edgewidth=0.2
upscale = 2
#plotting colourmax
# rosmax = 5.0#tubemodel.Wnt0#5#0.0
# dorsmax = 5.0#tubemodel.Wnt0#5#0.0
# wntmax = 5.0#tubemodel.Wnt0#5#0.0
# shhmax = 5.0#tubemodel.Wnt0#5#0.0
# unknownbase=5.0#tubemodel.Wnt0#5#0.0
rosmax = np.max([tubemodel.Wnt0,tubemodel.Shh0])#5#0.0
dorsmax = np.max([tubemodel.Wnt0,tubemodel.Shh0])#tubemodel.Wnt0#5#0.0
wntmax = np.max([tubemodel.Wnt0,tubemodel.Shh0])#tubemodel.Wnt0#5#0.0
shhmax = np.max([tubemodel.Wnt0,tubemodel.Shh0])#tubemodel.Wnt0#5#0.0
unknownbase= np.max([tubemodel.Wnt0,tubemodel.Shh0])#tubemodel.Wnt0#5#0.0

plotmode='Voxel'#Voxel for voxel
scale = 0.5
depthshade=True
seeThrough=False
WntColour =(72/255, 93/255, 127/255)

ShhColour =(97/255, 130/255, 31/255)

def plotsaved(plotFrom, time,Wnt=True,Shh=True,Rostro=True,Dorso=True,Mix=True,savePic=False):
    wntDir = plotFrom + '/Wnt/'
    shhDir = plotFrom + '/Shh/'
    rostroDir = plotFrom + '/rostro/'
    dorsoDir  = plotFrom + '/dorso/'
    mixDir = plotFrom + '/Mix/'
    baseLevels = np.load(plotFrom + '/BaseLevels.npy')
    allDir = plotFrom + '/allPictures/'
    if os.path.isdir(allDir) == False:
        os.mkdir(allDir)
    
    if os.path.isdir(mixDir) == False:
        os.mkdir(mixDir)
    
    if os.path.isdir(mixDir + '/pictures/') == False:
        os.mkdir(mixDir + '/pictures/')
    Wnt0, Shh0,FB0,MB0,HB0,P0,O0,N0 = baseLevels
    
    
    
    
    if Wnt == True:
        wntFile = wntDir +'T%1.1f' %time + '_Wnt'
        wntArray =np.load(wntFile +'.npy')       
        fig = plt.figure(figsize=figsize)
        axWnt = fig.add_subplot(1,1,1,projection='3d')    
        axWnt.set_title("WNT t = %1.1f s" % time)
        
        #print(np.max(wntArray))
        #plotarray(wntArray,axWnt,Wnt0)
        """
        if plotmode == 'Voxel':
            plotarrayVox(wntArray,axWnt,wntmax,red=WntColour[0],green=WntColour[1],blue=WntColour[2])
        else:
            plotarray(wntArray,axWnt,wntmax,red=WntColour[0],green=WntColour[1],blue=WntColour[2])
        """
        plotarray(wntArray,axWnt,wntmax,red=WntColour[0],green=WntColour[1],blue=WntColour[2],seeThrough=True)
        if savePic == True:
            #print(wntDir +'pictures/'+'T%1.1f_Wnt'  %time +'.png')
            """
            if plotmode == 'Voxel':
                plt.savefig(wntDir +'pictures/'+'T%1.1f_Wnt_Vox'  %time +'.png')
            else:
                plt.savefig(wntDir +'pictures/'+'T%1.1f_Wnt'  %time +'.png')
            """
            plt.savefig(wntDir +'pictures/'+'T%1.1f_Wnt'  %time +'.png')
            plt.savefig(allDir+'T%1.1f_Wnt'  %time +'.png')
            plt.close()    
          
    if Shh == True:
        shhFile = shhDir +'T%1.1f' %time + '_Shh'
        shhArray =np.load(shhFile +'.npy')       
        fig = plt.figure(figsize=figsize)
        axShh = fig.add_subplot(1,1,1,projection='3d')    
        axShh.set_title("SHH t = %1.1f s" % time)
        
        #print(np.max(shhArray))
        
        #plotarray(shhArray,axShh,Shh0)
        """
        if plotmode == 'Voxel':
            plotarrayVox(shhArray,axShh,shhmax,red=ShhColour[0],green=ShhColour[1],blue=ShhColour[2])
        else:
            plotarray(shhArray,axShh,shhmax,red=ShhColour[0],green=ShhColour[1],blue=ShhColour[2])
        """
        plotarray(shhArray,axShh,shhmax,red=ShhColour[0],green=ShhColour[1],blue=ShhColour[2],seeThrough=True)
        if savePic == True:
            """
            if plotmode == 'Voxel':
                plt.savefig(shhDir +'pictures/'+'T%1.1f' %time + '_Shh_Vox.png')
            else:
                plt.savefig(shhDir +'pictures/'+'T%1.1f' %time + '_Shh.png')
            """
            plt.savefig(shhDir +'pictures/'+'T%1.1f' %time + '_Shh.png')
            plt.savefig(allDir +'T%1.1f' %time + '_Shh.png')
            plt.close()    
            
    if Rostro == True:
        fbFile = rostroDir + 'T%1.1f' %time + '_FB'
        mbFile = rostroDir + 'T%1.1f' %time + '_MB'
        hbFile = rostroDir + 'T%1.1f' %time + '_HB'
        rcFile = rostroDir + 'T%1.1f' %time + '_rComp.npy'
        
        fbArray =np.load(fbFile +'.npy')
        mbArray =np.load(mbFile +'.npy')
        hbArray =np.load(hbFile +'.npy')
        
        
        if os.path.isfile(rcFile):
            rComp = np.load(rcFile)
        else:
            rComp = compare([fbArray,mbArray,hbArray])
            np.save(rcFile,rComp)
        #print(np.max(rComp))
        
        
        fig = plt.figure(figsize=figsize)
        axRos = fig.add_subplot(1,1,1,projection='3d')
        axRos.set_title("Rostrocaudal network (Brambach)  t = %1.1f s" % time)
        for i in range(len(rComp)):
            colours = ros.colours[i]
            if plotmode == 'Voxel':
                plotarrayVox(rComp[i],axRos,rosmax,red=colours[0],green=colours[1],blue=colours[2])
            else:
                plotarray(rComp[i],axRos,rosmax,red=colours[0],green=colours[1],blue=colours[2])
         
        proxy0 = plt.Rectangle((0, 0), 1, 1, fc=ros.colours[0])
        proxy1 = plt.Rectangle((0, 0), 1, 1, fc=ros.colours[1])
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=ros.colours[2])

        axRos.legend([proxy0, proxy1, proxy2],['FB', 'MB', 'HB'],bbox_to_anchor=(0.98, 1), loc=1, borderaxespad=0.1)  
            
        if savePic == True:
            if plotmode == 'Voxel':
                plt.savefig(rostroDir +'pictures/'+'T%1.1f' %time + '_Ros_Vox.png')
                plt.savefig(allDir +'T%1.1f' %time + '_Ros_Vox.png')
            else:
                plt.savefig(rostroDir +'pictures/'+'T%1.1f' %time + '_Ros.png')
                plt.savefig(allDir +'T%1.1f' %time + '_Ros.png')
            plt.close()    
           
    if Dorso == True:
        pFile = dorsoDir + 'T%1.1f' %time + '_P'
        oFile = dorsoDir + 'T%1.1f' %time + '_O'
        nFile = dorsoDir + 'T%1.1f' %time + '_N'
        dcFile = dorsoDir + 'T%1.1f' %time + '_dComp.npy'
        
        
        pArray =np.load(pFile +'.npy')
        oArray =np.load(oFile +'.npy')
        nArray =np.load(nFile +'.npy')
        
        
        
        if os.path.isfile(dcFile):
            dComp = np.load(dcFile)
        else:
            dComp = compare([pArray,oArray,nArray])
            np.save(dcFile,dComp)
            
        #print(np.max(dComp))
        
        
        fig = plt.figure(figsize=figsize)
        axDors = fig.add_subplot(2,1,1,projection='3d')
        axDorsDownside = fig.add_subplot(2,1,2,projection='3d')
        axDors.set_title("Dorsoventral network (Balaskas)  t = %1.1f s" % time)
        for i in range(len(dComp)):
            colours = dors.colours[i]
            if plotmode == 'Voxel':
                plotarrayVox(dComp[i],axDors,dorsmax,red=colours[0],green=colours[1],blue=colours[2])
                plotarrayVox(np.flip(dComp[i],2),axDorsDownside,dorsmax,red=colours[0],green=colours[1],blue=colours[2])
            else:
                plotarray(dComp[i],axDors,dorsmax,red=colours[0],green=colours[1],blue=colours[2])  
                plotarray(np.flip(dComp[i],2),axDorsDownside,dorsmax,red=colours[0],green=colours[1],blue=colours[2]) 
        

        proxy0 = plt.Rectangle((0, 0), 1, 1, fc=dors.colours[0])
        proxy1 = plt.Rectangle((0, 0), 1, 1, fc=dors.colours[1])
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=dors.colours[2])

        axDors.legend([proxy0, proxy1, proxy2],['P', 'O', 'N'],bbox_to_anchor=(1.10, 1), loc=2, borderaxespad=0.5)        
                
        if savePic == True:
            if plotmode == 'Voxel':
                plt.savefig(dorsoDir +'pictures/'+'T%1.1f' %time + '_Dors_Vox.png')
                plt.savefig(allDir +'T%1.1f' %time + '_Dors_Vox.png')
            
            else:
                plt.savefig(dorsoDir +'pictures/'+'T%1.1f' %time + '_Dors.png')
                plt.savefig(allDir +'T%1.1f' %time + '_Dors.png')
            plt.close()
            
    if Mix == True:
        """Plot gene combinations with a different colour for each combination."""
        
        pFile = dorsoDir + 'T%1.1f' %time + '_P'
        oFile = dorsoDir + 'T%1.1f' %time + '_O'
        nFile = dorsoDir + 'T%1.1f' %time + '_N'
        dcFile = dorsoDir + 'T%1.1f' %time + '_dComp.npy'
        
        
        pArray =np.load(pFile +'.npy')
        oArray =np.load(oFile +'.npy')
        nArray =np.load(nFile +'.npy')
        
        
        if os.path.isfile(dcFile):
            dComp = np.load(dcFile)
        else:
            dComp = compare([pArray,oArray,nArray])
            np.save(dcFile,dComp)
            
        fbFile = rostroDir + 'T%1.1f' %time + '_FB'
        mbFile = rostroDir + 'T%1.1f' %time + '_MB'
        hbFile = rostroDir + 'T%1.1f' %time + '_HB'
        rcFile = rostroDir + 'T%1.1f' %time + '_rComp.npy'
        
        fbArray =np.load(fbFile +'.npy')
        mbArray =np.load(mbFile +'.npy')
        hbArray =np.load(hbFile +'.npy')
        
        
        if os.path.isfile(rcFile):
            rComp = np.load(rcFile)
        else:
            rComp = compare([fbArray,mbArray,hbArray])
            np.save(rcFile,rComp)
        
        dimX = len(rComp[0])
        dimY = len(rComp[0][0])
        dimZ = len(rComp[0][0][0])
        
        mixArray = np.zeros((len(colmix.colours),dimX,dimY,dimZ))
        
        i=0
        for pon in dComp:
            for fbmbhb in rComp:
                an = np.transpose(np.nonzero(pon))
                bn = np.transpose(np.nonzero(fbmbhb))
                anl = an.tolist()
                bnl = bn.tolist()

                incommon = set(str(x) for x in anl) & set(str(y) for y in bnl)
                incommon = np.asarray([ast.literal_eval(i) for i in incommon])
                
                for coord in incommon:
                    #print(coord)
                    mixArray[i][coord[0]][coord[1]][coord[2]] = 1
                
                i+=1
            
        
        fig = plt.figure(figsize=figsize)
        fug = plt.figure(figsize=figsize)
        axMix = fig.add_subplot(1,1,1,projection='3d')
        axMixDownside = fug.add_subplot(1,1,1,projection='3d')
        
        
        for i in range(len(mixArray)):
            colours = colmix.colours[i]
            if plotmode == 'Voxel':
                plotarrayVox(mixArray[i],axMix,1,red=colours[0],green=colours[1],blue=colours[2])
                plotarrayVox(np.flip(mixArray[i],2),axMixDownside,1,red=colours[0],green=colours[1],blue=colours[2])
            else:
                plotarray(mixArray[i],axMix,1,red=colours[0],green=colours[1],blue=colours[2])  
                plotarray(np.flip(mixArray[i],2),axMixDownside,1,red=colours[0],green=colours[1],blue=colours[2]) 
        
        proxy0 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[0])
        proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[1])
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[2])
        proxy3 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[3])
        proxy4 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[4])
        proxy5 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[5])
        proxy6 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[6])
        proxy7 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[7])
        proxy8 = plt.Rectangle((0, 0), 1, 1, fc=colmix.colours[8])
        axMixDownside.legend([proxy0, proxy3, proxy6, proxy1, proxy4, proxy7, proxy2,proxy5,proxy8],['FB-P', 'FB-O', 'FB-N', 'MB-P', 'MB-O', 'MB-N', 'HB-P', 'HB-O','HB-N'],bbox_to_anchor=(1.12, 1), loc=2, borderaxespad=0.5)
        
        
        axMix.set_title("                                                                                                                                       t = %1.1f s" % time, fontsize=16,verticalalignment='bottom')
        axMixDownside.set_title("                                                                                                                                       t = %1.1f s" % time, fontsize=16,verticalalignment='bottom')
        #fig.subplots_adjust(top=1, bottom=0, left=-0.1, right=1)
        #ax.dist =3
        #axMix.dist = 3
        if savePic == True:
            if plotmode == 'Voxel':
                fig.savefig(mixDir +'pictures/'+'T%1.1f' %time + '_Mix_Vox.png',dpi=500)
                fig.savefig(allDir +'T%1.1f' %time + '_Mix_Vox.png',dpi=500)
                fug.savefig(mixDir +'pictures/'+'dT%1.1f' %time + '_Mix_Vox.png',dpi=500)
                fug.savefig(allDir +'dT%1.1f' %time + '_Mix_Vox.png',dpi=500)
            
            else:
                fig.savefig(mixDir +'pictures/'+'T%1.1f' %time + '_Mix.png',dpi=500)
                fig.savefig(allDir +'T%1.1f' %time + '_Mix.png',dpi=500)
                fug.savefig(mixDir +'pictures/'+'dT%1.1f' %time + '_Mix.png',dpi=500)
                fug.savefig(allDir +'dT%1.1f' %time + '_Mix.png',dpi=500)
            plt.close()

def axLabel(ax):
    ax.view_init(azim=120, elev=30)
    
    #ax.view_init(azim=60, elev=-40)
    stepsizex=20
    stepsizeyz = stepsizex/2
    startx, endx = ax.get_xlim()

    xticks = np.arange(0, endx, stepsizex)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in xticks])
    
    starty, endy = ax.get_ylim()
    yticks = np.arange(0, endy, stepsizeyz)
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in yticks])
    
    startz, endz = ax.get_zlim()
    zticks = np.arange(0, endz, stepsizeyz)
    ax.zaxis.set_ticks(zticks)
    ax.zaxis.set_ticklabels(['%d' %(z*10/scale) for z in zticks])
    
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([ upscale*0.3,upscale*0.1,upscale*0.6, 1]))
    
    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([ upscale*0.75,upscale*0.3,upscale* 0.5, 1]))
    ax.set_xlabel('\n[µm]', linespacing=3.2)

    #ax.set_proj_type('iso')
def plotarray(array,ax,maximum,red=0.47,green=0.0,blue=1.0, seeThrough= seeThrough):
    ax.set_aspect('equal')

    if np.all(array ==0):
        return
    #zgrid = np.asarray([[[z/grid.baselevel for z in x] for x in y] for y in grid.grid])    
    #colorgrid=np.asarray([[[matplotlib.colors.to_hex([ r, g, b,z/maximum ], keep_alpha=True) for z in x] for x in y] for y in array]) 
    
    if seeThrough == True:
        colorgrid=np.asarray([[[matplotlib.colors.to_hex([ red, green, blue,z/maximum], keep_alpha=True) for z in x] for x in y] for y in array])  
        fc = (colorgrid).flatten()        
        gridindices = np.where(np.ones_like(array))       
        ax.scatter(gridindices[0],gridindices[1],gridindices[2],marker = 'o',c=fc,linewidth=0,vmin=0,vmax=maximum,depthshade=depthshade,s=spheresize )
    else:
        gridindices = np.where(array)       
        ax.scatter(gridindices[0],gridindices[1],gridindices[2],marker = 'o',c=(red,green,blue),linewidth=0,vmin=0,vmax=maximum,depthshade=depthshade,s=spheresize )
    axLabel(ax)
def plotarrayVox(array,ax,maximum,red=0.47,green=0.0,blue=1.0,seeThrough= seeThrough):
    #ax.set_aspect('equal')

    if np.all(array ==0):
        return
    #zgrid = np.asarray([[[z/grid.baselevel for z in x] for x in y] for y in grid.grid])    
    #colorgrid=np.asarray([[[matplotlib.colors.to_hex([ r, g, b,z/maximum ], keep_alpha=True) for z in x] for x in y] for y in array]) 
    
    if seeThrough == True:
        print("yo")
        colorgrid=np.asarray([[[matplotlib.colors.to_hex([ red, green, blue,z/(5*maximum)], keep_alpha=True) for z in x] for x in y] for y in array])  
        filled_2=explode(array)

        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.01
        y[:, 0::2, :] += 0.01
        z[:, :, 0::2] += 0.01
        x[1::2, :, :] += 0.99
        y[:, 1::2, :] += 0.99
        z[:, :, 1::2] += 0.99

        #ax.view_init(0,180)
        ax.voxels(x,y,z,filled_2,facecolors=explode(colorgrid), edgecolor= matplotlib.colors.to_hex([ 183/255, 183/255, 183/255,0.3], keep_alpha=True), linewidth=edgewidth)
    #     fc = (colorgrid).flatten()        
    #     grid = np.zeros_like(array)
    #     grid[np.where(array)] =1
    #     ax.voxels(grid, facecolors=fc, edgecolor="")
        
    else:
        grid = np.zeros_like(array)
        grid[np.where(array)] =1
            
        ax.voxels(grid, facecolors=(red,green,blue), edgecolor=edgecolor, linewidth=edgewidth)
    axLabel(ax)

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def compare(matrices):
    dimy = len(matrices[0])
    dimx = len(matrices[0][0])
    dimz = len(matrices[0][0][0])
    
    show= np.zeros_like(matrices)
    for i in range(dimy):
        for j in range(dimx):
            for k in range(dimz):
                comparevalues =[m[i][j][k] for m in matrices]
                gene = np.argmax(comparevalues)
                show[gene][i][j][k] = np.max(comparevalues)
    return show      


def makeMovie(plotFrom,tstep):
    t=0
    testpath = plotFrom + '/Wnt/' +'T%1.1f' %t + '_Wnt.npy'
    #while t <50001:
    if t in [10,90,400,800,3000,8000]:
        
        if os.path.isfile(testpath):
            print("Time",t)
            plt.close("all")
            plotsaved(plotFrom,t,savePic=True)
            #plotsaved(plotFrom, t,Wnt=False,Shh=False,Rostro=False,Dorso=False,Mix=True,savePic=True)
            slicemov.plotSlices(t,10,5,5,save=True,plotmethod='square',dataPath=plotFrom)
        t+=tstep
        testpath = plotFrom + '/Wnt/' +'T%1.1f' %t + '_Wnt.npy'

def plotTime(plotFrom,time,wnt=False,shh=False,rostro=False,dorso=False,mix=False):
    plt.close("all")
    plotsaved(plotFrom,time,savePic=False,Wnt=wnt,Shh=shh,Rostro=rostro,Dorso=dorso,Mix=mix)

#makeMovie(dataPath,0.1)
plotTime(dataPath,8000.0,dorso=False,rostro=False,mix=True)
print("done")
plt.show()
