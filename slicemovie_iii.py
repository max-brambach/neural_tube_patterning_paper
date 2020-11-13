from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib 
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import colorConverter
import os
import ast
from scipy import ndimage
import paras_dorsoventral as dors
import paras_rostrocaudal as ros

import colourmix as colmix
matplotlib.rcParams.update({'font.size': 18})

time = 4000.0
slicenr = 5
tstep=50.0
axis = 'dorso'#'dorso'
runThrough = 'space'
scale = 0.5
StaticDataPath = 'cooltube_0.5_1'


if axis == 'dorso':
    fig = plt.figure(figsize = [6.5, 8])
if axis == 'rostro':
    fig = plt.figure(figsize = [4, 14])
ax1 = fig.add_subplot(111)

#DORSOVENTRAL
# generate the colors for your colormap
colorP = colorConverter.to_rgba(dors.colours[0])
colorO = colorConverter.to_rgba(dors.colours[1])
colorN = colorConverter.to_rgba(dors.colours[2])

white='white'
# make the colormaps
cmapP = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapP',[white,colorP],256)
cmapO = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapO',[white,colorO],256)
cmapN = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapN',[white,colorN],256)


cmapP._init()
cmapO._init()
cmapN._init() # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
dAlphas = np.linspace(0, 0.8, cmapO.N+3)
cmapP._lut[:,-1] = dAlphas
cmapO._lut[:,-1] = dAlphas
cmapN._lut[:,-1] = dAlphas

#ROSTROCAUDAL
colorFB = colorConverter.to_rgba(ros.colours[0])
colorMB = colorConverter.to_rgba(ros.colours[1])
colorHB = colorConverter.to_rgba(ros.colours[2])

white='white'
# make the colormaps
cmapFB = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapFB',[white,colorFB],256)
cmapMB = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapMB',[white,colorMB],256)
cmapHB = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapHB',[white,colorHB],256)


cmapFB._init()
cmapMB._init()
cmapHB._init() # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
rAlphas = np.linspace(0, 0.8, cmapMB.N+3)
cmapFB._lut[:,-1] = rAlphas
cmapMB._lut[:,-1] = rAlphas
cmapHB._lut[:,-1] = rAlphas




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

def getCut(axis,t=0,s=0,dataPath=StaticDataPath):

    if axis == 'dorso':
        dorsoDir  = dataPath + '/dorso/'
        dcFile = dorsoDir + 'T%1.1f' %t + '_dComp.npy'
        pFile = dorsoDir + 'T%1.1f' %t + '_P.npy'
        oFile = dorsoDir + 'T%1.1f' %t + '_O.npy'
        nFile = dorsoDir + 'T%1.1f' %t + '_N.npy'
        
        if os.path.isfile(dcFile):
            dComp = np.load(dcFile)
        else:
            pArray = np.load(pFile)
            oArray = np.load(oFile)
            nArray = np.load(nFile)
            dComp = compare([pArray,oArray,nArray])
            np.save(dcFile,dComp)
    
        arrA = dComp[0]
        arrB = dComp[1]
        arrC = dComp[2]
            
        arrA = arrA[s,:,:]
        arrB = arrB[s,:,:]
        arrC = arrC[s,:,:]    
            
            
    if axis == 'rostro':
        rostroDir  = dataPath + '/rostro/'
        rcFile = rostroDir  + 'T%1.1f' %t + '_rComp.npy'
        FBFile = rostroDir  + 'T%1.1f' %t + '_FB.npy'
        MBFile = rostroDir + 'T%1.1f' %t + '_MB.npy'
        HBFile = rostroDir  + 'T%1.1f' %t + '_HB.npy'
        
        if os.path.isfile(rcFile):
            rComp = np.load(rcFile)
        else:
            FBArray = np.load(FBFile)
            MBArray = np.load(MBFile)
            HBArray = np.load(HBFile)
            rComp = compare([FBArray,MBArray,HBArray])
            np.save(rcFile,rComp)
        
        arrA = rComp[0]
        arrB = rComp[1]
        arrC = rComp[2]
            
        # arrA = arrA[:,s,:]
        # arrB = arrB[:,s,:]
        # arrC = arrC[:,s,:]
        
        arrA = arrA[:,:,s]
        arrB = arrB[:,:,s]
        arrC = arrC[:,:,s]
            
    if axis == 'rostro2':
        rostroDir  = dataPath + '/rostro/'
        rcFile = rostroDir  + 'T%1.1f' %t + '_rComp.npy'
        FBFile = rostroDir  + 'T%1.1f' %t + '_FB'
        MBFile = rostroDir + 'T%1.1f' %t + '_MB'
        HBFile = rostroDir  + 'T%1.1f' %t + '_HB'
        
        if os.path.isfile(rcFile):
            rComp = np.load(rcFile)
        else:
            FBArray = np.load(FBFile)
            MBArray = np.load(MBFile)
            HBArray = np.load(HBFile)
            rComp = compare([FBArray,MBArray,HBArray])
            np.save(rcFile,rComp)
        
        arrA = rComp[0]
        arrB = rComp[1]
        arrC = rComp[2]
            
        arrA = arrA[:,s,:]
        arrB = arrB[:,s,:]
        arrC = arrC[:,s,:]

    return arrA,arrB,arrC

def getTS(ts, rate, t=time, s=slicenr):
    """ts = what is looped over in the animation"""
    if ts == 'time':
        t_ret = rate*tstep
        s_ret = slicenr
    
    if ts =='space':
        t_ret = t
        s_ret = rate

    
    return t_ret,s_ret
    
def update(rate):
    ax1.clear()    
    t,s = getTS(runThrough,rate)
    #print(rate,t,s)
    cut = getCut(axis,t,s)
    
    
    
    
    ax1.set_title("slice nr %d time %1.1f" %(s,t))
    #if t < len(data[0][0]):
        #ax1.matshow(data[:,t,:])
        #t+=1    
    #else:
        #t=0
    
    # ax1.imshow(arrFB[rate,:,:],interpolation='bilinear',cmap=cmap1)
    # ax1.imshow(arrMB[rate,:,:],interpolation='bilinear',cmap=cmap2)
    # ax1.imshow(arrHB[rate,:,:],interpolation='bilinear',cmap=cmap3)
    if axis == 'dorso':
        cmap1,cmap2,cmap3 = cmapP,cmapO,cmapN
        size = 500
    if axis == 'rostro':
        cmap1,cmap2,cmap3 = cmapFB,cmapMB,cmapHB
        size =100
    
    # ax1.imshow(cut[0],cmap=cmap1)
    # ax1.imshow(cut[1],cmap=cmap2)
    # ax1.imshow(cut[2],cmap=cmap3)
    """
    ax1.imshow(cut[0],interpolation='nearest',cmap=cmap1)
    ax1.imshow(cut[1],interpolation='nearest',cmap=cmap2)
    ax1.imshow(cut[2],interpolation='nearest',cmap=cmap3)
    """
    
    mapper1 = matplotlib.cm.ScalarMappable(cmap=cmap1)
    mapper2 = matplotlib.cm.ScalarMappable(cmap=cmap2)
    mapper3 = matplotlib.cm.ScalarMappable(cmap=cmap3)

    c1= np.where(cut[0])
    colors1 = mapper1.to_rgba(cut[0][c1])
    c2= np.where(cut[1])
    colors2 = mapper2.to_rgba(cut[1][c2])
    c3= np.where(cut[2])
    colors3 = mapper3.to_rgba(cut[2][c3])
    
    ax1.set_aspect('auto')
    
    ax1.set_xlim([-1,16])
    ax1.scatter(c1[0],c1[1],c=colors1,s=size)
    ax1.scatter(c2[0],c2[1],c=colors2,s=size)
    ax1.scatter(c3[0],c3[1],c=colors3, s=size)
    #plt.savefig('unsinnfig/t%d'% rate)

def plotSlices(time, dorsnr, rosnr, rosnr2, plotmethod='circle',save=True, dataPath=StaticDataPath):
    # fug = plt.figure(figsize=(8, 6))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    # axDors = fug.add_subplot(gs[0])
    # axRos = fug.add_subplot(gs[1])
    plt.close("all")
    
    fug = plt.figure(figsize = [7.5, 8])
    fag = plt.figure(figsize = [10, 14])
    axDors = fug.add_subplot(1,1,1)
    axRos = fag.add_subplot(1,2,1)
    axRos2 = fag.add_subplot(1,2,2)
    
    axDors.set_title("DV slice at \n x = %d µm, t = %1.1f " %(dorsnr*10/scale, time)) 
    axDors.set_xlabel("y [µm]")
    dxticks = np.arange(0,20,5)
    axDors.xaxis.set_ticks(dxticks)
    axDors.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in dxticks])    
    dyticks = np.arange(0,65,10)
    axDors.yaxis.set_ticks(dyticks)
    axDors.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in dyticks])
    axDors.set_ylabel("z [µm]")
    
    axRos.set_title("RC slice at \n z = %d µm, t = %1.1f " %(rosnr*10/scale, time))
    rxticks = dxticks
    axRos.xaxis.set_ticks(rxticks)
    axRos.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in rxticks]) 
    ryticks = np.arange(0,65,10)
    axRos.yaxis.set_ticks(ryticks)
    axRos.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in ryticks])
    axRos.set_xlabel("y [µm]")
    axRos.set_ylabel("x [µm]")
    
    
    axRos2.set_title("RC slice at \n y = %d µm, t = %1.1f " %(rosnr*10/scale, time))
    r2xticks = np.arange(0,65,10)
    axRos2.xaxis.set_ticks(r2xticks)
    axRos2.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in r2xticks]) 
    r2yticks = np.arange(0,65,10)
    axRos2.yaxis.set_ticks(r2yticks)
    axRos2.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in r2yticks])
    axRos2.set_xlabel("z [µm]")
    axRos2.set_ylabel("x [µm]")
    
    dataDors = getCut('dorso', t= time, s=dorsnr, dataPath = dataPath)
    dataRos = getCut('rostro', t= time, s=rosnr, dataPath = dataPath)
    dataRos2 = getCut('rostro2', t= time, s=rosnr2,dataPath = dataPath)
    
    for axtype in ['rostro','dorso']:
        if axtype == 'dorso':
            cmap1,cmap2,cmap3 = cmapP,cmapO,cmapN
            size = 500
            ax = axDors
            cut =dataDors
        if axtype == 'rostro':
            cmap1,cmap2,cmap3 = cmapFB,cmapMB,cmapHB
            size =100
            ax=axRos
            ax2=axRos2
            cut= dataRos
            cut2=dataRos2
        
        if plotmethod == 'circle':    
            mapper1 = matplotlib.cm.ScalarMappable(cmap=cmap1)
            mapper2 = matplotlib.cm.ScalarMappable(cmap=cmap2)
            mapper3 = matplotlib.cm.ScalarMappable(cmap=cmap3)
        
            c1= np.where(cut[0])
            colors1 = mapper1.to_rgba(cut[0][c1])
            c2= np.where(cut[1])
            colors2 = mapper2.to_rgba(cut[1][c2])
            c3= np.where(cut[2])
            colors3 = mapper3.to_rgba(cut[2][c3])
            
            ax.set_aspect('auto')
            
            #ax.set_xlim([-1,16])
            ax.scatter(c1[0],c1[1],c=colors1,s=size)
            ax.scatter(c2[0],c2[1],c=colors2,s=size)
            ax.scatter(c3[0],c3[1],c=colors3, s=size)
        
        if plotmethod == 'square': 
            # ax1.imshow(cut[0],cmap=cmap1)
            # ax1.imshow(cut[1],cmap=cmap2)
            # ax1.imshow(cut[2],cmap=cmap3)
            
            if axtype == 'rostro':
                
                
                ax.imshow(cut[0][:-1,:-1],interpolation='nearest',cmap=cmap1,origin = 'lower')
                ax.imshow(cut[1][:-1,:-1],interpolation='nearest',cmap=cmap2,origin = 'lower')
                ax.imshow(cut[2][:-1,:-1],interpolation='nearest',cmap=cmap3,origin = 'lower')
                
                ax2.imshow(ndimage.rotate(cut2[0][:-1,:-1],-90)[:,:-1],interpolation='nearest',cmap=cmap1,origin = 'lower')
                ax2.imshow(ndimage.rotate(cut2[1][:-1,:-1],-90)[:,:-1],interpolation='nearest',cmap=cmap2,origin = 'lower')
                ax2.imshow(ndimage.rotate(cut2[2][:-1,:-1],-90)[:,:-1],interpolation='nearest',cmap=cmap3,origin = 'lower')
                
                # rcut0 = ndimage.rotate(cut[0], 90)
                # rcut1 = ndimage.rotate(cut[1], 90)
                # rcut2 = ndimage.rotate(cut[2], 90)
                
                # ax.imshow(rcut0,interpolation='nearest',cmap=cmap1)
                # ax.imshow(rcut1,interpolation='nearest',cmap=cmap2)
                # ax.imshow(rcut2,interpolation='nearest',cmap=cmap3)
                
            if axtype == 'dorso':
                rcut0 = ndimage.rotate(cut[0], -90)
                rcut1 = ndimage.rotate(cut[1], -90)
                rcut2 = ndimage.rotate(cut[2], -90)
                
                ax.imshow(rcut0[:-1,1:],interpolation='nearest',cmap=cmap1,origin = 'lower')
                ax.imshow(rcut1[:-1,1:],interpolation='nearest',cmap=cmap2,origin = 'lower')
                ax.imshow(rcut2[:-1,1:],interpolation='nearest',cmap=cmap3,origin = 'lower')
    if save ==True:
        fug.savefig(dataPath + '/allPictures/T%1.1f_DV%d.png' %(time,dorsnr)   )
        fag.savefig(dataPath + '/allPictures/T%1.1f_RC%d_%d.png' %(time,rosnr,rosnr2)   )
  
def plotSliceMix(plotFrom, time, dorsnr, rosnr, rosnr2,save=True):
    dataPath = plotFrom
    """Plot gene combinations with a different colour for each combination."""
    wntDir = plotFrom + '/Wnt/'
    shhDir = plotFrom + '/Shh/'
    rostroDir = plotFrom + '/rostro/'
    dorsoDir  = plotFrom + '/dorso/'
    mixDir = plotFrom + '/Mix/'
    baseLevels = np.load(plotFrom + '/BaseLevels.npy')
    allDir = plotFrom + '/allPictures/'
    
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
        
    mixArray[mixArray==0] = np.nan
    #plt.close("all")
    
    fug = plt.figure(figsize = [7.5, 8])
    fag = plt.figure(figsize = [10, 14])
    axDors = fug.add_subplot(1,1,1)
    axRos = fag.add_subplot(1,2,1)
    axRos2 = fag.add_subplot(1,2,2)
    
    axDors.set_title("DV slice at \n x = %d µm, t = %1.1f " %(dorsnr*10/scale, time)) 
    axDors.set_xlabel("y [µm]")
    dxticks = np.arange(0,20,5)
    axDors.xaxis.set_ticks(dxticks)
    axDors.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in dxticks])    
    dyticks = np.arange(0,65,10)
    axDors.yaxis.set_ticks(dyticks)
    axDors.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in dyticks])
    axDors.set_ylabel("z [µm]")
    
    axRos.set_title("RC slice at \n z = %d µm, t = %1.1f " %(rosnr*10/scale, time))
    rxticks = dxticks
    axRos.xaxis.set_ticks(rxticks)
    axRos.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in rxticks]) 
    ryticks = np.arange(0,65,10)
    axRos.yaxis.set_ticks(ryticks)
    axRos.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in ryticks])
    axRos.set_xlabel("y [µm]")
    axRos.set_ylabel("x [µm]")
    
    
    axRos2.set_title("RC slice at \n y = %d µm, t = %1.1f " %(rosnr*10/scale, time))
    r2xticks = np.arange(0,65,10)
    axRos2.xaxis.set_ticks(r2xticks)
    axRos2.xaxis.set_ticklabels(['%d' %(x*10/scale) for x in r2xticks]) 
    r2yticks = np.arange(0,65,10)
    axRos2.yaxis.set_ticks(r2yticks)
    axRos2.yaxis.set_ticklabels(['%d' %(y*10/scale) for y in r2yticks])
    axRos2.set_xlabel("z [µm]")
    axRos2.set_ylabel("x [µm]")


    
    for axtype in ['rostro','dorso']:
        for i in range(len(mixArray)):
        #for i in range(3):
            
            colours = colmix.colours[i]
            #colours2 = colmix.colours[i+1]
            myCmap =  matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapP',[colours,colours],256)
            #myCmap2 =  matplotlib.colors.LinearSegmentedColormap.from_list('my_cmapP',['white',colours2],256)
            
            print(i, colours)
            if axtype == 'dorso':
                
                size = 500
                ax = axDors
                arr = getMixCut(axtype,mixArray[i],s=dorsnr)
                arr=(np.flip(np.transpose(arr),axis=1))[:-1,1:]
                cut = np.ma.masked_where(np.isnan(arr),arr)
                #cut= np.flip(cut)
                ax.set_aspect('equal')
              
            if axtype == 'rostro':
                size =100
                ax=axRos
                ax2=axRos2
                ax.set_aspect('equal')
                ax2.set_aspect('equal')
                
                arr= getMixCut('rostro',mixArray[i],s=rosnr)
                arr2=getMixCut('rostro2',mixArray[i],s=rosnr2)
                
                cut= np.ma.masked_where(np.isnan(arr),arr)

                cut2 = np.ma.masked_where(np.isnan(arr2),arr2)
                cut2 = (np.flip(np.transpose(cut2),axis=1))
                cut2= cut2[:,1:]
                
                

            # ax1.imshow(cut[0],cmap=cmap1)
            # ax1.imshow(cut[1],cmap=cmap2)
            # ax1.imshow(cut[2],cmap=cmap3)
            
            if axtype == 'rostro':
                print(cut[:-1,:-1])
                ax.pcolor(cut[:-1,:-1],cmap=myCmap)

                
                ax2.pcolor(cut2[:-1,:-1],cmap=myCmap)

                
                # rcut0 = ndimage.rotate(cut[0], 90)
                # rcut1 = ndimage.rotate(cut[1], 90)
                # rcut2 = ndimage.rotate(cut[2], 90)
                
                # ax.imshow(rcut0,interpolation='nearest',cmap=cmap1)
                # ax.imshow(rcut1,interpolation='nearest',cmap=cmap2)
                # ax.imshow(rcut2,interpolation='nearest',cmap=cmap3)
                
            if axtype == 'dorso':
                print("DORSO")
                rcut = ndimage.rotate(cut, -90)
               
                ax.pcolor(cut,cmap=myCmap)

        if save ==True:
            fug.savefig(dataPath + '/allPictures/T%1.1f_DV%d_Mix.png' %(time,dorsnr)   )
            fag.savefig(dataPath + '/allPictures/T%1.1f_RC%d_%d_Mix.png' %(time,rosnr,rosnr2)   )


def getMixCut(axis,mixArray_i,s=0):
    print(s, mixArray_i.size)
    if axis == 'dorso':           
        arrA = mixArray_i[s,:,:]

            
    if axis == 'rostro':
        arrA = mixArray_i[:,:,s]
           
    if axis == 'rostro2':          
        arrA = mixArray_i[:,s,:]
    print(arrA.shape)
    return arrA

def test():
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax3=fig.add_subplot(3,1,3)
    ax2 = fig.add_subplot(3,1,2)
    arr = np.random.rand(10,10)
    arr2 = np.copy(arr)
    arr[arr<=0.5] = np.nan
    arr2[arr2>0.5] = np.nan
    print(arr)

    
    m = np.ma.masked_where(np.isnan(arr),arr)
    m2 = np.ma.masked_where(np.isnan(arr2),arr2)
    ax.pcolor(m,cmap =cmapP)
    ax2.pcolor(m2,cmap=cmapO)
    ax3.pcolor(m,cmap=cmapP)
    ax3.pcolor(m2,cmap=cmapO)
#animation = FuncAnimation(fig, update, interval=700)
"""
myt= 8000
dnr=10
rnr=5
rnr2=4
plotSlices(myt,dnr,rnr,rnr2,save=True,plotmethod='square')
plotSliceMix(StaticDataPath,myt,dnr,rnr,rnr2)
#(StaticDataPath, 4000.0, 10,5,4)
#test()
plt.show()
"""