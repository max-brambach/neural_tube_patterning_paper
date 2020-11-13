# -*- coding: utf-8 -*-
from scipy.integrate import solve_ivp
import matplotlib
"""in case it's not working uncomment this: matplotlib.use('TkAgg') """
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import inv

from matplotlib import colors as mcolors




import paras_dorsoventral as dors
import paras_rostrocaudal as ros
import testround_difftest_set as r #for sparse matrix stuff
#import testround_difftest_backup as r
import stencil_import as tubemodel
import os

import plot_saved_v

d=10.0
dx=20
dt=0.1#10


maxtime = 10000 #TIME IN SECONDS!!!
"""
CHECKLIST
maxtime
dx according to model (10 ori, 20 0.5, 40 0.25 etc)
stencil_import paths
stencils in folder
plotsaved path
Wnt0, Shh0
delta_Wnt, delta_Shh
plotting colourmax here and in plotsaved
how often save?
spheresize according to model
"""



xlen =tubemodel.xmax
ylen =tubemodel.ymax
zlen = tubemodel.zmax
print(xlen,ylen,zlen)
spheresize = r.spheresize




D_Wnt = 150.7
D_Shh = 133.4

delta_Wnt = 0.04
delta_Shh = 0.1

Wnt0 = tubemodel.Wnt0
Shh0 = tubemodel.Shh0



#import the stencils for tubemodel, WNTsecretion and SHHsecretion points
stenc = tubemodel.stenc
WNTstenc= tubemodel.Wstenc
SHHstenc= tubemodel.Sstenc



#plotting colourmax
rosmax = tubemodel.Wnt0#5#0.0
dorsmax = tubemodel.Shh0#5#0.0
unknownbase=5.0

class Grid:
    def __init__(self,xdim,ydim,zdim, Name, seeds,Alpha,Baselevel):
        self.grid = np.zeros((xdim,ydim,zdim))        
        self.totalsites = np.sum(stenc.grid)
        self.name = Name
        self.xlen=xdim
        self.ylen=ydim
        self.zlen=zdim
        self.baselevel=Baselevel
        self.plantrandomseed(seeds)
        self.alpha=Alpha
        if Name =="Wnt":
            self.Amatr = A_Wnt
            self.b = b_Wnt
            self.delta = delta_Wnt
            print("deltawnt:",self.delta)
        if Name =="Shh":
            self.Amatr = A_Shh
            self.b = b_Shh
            self.delta = delta_Shh
    
    def show(self,ax):     
        plotgrid(self,ax)
                
          
    def plantseed(self,coordinates):
        for xyz in coordinates:
            x= xyz[0]
            y = xyz[1]
            z=xyz[2]
           
            self.grid[y][x][z] = self.baselevel
            
    
    def artificialseed(self,coordinates,level):
        for i in range(len(coordinates)):
            xyz = coordinates[i]
            x= xyz[0]
            y = xyz[1]
            z=xyz[2]
            self.grid[x][y][z] = level[i]*self.baselevel
            
    def plantrandomseed(self, seeds):
        n = seeds
        M = self.totalsites
        coords = np.transpose(np.where(stenc.grid))
       
        for c in coords:
            randomnr = np.random.uniform()
            if randomnr < n/M:                                               
                self.grid[c[0]][c[1]][c[2]] = self.baselevel#*np.random.uniform()
                n-=1
            M-=1
                    
    def diffusion(self,n):
        for i in range(n):
            deltaU,b = laplacian(self,self.Amatr,self.b)
            old = self.grid
            self.grid =old + dt*self.alpha*(deltaU +b)
            
    def degradation(self,n):
        for i in range(n):
            old = self.grid
            #print("degrmax",np.max(self.delta * self.grid *dt))
            self.grid = old - self.delta * old *dt
    

def rostrocaudal_reaction(rate,FB,MB,HB,Wnt):
    for i in range(rate):
        fb= (FB.grid).copy()
        mb= (MB.grid).copy()
        hb= (HB.grid).copy()
        gsk3= (GSK3.grid).copy() # Wnt modulates gsk3
  
        wnt= (Wnt.grid).copy()
        u = (U.grid).copy()

        FB.grid = fb + dt*( ros.c1*(gsk3**ros.n1)/(1+ ros.c1*(gsk3**ros.n1)+ ros.c2*(mb**ros.n2)+ ros.c3*(hb**ros.n3)) -ros.d1*fb )
        MB.grid = mb + dt*(ros.c4*(mb**ros.n4)/(1+ ros.c4*(mb**ros.n4)+ ros.c5*(fb**ros.n5)+ ros.c6*(hb**ros.n6)+ ros.c7*(gsk3**ros.n7)) -ros.d2*mb)
        HB.grid = hb + dt*( ros.c8*(hb**ros.n8)/(1 + ros.c8*(hb**ros.n8) + ros.c9*(fb**ros.n9) + ros.c10*(mb**ros.n10)+ ros.c11*(gsk3**ros.n11)) -ros.d3*hb  ) 
        GSK3.grid = gsk3 + dt*(ros.c12*(gsk3**ros.n12)/(1 + ros.c12*(gsk3**ros.n12)+ ros.c13*(u**ros.n13) ) -ros.d4*gsk3 )

        U.grid = u + dt*((ros.c14*(wnt**ros.n14) + ros.c15*(u**ros.n15))/( 1+ ros.c14*(wnt**ros.n14) + ros.c15*(u**ros.n15) + ros.c16*(u**ros.n16)) - ros.d5*u)    
        antistenc = np.ones_like(stenc.grid) - stenc.grid
        for c in np.transpose(np.where(antistenc)):
            FB.grid[c[0]][c[1]][c[2]] = 0
            MB.grid[c[0]][c[1]][c[2]] = 0
            HB.grid[c[0]][c[1]][c[2]] = 0
            GSK3.grid[c[0]][c[1]][c[2]] = 0
    
def dorsoventral_reaction(rate,P,O,N,G,S,W):
    for i in range(rate):
        p= (P.grid).copy()
        o= (O.grid).copy()
        n= (N.grid).copy()
        g= (G.grid).copy()
        s= (S.grid).copy()
        w= (W.grid).copy()
    
        P.grid = p + dt*( dors.alpha / (1.0 + (n/dors.NcritP)**dors.h1 + (o/dors.OcritP)**dors.h2 ) - dors.k1*p  )
        O.grid = o + dt*(( (dors.beta*g) / (1.0+g) ) * ( 1.0/(1.0+(n/dors.NcritO)**dors.h3) ) - dors.k2*o)
        N.grid = n + dt*( (dors.gamma*g/(1.0+g)) * (1.0/(1.0+ (o/dors.OcritN)**dors.h4 + (p/dors.PcritN)**dors.h5 )) - dors.k3*n)  
        G.grid = g + dt*(((dors.delta*s)/(1.0+s)) * (1.0/(1.0+ (w/dors.WcritG)**dors.h6 )) - dors.k4*g)
        
        antistenc = np.ones_like(stenc.grid) - stenc.grid
        for c in np.transpose(np.where(antistenc)):
            P.grid[c[0]][c[1]][c[2]] = 0
            O.grid[c[0]][c[1]][c[2]] = 0
            N.grid[c[0]][c[1]][c[2]] = 0
            G.grid[c[0]][c[1]][c[2]] = 0
        
 
def alldiffuse(rate,Wnt,Shh):
    for i in range(rate):
        Wnt.diffusion(1)
        Shh.diffusion(1)

def alldegrade(rate,Wnt,Shh):
    for i in range(rate):
        Wnt.degradation(1)
        Shh.degradation(1)

def plotgrid(grid,ax,r=0.47,g=0.0,b=1.0):
    if np.all(grid.grid ==0):
        return

    print("minmax",np.min(grid.grid),np.max(grid.grid))
    if grid.baselevel!=0:
        colorgrid=np.asarray([[[matplotlib.colors.to_hex([ r, g, b,z/unknownbase], keep_alpha=True) for z in x] for x in y] for y in grid.grid])     
    else:
        colorgrid=np.asarray([[[matplotlib.colors.to_hex([ r, g, b,z/unknownbase], keep_alpha=True) for z in x] for x in y] for y in grid.grid]) 
    fc = (colorgrid).flatten()        
    gridindices = np.where(np.ones_like(grid.grid))       
    ax.scatter(gridindices[0],gridindices[1],gridindices[2],marker = 'o',c=fc,linewidth=0,vmin=0,vmax=grid.baselevel,depthshade=False,s=spheresize )
    
    

def plotarray(array,ax,maximum,r=0.47,g=0.0,b=1.0):
    if np.all(array ==0):
        return
    colorgrid=np.asarray([[[matplotlib.colors.to_hex([ r, g, b,z/maximum ], keep_alpha=True) for z in x] for x in y] for y in array])       
    fc = (colorgrid).flatten()        
    gridindices = np.where(np.ones_like(array))       
    ax.scatter(gridindices[0],gridindices[1],gridindices[2],marker = 'o',c=fc,linewidth=0,vmin=0,vmax=maximum,depthshade=False,s=spheresize )


def plotarray_fixed_alpha(array,ax,maximum,alpha=0.3,r=0.47,g=0.0,b=1.0):
    if np.all(array ==0):
        return 
    colorgrid=np.asarray([[[matplotlib.colors.to_hex([ r, g, b,alpha ], keep_alpha=True) for z in x] for x in y] for y in array])       
    fc = (colorgrid).flatten()        
    gridindices = np.where(np.ones_like(array))       
    ax.scatter(gridindices[0],gridindices[1],gridindices[2],marker = 'o',c=fc,linewidth=0,vmin=0,vmax=maximum,depthshade=False,s=spheresize )

def secretion(rate,Wnt,Shh):
    for i in range(rate):
        Shh.artificialseed(SHHstenc.secretion_coords,SHHstenc.secretion_levels)
        Wnt.artificialseed(WNTstenc.secretion_coords,WNTstenc.secretion_levels) 
        

def run(maxt, savedirectory, save=True):
    for ax in [axWnt,axShh,axRos,axDors]:
        ax.clear()
        
    axRos.set_title("Rostrocaudal network (Max)")
    axDors.set_title("Dorsoventral network (Balaskas)")
    axWnt.set_title("Wnt")
    axShh.set_title("Shh ")
    
    if save == True:
        sd=savedirectory
      
        wntdir = sd + '/Wnt'
        shhdir = sd + '/Shh'
        rostrodir = sd + '/rostro'
        dorsodir  = sd + '/dorso'
        
        os.mkdir(wntdir)
        os.mkdir(shhdir)
        os.mkdir(rostrodir)
        os.mkdir(dorsodir)
        
        os.mkdir(wntdir + '/pictures')
        os.mkdir(shhdir + '/pictures')
        os.mkdir(rostrodir + '/pictures')
        os.mkdir(dorsodir + '/pictures')
    
    else:
        print('NOT SAVING')
        

    steps = int((maxt/dt +dt))
    print("steps:",steps)
    for step in range(steps):
        if save == True:
            if step in np.arange(0,3000,200) or step in np.arange(0,120000,20000) or step in np.arange(0,10000,1000): #step %1000 == 0 or step# and time % 100 == 0) or (save == True and time in np.arange(0,16,1)):
                time = step*dt
                save_networks(savedirectory,time,FB,MB,HB,P,O,N,G,Wnt,Shh)
                print("Saved time %f"% time)

        print("step",step,"/",steps)
        dorsoventral_reaction(1,P,O,N,G,Shh,Wnt)
        rostrocaudal_reaction(1,FB,MB,HB,Wnt)     
        alldiffuse(1,Wnt,Shh)
        secretion(1,Wnt,Shh)
        alldegrade(1,Wnt,Shh)


def sparsedot(A,v):
    """Dot product for sparse matrices"""
    w=np.zeros(len(v))
    for ija in A:
        
        i=ija[0]
        j=ija[1]
        a=ija[2]
        
        w[i] += v[j]*a
    return w
    
def laplacian(gridname,Amatr,b):
    v,c = r.grid_to_vector(stenc)
    c1,c2,c3 = np.transpose(c)
    
    u=(gridname.grid)[c1,c2,c3]

    
    if len(Amatr) == len(Amatr[0]):
        newu= np.dot(Amatr,u)
    else:
        newu= sparsedot(Amatr,u)

    L = r.vector_to_grid(newu,gridname,c)
    L[:,:,:] = L[:,:,:]/dx**2
    b = r.vector_to_grid(b,gridname,c)
    b = b*gridname.baselevel/dx**2 
    
    
    return L,b

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

def show_networks(FB,MB,HB,P,O,N,G,Wnt,Shh,axRos,axDors,axWnt,axShh, scale=False):
    for ax in [axWnt,axShh,axRos,axDors]:
        ax.clear()
    
    if scale == True:
        longest = max(xlen,ylen,zlen)
        for ax in [axWnt,axShh,axRos,axDors]:
            ax.set_xlim([0,longest])
            ax.set_ylim([0,longest])
            ax.set_zlim([0,longest])
    print("ah")
    
    plotgrid(Shh,axShh,r=92/255, g=121/255, b=168/255)
    plotgrid(Wnt,axWnt,r=14/255, g=27/255, b=48/255)
    print("oh")
    
    rostro = [FB.grid,MB.grid,HB.grid]
    ros_show = compare(rostro)
    print(np.max(ros_show),"=roshowmax")
    for i in range(len(ros_show)):
        colours = ros.colours[i]
        plotarray(ros_show[i],axRos,rosmax,r=colours[0],g=colours[1],b=colours[2])
        
    
    dorso = [P.grid,O.grid,N.grid]
    dors_show = compare(dorso)
    print(np.max(dors_show),"=dorsshowmax")
   
    for i in range(len(dors_show)):
        colours = dors.colours[i]
        plotarray(dors_show[i],axDors,dorsmax,r=colours[0],g=colours[1],b=colours[2])
    
    
    """
    #genes rostro
    farg = plt.figure()
    axtest = farg.add_subplot(2,2,1)
    axtest.set_title("genes rostro")
    FBax = FB.grid[:,0,-1]
    MBax = MB.grid[:,0,-1]
    HBax = HB.grid[:,0,-1]
    xr=np.arange(xlen)
    axtest.plot(xr,FBax,color=ros.colours[0],label='FB')
    axtest.plot(xr,MBax,color=ros.colours[1],label='MB')
    axtest.plot(xr,HBax,color=ros.colours[2],label='HB')
    
    #genes dorso
    axtest2 = farg.add_subplot(2,2,2)
    axtest2.set_title("genes dorso")
    xd=np.arange(zlen)
    Pax = P.grid[0,int(ylen/2),:]
    Oax = O.grid[0,int(ylen/2),:]
    Nax = N.grid[0,int(ylen/2),:]
    axtest2.plot(xd,Pax,color=dors.colours[0],label='P')
    axtest2.plot(xd,Oax,color=dors.colours[1],label='O')
    axtest2.plot(xd,Nax,color=dors.colours[2],label='N')
   
    
    #morphogens rostro
    axtest3 = farg.add_subplot(2,2,3)
    axtest3.set_title("morphogens rostro")
    Wntplotr = Wnt.grid[:,0,-1]
    Shhplotr = Shh.grid[:,0,-1]
    GSKplotr =  GSK3.grid[:,0,-1]
    axtest3.plot(xr,Wntplotr,color='k',label='Wnt')
    axtest3.plot(xr,Shhplotr,color='b',label='Shh')
    #axtest3.plot(xr,GSKplotr,color='r',label='GSK')
    
    #morphogens dorso
    axtest4 = farg.add_subplot(2,2,4)
    axtest4.set_title("morphogens dorso")
    Wntplotd = Wnt.grid[0,int(ylen/2),:]
    Shhplotd = Shh.grid[0,int(ylen/2),:]
    GSKplotd =  GSK3.grid[0,int(ylen/2),:]
    axtest4.plot(xd,Wntplotd,color='k',label='Wnt')
    axtest4.plot(xd,Shhplotd,color='b',label='Shh')
    #axtest4.plot(xd,GSKplotd,color='r',label='GSK')

    axtest.legend()
    axtest2.legend()
    axtest3.legend()
    axtest4.legend()
    """
    #plt.show()
 
def save_networks(savedir,t, FB,MB,HB,P,O,N,G,Wnt,Shh): 
    sd = savedir
    #if os.path.isdir(savedir):
        #print("directory already exists. creating new")
        #sd= savedir + '_1'
    
    wntdir = sd + '/Wnt'
    shhdir = sd + '/Shh'
    rostrodir = sd + '/rostro'
    dorsodir  = sd + '/dorso'
    
    infopath = sd + '/info.txt'
    
    if os.path.isfile(infopath) == False:
        f = open(infopath, 'w')
        info = "Model: %s \n Secretion Wnt: %s  \n Secretion Shh: %s\n" % (tubemodel.stencpath,tubemodel.wntsecrpath,tubemodel.shhsecrpath)
        info += "D_Wnt %f D_Shh %f delta_Wnt %f delta_Shh %f \n rosmax %f dorsmax %f unknownbase %f \n dx %f dt %f \n" % (D_Wnt, D_Shh, delta_Wnt, delta_Shh,  rosmax, dorsmax, unknownbase,dx,dt)
        info += "Baselevel: \n Wnt0 %f Shh0 %f \n FB %f MB %f HB %f \n P %f O %f N %f " % (Wnt0, Shh0,FB.baselevel,MB.baselevel,HB.baselevel,P.baselevel,O.baselevel,N.baselevel)
        np.savetxt(f,np.asarray([info]),fmt='%s') #.astype(int)
        f.close()
        
    
    
    #with baselevels
    #wntpath = wntdir + '/T%d_BL%d_Wnt' % (t,Wnt.baselevel) + '.npy'
    #shhpath = shhdir + '/T%d_BL%d_Shh' % (t,Shh.baselevel) + '.npy'
    #FBpath = rostrodir + '/T%d_BL%d_FB' % (t,FB.baselevel) + '.npy'
    #MBpath = rostrodir + '/T%d_BL%d_MB' % (t,MB.baselevel) + '.npy'
    #HBpath = rostrodir + '/T%d_BL%d_HB' % (t,HB.baselevel) + '.npy'
    #Ppath = dorsodir + '/T%d_BL%d_P' % (t,P.baselevel) + '.npy'
    #Opath = dorsodir + '/T%d_BL%d_O' % (t,O.baselevel) + '.npy'
    #Npath = dorsodir + '/T%d_BL%d_N' % (t,N.baselevel) + '.npy'
    
    #without BL
    wntpath = wntdir + '/T%1.1f_Wnt' % t + '.npy'
    shhpath = shhdir + '/T%1.1f_Shh' % t + '.npy'
    FBpath = rostrodir + '/T%1.1f_FB' % t + '.npy'
    MBpath = rostrodir + '/T%1.1f_MB' % t + '.npy'
    HBpath = rostrodir + '/T%1.1f_HB' % t + '.npy'
    Ppath = dorsodir + '/T%1.1f_P' % t + '.npy'
    Opath = dorsodir + '/T%1.1f_O' % t + '.npy'
    Npath = dorsodir + '/T%1.1f_N' % t + '.npy'
    BLpath = sd+ '/BaseLevels.npy'
    
    
    
    np.save(wntpath,Wnt.grid)
    np.save(shhpath,Shh.grid)
    np.save(FBpath,FB.grid)
    np.save(MBpath,MB.grid)
    np.save(HBpath,HB.grid)
    np.save(Ppath,P.grid)
    np.save(Opath,O.grid)
    np.save(Npath,N.grid)
    
    baselevels = np.asarray([Wnt0, Shh0,FB.baselevel,MB.baselevel,HB.baselevel,P.baselevel,O.baselevel,N.baselevel])
    np.save(BLpath,baselevels)
    
    

    

def AmatrCheck(pathA_Wnt,pathA_Shh,pathb_Wnt,pathb_Shh,wntStenc,shhStenc,stenc):
    if os.path.isfile(pathA_Wnt) and os.path.isfile(pathb_Wnt):
        print("WNT: Reading %s as Amatrix and %s as b" % (pathA_Wnt, pathb_Wnt))
        lines = np.loadtxt(pathA_Wnt)           
        A_Wnt = (np.asarray(lines)).astype('int')
        b_Wnt = np.load(pathb_Wnt)
    
    else:
      print("WNT: Creating %s and %s" % (pathA_Wnt, pathb_Wnt))
      f = open(pathA_Wnt, 'w')
      A_Wnt,b_Wnt= r.Amatrix_bs(stenc,WNTstenc)
      np.savetxt(f,A_Wnt,fmt='%i', delimiter='\t') #.astype(int)
      f.close()
      np.save(pathb_Wnt,b_Wnt)

    if os.path.isfile(pathA_Shh) and os.path.isfile(pathb_Shh):
        print("SHH: Reading %s as Amatrix and %s as b" % (pathA_Shh, pathb_Shh))
        lines = np.loadtxt(pathA_Shh)           
        A_Shh = (np.asarray(lines)).astype('int')
        b_Shh = np.load(pathb_Shh)   
    
    else:
      print("SHH: Creating %s and %s" % (pathA_Shh, pathb_Shh))
      g = open(pathA_Shh, 'w')   
      A_Shh,b_Shh= r.Amatrix_bs(stenc,SHHstenc)
      np.savetxt(g,A_Shh,fmt='%i', delimiter='\t') #.astype(int)  
      g.close() 
      np.save(pathb_Shh,b_Shh)
    
    
    return A_Wnt,b_Wnt,A_Shh,b_Shh
    


plt.close("all")
fig = plt.figure()

axRos = fig.add_subplot(2, 2, 1, projection='3d')
axDors = fig.add_subplot(2, 2, 2, projection='3d')
axWnt = fig.add_subplot(2, 2, 3, projection='3d')
axShh = fig.add_subplot(2, 2, 4, projection='3d')

"""
tube = r.tubecoords(xlen,ylen,zlen,bordersize=borders)
stenc = r.stencil(xlen,ylen,zlen,"stenc",tube) 
"""


saving_in = plot_saved_v.dataPath
os.mkdir(saving_in)

# pathA_Wnt = saving_in + '/'+ tubemodel.wntsecrpath + '_A.txt'
# pathA_Shh = saving_in + '/'+tubemodel.shhsecrpath + '_A.txt'
# pathb_Wnt = saving_in + '/'+tubemodel.wntsecrpath + '_b.npy'
# pathb_Shh = saving_in + '/'+tubemodel.shhsecrpath + '_b.npy'

pathA_Wnt =  tubemodel.wntsecrpath + '_A.txt'
pathA_Shh = tubemodel.shhsecrpath + '_A.txt'
pathb_Wnt = tubemodel.wntsecrpath + '_b.npy'
pathb_Shh = tubemodel.shhsecrpath + '_b.npy'

A_Wnt,b_Wnt,A_Shh,b_Shh = AmatrCheck(pathA_Wnt,pathA_Shh,pathb_Wnt,pathb_Shh,WNTstenc,SHHstenc,stenc) #create matrix A and vector b for matrix method

seeds = np.sum(stenc.grid)

"""#plot stencils
fug = plt.figure()
axStencWnt = fug.add_subplot(2, 1, 1, projection='3d')
axStencShh = fug.add_subplot(2, 1, 2, projection='3d')
axStencWnt.set_title("WNT secretion points")
axStencShh.set_title("SHH secretion points")


r.plotSecrStenc(WNTstenc,axStencWnt)
r.plotSecrStenc(SHHstenc,axStencShh)
"""


#rostrocaudal network grids
FB = Grid(xlen,ylen,zlen,"FB",seeds,0,ros.FB0)
MB = Grid(xlen,ylen,zlen,"MB",seeds,0,ros.MB0)
HB = Grid(xlen,ylen,zlen,"HB",seeds,0,ros.HB0)
GSK3 = Grid(xlen,ylen,zlen,"HB",seeds,0,ros.GSK30)
U = Grid(xlen,ylen,zlen,"U",seeds,0,ros.U0)
V = Grid(xlen,ylen,zlen,"U",seeds,0,ros.V0)

#dorsoventral network grids
G = Grid(xlen,ylen,zlen,"G",seeds,0,dors.Gli0)
P = Grid(xlen,ylen,zlen,"P",seeds,0,dors.P0)
O = Grid(xlen,ylen,zlen,"O",0,0,dors.O0)
N = Grid(xlen,ylen,zlen,"N",0,0,dors.N0)

#diffusion grids Shh and Wnt
Wnt = Grid(xlen,ylen,zlen,"Wnt",0,D_Wnt,Wnt0)
Shh = Grid(xlen,ylen,zlen,"Shh",0,D_Shh,Shh0)

networks = [FB,MB,HB,P,O,N,G,Wnt,Shh]

#secretion points
secretion(1,Wnt,Shh)




run(maxtime,saving_in,save=True)
plot_saved_v.makeMovie(saving_in,dt)


#animation = FuncAnimation(fig, update, interval=1)
#plt.show()
print('done')

