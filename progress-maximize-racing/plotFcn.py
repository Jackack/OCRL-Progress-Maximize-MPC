from tracks.readDataFcn import getTrack
from time2spatial import transformProj2Orig,transformOrig2Proj
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plotTrackProj(simX,filename='buggy_track.txt', T_opt=None):
    # load track
    s=simX[:,0]
    n=simX[:,1]
    e_psi=simX[:,2]
    v = np.linalg.norm(simX[:,3:5], axis=1)


    distance=10
    # transform data
    [x, y, _, _] = transformProj2Orig(s, n, e_psi, v,filename)
    # plot racetrack map

    #Setup plot
    plt.figure()
    plt.ylim(bottom=0,top=700)
    plt.xlim(left=0,right=700)
    plt.ylabel('y[m]')
    plt.xlabel('x[m]')

    # Plot center line
    [_,Xref,Yref,Psiref,_]=getTrack(filename)
    plt.plot(Xref,Yref,'--',color='k')

    # Draw Trackboundaries
    Xboundleft=Xref-distance*np.sin(Psiref)
    Yboundleft=Yref+distance*np.cos(Psiref)
    Xboundright=Xref+distance*np.sin(Psiref)
    Yboundright=Yref-distance*np.cos(Psiref)
    plt.plot(Xboundleft,Yboundleft,color='k',linewidth=1)
    plt.plot(Xboundright,Yboundright,color='k',linewidth=1)
    plt.plot(x,y, '-b')

    # Draw driven trajectory
    heatmap = plt.scatter(x,y, c=v, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("velocity in [m/s]")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.step(t, simU[:,:])
    plt.title('closed-loop simulation')
    plt.legend(['a','ddelta'])
    plt.ylabel('u')
    plt.xlabel('t')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.step(t, simX[:,6], color='b')
    plt.legend(['delta'])
    plt.ylabel('steering')
    plt.xlabel('t')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, simX[:,1:3])
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend(['r','e_psi'])
    plt.grid(True)

def plotalat(simX,simU,constraint,t):
    Nsim=t.shape[0]
    plt.figure()
    alat=np.zeros(Nsim)
    for i in range(Nsim):
        alat[i]=constraint.alat(simX[i,:],simU[i,:])
    plt.plot(t,alat)
    plt.plot([t[0],t[-1]],[constraint.alat_min, constraint.alat_min],'k--')
    plt.plot([t[0],t[-1]],[constraint.alat_max, constraint.alat_max],'k--')
    plt.legend(['alat','alat_min/max'])
    plt.xlabel('t')
    plt.ylabel('alat[m/s^2]')

