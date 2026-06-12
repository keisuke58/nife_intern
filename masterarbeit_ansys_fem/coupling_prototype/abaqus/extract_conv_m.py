from odbAccess import openOdb
import numpy as np
mu=5000/(2*1.45)
grid=np.linspace(0.1,0.8,15); prof={}
for N in [4,8,16,32]:
    o=openOdb('column_DHm_N%d.odb'%N); fr=o.steps['Step-1'].frames[-1]
    pairs=sorted([(s.elementLabel,s.data[0]) for s in fr.fieldOutputs['S'].values])
    s11=np.array([p[1] for p in pairs]); zref=(np.arange(N)+0.5)/N
    prof[N]=np.interp(grid,zref,s11); o.close()
pk={N:np.max(np.abs(prof[N])) for N in prof}
print("MODERATE growth (GSCALE=0.25), interior grid z/H in [0.1,0.8]:")
print("  N   peak|S11|  /mu     pairwise   ||S-S32||/||S32||")
for N in [4,8,16,32]:
    ch='' if N==4 else ' %+.2f%%'%(100*(pk[N]-pk[N//2])/pk[N//2])
    d=np.linalg.norm(prof[N]-prof[32])/np.linalg.norm(prof[32])
    print("  %2d  %8.1f  %.3f %s    %.4f"%(N,pk[N],pk[N]/mu,ch,d))
