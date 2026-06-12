from odbAccess import openOdb
import numpy as np
mu=5000/(2*1.45)
grid=np.linspace(0.1,0.8,15)   # interior reference-depth window (exclude free top & fixed bottom)
prof={}
for N in [4,8,16,32]:
    o=openOdb('column_DH_N%d.odb'%N)
    fr=o.steps['Step-1'].frames[-1]
    # element order = reference depth; sort by ELEMENT LABEL (bottom->top in our mesh)
    pairs=sorted([(s.elementLabel, s.data[0]) for s in fr.fieldOutputs['S'].values])
    s11=np.array([p[1] for p in pairs]); zref=(np.arange(N)+0.5)/N
    prof[N]=np.interp(grid, zref, s11)
    o.close()
print("Interp S11 on fixed interior reference grid z/H in [0.1,0.8]:")
print("  metric          N=4       N=8      N=16     N=32   |  16->32")
pk={N:np.max(np.abs(prof[N])) for N in prof}
l2={N:np.sqrt(np.mean(prof[N]**2)) for N in prof}
print("  peak|S11|    %8.1f %8.1f %8.1f %8.1f  |  %+.2f%%"%(pk[4],pk[8],pk[16],pk[32],100*(pk[32]-pk[16])/pk[16]))
print("  RMS S11      %8.1f %8.1f %8.1f %8.1f  |  %+.2f%%"%(l2[4],l2[8],l2[16],l2[32],100*(l2[32]-l2[16])/l2[16]))
print("  peak|S11|/mu %8.3f %8.3f %8.3f %8.3f"%(pk[4]/mu,pk[8]/mu,pk[16]/mu,pk[32]/mu))
# profile-to-profile L2 difference vs finest (N=32)
print("\nProfile convergence (||S11_N - S11_32||_2 / ||S11_32||_2):")
for N in [4,8,16]:
    d=np.linalg.norm(prof[N]-prof[32])/np.linalg.norm(prof[32])
    print("  N=%2d : %.4f"%(N,d))
