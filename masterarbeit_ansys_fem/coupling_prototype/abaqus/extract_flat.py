from odbAccess import openOdb
import numpy as np
mu=5000/(2*1.45); prev=None
print("UNIFORM phi_Pg=0.20 (moderate), interior plateau S11 (exclude top free-surf element):")
print("  N   interior_mean_S11   /mu    interior_std   pairwise   top(free)")
for N in [4,8,16,32]:
    o=openOdb('column_flat_N%d.odb'%N); fr=o.steps['Step-1'].frames[-1]
    pairs=sorted([(s.elementLabel,s.data[0]) for s in fr.fieldOutputs['S'].values])
    s11=np.array([p[1] for p in pairs])
    interior=s11[:-1]                    # drop top free-surface element
    m=interior.mean(); sd=interior.std()
    ch='' if prev is None else ' %+.2f%%'%(100*(m-prev)/prev)
    print("  %2d   %10.2f   %.4f   %8.2f %s   %8.1f"%(N,m,m/mu,sd,ch,s11[-1]))
    prev=m; o.close()
