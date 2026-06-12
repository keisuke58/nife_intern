from odbAccess import openOdb
import numpy as np
mu=5000/(2*1.45)
def botval(job,N):
    o=openOdb('%s.odb'%job); fr=o.steps['Step-1'].frames[-1]
    pairs=sorted([(s.elementLabel,s.data[0]) for s in fr.fieldOutputs['S'].values])
    s11=np.array([p[1] for p in pairs])
    o.close()
    nb=max(1,N//4)
    return s11[0], s11[:nb].mean()    # bottom element, bottom-quartile mean
for tag,patt in [("UNIFORM phi=0.20","column_flat_N%d"),("GRADED DH (moderate)","column_DHm_N%d")]:
    print("\n%s  -- confined bottom plateau (away from free surface):"%tag)
    print("  N   bottom_S11   /mu     bottomQ_mean   /mu     pairwise(bottomQ)")
    prev=None
    for N in [4,8,16,32]:
        b,q=botval(patt%N,N)
        ch='' if prev is None else ' %+.2f%%'%(100*(q-prev)/prev)
        print("  %2d  %9.1f  %.4f   %9.1f   %.4f %s"%(N,b,b/mu,q,q/mu,ch))
        prev=q
