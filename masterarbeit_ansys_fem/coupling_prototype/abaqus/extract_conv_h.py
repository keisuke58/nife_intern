from odbAccess import openOdb
mu=5000/(2*1.45)
print("  N   peak_interior|S11|   |S11|/mu   pairwise   top(free-surf)")
prev=None
for N in [4,8,16,32]:
    o=openOdb('column_DHH_N%d.odb'%N)
    fr=o.steps['Step-1'].frames[-1]
    rows=sorted([(c.data[2], s.data[0]) for s,c in zip(fr.fieldOutputs['S'].values, fr.fieldOutputs['COORD'].values)])
    s11=[r[1] for r in rows]; interior=s11[:-1]
    pk=max(abs(x) for x in interior)
    chg='' if prev is None else '  %+.1f%%'%(100*(pk-prev)/prev)
    print("%3d   %12.1f    %7.3f %s    %9.1f"%(N,pk,pk/mu,chg,s11[-1]))
    prev=pk; o.close()
