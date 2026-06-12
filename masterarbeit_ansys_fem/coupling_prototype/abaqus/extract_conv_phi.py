from odbAccess import openOdb
mu=5000/(2*1.45)
print("  N    <|S11|>(vol-avg)   |S11|/mu    peak_mid[0.1-0.6H]   top(free-surf)")
prevA=prevP=None
for N in [4,8,16,32]:
    o=openOdb('column_DH_N%d.odb'%N)
    fr=o.steps['Step-1'].frames[-1]
    rows=sorted([(c.data[2], s.data[0]) for s,c in zip(fr.fieldOutputs['S'].values, fr.fieldOutputs['COORD'].values)])
    zmax=rows[-1][0]
    s11=[r[1] for r in rows]
    avg=sum(abs(x) for x in s11)/len(s11)                       # volume-average |S11|
    mid=[r[1] for r in rows if 0.1*zmax<=r[0]<=0.6*zmax]        # interior window (no boundary layers)
    pk=max(abs(x) for x in mid) if mid else 0
    cA = '' if prevA is None else ' (%+.1f%%)'%(100*(avg-prevA)/prevA)
    cP = '' if prevP is None else ' (%+.1f%%)'%(100*(pk-prevP)/prevP)
    print("%3d   %12.1f%s   %7.3f   %12.1f%s   %9.1f"%(N,avg,cA,avg/mu,pk,cP,s11[-1]))
    prevA,prevP=avg,pk
    o.close()
