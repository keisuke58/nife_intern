from odbAccess import openOdb
for N in [8,16,32]:
    o=openOdb('column_DHm_N%d.odb'%N); fr=o.steps['Step-1'].frames[-1]
    pairs=sorted([(s.elementLabel,s.data[0]) for s in fr.fieldOutputs['S'].values])
    s11=[p[1] for p in pairs]
    # subsample to ~8 points for readability
    step=max(1,N//8)
    print("N=%2d:"%N, " ".join("%6.0f"%s11[i] for i in range(0,N,step)))
    o.close()
