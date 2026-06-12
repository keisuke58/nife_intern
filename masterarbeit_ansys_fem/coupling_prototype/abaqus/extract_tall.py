from odbAccess import openOdb
import numpy as np, json, sys
mu=5000/(2*1.45); H=4.0
def get(job,cond):
    o=openOdb('%s.odb'%job); fr=o.steps['Step-1'].frames[-1]
    rows=sorted([(s.elementLabel,s.data[0],s.data[2]) for s in fr.fieldOutputs['S'].values])
    o.close(); N=len(rows)
    zref=(np.arange(N)+0.5)/N                      # reference depth z/H in [0,1]
    s11=np.array([r[1] for r in rows]); s33=np.array([r[2] for r in rows])
    p=json.load(open('phipg_depth_%s.json'%cond)); phi=np.interp(zref,p['z_norm'],p['phi_Pg'])
    return zref,s11,s33,phi
for cond,job in [('DH','column_DHtall'),('CH','column_CHtall')]:
    z,s11,s33,phi=get(job,cond)
    f=open('coltall_%s.csv'%cond,'w'); f.write('zref,S11,S33,phi,S11_mu\n')
    for i in range(len(z)): f.write('%.5f,%.4f,%.4f,%.5f,%.5f\n'%(z[i],s11[i],s33[i],phi[i],s11[i]/mu))
    f.close()
    # interior window: exclude bottom 10% (fixed BC) and top 25% (free-surface relaxation ~1 width=0.25H)
    m=(z>=0.10)&(z<=0.75)
    r=np.corrcoef(s11[m],phi[m])[0,1]
    print('%s: interior[0.10-0.75H] corr(S11,phi)=%+.3f  peak|S11|/mu=%.2f  mean|S11|/mu=%.2f  (top-relax z>0.75)'%(
        cond,r,np.abs(s11[m]).max()/mu,np.abs(s11[m]).mean()/mu))
