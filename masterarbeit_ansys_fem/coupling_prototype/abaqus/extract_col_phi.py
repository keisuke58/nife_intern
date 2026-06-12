# abaqus python extract_col_phi.py column_phi_DH  -> writes column_phi_DH_profile.csv
from odbAccess import openOdb
import sys
job=sys.argv[1]
o=openOdb(job+'.odb')
fr=o.steps['Step-1'].frames[-1]
S=fr.fieldOutputs['S']; COORD=fr.fieldOutputs['COORD']; SDV3=fr.fieldOutputs['SDV3']; SDV1=fr.fieldOutputs['SDV1']
rows=[]
for s,c,p,l in zip(S.values,COORD.values,SDV3.values,SDV1.values):
    rows.append((c.data[2], s.data[0], s.data[2], p.data, l.data))  # z, S11, S33, phi, lam
rows.sort()
print("  z      S11      S33     phi_Pg   s(lam)")
for z,s11,s33,phi,lam in rows:
    print("%.3f  %8.2f %8.2f  %.4f   %.4f"%(z,s11,s33,phi,lam))
import io
f=open(job+'_profile.csv','w'); f.write('z,S11,S33,phi_Pg,lam\n')
for r in rows: f.write('%.4f,%.4f,%.4f,%.5f,%.5f\n'%r)
f.close()
s11=[r[1] for r in rows]
print('peak |S11| = %.1f  at z=%.2f'%(max(abs(x) for x in s11), rows[max(range(len(s11)),key=lambda i:abs(s11[i]))][0]))
print('S11 depth amplitude (max-min) = %.1f'%(max(s11)-min(s11)))
o.close()
