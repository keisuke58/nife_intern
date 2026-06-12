from odbAccess import openOdb
odb = openOdb('one_element_phi.odb')
fr = odb.steps['Step-1'].frames[-1]
U  = fr.fieldOutputs['U']
S  = fr.fieldOutputs['S']
SDV1 = fr.fieldOutputs['SDV1']; SDV2=fr.fieldOutputs['SDV2']; SDV3=fr.fieldOutputs['SDV3']
# top nodes 5-8 U3
u3=[]
for v in U.values:
    if v.nodeLabel in (5,6,7,8): u3.append(v.data[2])
import math
u3m=sum(u3)/len(u3)
sv=S.values[0].data   # S11 S22 S33 S12 S13 S23 (one int pt representative)
lam_z=SDV1.values[0].data; Je=SDV2.values[0].data; phi=SDV3.values[0].data
print("phi_Pg (SDV3)        = %.5f"%phi)
print("lam_z  (SDV1)        = %.5f   -> expected U3 = lam_z-1 = %.5f"%(lam_z, lam_z-1))
print("Je     (SDV2)        = %.5f   (free swelling -> ~lam_z)"%Je)
print("U3(top, mean)        = %.5f"%u3m)
print("S11,S22,S33          = %.4g, %.4g, %.4g"%(sv[0],sv[1],sv[2]))
print("max|S| (free swell -> ~0) = %.4g"%max(abs(x) for x in sv))
print("U3 error vs lam_z-1  = %.2e"%abs(u3m-(lam_z-1)))
odb.close()
