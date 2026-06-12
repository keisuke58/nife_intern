from odbAccess import openOdb
o=openOdb('one_element_phi_confined.odb')
fr=o.steps['Step-1'].frames[-1]
sv=fr.fieldOutputs['S'].values[0].data
lam_z=fr.fieldOutputs['SDV1'].values[0].data
print("lam_z = %.4f"%lam_z)
print("S11,S22,S33 = %.2f, %.2f, %.2f  (mu=E/2(1+nu)=%.1f)"%(sv[0],sv[1],sv[2],5000/(2*1.45)))
print("S33/mu = %.3f   (confined z-growth -> compressive)"%(sv[2]/(5000/(2*1.45))))
o.close()
