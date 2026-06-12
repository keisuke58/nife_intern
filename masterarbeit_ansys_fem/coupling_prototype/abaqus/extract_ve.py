from odbAccess import openOdb
import numpy as np
odb = openOdb('biofilm_viscoelastic.odb')
t0=0.0; rows=[]
for sname in ['Step-1','Step-2']:
    st=odb.steps[sname]
    rk=[k for k in st.historyRegions.keys() if 'Int Point 1' in k][0]
    s13=st.historyRegions[rk].historyOutputs['S13'].data
    for (t,v) in s13:
        rows.append((t0+t, v))
    t0+=st.timePeriod
rows=np.array(rows)
absS=np.abs(rows[:,1]); ipk=absS.argmax(); peak=absS[ipk]; final=rows[-1,1]
print("time[s]      S13       S13/peak")
step=max(1,len(rows)//18)
for t,v in rows[::step]:
    print("%10.2f  %9.5f   %7.4f" % (t,v,v/rows[ipk,1]))
print("---")
print("instantaneous |S13| (t=0+) = %.5f" % peak)
print("final S13 (t=4500s)        = %.5f" % final)
print("final/instantaneous        = %.4f   (theory g_inf = 0.0500)" % (final/rows[ipk,1]))
print("relaxed fraction           = %.4f   (theory 1-g_inf = 0.9500)" % (1-final/rows[ipk,1]))
np.savetxt('s13_relaxation.csv', rows, delimiter=',', header='time_s,S13', comments='')
print("wrote s13_relaxation.csv (%d pts)"%len(rows))
odb.close()
