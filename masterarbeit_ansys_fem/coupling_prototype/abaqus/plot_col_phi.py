import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
mu=5000/(2*1.45)
def load(j):
    d=np.genfromtxt(j+'_profile.csv',delimiter=',',names=True)
    n=len(d); zref=(np.arange(n)+0.5)/n   # reference (undeformed) normalized depth
    return zref, d['S11']/mu, d['phi_Pg']
zD,sD,pD=load('column_phi_DH'); zC,sC,pC=load('column_phi_CH')
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.4,3.6),sharey=True)
# left: residual stress sigma_xx/mu vs depth
a1.plot(sD,zD,'o-',color='C3',label='DH (dysbiotic)')
a1.plot(sC,zC,'s--',color='C0',label='CH (commensal)')
a1.set_xlabel(r'residual stress  $\sigma_{xx}/\mu$'); a1.set_ylabel('reference depth  $z/H$')
a1.set_title('Socket-free residual-stress column',fontsize=10)
a1.legend(fontsize=8,frameon=False); a1.grid(alpha=0.25); a1.invert_yaxis()
# right: phi_Pg(z) driver
a2.plot(pD,zD,'o-',color='C3',label='DH'); a2.plot(pC,zC,'s--',color='C0',label='CH')
a2.set_xlabel(r'composition  $\phi_{Pg}(z)$'); a2.set_title('Growth driver (CLSM)',fontsize=10)
a2.grid(alpha=0.25)
fig.suptitle(r'$\sigma_{xx}(z)$ tracks $\phi_{Pg}(z)$  (umat\_growth\_phi.f, field-driven, $\gamma=1$)',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('../fig_column_phi_DHvsCH.pdf'); fig.savefig('../fig_column_phi_DHvsCH.png',dpi=150)
print('DH peak sxx/mu=%.2f amp=%.2f | CH peak=%.2f amp=%.2f'%(
  abs(sD).max(),sD.max()-sD.min(),abs(sC).max(),sC.max()-sC.min()))
print('Pearson corr sigma_xx vs phi_Pg: DH=%.3f CH=%.3f'%(
  np.corrcoef(sD,pD)[0,1],np.corrcoef(sC,pC)[0,1]))
print('wrote fig_column_phi_DHvsCH.{pdf,png}')
