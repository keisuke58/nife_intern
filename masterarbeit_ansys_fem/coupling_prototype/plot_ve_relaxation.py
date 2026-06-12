import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
fe = np.loadtxt('abaqus/s13_relaxation.csv', delimiter=',', skiprows=1)
t, s13 = fe[:,0], fe[:,1]
ipk = np.abs(s13).argmax()
norm = s13/s13[ipk]
tprime = t - t[ipk]                      # relaxation clock starts when strain is fully applied
P = json.load(open('prony_biofilm.json'))
g_inf = 0.05
ssum = sum(q['g_i'] for q in P['prony'])
gi = [p['g_i']/ssum*(1-g_inf) for p in P['prony']]
ti = [p['tau_i_s'] for p in P['prony']]
g = lambda x: g_inf + sum(gg*np.exp(-x/tau) for gg,tau in zip(gi,ti))
tt = np.logspace(-2, np.log10(tprime.max()), 400)
m = tprime > 1e-6
fig, ax = plt.subplots(figsize=(5.0,3.4))
ax.semilogx(tt, g(tt), '-', color='C3', lw=2, label='analytic Prony  g(t)')
ax.semilogx(tprime[m], norm[m], 'o', ms=4, mfc='none', color='C0', label='Abaqus FE  $S_{13}/S_{13}(0^+)$')
ax.axhline(g_inf, ls=':', color='gray', lw=1); ax.text(0.02, g_inf+0.02, r'$g_\infty=0.05$', fontsize=8, color='gray')
for tau,lab in [(ti[0],r'$\tau_K$'),(ti[1],r'$\tau_M$')]:
    ax.axvline(tau, ls='--', color='gray', lw=0.7, alpha=0.6); ax.text(tau*1.05,0.92,lab,fontsize=8,color='gray')
ax.set_xlabel('relaxation time  $t-t_0$  [s]'); ax.set_ylabel('normalized shear stress')
ax.set_title('Biofilm shear-stress relaxation (Burgers$\\to$Prony)', fontsize=10)
ax.legend(fontsize=8, frameon=False); ax.set_ylim(0,1.05); ax.grid(alpha=0.25, which='both')
fig.tight_layout(); fig.savefig('fig_ve_relaxation.pdf'); fig.savefig('fig_ve_relaxation.png', dpi=150)
rms = float(np.sqrt(np.mean((norm[m]-g(tprime[m]))**2)))
print('rms(FE-theory, shifted origin) = %.4f'%rms)
print('wrote fig_ve_relaxation.{pdf,png}')
