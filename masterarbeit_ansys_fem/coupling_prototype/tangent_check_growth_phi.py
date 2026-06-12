#!/usr/bin/env python3
"""Finite-difference consistency check of the umat_growth_phi.f consistent tangent (DDSDDE).

The socket-free UMAT supplies the mholla/Holland "elastic + geometric" Jaumann tangent. This script
replicates the UMAT's Cauchy stress and DDSDDE in numpy (bit-for-bit the same algebra) and verifies
the analytic 6x6 against a finite-difference tangent, using the Abaqus convention:

    DDSDDE[i,j] = d(sigma_i)/d(eps_j)        eps = [e11,e22,e33, g12,g13,g23]  (engineering shear)

FD column j: perturb the *current* configuration by a small symmetric strain dE_j (Sun et al. 2008),
F+ = (I + dE_j) F, recompute sigma, column = (sigma(F+) - sigma(F)) / h. This is exactly the
Jaumann-rate tangent Abaqus expects, so a correct geometric tangent matches it to ~O(h).

Checked at several random F and growth states (mode 0 depth and mode 1 iso volume-matched).
"""
import numpy as np

np.random.seed(0)
I3 = np.eye(3)
# Voigt index pairs (Abaqus order): 11 22 33 12 13 23
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def lame(E, nu):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lam, mu


def Fg_of(phi, beta, ic, mode):
    growth = max(0.0, beta * phi + ic)
    lam_z = 1.0 + growth
    if mode == 1:                       # iso, volume-matched: s = Jg^(1/3)
        s = lam_z ** (1.0 / 3.0)
        return np.diag([s, s, s])
    return np.diag([1.0, 1.0, lam_z])   # depth (z)


def cauchy(F, Fg, E, nu):
    """Cauchy stress as the UMAT computes it (Voigt 6: 11 22 33 12 13 23)."""
    lam, mu = lame(E, nu)
    Fe = F @ np.linalg.inv(Fg)
    Je = np.linalg.det(Fe)
    be = Fe @ Fe.T
    lnJe = np.log(Je)
    xi = np.array([1, 1, 1, 0, 0, 0.0])
    beV = np.array([be[0, 0], be[1, 1], be[2, 2], be[0, 1], be[0, 2], be[1, 2]])
    return ((lam * lnJe - mu) * xi + mu * beV) / Je


def ddsdde_analytic(F, Fg, E, nu):
    """The mholla geometric tangent exactly as in umat_growth_phi.f."""
    lam, mu = lame(E, nu)
    Fe = F @ np.linalg.inv(Fg)
    Je = np.linalg.det(Fe)
    lnJe = np.log(Je)
    s = cauchy(F, Fg, E, nu)                 # s[0..5]
    C = np.zeros((6, 6))
    a = (lam - 2.0 * (lam * lnJe - mu)) / Je
    C[0, 0] = a + 2 * s[0]; C[1, 1] = a + 2 * s[1]; C[2, 2] = a + 2 * s[2]
    C[0, 1] = C[1, 0] = lam / Je
    C[0, 2] = C[2, 0] = lam / Je
    C[1, 2] = C[2, 1] = lam / Je
    C[0, 3] = C[3, 0] = s[3]; C[1, 3] = C[3, 1] = s[3]; C[2, 3] = C[3, 2] = 0.0
    C[3, 3] = -(lam * lnJe - mu) / Je + (s[0] + s[1]) / 2.0
    C[0, 4] = C[4, 0] = s[4]; C[1, 4] = C[4, 1] = 0.0; C[2, 4] = C[4, 2] = s[4]
    C[0, 5] = C[5, 0] = 0.0; C[1, 5] = C[5, 1] = s[5]; C[2, 5] = C[5, 2] = s[5]
    C[4, 4] = -(lam * lnJe - mu) / Je + (s[0] + s[2]) / 2.0
    C[5, 5] = -(lam * lnJe - mu) / Je + (s[1] + s[2]) / 2.0
    C[3, 4] = C[4, 3] = s[5] / 2.0
    C[3, 5] = C[5, 3] = s[4] / 2.0
    C[4, 5] = C[5, 4] = s[3] / 2.0
    return C


def _Jsigma(F, Fg, E, nu):
    """Kirchhoff stress tau = J*sigma (3x3) — the correct measure for the Jaumann FD."""
    lam, mu = lame(E, nu)
    Fe = F @ np.linalg.inv(Fg)
    Je = np.linalg.det(Fe)
    b = Fe @ Fe.T
    sig = (mu / Je) * (b - I3) + (lam / Je) * np.log(Je) * I3
    return Je * sig, Je


def ddsdde_fd(F, Fg, E, nu, h=1e-7):
    """Consistent Abaqus (Jaumann) tangent by FD: C = (1/J) d(J*sigma)/dE with a symmetric
    stretch perturbation F+ = (I + dE) F. Using the Kirchhoff stress tau = J*sigma is essential —
    a naive Cauchy-stress dsigma/dE differs from the Jaumann tangent by O(sigma) (objective-rate
    convention) and is NOT a valid check at finite stress."""
    tau0, J0 = _Jsigma(F, Fg, E, nu)
    C = np.zeros((6, 6))
    for j, (a, b) in enumerate(PAIRS):
        dE = np.zeros((3, 3))
        if a == b:
            dE[a, a] = h
        else:
            dE[a, b] = dE[b, a] = h / 2.0      # engineering shear: g = 2*e = h
        Fp = (I3 + dE) @ F
        taup, _ = _Jsigma(Fp, Fg, E, nu)
        dtau = (taup - tau0) / h / J0
        C[:, j] = [dtau[0, 0], dtau[1, 1], dtau[2, 2], dtau[0, 1], dtau[0, 2], dtau[1, 2]]
    return C


def run():
    E, nu = 5000.0, 0.45
    cases = [
        ("mode0 depth, phi=0.20", 0, 0.20, 17.18, -1.337),
        ("mode1 iso,   phi=0.30", 1, 0.30, 17.18, -1.337),
        ("mode1 iso,   phi=0.34", 1, 0.34, 17.18, -1.337),
        ("mode0 depth, CH phi=0.18", 0, 0.18, -10.41, 2.166),
    ]
    print("FD consistency of umat_growth_phi.f tangent (Abaqus Jaumann convention):")
    print("  %-26s  max|C_an-C_fd|   rel-err" % "case (random F)")
    worst = 0.0
    for name, mode, phi, beta, ic in cases:
        F = I3 + 0.08 * (np.random.rand(3, 3) - 0.5)     # random moderate deformation
        Fg = Fg_of(phi, beta, ic, mode)
        Ca = ddsdde_analytic(F, Fg, E, nu)
        Cf = ddsdde_fd(F, Fg, E, nu)
        amax = np.abs(Ca - Cf).max()
        rel = amax / np.abs(Ca).max()
        worst = max(worst, rel)
        print("  %-26s  %12.4e   %9.2e" % (name, amax, rel))
    print("  ---\n  worst rel-err = %.2e  ->  %s" % (
        worst, "PASS (consistent geometric tangent)" if worst < 1e-4 else "CHECK"))


if __name__ == "__main__":
    run()
