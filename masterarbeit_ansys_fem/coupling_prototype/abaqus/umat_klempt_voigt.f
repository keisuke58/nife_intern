c =====================================================================
c  umat_klempt_voigt.f  --  Option A: Klempt UMAT + Voigt E(phi_i) v2
c
c  Fixes academic hole #1 (phi^2-gated stiffness):
c    Klempt 2024 Eq.20: Psi = phi^2 * mu/2 * (I:Ce-3) + ...
c    => E_eff = phi_local^2 / phi_total^2 * E_Voigt   (Klempt exact)
c    phi_local varies by depth (inner face=full biofilm, outer=planktonic)
c    phi_local provided via PREDEF(7) (field variable 7)
c
c  Constitutive model:
c    F = Fe . Fg   (multiplicative growth split, Klempt 2024)
c    Fg = s * I    (isotropic, s = 1 + alpha)
c    alpha     = PREDEF(1)    [ramped via *Field + AMPLITUDE]
c    phi_i     = PREDEF(2..6) [TMCMC species fractions, fixed per condition]
c    phi_local = PREDEF(7)    [depth-varying total phi from PDE, fixed]
c    E_Voigt   = sum_i phi_i * E_i               [condition-level Voigt]
c    E_gated   = (phi_local/phi_total)^2 * E_Voigt  [Klempt Eq.20 exact]
c    nu        = PROPS(1) = 0.49
c
c  Neo-Hookean on elastic part:
c    W = (mu/2)*(tr(be)-3) + (lam/2)*ln(Je)^2 - mu*ln(Je)
c    sigma = ((lam*lnJe - mu)*I + mu*be) / Je
c
c  Species stiffness [MPa]:
c    So=1e-3, An=8e-4, Vd=6e-4, Fn=2e-4, Pg=1e-5
c
c  PROPS(1) = nu = 0.49
c  CONSTANTS=1
c
c  PREDEF (field variables):
c    PREDEF(1) = alpha_total  (ramped via *Field amplitude)
c    PREDEF(2) = phi_So, PREDEF(3)=phi_An, PREDEF(4)=phi_Vd
c    PREDEF(5) = phi_Fn, PREDEF(6) = phi_Pg
c    PREDEF(7) = phi_local_total  (depth-varying, fixed)
c
c  DEPVAR 5:
c    SDV1 = s = 1+alpha
c    SDV2 = Je = det(Fe)
c    SDV3 = alpha
c    SDV4 = E_gated [MPa]   (phi^2-gated Voigt E)
c    SDV5 = phi_gate = (phi_local/phi_total)^2
c
c  References:
c    Klempt 2024 BMMB Eq.20 (phi^2-gated energy density)
c    Phase 3b Voigt: E_SPECIES = {So:1e3, An:8e2, Vd:6e2, Fn:2e2, Pg:10} Pa
c =====================================================================
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,
     2 PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,
     3 DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 TIME(2),PREDEF(*),DPRED(*),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)

      real*8 nu, lam, mu_sh
      real*8 E_voigt, E_voigt_norm, E_gated, phi_gate
      real*8 alpha, s_iso
      real*8 phi_So, phi_An, phi_Vd, phi_Fn, phi_Pg
      real*8 phi_total_cond, phi_local
      real*8 fe(3,3), be(6), xi(6), detfe, lnJe
      integer i, j

c --- species stiffness [MPa] (tooth mesh units: mm/MPa/N) ----------------
      real*8 E_So, E_An, E_Vd, E_Fn, E_Pg
      parameter (E_So = 1.0d-3)
      parameter (E_An = 8.0d-4)
      parameter (E_Vd = 6.0d-4)
      parameter (E_Fn = 2.0d-4)
      parameter (E_Pg = 1.0d-5)

      data xi/1.d0,1.d0,1.d0,0.d0,0.d0,0.d0/

c --- Poisson ratio -------------------------------------------------------
      nu = PROPS(1)

c === SPECIES FRACTIONS (PREDEF 2..6) — TMCMC condition-level fractions ===
      phi_So = max(0.d0, PREDEF(2) + DPRED(2))
      phi_An = max(0.d0, PREDEF(3) + DPRED(3))
      phi_Vd = max(0.d0, PREDEF(4) + DPRED(4))
      phi_Fn = max(0.d0, PREDEF(5) + DPRED(5))
      phi_Pg = max(0.d0, PREDEF(6) + DPRED(6))

c === Voigt E (condition-level, at phi_total=1) [MPa] ====================
      E_voigt = phi_So*E_So + phi_An*E_An + phi_Vd*E_Vd
     #        + phi_Fn*E_Fn + phi_Pg*E_Pg
      if (E_voigt .lt. 1.0d-12) E_voigt = 1.0d-12

c === phi^2-GATED E (Klempt Eq.20: Psi = phi^2 * mu/2 * ...) ============
c     phi_local in [0,1]: PDE total biofilm density (depth-varying, PREDEF 7)
c     phi_total_cond: TMCMC sum(phi_i), may be <1 (not all volume is biofilm)
c     E_voigt_norm: Voigt E normalized to phi_total=1 (E at full coverage)
c     phi_gate = phi_local^2 in [0,1]  (Klempt Eq.20 exact)
      phi_local      = min(1.0d0, max(0.d0, PREDEF(7) + DPRED(7)))
      phi_total_cond = phi_So + phi_An + phi_Vd + phi_Fn + phi_Pg
      E_voigt_norm   = E_voigt / max(phi_total_cond, 1.0d-10)
      phi_gate       = phi_local * phi_local
      E_gated        = phi_gate * E_voigt_norm
      if (E_gated .lt. 1.0d-12) E_gated = 1.0d-12

c === Lame constants from gated E =========================================
      mu_sh = E_gated / (2.d0*(1.d0+nu))
      lam   = E_gated*nu / ((1.d0+nu)*(1.d0-2.d0*nu))

c === GROWTH VARIABLE alpha (PREDEF 1, ramped via *Field) ================
      alpha = max(0.d0, PREDEF(1) + DPRED(1))
      s_iso = 1.d0 + alpha

c === Fe = F / s ==========================================================
      do i=1,3
        do j=1,3
          fe(i,j) = DFGRD1(i,j) / s_iso
        end do
      end do

c === Je = det(Fe) ===========================================================
      detfe = +fe(1,1)*(fe(2,2)*fe(3,3)-fe(2,3)*fe(3,2))
     #        -fe(1,2)*(fe(2,1)*fe(3,3)-fe(2,3)*fe(3,1))
     #        +fe(1,3)*(fe(2,1)*fe(3,2)-fe(2,2)*fe(3,1))
      if (detfe .lt. 1.d-15) detfe = 1.d-15
      lnJe = dlog(detfe)

c === be = Fe.Fe^T (Voigt: 11 22 33 12 13 23) ================================
      be(1) = fe(1,1)*fe(1,1)+fe(1,2)*fe(1,2)+fe(1,3)*fe(1,3)
      be(2) = fe(2,1)*fe(2,1)+fe(2,2)*fe(2,2)+fe(2,3)*fe(2,3)
      be(3) = fe(3,1)*fe(3,1)+fe(3,2)*fe(3,2)+fe(3,3)*fe(3,3)
      be(4) = fe(1,1)*fe(2,1)+fe(1,2)*fe(2,2)+fe(1,3)*fe(2,3)
      be(5) = fe(1,1)*fe(3,1)+fe(1,2)*fe(3,2)+fe(1,3)*fe(3,3)
      be(6) = fe(2,1)*fe(3,1)+fe(2,2)*fe(3,2)+fe(2,3)*fe(3,3)

c === Cauchy stress ===========================================================
      do i=1,NTENS
        STRESS(i) = ((lam*lnJe-mu_sh)*xi(i) + mu_sh*be(i)) / detfe
      end do

c === Consistent tangent ======================================================
      do i=1,NTENS
        do j=1,NTENS
          DDSDDE(i,j) = 0.d0
        end do
      end do
      DDSDDE(1,1)=(lam-2.d0*(lam*lnJe-mu_sh))/detfe+2.d0*STRESS(1)
      DDSDDE(2,2)=(lam-2.d0*(lam*lnJe-mu_sh))/detfe+2.d0*STRESS(2)
      DDSDDE(3,3)=(lam-2.d0*(lam*lnJe-mu_sh))/detfe+2.d0*STRESS(3)
      DDSDDE(1,2)=lam/detfe
      DDSDDE(1,3)=lam/detfe
      DDSDDE(2,3)=lam/detfe
      DDSDDE(1,4)=STRESS(4)
      DDSDDE(2,4)=STRESS(4)
      DDSDDE(3,4)=0.d0
      DDSDDE(4,4)=-(lam*lnJe-mu_sh)/detfe+(STRESS(1)+STRESS(2))/2.d0
      if (NTENS.eq.6) then
        DDSDDE(1,5)=STRESS(5)
        DDSDDE(2,5)=0.d0
        DDSDDE(3,5)=STRESS(5)
        DDSDDE(1,6)=0.d0
        DDSDDE(2,6)=STRESS(6)
        DDSDDE(3,6)=STRESS(6)
        DDSDDE(5,5)=-(lam*lnJe-mu_sh)/detfe+(STRESS(1)+STRESS(3))/2.d0
        DDSDDE(6,6)=-(lam*lnJe-mu_sh)/detfe+(STRESS(2)+STRESS(3))/2.d0
        DDSDDE(4,5)=STRESS(6)/2.d0
        DDSDDE(4,6)=STRESS(5)/2.d0
        DDSDDE(5,6)=STRESS(4)/2.d0
      end if
      do i=2,NTENS
        do j=1,i-1
          DDSDDE(i,j)=DDSDDE(j,i)
        end do
      end do

c === State variables =========================================================
      SSE = (lam*lnJe**2
     #      + mu_sh*(be(1)+be(2)+be(3)-3.d0-2.d0*lnJe)) / 2.d0
      if (NSTATV.ge.1) STATEV(1) = s_iso
      if (NSTATV.ge.2) STATEV(2) = detfe
      if (NSTATV.ge.3) STATEV(3) = alpha
      if (NSTATV.ge.4) STATEV(4) = E_gated
      if (NSTATV.ge.5) STATEV(5) = phi_gate

      RETURN
      END
