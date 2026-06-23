c =====================================================================
c  umat_klempt2025.f  --  Option C: Klempt 2025 multi-species UMAT
c
c  Extension of Klempt 2024 to N_SP=5 species (Klempt 2025 arXiv:2509.01274).
c
c  Multi-species growth decomposition:
c    Each species s has its own growth variable alpha_s (from per-species PDE).
c    Additive total growth (Klempt 2025 additive Mandel decomposition):
c      alpha_total = sum_s alpha_s
c      Fg = (1 + alpha_total) * I       (isotropic, additive species growth)
c      Fe = F * Fg^{-1} = F / (1 + alpha_total)
c
c  Voigt stiffness:
c      E_Voigt = sum_s phi_s * E_s      (species-composition-dependent stiffness)
c
c  Full constitutive chain:
c      F = Fe * Fg  →  Fe = F/s, s=1+alpha_total
c      W = (mu/2)*(trbe-3) + (lam/2)*(lnJe)^2 - mu*lnJe
c      sigma = ((lam*lnJe-mu)*I + mu*be) / Je
c
c  Closes ALL academic holes:
c    [Hole 1-3] alpha_s from per-species PDE (klempt_pde_multispecies.py)
c    [Hole 4]   multi-species UMAT (N=5 species, Klempt 2025 formulation)
c    [Hole 5]   E = Voigt(phi_i) not constant
c    [Hole 6]   alpha_s PDE uses exact Eq.36 (no Monod in alpha equation)
c
c  PREDEF (10 field variables):
c    PREDEF(1)  = alpha_So   (from klempt_alpha_s_final_{cond}_So.npy)
c    PREDEF(2)  = alpha_An
c    PREDEF(3)  = alpha_Vd
c    PREDEF(4)  = alpha_Fn
c    PREDEF(5)  = alpha_Pg
c    PREDEF(6)  = phi_So     (TMCMC species fraction, uniform per condition)
c    PREDEF(7)  = phi_An
c    PREDEF(8)  = phi_Vd
c    PREDEF(9)  = phi_Fn
c    PREDEF(10) = phi_Pg
c
c  PROPS(1) = nu  (Poisson ratio = 0.49, fixed from Klempt Table 2)
c  CONSTANTS=1
c
c  DEPVAR 6:
c    SDV1 = alpha_total = sum_s alpha_s
c    SDV2 = Je = det(Fe)
c    SDV3 = E_Voigt [MPa]
c    SDV4 = s = 1 + alpha_total (growth stretch)
c    SDV5 = alpha_So   (largest contributor in commensal)
c    SDV6 = alpha_Vd   (largest contributor in dysbiotic HOBiC)
c
c  References:
c    Klempt et al. 2024 BMMB (F=FeFg, neo-Hookean, Table 2)
c    Klempt et al. 2025 arXiv:2509.01274 (multi-species additive extension)
c    Phase 3b Voigt stiffness: E_SPECIES = {So:1000,An:800,Vd:600,Fn:200,Pg:10} Pa
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

      real*8 nu, lam, mu_sh, E_voigt
      real*8 alpha_So, alpha_An, alpha_Vd, alpha_Fn, alpha_Pg
      real*8 phi_So, phi_An, phi_Vd, phi_Fn, phi_Pg
      real*8 alpha_total, s_iso
      real*8 fe(3,3), be(6), xi(6), detfe, lnJe
      integer i, j

c --- species stiffness [MPa] (tooth mesh units: mm/MPa) -------------------
      real*8 E_So, E_An, E_Vd, E_Fn, E_Pg
      parameter (E_So = 1.0d-3)   ! 1000 Pa
      parameter (E_An = 8.0d-4)   !  800 Pa
      parameter (E_Vd = 6.0d-4)   !  600 Pa
      parameter (E_Fn = 2.0d-4)   !  200 Pa
      parameter (E_Pg = 1.0d-5)   !   10 Pa

      data xi/1.d0,1.d0,1.d0,0.d0,0.d0,0.d0/

c --- Poisson ratio (fixed: Klempt Table 2 nu=0.49) ----------------------
      nu = PROPS(1)

c === READ PER-SPECIES GROWTH VARIABLES (PREDEF 1..5) ====================
c     alpha_s from condition-specific Klempt per-species PDE (Option B+C)
      alpha_So = max(0.d0, PREDEF(1)  + DPRED(1))
      alpha_An = max(0.d0, PREDEF(2)  + DPRED(2))
      alpha_Vd = max(0.d0, PREDEF(3)  + DPRED(3))
      alpha_Fn = max(0.d0, PREDEF(4)  + DPRED(4))
      alpha_Pg = max(0.d0, PREDEF(5)  + DPRED(5))

c === READ SPECIES FRACTIONS (PREDEF 6..10) ================================
      phi_So = max(0.d0, PREDEF(6)  + DPRED(6))
      phi_An = max(0.d0, PREDEF(7)  + DPRED(7))
      phi_Vd = max(0.d0, PREDEF(8)  + DPRED(8))
      phi_Fn = max(0.d0, PREDEF(9)  + DPRED(9))
      phi_Pg = max(0.d0, PREDEF(10) + DPRED(10))

c === VOIGT STIFFNESS ======================================================
      E_voigt = phi_So*E_So + phi_An*E_An + phi_Vd*E_Vd
     #        + phi_Fn*E_Fn + phi_Pg*E_Pg
      if (E_voigt .lt. 1.0d-10) E_voigt = 1.0d-10
      mu_sh = E_voigt / (2.d0*(1.d0+nu))
      lam   = E_voigt*nu / ((1.d0+nu)*(1.d0-2.d0*nu))

c === TOTAL GROWTH (Klempt 2025 additive Mandel decomposition) ============
c     alpha_total = sum_s alpha_s
c     Fg = (1 + alpha_total) * I  (isotropic, commuting species Fg_s)
      alpha_total = alpha_So + alpha_An + alpha_Vd + alpha_Fn + alpha_Pg
      s_iso       = 1.d0 + alpha_total

c === ELASTIC DEFORMATION GRADIENT Fe = F . Fg^{-1} = F / s ==============
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

c === LEFT CAUCHY-GREEN be = Fe.Fe^T (Voigt notation: 11 22 33 12 13 23) ===
      be(1) = fe(1,1)*fe(1,1)+fe(1,2)*fe(1,2)+fe(1,3)*fe(1,3)
      be(2) = fe(2,1)*fe(2,1)+fe(2,2)*fe(2,2)+fe(2,3)*fe(2,3)
      be(3) = fe(3,1)*fe(3,1)+fe(3,2)*fe(3,2)+fe(3,3)*fe(3,3)
      be(4) = fe(1,1)*fe(2,1)+fe(1,2)*fe(2,2)+fe(1,3)*fe(2,3)
      be(5) = fe(1,1)*fe(3,1)+fe(1,2)*fe(3,2)+fe(1,3)*fe(3,3)
      be(6) = fe(2,1)*fe(3,1)+fe(2,2)*fe(3,2)+fe(2,3)*fe(3,3)

c === CAUCHY STRESS (neo-Hookean, Voigt E) ==================================
      do i=1,NTENS
        STRESS(i) = ((lam*lnJe-mu_sh)*xi(i)+mu_sh*be(i)) / detfe
      end do

c === CONSISTENT TANGENT =====================================================
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

c === STATE VARIABLES ========================================================
      SSE = (lam*lnJe**2
     #      + mu_sh*(be(1)+be(2)+be(3)-3.d0-2.d0*lnJe)) / 2.d0
      if (NSTATV.ge.1) STATEV(1) = alpha_total
      if (NSTATV.ge.2) STATEV(2) = detfe
      if (NSTATV.ge.3) STATEV(3) = E_voigt
      if (NSTATV.ge.4) STATEV(4) = s_iso
      if (NSTATV.ge.5) STATEV(5) = alpha_So
      if (NSTATV.ge.6) STATEV(6) = alpha_Vd

      RETURN
      END
