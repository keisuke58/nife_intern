c =====================================================================
c  umat_klempt_voigt.f  --  Option A: Klempt UMAT + Voigt E(phi_i)
c
c  Fixes:
c    - E = const  →  E = Voigt(phi_i) = sum_i phi_i * E_i   (species-composition)
c    - alpha from condition-specific Klempt PDE (klempt_pde_multispecies.py)
c
c  Constitutive model:
c    F = Fe . Fg   (multiplicative growth split, Klempt 2024)
c    Fg = s * I    (isotropic, s = 1 + alpha)
c    alpha         = PREDEF(1)   [from klempt_alpha_final_{cond}.npy via FIELD]
c    phi_i         = PREDEF(2..6) [from TMCMC ref_0d_{cond}.json via FIELD]
c    E_Voigt       = sum_i phi_i * E_i  [Voigt homogenisation, in MPa]
c    nu            = PROPS(1)  [fixed = 0.49]
c
c  Neo-Hookean on elastic part:
c    W = (mu/2)*(tr(be)-3) + (lam/2)*ln(Je)^2 - mu*ln(Je)
c    sigma = ((lam*lnJe - mu)*I + mu*be) / Je
c
c  Species stiffness E_i [Pa] → E_i [MPa]:
c    So (S.oralis)        E_So = 1000 Pa = 1.00e-3 MPa  (EPS-rich, stiff)
c    An (A.naeslundii)    E_An =  800 Pa = 8.00e-4 MPa
c    Vd (V.dispar)        E_Vd =  600 Pa = 6.00e-4 MPa  (no EPS, softer)
c    Fn (F.nucleatum)     E_Fn =  200 Pa = 2.00e-4 MPa
c    Pg (P.gingivalis)    E_Pg =   10 Pa = 1.00e-5 MPa  (keystone pathogen)
c  Source: Voigt analogy from Phase 3b (phase3b_voigt_stress.py)
c
c  PROPS(1) = nu  (Poisson ratio, default 0.49)
c  CONSTANTS=1
c
c  PREDEF (field variables, set via *INITIAL CONDITIONS TYPE=FIELD):
c    PREDEF(1) = alpha       (condition-specific Klempt α, from FIELD variable 1)
c    PREDEF(2) = phi_So      (TMCMC species fraction, FIELD variable 2)
c    PREDEF(3) = phi_An
c    PREDEF(4) = phi_Vd
c    PREDEF(5) = phi_Fn
c    PREDEF(6) = phi_Pg
c
c  DEPVAR 4:
c    SDV1 = growth stretch s = 1+alpha
c    SDV2 = Je = det(Fe)
c    SDV3 = alpha
c    SDV4 = E_Voigt [MPa]
c
c  Reference: Klempt et al. 2024 Biomech Model Mechanobiol (Felix Table 2)
c             Phase 3b Voigt E_SPECIES assignment
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
      real*8 E_voigt
      real*8 alpha, s_iso
      real*8 fe(3,3), be(6), xi(6), detfe, lnJe
      real*8 phi_So, phi_An, phi_Vd, phi_Fn, phi_Pg
      integer i, j

c --- species stiffness [MPa] (1 Pa = 1e-6 MPa; tooth mesh in mm/MPa) ---------
c     E_SPECIES [Pa] = {So:1000, An:800, Vd:600, Fn:200, Pg:10}
      real*8 E_So, E_An, E_Vd, E_Fn, E_Pg
      parameter (E_So = 1.0d-3)   ! 1000 Pa
      parameter (E_An = 8.0d-4)   !  800 Pa
      parameter (E_Vd = 6.0d-4)   !  600 Pa
      parameter (E_Fn = 2.0d-4)   !  200 Pa
      parameter (E_Pg = 1.0d-5)   !   10 Pa

      data xi/1.d0,1.d0,1.d0,0.d0,0.d0,0.d0/

c --- fixed Poisson ratio -------------------------------------------------
      nu = PROPS(1)

c --- species fractions from TMCMC (field variables 2..6) -----------------
      phi_So = max(0.d0, PREDEF(2) + DPRED(2))
      phi_An = max(0.d0, PREDEF(3) + DPRED(3))
      phi_Vd = max(0.d0, PREDEF(4) + DPRED(4))
      phi_Fn = max(0.d0, PREDEF(5) + DPRED(5))
      phi_Pg = max(0.d0, PREDEF(6) + DPRED(6))

c --- Voigt Young's modulus [MPa] -----------------------------------------
      E_voigt = phi_So*E_So + phi_An*E_An + phi_Vd*E_Vd
     #        + phi_Fn*E_Fn + phi_Pg*E_Pg
      if (E_voigt .lt. 1.0d-10) E_voigt = 1.0d-10   ! numerical floor

c --- Lame constants from Voigt E ------------------------------------------
      mu_sh = E_voigt / (2.d0*(1.d0+nu))
      lam   = E_voigt*nu / ((1.d0+nu)*(1.d0-2.d0*nu))

c --- growth variable alpha from Klempt PDE (field variable 1) ------------
      alpha = PREDEF(1) + DPRED(1)
      if (alpha .lt. 0.d0) alpha = 0.d0
      s_iso = 1.d0 + alpha

c --- elastic deformation gradient Fe = F / s ----------------------------
      do i=1,3
        do j=1,3
          fe(i,j) = DFGRD1(i,j) / s_iso
        end do
      end do

c --- Je = det(Fe) -----------------------------------------------------------
      detfe = +fe(1,1)*(fe(2,2)*fe(3,3)-fe(2,3)*fe(3,2))
     #        -fe(1,2)*(fe(2,1)*fe(3,3)-fe(2,3)*fe(3,1))
     #        +fe(1,3)*(fe(2,1)*fe(3,2)-fe(2,2)*fe(3,1))
      if (detfe .lt. 1.d-15) detfe = 1.d-15
      lnJe = dlog(detfe)

c --- elastic left Cauchy-Green be = Fe.Fe^T (Voigt: 11 22 33 12 13 23) ---
      be(1) = fe(1,1)*fe(1,1)+fe(1,2)*fe(1,2)+fe(1,3)*fe(1,3)
      be(2) = fe(2,1)*fe(2,1)+fe(2,2)*fe(2,2)+fe(2,3)*fe(2,3)
      be(3) = fe(3,1)*fe(3,1)+fe(3,2)*fe(3,2)+fe(3,3)*fe(3,3)
      be(4) = fe(1,1)*fe(2,1)+fe(1,2)*fe(2,2)+fe(1,3)*fe(2,3)
      be(5) = fe(1,1)*fe(3,1)+fe(1,2)*fe(3,2)+fe(1,3)*fe(3,3)
      be(6) = fe(2,1)*fe(3,1)+fe(2,2)*fe(3,2)+fe(2,3)*fe(3,3)

c --- Cauchy stress sigma = ((lam*lnJe-mu)*I + mu*be) / Je ---------------
      do i=1,NTENS
        STRESS(i) = ((lam*lnJe-mu_sh)*xi(i) + mu_sh*be(i)) / detfe
      end do

c --- consistent elastic+geometric tangent ---------------------------------
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

c --- state variables -------------------------------------------------------
      SSE = (lam*lnJe**2
     #      + mu_sh*(be(1)+be(2)+be(3)-3.d0-2.d0*lnJe)) / 2.d0
      if (NSTATV.ge.1) STATEV(1) = s_iso
      if (NSTATV.ge.2) STATEV(2) = detfe
      if (NSTATV.ge.3) STATEV(3) = alpha
      if (NSTATV.ge.4) STATEV(4) = E_voigt

      RETURN
      END
