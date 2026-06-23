c =====================================================================
c  umat_klempt_alpha.f  --  Self-contained, socket-free Abaqus UMAT
c  for Felix Klempt 2024 (Biomech Model Mechanobiol) biofilm growth.
c
c  Constitutive model:
c    F = Fe . Fg   (multiplicative growth split)
c    Fg = s * I    (isotropic, s = 1 + alpha)
c    alpha = accumulated growth variable from JAXFEM PDE (Eq.36)
c          = PREDEF(1) delivered as field variable 1 per Gauss point
c
c  Neo-Hookean on the elastic part:
c    W = (mu/2)*(tr(be)-3) + (lam/2)*ln(Je)^2 - mu*ln(Je)
c    sigma = ((lam*lnJe - mu)*I + mu*be) / Je      (Cauchy stress)
c
c  Tangent: consistent geometric tangent (Holland / mholla form).
c  Based on umat_growth_phi.f with beta/ic/mode parameters removed;
c  alpha is the per-Gauss-point field variable, not a fitted function of phi.
c
c  PROPS(1)=E   PROPS(2)=nu   (CONSTANTS=2)
c  DEPVAR 3:  SDV1 = stretch s = 1+alpha
c             SDV2 = Je = det(Fe)
c             SDV3 = alpha (field variable copy)
c
c  Use: *USER MATERIAL, CONSTANTS=2
c       *DEPVAR
c        3
c       *INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=1
c        <node_id>, <alpha_value>   (alpha from JAXFEM)
c       *STEP, NLGEOM=YES
c       *STATIC
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

      real*8 E, nu, lam, mu, alpha, s_iso
      real*8 fe(3,3), be(6), xi(6), detfe, lnJe
      integer i, j

      data xi/1.d0,1.d0,1.d0,0.d0,0.d0,0.d0/

c --- material parameters (Felix Table 2: E=10 Pa, nu=0.49) -----------
      E   = PROPS(1)
      nu  = PROPS(2)
      mu  = E / (2.d0*(1.d0+nu))
      lam = E*nu / ((1.d0+nu)*(1.d0-2.d0*nu))

c --- accumulated growth variable from JAXFEM Eq.36 -------------------
c     PREDEF(1) = alpha at start of increment (from *INITIAL CONDITIONS)
c     DPRED(1)  = increment of field variable  (=0 for static snapshot)
      alpha = PREDEF(1) + DPRED(1)
      if (alpha .lt. 0.d0) alpha = 0.d0

c --- isotropic growth: Fg = s*I, s = 1 + alpha -----------------------
c     (Felix: Fg = alpha*I with alpha_0=1; here alpha is JAXFEM value
c      starting from 0, so s = 1 + alpha matches Felix's Fg diagonal.)
      s_iso = 1.d0 + alpha

c --- elastic deformation gradient Fe = F . Fg^{-1} = F / s -----------
      do i=1,3
        do j=1,3
          fe(i,j) = DFGRD1(i,j) / s_iso
        end do
      end do

c --- Je = det(Fe) -------------------------------------------------------
      detfe = +fe(1,1)*(fe(2,2)*fe(3,3)-fe(2,3)*fe(3,2))
     #        -fe(1,2)*(fe(2,1)*fe(3,3)-fe(2,3)*fe(3,1))
     #        +fe(1,3)*(fe(2,1)*fe(3,2)-fe(2,2)*fe(3,1))
      lnJe = dlog(detfe)

c --- elastic left Cauchy-Green be = Fe.Fe^T (Voigt: 11 22 33 12 13 23)
      be(1) = fe(1,1)*fe(1,1)+fe(1,2)*fe(1,2)+fe(1,3)*fe(1,3)
      be(2) = fe(2,1)*fe(2,1)+fe(2,2)*fe(2,2)+fe(2,3)*fe(2,3)
      be(3) = fe(3,1)*fe(3,1)+fe(3,2)*fe(3,2)+fe(3,3)*fe(3,3)
      be(4) = fe(1,1)*fe(2,1)+fe(1,2)*fe(2,2)+fe(1,3)*fe(2,3)
      be(5) = fe(1,1)*fe(3,1)+fe(1,2)*fe(3,2)+fe(1,3)*fe(3,3)
      be(6) = fe(2,1)*fe(3,1)+fe(2,2)*fe(3,2)+fe(2,3)*fe(3,3)

c --- Cauchy stress sigma = ((lam*lnJe-mu)*I + mu*be) / Je ----------
      do i=1,NTENS
        STRESS(i) = ((lam*lnJe-mu)*xi(i) + mu*be(i)) / detfe
      end do

c --- consistent elastic+geometric tangent (Holland/mholla) ------------
      do i=1,NTENS
        do j=1,NTENS
          DDSDDE(i,j) = 0.d0
        end do
      end do
      DDSDDE(1,1) = (lam-2.d0*(lam*lnJe-mu))/detfe + 2.d0*STRESS(1)
      DDSDDE(2,2) = (lam-2.d0*(lam*lnJe-mu))/detfe + 2.d0*STRESS(2)
      DDSDDE(3,3) = (lam-2.d0*(lam*lnJe-mu))/detfe + 2.d0*STRESS(3)
      DDSDDE(1,2) = lam/detfe
      DDSDDE(1,3) = lam/detfe
      DDSDDE(2,3) = lam/detfe
      DDSDDE(1,4) = STRESS(4)
      DDSDDE(2,4) = STRESS(4)
      DDSDDE(3,4) = 0.d0
      DDSDDE(4,4) = -(lam*lnJe-mu)/detfe + (STRESS(1)+STRESS(2))/2.d0
      if (NTENS.eq.6) then
        DDSDDE(1,5) = STRESS(5)
        DDSDDE(2,5) = 0.d0
        DDSDDE(3,5) = STRESS(5)
        DDSDDE(1,6) = 0.d0
        DDSDDE(2,6) = STRESS(6)
        DDSDDE(3,6) = STRESS(6)
        DDSDDE(5,5) = -(lam*lnJe-mu)/detfe+(STRESS(1)+STRESS(3))/2.d0
        DDSDDE(6,6) = -(lam*lnJe-mu)/detfe+(STRESS(2)+STRESS(3))/2.d0
        DDSDDE(4,5) = STRESS(6)/2.d0
        DDSDDE(4,6) = STRESS(5)/2.d0
        DDSDDE(5,6) = STRESS(4)/2.d0
      end if
      do i=2,NTENS
        do j=1,i-1
          DDSDDE(i,j) = DDSDDE(j,i)
        end do
      end do

c --- strain energy + state variable output ----------------------------
      SSE = (lam*lnJe**2 + mu*(be(1)+be(2)+be(3)-3.d0-2.d0*lnJe))/2.d0
      if (NSTATV.ge.1) STATEV(1) = s_iso    ! growth stretch 1+alpha
      if (NSTATV.ge.2) STATEV(2) = detfe    ! Je = det(Fe)
      if (NSTATV.ge.3) STATEV(3) = alpha    ! JAXFEM alpha field

      RETURN
      END
