c =====================================================================
c  umat_growth_phi.f  --  SOCKET-FREE finite-strain biofilm growth UMAT.
c
c  Self-contained: NO Python server, NO TCP socket. The composition field
c  phi_Pg is delivered per Gauss point as a PREDEFINED FIELD VARIABLE
c  (PREDEF(1)) -- the offline-tabulated-surrogate path, and the exact
c  pattern an ANSYS USERMAT uses (field variable -> growth stretch).
c
c  Constitutive model: compressible neo-Hookean on the elastic part of a
c  multiplicative growth split  F = Fe . Fg.
c    mode 0 (depth): anisotropic substratum-normal growth
c        Fg = diag(1, 1, lam_z),  lam_z = 1 + max(0, beta*phi + ic)
c        (= mholla umat_fiber_morph with fiber n = e_z; CLSM-calibrated beta,ic).
c        NOTE: in a laterally-confined, free-top column this gives sigma_xx ~ 0
c        (the column grows taller freely) -> use mode 1 for in-plane residual stress.
c    mode 1 (iso, VOLUME-MATCHED to CLSM): Fg = s*I, s = Jg^(1/3),
c        Jg = 1 + max(0, beta*phi + ic) = the CLSM-measured volume/thickness factor
c        (lateral FOV fixed -> thickness ratio = volume ratio = lam_z). The isotropic
c        model thus reproduces the measured volume increase; magnitude is model-
c        dependent because the true growth is anisotropic (z) -- thesis caveat (3).
c
c  Stress + CONSISTENT GEOMETRIC TANGENT are the textbook neo-Hookean+growth
c  forms from Holland, "Hitchhiker's Guide to Abaqus" (mholla/growth,
c  Zenodo 1243270) -- cite if used. They depend only on (lam,mu,Je,be,sigma),
c  so the SAME tangent code is valid for any Fg (iso or depth).
c
c  PROPS: 1=E  2=nu  3=beta_depth  4=intercept  5=mode(0/1)
c  DEPVAR 3:  SDV1=lam_z(or s)   SDV2=Je   SDV3=phi_Pg
c  Use with *USER MATERIAL, CONSTANTS=5 ; *DEPVAR 3 ; NLGEOM=YES ; *FIELD VAR=1.
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

      real*8 E, nu, lam, mu, beta, ic, phi, growth, lam_z, s_iso
      real*8 fe(3,3), be(6), xi(6), detfe, lnJe
      integer i, j, mode

      data xi/1.d0,1.d0,1.d0,0.d0,0.d0,0.d0/

c...  material parameters
      E    = PROPS(1)
      nu   = PROPS(2)
      beta = PROPS(3)
      ic   = PROPS(4)
      mode = nint(PROPS(5))
      mu   = E/(2.d0*(1.d0+nu))
      lam  = E*nu/((1.d0+nu)*(1.d0-2.d0*nu))

c...  composition field at this Gauss point (end-of-increment value)
      phi = PREDEF(1) + DPRED(1)

c...  growth stretch from the CLSM calibration (clamped at no-growth)
      growth = beta*phi + ic
      if (growth .lt. 0.d0) growth = 0.d0
      lam_z = 1.d0 + growth

c...  elastic deformation gradient Fe = F . Fg^{-1}
      if (mode .eq. 1) then
c       isotropic, volume-matched: Jg = lam_z (CLSM volume factor), s = Jg^(1/3)
        s_iso = lam_z**(1.d0/3.d0)
        do i=1,3
          do j=1,3
            fe(i,j) = DFGRD1(i,j) / s_iso
          end do
        end do
      else
c       depth (z): Fg = diag(1,1,lam_z) -> scale only the 3rd column of F
        do i=1,3
          fe(i,1) = DFGRD1(i,1)
          fe(i,2) = DFGRD1(i,2)
          fe(i,3) = DFGRD1(i,3) / lam_z
        end do
      end if

c...  Je = det(Fe)
      detfe = +fe(1,1)*(fe(2,2)*fe(3,3)-fe(2,3)*fe(3,2))
     #        -fe(1,2)*(fe(2,1)*fe(3,3)-fe(2,3)*fe(3,1))
     #        +fe(1,3)*(fe(2,1)*fe(3,2)-fe(2,2)*fe(3,1))
      lnJe = dlog(detfe)

c...  elastic left Cauchy-Green be = Fe.Fe^T (Voigt 11 22 33 12 13 23)
      be(1) = fe(1,1)*fe(1,1)+fe(1,2)*fe(1,2)+fe(1,3)*fe(1,3)
      be(2) = fe(2,1)*fe(2,1)+fe(2,2)*fe(2,2)+fe(2,3)*fe(2,3)
      be(3) = fe(3,1)*fe(3,1)+fe(3,2)*fe(3,2)+fe(3,3)*fe(3,3)
      be(4) = fe(1,1)*fe(2,1)+fe(1,2)*fe(2,2)+fe(1,3)*fe(2,3)
      be(5) = fe(1,1)*fe(3,1)+fe(1,2)*fe(3,2)+fe(1,3)*fe(3,3)
      be(6) = fe(2,1)*fe(3,1)+fe(2,2)*fe(3,2)+fe(2,3)*fe(3,3)

c...  Cauchy stress  sigma = ((lam*lnJe - mu)*I + mu*be)/Je
      do i=1,NTENS
        STRESS(i) = ((lam*lnJe-mu)*xi(i) + mu*be(i))/detfe
      end do

c...  consistent elastic + geometric tangent (Holland/mholla form)
      do i=1,NTENS
        do j=1,NTENS
          DDSDDE(i,j) = 0.d0
        end do
      end do
      DDSDDE(1,1) = (lam - 2.d0*(lam*lnJe-mu))/detfe + 2.d0*STRESS(1)
      DDSDDE(2,2) = (lam - 2.d0*(lam*lnJe-mu))/detfe + 2.d0*STRESS(2)
      DDSDDE(3,3) = (lam - 2.d0*(lam*lnJe-mu))/detfe + 2.d0*STRESS(3)
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
        DDSDDE(5,5) = -(lam*lnJe-mu)/detfe + (STRESS(1)+STRESS(3))/2.d0
        DDSDDE(6,6) = -(lam*lnJe-mu)/detfe + (STRESS(2)+STRESS(3))/2.d0
        DDSDDE(4,5) = STRESS(6)/2.d0
        DDSDDE(4,6) = STRESS(5)/2.d0
        DDSDDE(5,6) = STRESS(4)/2.d0
      end if
      do i=2,NTENS
        do j=1,i-1
          DDSDDE(i,j) = DDSDDE(j,i)
        end do
      end do

c...  strain energy + state output
      SSE = (lam*lnJe**2 + mu*(be(1)+be(2)+be(3)-3.d0-2.d0*lnJe))/2.d0
      if (NSTATV.ge.1) STATEV(1) = lam_z
      if (NSTATV.ge.2) STATEV(2) = detfe
      if (NSTATV.ge.3) STATEV(3) = phi

      RETURN
      END
