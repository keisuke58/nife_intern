c =====================================================================
c  uel_phasefield_at2.f — Abaqus UEL for the coupled u-d AT2 phase-field
c  brittle fracture (Miehe-Hofacker 2010, Borden et al. 2012).
c
c  Element : 4-node bilinear quadrilateral, plane strain.
c  DOFs    : 3 per node = (u_x, u_y, d). NDOFEL = 12. JTYPE = 1 → use
c            *USER ELEMENT, NODES=4, TYPE=U1, COORDINATES=2, PROPERTIES=4,
c            VARIABLES=8 (history H, undivided positive elastic energy, per Gauss point × 4).
c  IPROPS  : (none).  PROPS(1..4) = (E, nu, Gc, l)
c             E   - Young's modulus
c             nu  - Poisson's ratio
c             Gc  - Griffith critical energy release rate
c             l   - phase-field regularisation length (≈ 1.5 * h)
c  SVARS   : per Gauss point, two doubles in this order:
c             SVARS(2*(ip-1)+1) = H_n      (history of max positive elastic energy)
c             SVARS(2*(ip-1)+2) = psi_pos  (current positive elastic energy, output only)
c
c  Driving force : tension-only undecomposed elastic energy
c                   ψ_+(ε) = 0.5 ε : C : ε   if  tr(ε) ≥ 0
c                          = 0                else
c                  Robust tension-driven crack; classical Amor / Miehe decomposition can be
c                  swapped in by replacing `psi_pos` (see below).
c
c  Tangent       : Standard split AMATRX = | Kuu  Kud |
c                                          | Kdu  Kdd |
c                  Kud = ∂R_u/∂d uses the history H formulation:
c                  damage modifies stress as σ = g(d) C ε with g(d) = (1-d)^2 + η,
c                  so Kud(I,J) = ∫ B_I^T (∂g/∂d) C ε N_J dV    only when active (tension)
c                  Kdu(I,J) = 0 in the history-variable formulation (history is frozen at the
c                             max of ψ_+ over time so it has no dependence on the current u).
c
c  Numerical quadrature : 2×2 Gauss = 4 IPs.
c
c  Geometry / load assembly (the host calls this UEL only — boundary conditions, load steps,
c  damage initiation through *INITIAL CONDITIONS, USER OR FIELD live in the input deck).
c =====================================================================

      SUBROUTINE UEL(
     1  RHS, AMATRX, SVARS, ENERGY, NDOFEL, NRHS, NSVARS,
     2  PROPS, NPROPS, COORDS, MCRD, NNODE, U, DU, V, A,
     3  JTYPE, TIME, DTIME, KSTEP, KINC, JELEM, PARAMS,
     4  NDLOAD, JDLTYP, ADLMAG, PREDEF, NPREDF, LFLAGS,
     5  MLVARX, DDLMAG, MDLOAD, PNEWDT, JPROPS, NJPROP,
     6  PERIOD)

      INCLUDE 'ABA_PARAM.INC'

      DIMENSION RHS(MLVARX,*), AMATRX(NDOFEL,NDOFEL),
     1          SVARS(NSVARS), ENERGY(8), PROPS(NPROPS),
     2          COORDS(MCRD,NNODE), U(NDOFEL), DU(MLVARX,*),
     3          V(NDOFEL), A(NDOFEL), TIME(2), PARAMS(*),
     4          JDLTYP(MDLOAD,*), ADLMAG(MDLOAD,*),
     5          PREDEF(2,NPREDF,NNODE), LFLAGS(*),
     6          DDLMAG(MDLOAD,*), JPROPS(*)

c     ------------------------------------------------------------------
c     local declarations
c     ------------------------------------------------------------------
      INTEGER, PARAMETER :: NGP = 4
      INTEGER, PARAMETER :: NEN = 4
      INTEGER, PARAMETER :: NSD = 2
      DOUBLE PRECISION   :: E_MOD, NU, GC, ELL
      DOUBLE PRECISION   :: ETA
      DOUBLE PRECISION   :: GP(NSD, NGP), GW(NGP)
      DOUBLE PRECISION   :: XI, ETAGP
      DOUBLE PRECISION   :: SF(NEN), DSFDXI(NSD, NEN)
      DOUBLE PRECISION   :: JAC(NSD,NSD), JINV(NSD,NSD), DETJ
      DOUBLE PRECISION   :: DSFDX(NSD, NEN)
      DOUBLE PRECISION   :: BMAT(3, 2*NEN)
      DOUBLE PRECISION   :: CMAT(3,3)
      DOUBLE PRECISION   :: U_NODE(NEN, NSD), D_NODE(NEN)
      DOUBLE PRECISION   :: STRAIN(3), STRESS(3)
      DOUBLE PRECISION   :: D_GP, GRADD(NSD)
      DOUBLE PRECISION   :: GFUN, DGFUN
      DOUBLE PRECISION   :: PSI_POS, TRACE_E, H_HIST
      DOUBLE PRECISION   :: WDV
      DOUBLE PRECISION   :: KUU(2*NEN, 2*NEN), KUD(2*NEN, NEN)
      DOUBLE PRECISION   :: KDD(NEN, NEN), RU(2*NEN), RD(NEN)
      DOUBLE PRECISION   :: TMP, COEF
      INTEGER            :: IP, II, JJ, A1, B1, IDX1, IDX2

c     ------------------------------------------------------------------
c     read material props (default safety values if not supplied)
c     ------------------------------------------------------------------
      E_MOD = PROPS(1)
      NU    = PROPS(2)
      GC    = PROPS(3)
      ELL   = PROPS(4)
      ETA   = 1.0D-08

c     plane-strain constitutive matrix
      TMP = E_MOD / ((1.0D0 + NU) * (1.0D0 - 2.0D0*NU))
      CMAT = 0.0D0
      CMAT(1,1) = TMP * (1.0D0 - NU)
      CMAT(1,2) = TMP * NU
      CMAT(2,1) = TMP * NU
      CMAT(2,2) = TMP * (1.0D0 - NU)
      CMAT(3,3) = TMP * (1.0D0 - 2.0D0*NU) / 2.0D0

c     ------------------------------------------------------------------
c     2x2 Gauss points
c     ------------------------------------------------------------------
      GP(1,1) = -1.0D0/DSQRT(3.0D0); GP(2,1) = -1.0D0/DSQRT(3.0D0)
      GP(1,2) =  1.0D0/DSQRT(3.0D0); GP(2,2) = -1.0D0/DSQRT(3.0D0)
      GP(1,3) =  1.0D0/DSQRT(3.0D0); GP(2,3) =  1.0D0/DSQRT(3.0D0)
      GP(1,4) = -1.0D0/DSQRT(3.0D0); GP(2,4) =  1.0D0/DSQRT(3.0D0)
      GW(1) = 1.0D0; GW(2) = 1.0D0; GW(3) = 1.0D0; GW(4) = 1.0D0

c     ------------------------------------------------------------------
c     unpack nodal DOFs: U = [ux1,uy1,d1, ux2,uy2,d2, ...]
c     ------------------------------------------------------------------
      DO II = 1, NEN
        U_NODE(II,1) = U(3*(II-1)+1)
        U_NODE(II,2) = U(3*(II-1)+2)
        D_NODE(II)   = U(3*(II-1)+3)
      END DO

      RHS    = 0.0D0
      AMATRX = 0.0D0
      KUU = 0.0D0; KUD = 0.0D0; KDD = 0.0D0
      RU  = 0.0D0; RD  = 0.0D0

c     ------------------------------------------------------------------
c     loop over Gauss points
c     ------------------------------------------------------------------
      DO IP = 1, NGP
        XI    = GP(1, IP)
        ETAGP = GP(2, IP)

c       shape functions
        SF(1) = 0.25D0*(1.0D0-XI)*(1.0D0-ETAGP)
        SF(2) = 0.25D0*(1.0D0+XI)*(1.0D0-ETAGP)
        SF(3) = 0.25D0*(1.0D0+XI)*(1.0D0+ETAGP)
        SF(4) = 0.25D0*(1.0D0-XI)*(1.0D0+ETAGP)

c       dN/dxi
        DSFDXI(1,1) = -0.25D0*(1.0D0-ETAGP); DSFDXI(2,1) = -0.25D0*(1.0D0-XI)
        DSFDXI(1,2) =  0.25D0*(1.0D0-ETAGP); DSFDXI(2,2) = -0.25D0*(1.0D0+XI)
        DSFDXI(1,3) =  0.25D0*(1.0D0+ETAGP); DSFDXI(2,3) =  0.25D0*(1.0D0+XI)
        DSFDXI(1,4) = -0.25D0*(1.0D0+ETAGP); DSFDXI(2,4) =  0.25D0*(1.0D0-XI)

c       Jacobian
        JAC = 0.0D0
        DO II = 1, NEN
          JAC(1,1) = JAC(1,1) + DSFDXI(1,II) * COORDS(1, II)
          JAC(1,2) = JAC(1,2) + DSFDXI(1,II) * COORDS(2, II)
          JAC(2,1) = JAC(2,1) + DSFDXI(2,II) * COORDS(1, II)
          JAC(2,2) = JAC(2,2) + DSFDXI(2,II) * COORDS(2, II)
        END DO
        DETJ = JAC(1,1)*JAC(2,2) - JAC(1,2)*JAC(2,1)
        IF (DETJ .LE. 0.0D0) THEN
          WRITE(*,*) 'UEL_PHASEFIELD: bad Jacobian at IP=', IP
          PNEWDT = 0.5D0
          RETURN
        END IF
        JINV(1,1) =  JAC(2,2)/DETJ; JINV(1,2) = -JAC(1,2)/DETJ
        JINV(2,1) = -JAC(2,1)/DETJ; JINV(2,2) =  JAC(1,1)/DETJ

c       dN/dx
        DO II = 1, NEN
          DSFDX(1, II) = JINV(1,1)*DSFDXI(1,II) + JINV(1,2)*DSFDXI(2,II)
          DSFDX(2, II) = JINV(2,1)*DSFDXI(1,II) + JINV(2,2)*DSFDXI(2,II)
        END DO

c       B matrix (plane-strain Voigt order: εxx, εyy, 2εxy)
        BMAT = 0.0D0
        DO II = 1, NEN
          BMAT(1, 2*(II-1)+1) = DSFDX(1, II)
          BMAT(2, 2*(II-1)+2) = DSFDX(2, II)
          BMAT(3, 2*(II-1)+1) = DSFDX(2, II)
          BMAT(3, 2*(II-1)+2) = DSFDX(1, II)
        END DO

c       interpolate d and grad(d) and strain
        D_GP   = 0.0D0
        GRADD  = 0.0D0
        DO II = 1, NEN
          D_GP   = D_GP   + SF(II) * D_NODE(II)
          GRADD(1) = GRADD(1) + DSFDX(1, II) * D_NODE(II)
          GRADD(2) = GRADD(2) + DSFDX(2, II) * D_NODE(II)
        END DO
        STRAIN = 0.0D0
        DO II = 1, NEN
          STRAIN(1) = STRAIN(1) + DSFDX(1,II) * U_NODE(II,1)
          STRAIN(2) = STRAIN(2) + DSFDX(2,II) * U_NODE(II,2)
          STRAIN(3) = STRAIN(3) + DSFDX(2,II)*U_NODE(II,1)
     1                          + DSFDX(1,II)*U_NODE(II,2)
        END DO

c       degradation function g(d) = (1-d)^2 + eta
        GFUN  = (1.0D0 - D_GP)**2 + ETA
        DGFUN = -2.0D0 * (1.0D0 - D_GP)

c       elastic stress = g(d) C strain
        STRESS(1) = GFUN * (CMAT(1,1)*STRAIN(1) + CMAT(1,2)*STRAIN(2))
        STRESS(2) = GFUN * (CMAT(2,1)*STRAIN(1) + CMAT(2,2)*STRAIN(2))
        STRESS(3) = GFUN * CMAT(3,3) * STRAIN(3)

c       tension-only driving force: ψ_+ = 0.5 ε C ε  if tr(ε) >= 0 else 0
        TRACE_E = STRAIN(1) + STRAIN(2)
        IF (TRACE_E .GE. 0.0D0) THEN
          PSI_POS = 0.5D0 * ( STRAIN(1)*(CMAT(1,1)*STRAIN(1)+CMAT(1,2)*STRAIN(2))
     1                      + STRAIN(2)*(CMAT(2,1)*STRAIN(1)+CMAT(2,2)*STRAIN(2))
     2                      + STRAIN(3)*CMAT(3,3)*STRAIN(3) )
        ELSE
          PSI_POS = 0.0D0
        END IF

c       history H = max(H_n, ψ_+)   — monotone increasing (Miehe-Hofacker)
        H_HIST = SVARS(2*(IP-1)+1)
        IF (PSI_POS .GT. H_HIST) H_HIST = PSI_POS
        SVARS(2*(IP-1)+1) = H_HIST
        SVARS(2*(IP-1)+2) = PSI_POS

        WDV = DETJ * GW(IP)

c       --- assemble Kuu and Ru -----------------------------------------
        DO II = 1, 2*NEN
          DO JJ = 1, 2*NEN
            TMP = 0.0D0
            DO A1 = 1, 3
              DO B1 = 1, 3
                TMP = TMP + BMAT(A1,II) * CMAT(A1,B1) * BMAT(B1,JJ)
              END DO
            END DO
            KUU(II,JJ) = KUU(II,JJ) + GFUN * TMP * WDV
          END DO
          RU(II) = RU(II) - ( BMAT(1,II)*STRESS(1)
     1                      + BMAT(2,II)*STRESS(2)
     2                      + BMAT(3,II)*STRESS(3) ) * WDV
        END DO

c       --- assemble Kud (coupling u → d) -------------------------------
        DO II = 1, 2*NEN
          DO JJ = 1, NEN
            TMP = BMAT(1,II)*(CMAT(1,1)*STRAIN(1)+CMAT(1,2)*STRAIN(2))
     1          + BMAT(2,II)*(CMAT(2,1)*STRAIN(1)+CMAT(2,2)*STRAIN(2))
     2          + BMAT(3,II)*(CMAT(3,3)*STRAIN(3))
            KUD(II,JJ) = KUD(II,JJ) + DGFUN * SF(JJ) * TMP * WDV
          END DO
        END DO

c       --- assemble Kdd and Rd -----------------------------------------
c        Kdd_IJ = ∫ [ (Gc/l + 2H) N_I N_J + Gc l (∇N_I)·(∇N_J) ] dV
c        Rd_I   = -∫ [ (Gc/l + 2H) d N_I + Gc l (∇d)·(∇N_I) - 2H N_I ] dV
        COEF = GC/ELL + 2.0D0 * H_HIST
        DO II = 1, NEN
          DO JJ = 1, NEN
            KDD(II,JJ) = KDD(II,JJ) +
     1        ( COEF * SF(II) * SF(JJ)
     2        + GC * ELL * ( DSFDX(1,II)*DSFDX(1,JJ)
     3                     + DSFDX(2,II)*DSFDX(2,JJ) ) ) * WDV
          END DO
          RD(II) = RD(II) - ( COEF * D_GP * SF(II)
     1                      + GC * ELL * ( GRADD(1)*DSFDX(1,II)
     2                                   + GRADD(2)*DSFDX(2,II) )
     3                      - 2.0D0 * H_HIST * SF(II) ) * WDV
        END DO
      END DO

c     ------------------------------------------------------------------
c     scatter sub-blocks into AMATRX and RHS using node-major DOF layout
c     DOF ordering per node: (ux, uy, d) → (3*(I-1)+1, 3*(I-1)+2, 3*(I-1)+3)
c     ------------------------------------------------------------------
      DO II = 1, NEN
        DO JJ = 1, NEN
c         Kuu block (2×2)
          DO A1 = 1, 2
            DO B1 = 1, 2
              IDX1 = 3*(II-1) + A1
              IDX2 = 3*(JJ-1) + B1
              AMATRX(IDX1, IDX2) = AMATRX(IDX1, IDX2)
     1            + KUU(2*(II-1)+A1, 2*(JJ-1)+B1)
            END DO
          END DO
c         Kud block (2×1)
          DO A1 = 1, 2
            IDX1 = 3*(II-1) + A1
            IDX2 = 3*(JJ-1) + 3
            AMATRX(IDX1, IDX2) = AMATRX(IDX1, IDX2)
     1          + KUD(2*(II-1)+A1, JJ)
          END DO
c         Kdd block (1×1)
          IDX1 = 3*(II-1) + 3
          IDX2 = 3*(JJ-1) + 3
          AMATRX(IDX1, IDX2) = AMATRX(IDX1, IDX2) + KDD(II, JJ)
        END DO
c       RHS (residual) — split into u (2 entries) and d (1 entry) per node
        RHS(3*(II-1)+1, 1) = RU(2*(II-1)+1)
        RHS(3*(II-1)+2, 1) = RU(2*(II-1)+2)
        RHS(3*(II-1)+3, 1) = RD(II)
      END DO

      RETURN
      END
