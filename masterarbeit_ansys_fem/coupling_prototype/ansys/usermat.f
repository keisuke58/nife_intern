c =====================================================================
c  USERMAT skeleton — ANSYS Mechanical APDL user material that delegates
c  the constitutive update to the SAME Python server (umat_server_nsp.py)
c  over TCP, reusing bridge_client.c unchanged.
c
c  Status: SKELETON for the IKM ANSYS machine (Felix's framework). Mirrors
c  abaqus/umat.f 1:1 — the wire protocol and bridge_client.c are identical;
c  only the subroutine signature and the Voigt SHEAR ordering differ.
c
c  Build (UPF): place in the ANSYS user routine, then
c     ANSUSERMAT=1, link bridge_client.o, rebuild ANS solver, run with
c     TB,USER + TB,STATE.  See ../README.md and ../BUILD_ANSYS.md.
c
c  *** Voigt ordering caveat ***
c     ANSYS ncomp order : (xx, yy, zz, xy, yz, xz)
c     Python/Abaqus order: (11, 22, 33, 12, 13, 23) = (xx,yy,zz,xy,xz,yz)
c     => shear slots 5 and 6 are SWAPPED. Reorder on send and on receive
c        (done explicitly below) so the Python model always sees Abaqus order.
c =====================================================================
      subroutine usermat(
     &  matId, elemId, kDomIntPt, kLayer, kSectPt,
     &  ldstep, isubst, keycut,
     &  nDirect, nShear, ncomp, nStatev, nProp,
     &  Time, dTime, Temp, dTemp,
     &  stress, ustatev, dsdePl, sedEl, sedPl, epseq,
     &  Strain, dStrain, epsPl, prop, coords,
     &  var0, defGrad_t, defGrad,
     &  tsstif, epsZZ, var1, var2, var3, var4, var5,
     &  var6, var7, var8)

#include "impcom.inc"
      INTEGER matId, elemId, kDomIntPt, kLayer, kSectPt,
     &        ldstep, isubst, keycut, nDirect, nShear, ncomp,
     &        nStatev, nProp
      DOUBLE PRECISION Time, dTime, Temp, dTemp, sedEl, sedPl, epseq,
     &        epsZZ
      DOUBLE PRECISION stress(ncomp), ustatev(nStatev),
     &        dsdePl(ncomp,ncomp), Strain(ncomp), dStrain(ncomp),
     &        epsPl(ncomp), prop(nProp), coords(3),
     &        defGrad_t(3,3), defGrad(3,3), tsstif(2)

      DOUBLE PRECISION REQ(2+6+6+64), RESP(6+36+64)
      DOUBLE PRECISION ST6(6), DS6(6), Ea, XNU
      INTEGER I, J, NREQ, NRESP, IRC, MAP(6)
      EXTERNAL BRIDGE_EVAL
      INTEGER  BRIDGE_EVAL

c     map ANSYS (xx,yy,zz,xy,yz,xz) -> Abaqus (xx,yy,zz,xy,xz,yz)
      DATA MAP /1,2,3,4,6,5/

      Ea  = prop(1)
      XNU = prop(2)

c     --- reorder strain/dstrain into Abaqus order ---
      DO I = 1, 6
        IF (I .LE. ncomp) THEN
          ST6(I) = Strain(MAP(I))
          DS6(I) = dStrain(MAP(I))
        ELSE
          ST6(I) = 0.0d0
          DS6(I) = 0.0d0
        END IF
      END DO

c     --- pack request: NSTATV, STRAN(6), DSTRAN(6), STATEV, E, nu ---
      REQ(1) = DBLE(nStatev)
      DO I = 1, 6
        REQ(1+I) = ST6(I)
        REQ(7+I) = DS6(I)
      END DO
      DO I = 1, nStatev
        REQ(13+I) = ustatev(I)
      END DO
      REQ(14+nStatev) = Ea
      REQ(15+nStatev) = XNU
      NREQ = 15 + nStatev

      NRESP = 6 + 36 + nStatev
      IRC = BRIDGE_EVAL(REQ, NREQ, RESP, NRESP)
      IF (IRC .NE. 0) THEN
         keycut = 1            ! signal ANSYS to cut the substep
         RETURN
      END IF

c     --- unpack, reorder back to ANSYS order ---
      DO I = 1, ncomp
         stress(I) = RESP(MAP(I))
      END DO
      DO I = 1, ncomp
        DO J = 1, ncomp
           dsdePl(I,J) = RESP(6 + (MAP(I)-1)*6 + MAP(J))
        END DO
      END DO
      DO I = 1, nStatev
         ustatev(I) = RESP(42 + I)
      END DO

      RETURN
      END
