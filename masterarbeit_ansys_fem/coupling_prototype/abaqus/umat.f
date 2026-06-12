C =====================================================================
C  UMAT skeleton — Abaqus user material that delegates the constitutive
C  update to the Python umat_server.py over a TCP socket.
C
C  Status: SKELETON. Compile + run on the Abaqus machine (cannot be
C  tested here). The wire protocol matches umat_server.py exactly:
C     send : NSTATV  STRAN(6)  DSTRAN(6)  STATEV(NSTATV)  E  nu
C     recv : STRESS(6)  DDSDDE(36 row-major)  STATEV_new(NSTATV)
C
C  The actual TCP send/recv lives in bridge_client.c (ISO_C_BINDING),
C  so this Fortran stays portable. Build, e.g.:
C     abaqus make library=umat.f                 ! + link bridge_client.o
C  or  abaqus job=one_element user=umat.f interactive
C  (ensure bridge_client.o is on the link line; see README).
C
C  PROPS(1)=E, PROPS(2)=nu   (placeholder law; real model reads its own)
C =====================================================================
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,
     2 PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,
     3 DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)

      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 TIME(2),PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)

C     --- request/response buffers (fixed 6-component / 6x6 layout) ---
      DOUBLE PRECISION REQ(2 + 6 + 6 + 64)
      DOUBLE PRECISION RESP(6 + 36 + 64)
      INTEGER I, J, NREQ, NRESP, IRC
      DOUBLE PRECISION E, XNU

C     ISO_C_BINDING entry implemented in bridge_client.c:
C       int bridge_eval(double* req, int nreq, double* resp, int nresp)
      EXTERNAL BRIDGE_EVAL
      INTEGER  BRIDGE_EVAL

      E   = PROPS(1)
      XNU = PROPS(2)

C     --- pack request: NSTATV, STRAN(6), DSTRAN(6), STATEV, E, nu ---
      REQ(1) = DBLE(NSTATV)
      DO I = 1, 6
        REQ(1+I)   = STRAN(I)
        REQ(7+I)   = DSTRAN(I)
      END DO
      DO I = 1, NSTATV
        REQ(13+I)  = STATEV(I)
      END DO
      REQ(14+NSTATV) = E
      REQ(15+NSTATV) = XNU
      NREQ = 15 + NSTATV

C     --- call Python over socket; resp = STRESS(6) DDSDDE(36) STATEV ---
      NRESP = 6 + 36 + NSTATV
      IRC = BRIDGE_EVAL(REQ, NREQ, RESP, NRESP)
      IF (IRC .NE. 0) THEN
         WRITE(*,*) 'UMAT bridge_eval failed, rc=', IRC
         CALL XIT          ! abort the Abaqus job
      END IF

C     --- unpack ---
      DO I = 1, 6
         STRESS(I) = RESP(I)
      END DO
      DO I = 1, 6
         DO J = 1, 6
            DDSDDE(I,J) = RESP(6 + (I-1)*6 + J)   ! row-major from Python
         END DO
      END DO
      DO I = 1, NSTATV
         STATEV(I) = RESP(42 + I)
      END DO

      RETURN
      END
