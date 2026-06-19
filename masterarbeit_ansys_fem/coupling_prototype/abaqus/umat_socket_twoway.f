c =====================================================================
c  umat_socket_twoway.f — small-strain UMAT bridging Abaqus to the two-way
c  Python server (umat_server_twoway.py).
c
c  Same wire/socket layout as umat_socket.f but the request appends the
c  previous-increment Cauchy stress so the Python side can throttle the
c  composition-driven growth by mechanotransduction:
c
c    send : NSTATV  STRAN(6)  DSTRAN(6)  STATEV(NSTATV)  E  nu  STRESS(6)
c    recv : STRESS(6) DDSDDE(36 row-major) STATEV_new(NSTATV)
c
c  Server  : 127.0.0.1:50008
c  Deck    : *DEPVAR 12 (NSP state), *USER MATERIAL, CONSTANTS=2 (E, nu),
c            NLGEOM=NO (small-strain version).
c
c  Use umat_socket_fs.f for the finite-strain variant.
c =====================================================================
      module bridge_mod_tw
      use iso_c_binding
      implicit none
      interface
        function c_socket(d,t,p) bind(C,name="socket")
          import :: c_int
          integer(c_int),value :: d,t,p
          integer(c_int) :: c_socket
        end function
        function c_connect(fd,addr,ln) bind(C,name="connect")
          import :: c_int,c_char
          integer(c_int),value :: fd,ln
          character(kind=c_char),dimension(*) :: addr
          integer(c_int) :: c_connect
        end function
        function c_write(fd,buf,n) bind(C,name="write")
          import :: c_int,c_char,c_size_t,c_long
          integer(c_int),value :: fd
          character(kind=c_char),dimension(*) :: buf
          integer(c_size_t),value :: n
          integer(c_long) :: c_write
        end function
        function c_read(fd,buf,n) bind(C,name="read")
          import :: c_int,c_char,c_size_t,c_long
          integer(c_int),value :: fd
          character(kind=c_char),dimension(*) :: buf
          integer(c_size_t),value :: n
          integer(c_long) :: c_read
        end function
        function c_close(fd) bind(C,name="close")
          import :: c_int
          integer(c_int),value :: fd
          integer(c_int) :: c_close
        end function
      end interface
      contains

      integer function bridge_call(req, resp, nresp)
      use iso_c_binding
      implicit none
      character(len=*),intent(in) :: req
      integer,intent(in) :: nresp
      double precision,intent(out) :: resp(nresp)
      integer(c_int) :: fd, ok, AF_INET, SOCK_STREAM
      character(kind=c_char,len=16) :: addr_struct
      character(kind=c_char,len=65536) :: rxbuf
      integer(c_long) :: nrx, nwr
      integer :: i, k
      double precision :: tmp
      AF_INET     = 2
      SOCK_STREAM = 1

c     IPv4 127.0.0.1 : 50008 (port 50008 → 0xC368, network byte order C3 68)
      addr_struct = char(2)//char(0)//char(195)//char(104)//
     1              char(127)//char(0)//char(0)//char(1)//
     2              char(0)//char(0)//char(0)//char(0)//
     3              char(0)//char(0)//char(0)//char(0)

      fd = c_socket(AF_INET, SOCK_STREAM, 0)
      ok = c_connect(fd, addr_struct, 16)
      if (ok .ne. 0) then
        bridge_call = -1
        return
      end if
      nwr = c_write(fd, req, int(len_trim(req), c_size_t))
      nrx = c_read(fd, rxbuf, int(65536, c_size_t))
      ok = c_close(fd)

c     parse whitespace-separated doubles from rxbuf into resp
      k = 1
      i = 1
      do while (k .le. nresp .and. i .le. nrx)
        do while (i .le. nrx .and. (rxbuf(i:i) == ' ' .or.
     1                              rxbuf(i:i) == char(10) .or.
     2                              rxbuf(i:i) == char(13) .or.
     3                              rxbuf(i:i) == char(9)))
          i = i + 1
        end do
        if (i .gt. nrx) exit
        read(rxbuf(i:nrx), *, iostat=ok) tmp
        if (ok .ne. 0) exit
        resp(k) = tmp
        k = k + 1
        do while (i .le. nrx .and. rxbuf(i:i) /= ' ' .and.
     1            rxbuf(i:i) /= char(10) .and.
     2            rxbuf(i:i) /= char(13) .and.
     3            rxbuf(i:i) /= char(9))
          i = i + 1
        end do
      end do
      bridge_call = 0
      end function bridge_call
      end module bridge_mod_tw

c =====================================================================
      SUBROUTINE UMAT(
     1  STRESS, STATEV, DDSDDE, SSE, SPD, SCD,
     2  RPL, DDSDDT, DRPLDE, DRPLDT,
     3  STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED,
     4  CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS,
     5  COORDS, DROT, PNEWDT, CELENT,
     6  DFGRD0, DFGRD1, NOEL, NPT, LAYER, KSPT, KSTEP, KINC)

      use bridge_mod_tw
      INCLUDE 'ABA_PARAM.INC'

      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS), STATEV(NSTATV), DDSDDE(NTENS,NTENS),
     1          DDSDDT(NTENS), DRPLDE(NTENS), STRAN(NTENS),
     2          DSTRAN(NTENS), TIME(2), PREDEF(1), DPRED(1),
     3          PROPS(NPROPS), COORDS(3), DROT(3,3),
     4          DFGRD0(3,3), DFGRD1(3,3)

      DOUBLE PRECISION :: E_MOD, NU
      DOUBLE PRECISION :: REQ_FIELDS(2 + 6 + 6 + 12 + 2 + 6)
      DOUBLE PRECISION :: RESP(6 + 36 + 12)
      CHARACTER*4096 :: REQ_TEXT
      INTEGER :: I, J, BS, K
      INTEGER :: NSEND
      DOUBLE PRECISION :: VAL

      E_MOD = PROPS(1)
      NU    = PROPS(2)

c     assemble the wire payload as a single space-separated record. STRESS at
c     the entry point of UMAT is the σ_n (last converged Cauchy stress), so it
c     is forwarded to the server unchanged.
      WRITE(REQ_TEXT, '(I0, 1X)') NSTATV
      BS = len_trim(REQ_TEXT) + 1
      DO I = 1, 6
        IF (I .LE. NTENS) THEN
          VAL = STRAN(I)
        ELSE
          VAL = 0.0D0
        END IF
        WRITE(REQ_TEXT(BS:), '(ES23.16, 1X)') VAL
        BS = len_trim(REQ_TEXT) + 1
      END DO
      DO I = 1, 6
        IF (I .LE. NTENS) THEN
          VAL = DSTRAN(I)
        ELSE
          VAL = 0.0D0
        END IF
        WRITE(REQ_TEXT(BS:), '(ES23.16, 1X)') VAL
        BS = len_trim(REQ_TEXT) + 1
      END DO
      DO I = 1, NSTATV
        WRITE(REQ_TEXT(BS:), '(ES23.16, 1X)') STATEV(I)
        BS = len_trim(REQ_TEXT) + 1
      END DO
      WRITE(REQ_TEXT(BS:), '(ES23.16, 1X, ES23.16, 1X)') E_MOD, NU
      BS = len_trim(REQ_TEXT) + 1
      DO I = 1, 6
        IF (I .LE. NTENS) THEN
          VAL = STRESS(I)
        ELSE
          VAL = 0.0D0
        END IF
        WRITE(REQ_TEXT(BS:), '(ES23.16, 1X)') VAL
        BS = len_trim(REQ_TEXT) + 1
      END DO
      WRITE(REQ_TEXT(BS:BS), '(A)') CHAR(10)
      BS = BS + 1

      NSEND = 6 + 36 + NSTATV
      IF (bridge_call(REQ_TEXT(1:BS-1), RESP, NSEND) .NE. 0) THEN
        PNEWDT = 0.5D0
        RETURN
      END IF

      DO I = 1, NTENS
        STRESS(I) = RESP(I)
      END DO
      K = 6
      DO I = 1, 6
        DO J = 1, 6
          K = K + 1
          IF (I .LE. NTENS .AND. J .LE. NTENS) THEN
            DDSDDE(I,J) = RESP(K)
          END IF
        END DO
      END DO
      DO I = 1, NSTATV
        STATEV(I) = RESP(6 + 36 + I)
      END DO

      RETURN
      END
