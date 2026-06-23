c =====================================================================
c  umat_socket_ve.f — same as umat_socket_col.f but also sends DTIME
c  so the Python server can integrate the Prony viscoelastic arms.
c     send : NSTATV  COORDS(3)  DFGRD1(9,row-major)  STATEV(NSTATV)  E  nu  DTIME \n
c     recv : STRESS(6) DDSDDE(36) STATEV_new(NSTATV)
c  Server 127.0.0.1:50007 (umat_server_ve.py). *DEPVAR 19, NLGEOM=YES.
c  2 Maxwell arms (from prony_biofilm.json); STATEV layout:
c    [1]      = growth ramp counter
c    [2..7]   = h1_dev (Prony arm 1 deviatoric overstress, Voigt)
c    [8..13]  = h2_dev (Prony arm 2 deviatoric overstress, Voigt)
c    [14..19] = sigma_e_old (previous elastic Cauchy stress, Voigt)
c =====================================================================
      module bridge_mod_ve
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
      character(kind=c_char) :: addr(16), rbuf(1)
      character(len=16384) :: rstr, sbuf
      integer(c_int) :: fd, rc
      integer(c_long) :: nw, nr
      integer :: i, pos, sendlen
      fd = c_socket(2_c_int, 1_c_int, 0_c_int)
      if (fd < 0) then; bridge_call = 1; return; end if
      do i=1,16; addr(i)=char(0); end do
      addr(1)=char(2);  addr(2)=char(0)
      addr(3)=char(195); addr(4)=char(87)        ! port 50007
      addr(5)=char(127); addr(8)=char(1)         ! 127.0.0.1
      rc = c_connect(fd, addr, 16_c_int)
      if (rc /= 0) then; rc=c_close(fd); bridge_call=2; return; end if
      sbuf = trim(req)//char(10)//char(0)
      sendlen = len_trim(req)+1
      nw = c_write(fd, sbuf, int(sendlen,c_size_t))
      pos = 0; rstr = ' '
      do
        nr = c_read(fd, rbuf, 1_c_size_t)
        if (nr <= 0) exit
        if (rbuf(1) == char(10)) exit
        pos = pos+1; rstr(pos:pos) = rbuf(1)
        if (pos >= 16383) exit
      end do
      rc = c_close(fd)
      read(rstr(1:pos), *, err=900) (resp(i), i=1,nresp)
      bridge_call = 0; return
 900  bridge_call = 3; return
      end function
      end module
c ---------------------------------------------------------------------
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,
     2 PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,
     3 DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
      use bridge_mod_ve
      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 TIME(2),PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)
      character(len=16384) :: req, tmp
      double precision :: resp(6+36+64)
      integer :: i, j, irc, nresp

      write(req,'(I0)') NSTATV
      do i=1,3
        write(tmp,'(1X,ES24.16)') COORDS(i); req=trim(req)//trim(tmp)
      end do
      do i=1,3
        do j=1,3
          write(tmp,'(1X,ES24.16)') DFGRD1(i,j); req=trim(req)//trim(tmp)
        end do
      end do
      do i=1,NSTATV
        write(tmp,'(1X,ES24.16)') STATEV(i); req=trim(req)//trim(tmp)
      end do
      write(tmp,'(1X,ES24.16)') PROPS(1); req=trim(req)//trim(tmp)
      write(tmp,'(1X,ES24.16)') PROPS(2); req=trim(req)//trim(tmp)
      write(tmp,'(1X,ES24.16)') DTIME; req=trim(req)//trim(tmp)  ! VE extension

      nresp = 6 + 36 + NSTATV
      irc = bridge_call(req, resp, nresp)
      if (irc /= 0) then
        write(*,*) 'UMAT_VE bridge_call failed rc=', irc
        call XIT
      end if
      do i=1,6
        STRESS(i) = resp(i)
      end do
      do i=1,6
        do j=1,6
          DDSDDE(i,j) = resp(6 + (i-1)*6 + j)
        end do
      end do
      do i=1,NSTATV
        STATEV(i) = resp(42 + i)
      end do
      RETURN
      END
