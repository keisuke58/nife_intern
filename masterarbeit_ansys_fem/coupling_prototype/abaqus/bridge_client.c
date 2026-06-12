/* bridge_client.c — TCP client called from umat.f via ISO_C_BINDING.
 *
 * Status: SKELETON for the Abaqus machine. Sends one ASCII request record to
 * umat_server.py and parses one response record. Matches umat_server.py's
 * protocol:  send "NSTATV STRAN(6) DSTRAN(6) STATEV E nu\n", recv floats.
 *
 * Fortran calls:  int bridge_eval(double* req, int nreq, double* resp, int nresp)
 * Returns 0 on success, nonzero on any socket/parse error.
 *
 * Host/port come from env BRIDGE_HOST (default 127.0.0.1) / BRIDGE_PORT (50007),
 * so the same binary works on the Abaqus box without recompiling.
 *
 * Build alongside umat.f, e.g.:
 *   cc -c -fPIC bridge_client.c -o bridge_client.o
 *   abaqus job=one_element user=umat.f interactive   # ensure .o is linked
 * (Abaqus name-mangles Fortran externals; bridge_eval is declared lowercase
 *  with a trailing underscore for gfortran/ifort. Adjust per compiler.)
 *
 * NOTE: opening a fresh TCP connection per Gauss point is fine for a
 * correctness prototype but slow for production — keep a persistent socket
 * (cache the fd in a static) or move to in-process / tabulated once trusted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static int connect_server(void) {
    const char *host = getenv("BRIDGE_HOST"); if (!host) host = "127.0.0.1";
    const char *ps   = getenv("BRIDGE_PORT"); int port = ps ? atoi(ps) : 50007;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET; sa.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) { close(fd); return -1; }
    if (connect(fd, (struct sockaddr*)&sa, sizeof sa) < 0) { close(fd); return -1; }
    return fd;
}

/* trailing underscore = gfortran/ifort default mangling for EXTERNAL BRIDGE_EVAL */
int bridge_eval_(double *req, int *nreq, double *resp, int *nresp) {
    int fd = connect_server();
    if (fd < 0) return 1;

    char buf[8192]; int off = 0;
    for (int i = 0; i < *nreq; ++i)
        off += snprintf(buf + off, sizeof buf - off, "%.16e ", req[i]);
    off += snprintf(buf + off, sizeof buf - off, "\n");
    if (write(fd, buf, off) != off) { close(fd); return 2; }

    /* read a full line */
    int n = 0, r;
    while (n < (int)sizeof buf - 1 && (r = read(fd, buf + n, 1)) == 1) {
        if (buf[n] == '\n') { n++; break; }
        n++;
    }
    buf[n] = '\0';
    close(fd);

    int k = 0; char *tok = strtok(buf, " \t\n");
    while (tok && k < *nresp) { resp[k++] = atof(tok); tok = strtok(NULL, " \t\n"); }
    return (k == *nresp) ? 0 : 3;
}
