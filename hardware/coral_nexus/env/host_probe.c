#include <stdio.h>

/* Compiled and executed inside the locked container by verify_inside.sh.
 * The exact stdout line "host-probe-ok 44" is asserted byte-for-byte. */
int main(void) {
  int acc = 0;
  for (int i = 1; i <= 8; i++) {
    acc += i;
  }
  printf("host-probe-ok %d\n", acc + 8); /* 36 + 8 = 44 */
  return 0;
}
