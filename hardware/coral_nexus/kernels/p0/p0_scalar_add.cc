// P0 scalar workload (C005): runs on the UNMODIFIED upstream Coral core.
//
// Inputs are nonzero, independently hand-computable float pairs; the kernel
// self-checks every sum and returns 0 only when all eight outputs match the
// expected bit patterns. The core's tohost convention maps a 0 return to
// tohost==1 (success); any mismatch makes core_mini_axi_sim exit non-zero.
//
// Expected outputs (hand computed, IEEE-754 binary32):
//   1.0f + 0.5f  = 1.5f   (0x3FC00000)   5.0f + 0.25f = 5.25f  (0x40A80000)
//   2.0f + 1.0f  = 3.0f   (0x40400000)   6.0f + 0.5f  = 6.5f   (0x40D00000)
//   3.0f + 1.5f  = 4.5f   (0x40900000)   7.0f + 1.0f  = 8.0f   (0x41000000)
//   4.0f + 2.0f  = 6.0f   (0x40C00000)   8.0f + 2.0f  = 10.0f  (0x41200000)

#include <string.h>

float input1[8] __attribute__((section(".data"))) = {
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
};
float input2[8] __attribute__((section(".data"))) = {
  0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.5f, 1.0f, 2.0f,
};
float output[8] __attribute__((section(".data")));

int main() {
  for (int i = 0; i < 8; i++) {
    output[i] = input1[i] + input2[i];
  }
  const float expected[8] = {
    1.5f, 3.0f, 4.5f, 6.0f, 5.25f, 6.5f, 8.0f, 10.0f,
  };
  int ok = 1;
  for (int i = 0; i < 8; i++) {
    if (memcmp(&output[i], &expected[i], sizeof(float)) != 0) {
      ok = 0;
    }
  }
  return ok ? 0 : 2;
}
