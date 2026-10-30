// SPDX-License-Identifier: BSD-3-Clause
//
// P03 HardFloat primitive probe TSV batch harness (VNumericProbe).
//
// Usage:
//   numeric_probe <input.tsv> <output.tsv>
//
// Input rows (whitespace-separated, exactly 5 tokens, no header):
//   id op a b c
//   id   non-whitespace token, echoed verbatim
//   op   one of: fma add mul bf16 i32 (lowercase)
//   a,b,c  IEEE bit-pattern hex words: optional 0x/0X prefix, 1..8 hex digits
// Unary ops (bf16, i32) consume a; b and c are still required tokens and are
// validated but ignored. Empty trailing line (final newline) is fine; any
// interior blank line, wrong token count, bad word, or unknown op is a
// malformed input -> exit 2. The output file may be partial after an error.
//
// Output rows (complete batch, one per input row, same order):
//   id op bits flags
//   bits  fma/add/mul/i32: 0x + 8 lowercase hex digits; bf16: 0x + 4 lowercase
//         hex digits (16-bit pattern)
//   flags 0x + 2 lowercase hex digits, standard 5-bit layout
//   NV16/DZ8/OF4/UF2/NX1
// Measured outputs only: no golden values, no post-hoc normalization of DUT
// results, no process launches, no environment data in the file.
//
// Model/IO discipline (parent contract): exactly one VNumericProbe instance
// and one of each stream buffer for the whole batch; line/token buffers and
// the output string are created once and reused (capacity grows once at
// most); no per-vector heap allocation and no new/delete. RecFNToIN is a
// chisel3.Module, so clock/reset are genuinely driven: one reset pulse, then
// a settled low->high clock edge pair per vector (all probe submodules are
// combinational).
//
// Exit codes: 0 ok, 1 usage, 2 malformed input/unknown op, 3 write/IO error.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "VNumericProbe.h"
#include "verilated.h"

namespace {

constexpr uint32_t kFlushBytes = 1u << 16;

enum Op : uint8_t {
  OP_FMA = 0,
  OP_ADD = 1,
  OP_MUL = 2,
  OP_BF16 = 3,
  OP_I32 = 4
};

struct Token {
  char* ptr;
  size_t len;
};

bool op_encoding(const char* s, size_t n, Op* enc) {
  if (n == 3 && std::memcmp(s, "fma", 3) == 0) {
    *enc = OP_FMA;
    return true;
  }
  if (n == 3 && std::memcmp(s, "add", 3) == 0) {
    *enc = OP_ADD;
    return true;
  }
  if (n == 3 && std::memcmp(s, "mul", 3) == 0) {
    *enc = OP_MUL;
    return true;
  }
  if (n == 4 && std::memcmp(s, "bf16", 4) == 0) {
    *enc = OP_BF16;
    return true;
  }
  if (n == 3 && std::memcmp(s, "i32", 3) == 0) {
    *enc = OP_I32;
    return true;
  }
  return false;
}

bool is_hex_digit(char ch) {
  return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
         (ch >= 'A' && ch <= 'F');
}

int hex_value(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  return ch - 'A' + 10;
}

// Optional 0x/0X prefix, 1..8 hex digits, at most 32 bits of payload.
bool parse_word(const char* s, size_t n, uint32_t* out) {
  size_t i = 0;
  if (n >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) i = 2;
  if (i >= n || n - i > 8) return false;
  uint32_t v = 0;
  for (; i < n; ++i) {
    if (!is_hex_digit(s[i])) return false;
    v = (v << 4) | static_cast<uint32_t>(hex_value(s[i]));
  }
  *out = v;
  return true;
}

bool is_space(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\v' || ch == '\f';
}

// Splits the line in place into at most 6 token slices; false when more
// than 5 tokens are present or fewer than 5 tokens on a non-empty line.
bool split_tokens(char* line, size_t n, Token* toks, size_t* count) {
  size_t i = 0;
  size_t k = 0;
  while (i < n) {
    while (i < n && is_space(line[i])) ++i;
    if (i >= n) break;
    size_t start = i;
    while (i < n && !is_space(line[i])) ++i;
    if (k == 6) return false;  // more than 5 tokens
    toks[k].ptr = line + start;
    toks[k].len = i - start;
    ++k;
  }
  if (k != 5) return false;
  *count = k;
  return true;
}

// Drives one vector through the single probe instance with a settled
// clock edge pair; the design is combinational, so outputs are stable
// after the first eval of each phase.
void drive(VNumericProbe& dut, Op op, uint32_t a, uint32_t b, uint32_t c,
           uint32_t* out, uint8_t* flags) {
  dut.op = static_cast<uint8_t>(op);
  dut.a = a;
  dut.b = b;
  dut.c = c;
  dut.clock = 0;
  dut.eval();
  dut.clock = 1;
  dut.eval();
  *out = static_cast<uint32_t>(dut.out);
  *flags = static_cast<uint8_t>(dut.flags & 0x1F);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: numeric_probe <input.tsv> <output.tsv>\n");
    return 1;
  }
  Verilated::commandArgs(argc, argv);

  std::FILE* in = std::fopen(argv[1], "rb");
  if (in == nullptr) {
    std::fprintf(stderr, "numeric_probe: cannot open input %s\n", argv[1]);
    return 1;
  }
  std::FILE* outf = std::fopen(argv[2], "wb");
  if (outf == nullptr) {
    std::fprintf(stderr, "numeric_probe: cannot open output %s\n", argv[2]);
    std::fclose(in);
    return 1;
  }

  VNumericProbe dut;
  // Genuine reset pulse for the Module top (RecFNToIN is a Module), then idle
  // clock low before the per-vector settle edges.
  dut.reset = 1;
  dut.clock = 0;
  dut.eval();
  dut.clock = 1;
  dut.eval();
  dut.reset = 0;

  std::string line;    // reused input line buffer
  std::string outbuf;  // reused output buffer, flushed in kFlushBytes chunks
  line.reserve(4096);
  outbuf.reserve(kFlushBytes + 4096);

  char cell[64];
  Token toks[6];
  size_t ntok = 0;
  int lineno = 0;
  int rc = 0;
  int got = 0;
  while ((got = std::fgetc(in)) != EOF) {
    ++lineno;
    line.clear();
    while (got != EOF && got != '\n') {
      line.push_back(static_cast<char>(got));
      got = std::fgetc(in);
    }
    const size_t len = line.size();

    // Blank line: only a trailing newline is tolerated (the loop above never
    // re-enters after EOF); any interior blank line is malformed.
    bool blank = true;
    for (size_t i = 0; i < len; ++i) {
      if (!is_space(line[i])) {
        blank = false;
        break;
      }
    }
    if (blank) {
      std::fprintf(stderr, "numeric_probe: line %d: blank line\n", lineno);
      rc = 2;
      break;
    }

    if (!split_tokens(&line[0], len, toks, &ntok)) {
      std::fprintf(stderr, "numeric_probe: line %d: expected 5 tokens\n",
                   lineno);
      rc = 2;
      break;
    }
    Op op = OP_FMA;
    if (!op_encoding(toks[1].ptr, toks[1].len, &op)) {
      std::fprintf(stderr, "numeric_probe: line %d: unknown op\n", lineno);
      rc = 2;
      break;
    }
    uint32_t words[3] = {0, 0, 0};
    bool words_ok = true;
    for (size_t w = 0; w < 3; ++w) {
      words_ok = parse_word(toks[2 + w].ptr, toks[2 + w].len, &words[w]);
      if (!words_ok) break;
    }
    if (!words_ok) {
      std::fprintf(stderr, "numeric_probe: line %d: bad hex word\n", lineno);
      rc = 2;
      break;
    }

    uint32_t bits = 0;
    uint8_t flags = 0;
    drive(dut, op, words[0], words[1], words[2], &bits, &flags);

    outbuf.append(toks[0].ptr, toks[0].len);
    outbuf.push_back('\t');
    outbuf.append(toks[1].ptr, toks[1].len);
    outbuf.push_back('\t');
    int n = 0;
    if (op == OP_BF16) {
      n = std::snprintf(cell, sizeof cell, "0x%04x\t0x%02x\n", bits & 0xFFFFu,
                        flags);
    } else {
      n = std::snprintf(cell, sizeof cell, "0x%08x\t0x%02x\n", bits, flags);
    }
    if (n <= 0) {
      std::fprintf(stderr, "numeric_probe: line %d: format failure\n", lineno);
      rc = 3;
      break;
    }
    outbuf.append(cell, static_cast<size_t>(n));
    if (outbuf.size() >= kFlushBytes) {
      if (std::fwrite(outbuf.data(), 1, outbuf.size(), outf) != outbuf.size()) {
        std::fprintf(stderr, "numeric_probe: write failure\n");
        rc = 3;
        break;
      }
      outbuf.clear();
    }
  }
  if (rc == 0 && std::ferror(in)) {
    std::fprintf(stderr, "numeric_probe: read failure\n");
    rc = 3;
  }
  if (rc != 0 && !outbuf.empty()) {
    // Best-effort: keep the measured rows that preceded the error so the
    // partial output ends exactly at the failing line.
    (void)std::fwrite(outbuf.data(), 1, outbuf.size(), outf);
  }
  if (rc == 0) {
    if (!outbuf.empty() &&
        std::fwrite(outbuf.data(), 1, outbuf.size(), outf) != outbuf.size()) {
      std::fprintf(stderr, "numeric_probe: write failure\n");
      rc = 3;
    }
    if (std::fflush(outf) != 0 || std::fclose(outf) != 0) {
      std::fprintf(stderr, "numeric_probe: output close failure\n");
      std::fclose(in);
      dut.final();
      return 3;
    }
    std::fclose(in);
    dut.final();
    return 0;
  }
  std::fflush(outf);
  std::fclose(outf);
  std::fclose(in);
  dut.final();
  return rc;
}
