// C receive model of one Tile link (C017). Built only from the GENERATED
// cn_messages.h; it re-implements the rules of the generated Python Rx
// independently so a layout or rule drift shows up as a line mismatch. stdin :
// one message per line "<channel index> <dir index> <hex bytes>" (hex may be
// empty) stdout: one canonical line per message, identical to wire.Rx.line().
#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cn_messages.h"

#define MAX_BYTES 4096

typedef struct {
  uint64_t txn;
  int msg;
  uint64_t stage;
} dd_t;

static uint64_t next_seq[CN_LINKS];
static dd_t dedup[CN_DEDUP_ENTRIES];
static int dedup_n;
static struct {
  uint64_t txn;
  int msg;
} reqs[CN_DEDUP_ENTRIES];
static int reqs_n;
static uint64_t live[CN_CONTEXT_SLOTS];
static uint64_t effects;

static uint64_t word(const uint8_t* b, int i) {
  uint64_t v = 0;
  for (int k = 7; k >= 0; --k) v = (v << 8) | b[i * 8 + k];
  return v;
}

static int has_req(uint64_t txn, int msg) {
  for (int i = 0; i < reqs_n; ++i)
    if (reqs[i].txn == txn && reqs[i].msg == msg) return 1;
  return 0;
}

static int deliver(int ch, int dir, const uint8_t* b, size_t n, uint64_t* w,
                   int* nw) {
  *nw = 0;
  if (n < CN_HEADER_WORDS * 8) return CN_ERR_TRUNCATED;
  uint64_t type = word(b, 0), payload = word(b, 1), seq = word(b, 2),
           reserved = word(b, 3);
  if (type >= CN_MSG_COUNT) return CN_ERR_UNKNOWN_MSG;
  if (reserved != 0) return CN_ERR_RESERVED;
  const cn_msg_t* m = &cn_msgs[type];
  if (payload != m->payload_words) return CN_ERR_LENGTH_MISMATCH;
  size_t total = (size_t)(CN_HEADER_WORDS + m->payload_words) * 8;
  if (n < total) return CN_ERR_TRUNCATED;
  if (n > total) return CN_ERR_TRAILING;
  if (m->channel != ch || m->dir != dir) return CN_ERR_WRONG_PORT;
  for (int i = 0; i < m->payload_words; ++i)
    w[i] = word(b, CN_HEADER_WORDS + i);
  *nw = m->payload_words;
  for (int i = 0; i < m->nfields; ++i) {
    const cn_field_t* f = &m->fields[i];
    uint64_t v = w[f->offset];
    if (f->kind == CN_KIND_ENUM && v >= f->limit) return CN_ERR_ENUM_RANGE;
    if (f->kind == CN_KIND_BOOL && v > 1) return CN_ERR_BOOL_RANGE;
    if (f->kind == CN_KIND_BITSET && (v & ~f->limit))
      return CN_ERR_BITSET_RANGE;
  }
  for (int i = 0; i < m->nconds; ++i) {
    const cn_cond_t* c = &m->conds[i];
    if (c->kind == 0 && w[c->a] != w[c->b]) return CN_ERR_CONDITION;
    if (c->kind == 1 && ((w[c->a] != 0) != (w[c->b] == c->value)))
      return CN_ERR_CONDITION;
  }
  // stateful rules
  int64_t slot = m->ctx_slot >= 0 ? (int64_t)w[m->ctx_slot] : -1;
  if (m->ctx_slot >= 0 && w[m->ctx_slot] >= CN_CONTEXT_SLOTS)
    return CN_ERR_CAPACITY;
  int link = ch * CN_DIRS + dir;
  if (seq < next_seq[link]) return CN_ERR_TRANSPORT_REPLAY;
  if (seq > next_seq[link]) return CN_ERR_INCOMPLETE;
  next_seq[link]++;
  uint64_t gen = m->ctx_gen >= 0 ? w[m->ctx_gen] : 0;
  if (slot >= 0 && gen < live[slot]) return CN_ERR_STALE_GENERATION;
  int has_txn = m->txn >= 0;
  uint64_t txn = has_txn ? w[m->txn] : 0;
  if (!strcmp(m->name, "ACK") && has_txn && !has_req(txn, (int)w[m->ack_for]))
    return CN_ERR_ACK_MISMATCH;
  if (has_txn) {
    uint64_t stage = m->stage >= 0 ? w[m->stage] : 0;
    for (int i = 0; i < dedup_n; ++i)
      if (dedup[i].txn == txn && dedup[i].msg == (int)type &&
          dedup[i].stage == stage)
        return CN_ERR_DUPLICATE_STAGE;
    if (dedup_n >= CN_DEDUP_ENTRIES) return CN_ERR_CAPACITY;
    dedup[dedup_n++] = (dd_t){txn, (int)type, stage};
    if (!has_req(txn, (int)type)) {
      reqs[reqs_n].txn = txn;
      reqs[reqs_n].msg = (int)type;
      reqs_n++;
    }
  }
  if (slot >= 0 && m->establishes) live[slot] = gen;
  effects++;
  return CN_ERR_OK;
}

int main(void) {
  static char line[MAX_BYTES * 2 + 64];
  while (fgets(line, sizeof line, stdin)) {
    int ch, dir, pos = 0;
    if (sscanf(line, "%d %d %n", &ch, &dir, &pos) < 2) return 2;
    uint8_t blob[MAX_BYTES];
    size_t n = 0;
    for (const char* p = line + pos;
         isxdigit((unsigned char)p[0]) && isxdigit((unsigned char)p[1]);
         p += 2) {
      char pair[3] = {p[0], p[1], 0};
      blob[n++] = (uint8_t)strtoul(pair, NULL, 16);
    }
    uint64_t w[64];
    int nw;
    int code = deliver(ch, dir, blob, n, w, &nw);
    uint64_t sum = 0;
    for (int i = 0; i < CN_CONTEXT_SLOTS; ++i) sum += live[i];
    printf("%s eff=%" PRIu64 " dd=%d seq=", cn_error_names[code], effects,
           dedup_n);
    for (int i = 0; i < CN_LINKS; ++i)
      printf(i ? ",%" PRIu64 : "%" PRIu64, next_seq[i]);
    printf(" live=%" PRIu64 " f=", sum);
    if (code != CN_ERR_OK) {
      printf("-");
    } else {
      for (int i = 0; i < nw; ++i) printf(i ? ",%" PRIx64 : "%" PRIx64, w[i]);
    }
    printf("\n");
  }
  return 0;
}
