/*
 * P10 BOA wire protocol selftest driver (C side).
 *
 * Consumes only the generated header boa_protocol.h:
 *
 *   roundtrip <in.bin> <out.bin>    decode every frame, re-encode, write out
 *                                   and print "frames=N kinds=NAME:count,..."
 *                                   (names sorted; the Python and Scala
 *                                   drivers print the same string)
 *   expect-fail <in.bin> <field>    decode must fail and name <field>
 *
 * Exit codes: 0 success, 1 unexpected success, 2 usage/IO error,
 * 3 protocol rejection, 4 rejection names a different field.
 */
#define BOA_PROTOCOL_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boa_protocol.h"

#define BOA_MAX_KIND 16
#define BOA_COUNT_BUFFER 512

static unsigned char* read_file(const char* path, size_t* length) {
  FILE* handle = fopen(path, "rb");
  if (handle == NULL) return NULL;
  if (fseek(handle, 0, SEEK_END) != 0) {
    fclose(handle);
    return NULL;
  }
  long size = ftell(handle);
  if (size < 0) {
    fclose(handle);
    return NULL;
  }
  rewind(handle);
  unsigned char* data = malloc((size_t)size + 1);
  if (data == NULL) {
    fclose(handle);
    return NULL;
  }
  if (fread(data, 1, (size_t)size, handle) != (size_t)size) {
    free(data);
    fclose(handle);
    return NULL;
  }
  fclose(handle);
  *length = (size_t)size;
  return data;
}

static int write_file(const char* path, const unsigned char* data,
                      size_t length) {
  FILE* handle = fopen(path, "wb");
  if (handle == NULL) return -1;
  size_t written = fwrite(data, 1, length, handle);
  fclose(handle);
  return written == length ? 0 : -1;
}

typedef struct {
  const char* name;
  int count;
} kind_count;

static int compare_kinds(const void* left, const void* right) {
  return strcmp(((const kind_count*)left)->name,
                ((const kind_count*)right)->name);
}

static void print_counts(const kind_count* counts, size_t unique,
                         size_t total) {
  char buffer[BOA_COUNT_BUFFER];
  size_t used = 0;
  used += (size_t)snprintf(buffer + used, sizeof(buffer) - used,
                           "frames=%zu kinds=", total);
  for (size_t index = 0; index < unique && used < sizeof(buffer); index++)
    used += (size_t)snprintf(buffer + used, sizeof(buffer) - used, "%s%s:%d",
                             index == 0 ? "" : ",", counts[index].name,
                             counts[index].count);
  printf("%s\n", buffer);
}

static int run_roundtrip(const char* inPath, const char* outPath) {
  boa_err err = {NULL, NULL};
  size_t length = 0;
  unsigned char* data = read_file(inPath, &length);
  if (data == NULL) {
    fprintf(stderr, "io-error: cannot read %s\n", inPath);
    return 2;
  }
  kind_count counts[BOA_MAX_KIND];
  memset(counts, 0, sizeof(counts));
  size_t unique = 0;
  size_t total = 0;
  unsigned char* out = calloc(length + 64, 1);
  size_t out_used = 0;
  if (out == NULL) {
    fprintf(stderr, "io-error: out of memory\n");
    free(data);
    return 2;
  }
  size_t pos = 0;
  while (pos < length) {
    boa_msg message;
    long used = boa_decode_frame(data + pos, length - pos, &message, &err);
    if (used < 0) {
      fprintf(stderr, "field=%s reason=%s\n", err.field, err.reason);
      free(data);
      free(out);
      return 3;
    }
    const char* name = boa_kind_name(message.kind);
    size_t slot = 0;
    while (slot < unique && strcmp(counts[slot].name, name) != 0) slot++;
    if (slot == unique) {
      counts[unique].name = name;
      counts[unique].count = 0;
      unique++;
    }
    counts[slot].count++;
    total++;
    long written = boa_encode_frame(&message, out + out_used,
                                    length + 64 - out_used, &err);
    boa_msg_free(&message);
    if (written < 0) {
      fprintf(stderr, "field=%s reason=%s\n", err.field, err.reason);
      free(data);
      free(out);
      return 3;
    }
    out_used += (size_t)written;
    pos += (size_t)used;
  }
  if (write_file(outPath, out, out_used) != 0) {
    fprintf(stderr, "io-error: cannot write %s\n", outPath);
    free(data);
    free(out);
    return 2;
  }
  qsort(counts, unique, sizeof(kind_count), compare_kinds);
  print_counts(counts, unique, total);
  free(data);
  free(out);
  return 0;
}

static int run_expect_fail(const char* inPath, const char* field) {
  boa_err err = {NULL, NULL};
  size_t length = 0;
  unsigned char* data = read_file(inPath, &length);
  if (data == NULL) {
    fprintf(stderr, "io-error: cannot read %s\n", inPath);
    return 2;
  }
  size_t pos = 0;
  int rejected = 0;
  const char* named = NULL;
  const char* reason = NULL;
  while (pos < length) {
    boa_msg message;
    long used = boa_decode_frame(data + pos, length - pos, &message, &err);
    if (used < 0) {
      rejected = 1;
      named = err.field;
      reason = err.reason;
      break;
    }
    boa_msg_free(&message);
    pos += (size_t)used;
  }
  free(data);
  if (!rejected) {
    fprintf(stderr, "unexpected-success: decode accepted %s\n", inPath);
    return 1;
  }
  printf("rejected field=%s reason=%s\n", named, reason);
  if (strstr(named, field) == NULL) {
    fprintf(stderr, "expected field substring '%s' not named\n", field);
    return 4;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 4 && strcmp(argv[1], "roundtrip") == 0)
    return run_roundtrip(argv[2], argv[3]);
  if (argc == 4 && strcmp(argv[1], "expect-fail") == 0)
    return run_expect_fail(argv[2], argv[3]);
  fprintf(stderr, "usage: %s roundtrip <in> <out> | expect-fail <in> <field>\n",
          argv[0]);
  return 2;
}
