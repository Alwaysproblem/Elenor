// C021 pinned Spike reference C ABI shim.
//
// Wraps coralnpu::sim::SpikeSimulator (pinned upstream
// sw/coralnpu_sim/spike_simulator.{h,cc}) so the Python reference driver
// executes the exact pinned Spike ISA semantics via LoadProgram / Step /
// IsHalted / ReadRegister / WriteRegister / ReadMemory / WriteMemory /
// ReadVectorRegister / WriteVectorRegister / ReadCurrentMatrixTile / Reset.
// No DUT RTL, no Python float, and no unverified Nexus-private instructions are
// invented here.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sw/coralnpu_sim/spike_simulator.h"

namespace {

struct Handle {
  std::unique_ptr<coralnpu::sim::SpikeSimulator> sim;
  std::string last_error;
};

}  // namespace

extern "C" {

struct cn_mem_region {
  uint64_t start_address;
  uint64_t length;
};

void* cn_spike_create(const char* isa, const char* priv,
                      const cn_mem_region* regions, size_t num_regions) {
  try {
    coralnpu::sim::SpikeSimulatorOptions opts;
    if (isa != nullptr && isa[0] != '\0') {
      opts.isa = isa;
    }
    if (priv != nullptr && priv[0] != '\0') {
      opts.priv = priv;
    }
    if (regions != nullptr && num_regions > 0) {
      opts.memory_regions.reserve(num_regions);
      for (size_t i = 0; i < num_regions; ++i) {
        opts.memory_regions.push_back(
            {regions[i].start_address, static_cast<size_t>(regions[i].length)});
      }
    }
    auto* h = new Handle();
    h->sim = std::make_unique<coralnpu::sim::SpikeSimulator>(opts);
    return h;
  } catch (...) {
    return nullptr;
  }
}

void cn_spike_destroy(void* raw) { delete static_cast<Handle*>(raw); }

int cn_spike_reset(void* raw) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim) {
    return -1;
  }
  try {
    h->sim->Reset();
    return 0;
  } catch (const std::exception& e) {
    h->last_error = e.what();
    return -1;
  } catch (...) {
    h->last_error = "unknown reset exception";
    return -1;
  }
}

int cn_spike_load_program(void* raw, const char* elf_path, int has_entry,
                          uint32_t entry_point) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || elf_path == nullptr) {
    return -1;
  }
  try {
    std::optional<uint32_t> entry = std::nullopt;
    if (has_entry != 0) {
      entry = entry_point;
    }
    return h->sim->LoadProgram(elf_path, entry) ? 0 : -1;
  } catch (const std::exception& e) {
    h->last_error = e.what();
    return -1;
  } catch (...) {
    h->last_error = "unknown load exception";
    return -1;
  }
}

int cn_spike_step(void* raw, int num_steps) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim) {
    return -1;
  }
  try {
    return h->sim->Step(num_steps);
  } catch (const std::exception& e) {
    h->last_error = e.what();
    return -2;
  } catch (...) {
    h->last_error = "unknown step exception";
    return -2;
  }
}

int cn_spike_is_halted(void* raw) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim) {
    return 1;
  }
  return h->sim->IsHalted() ? 1 : 0;
}

uint64_t cn_spike_get_pc(void* raw) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim) {
    return 0;
  }
  return h->sim->GetPC();
}

uint64_t cn_spike_get_cycle_count(void* raw) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim) {
    return 0;
  }
  return h->sim->GetCycleCount();
}

int cn_spike_read_register(void* raw, const char* name, uint64_t* out) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || name == nullptr || out == nullptr) {
    return -1;
  }
  try {
    *out = h->sim->ReadRegister(name);
    return 0;
  } catch (const std::exception& e) {
    h->last_error = e.what();
    return -1;
  } catch (...) {
    h->last_error = "unknown read_register exception";
    return -1;
  }
}

int cn_spike_write_register(void* raw, const char* name, uint64_t value) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || name == nullptr) {
    return -1;
  }
  try {
    h->sim->WriteRegister(name, value);
    return 0;
  } catch (const std::exception& e) {
    h->last_error = e.what();
    return -1;
  } catch (...) {
    h->last_error = "unknown write_register exception";
    return -1;
  }
}

int cn_spike_read_vector_register(void* raw, int v_idx, uint32_t words_out[4]) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || words_out == nullptr) {
    return -1;
  }
  return h->sim->ReadVectorRegister(v_idx, words_out) ? 0 : -1;
}

int cn_spike_write_vector_register(void* raw, int v_idx,
                                   const uint32_t words_in[4]) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || words_in == nullptr) {
    return -1;
  }
  return h->sim->WriteVectorRegister(v_idx, words_in) ? 0 : -1;
}

int cn_spike_read_matrix_tile(void* raw, uint32_t tile, uint32_t* tew_out,
                              uint32_t* words_out, size_t max_words,
                              size_t* actual_words) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || tew_out == nullptr ||
      actual_words == nullptr) {
    return -1;
  }
  std::vector<uint32_t> words;
  if (!h->sim->ReadCurrentMatrixTile(tile, tew_out, &words)) {
    return -1;
  }
  *actual_words = words.size();
  if (words_out != nullptr) {
    size_t copy_n = words.size() < max_words ? words.size() : max_words;
    std::memcpy(words_out, words.data(), copy_n * sizeof(uint32_t));
  }
  return 0;
}

int cn_spike_read_memory(void* raw, uint64_t address, void* buffer,
                         size_t length) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || (buffer == nullptr && length > 0)) {
    return -1;
  }
  return h->sim->ReadMemory(address, buffer, length) ? 0 : -1;
}

int cn_spike_write_memory(void* raw, uint64_t address, const void* buffer,
                          size_t length) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr || !h->sim || (buffer == nullptr && length > 0)) {
    return -1;
  }
  return h->sim->WriteMemory(address, buffer, length) ? 0 : -1;
}

const char* cn_spike_last_error(void* raw) {
  auto* h = static_cast<Handle*>(raw);
  if (h == nullptr) {
    return "null handle";
  }
  return h->last_error.c_str();
}

}  // extern "C"
