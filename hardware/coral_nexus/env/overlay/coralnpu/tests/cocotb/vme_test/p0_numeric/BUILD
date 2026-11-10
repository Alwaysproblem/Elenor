# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//rules:coralflow_test.bzl", "coralflow_test")

# C011 numeric probe: runs directed BF16 vectors through the UNMODIFIED upstream
# VmeCoreMiniAxi Verilator model and records input/actual bits. Overlay package; the
# upstream tree is never modified. It reuses the ELF of //tests/cocotb/vme_test.
coralflow_test(
    name = "p0_numeric_probe",
    size = "medium",
    srcs = ["p0_numeric_probe_bench.py"],
    hdl_toplevel = "VmeCoreMiniAxi",
    targets = ["verilator"],
    deps = [
        "//coralnpu_test_utils/sim_backends:verilator_test_fixture",
    ],
)
