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

# P0 BF16 KAT capture (C006): runs the EXACTLY-PRECISE bf16 KAT only and writes
# /out/coralnpu_kat.json from inside the simulation process. This package is
# an overlay onto the pinned upstream work root; the upstream tree itself is
# never modified.
coralflow_test(
    name = "p0_bf16_kat",
    size = "medium",
    srcs = ["p0_bf16_kat_bench.py"],
    hdl_toplevel = "VmeCoreMiniAxi",
    targets = ["verilator"],
    deps = [
        "//coralnpu_test_utils/sim_backends:verilator_test_fixture",
    ],
)
