# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-specific configuration for SOL ExecBench benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClockPreset:
    """GPU and DRAM clock frequencies for stable benchmarking."""

    gpu_clk_mhz: int
    dram_clk_mhz: int


CLOCK_LOCK_PRESETS: dict[str, ClockPreset] = {
    "NVIDIA B200": ClockPreset(gpu_clk_mhz=1500, dram_clk_mhz=3996),
    "NVIDIA H100": ClockPreset(gpu_clk_mhz=1410, dram_clk_mhz=1593),
    "NVIDIA A100": ClockPreset(gpu_clk_mhz=1065, dram_clk_mhz=1215),
}


def get_clock_preset(device_name: str) -> Optional[ClockPreset]:
    """Get the clock preset for a given GPU device name.

    Returns None if the device is not in the preset table.

    Parameters
    ----------
    device_name : str
        The GPU device name string (e.g., from torch.cuda.get_device_name()).

    Returns
    -------
    Optional[ClockPreset]
        Clock preset with GPU and DRAM frequencies, or None if not in presets.
    """
    for key, preset in CLOCK_LOCK_PRESETS.items():
        if key in device_name:
            return preset
    return None
