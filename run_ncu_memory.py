#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This module wraps three tasks:
1) Collect core metrics for specified CUDA kernels with Nsight Compute into CSV (`profile_bench`).
2) Extract and clean those metrics into a DataFrame from the CSV (`load_ncu_metrics`).
3) Convert the metrics table into a string suitable for inclusion in an LLM prompt (`metrics_to_prompt`).

Typical usage:
    from gpu_profile_utils import profile_bench, load_ncu_metrics, metrics_to_prompt

    kernel_names = extract_cuda_kernel_names(test_kernel)
    csv_path = profile_bench(kernel_names=kernel_names)
    df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",))
    prompt_block = metrics_to_prompt(df)
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence, Union, Any, Dict, Tuple
import json, math
import pandas as pd
import numpy as np


__all__ = [
    "METRICS",
    "METRICS_NEW",
    "SECTIONS",
    "METRIC_COLUMNS",
    "METRIC_COLUMNS_NEW",
    "profile_bench",
    "load_ncu_metrics",
    "metrics_to_prompt",
]

# Keep only the core "kernel performance related" metrics (aligned with `ncu --metrics`)
METRICS = ",".join([
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    # "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    # "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # "l1tex__t_sector_hit_rate.pct",
    # "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # "lts__t_sector_hit_rate.pct",
    # "lts__throughput.avg.pct_of_peak_sustained_active",
    # "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    # "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    # "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    # "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    # "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    # "smsp__sass_average_branch_targets_threads_uniform.pct",
])


METRICS_NEW = ",".join([
            # Time / cycles
        "gpu__time_duration.avg",
        "sm__cycles_active.avg",
        "sm__cycles_elapsed.avg",

        # Compute vs memory
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",

        # Warp / occupancy
        "sm__warps_active.avg.pct_of_peak_sustained_active",

        # Memory hierarchy
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",

        # Launch / limits
        "launch__registers_per_thread",
        "launch__occupancy_limit_registers",        
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
        "launch__occupancy_per_register_count",
        "launch__occupancy_per_shared_mem_size",
        "launch__occupancy_per_block_size",
        "launch__block_size",
        "launch__grid_size",
        "launch__shared_mem_per_block",
        "launch__waves_per_multiprocessor",

        # Warp stall metrics (base names - will match all variants: .avg, .max_rate, .pct, .ratio)
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active",
        "smsp__warp_issue_stalled_no_instruction_per_warp_active",
        "smsp__warp_issue_stalled_not_selected_per_warp_active",

        # Branch metrics (base names - will match all variants)
        "smsp__sass_branch_targets_threads_divergent",
        "smsp__sass_branch_targets_threads_uniform",
        "smsp__thread_inst_executed_pred_on_per_inst_executed",
        "smsp__warps_eligible",

        # Instruction mix. Without these the profile cannot express HOW MUCH WORK
        # each issued instruction does, only how busy the SM is -- and on a
        # tensor-core kernel that is the difference between winning and losing.
        # Measured on 002/own_winograd (wmma m16n16k8) against the cuDNN kernel
        # it was replacing, same conv, same GPU:
        #     tensor inst   33,554,432  vs   2,359,296   (14x MORE for 2.25x LESS math)
        #     total inst   977,600,512  vs  46,712,360
        #     -> 512 MACs per tensor instruction vs 16,384; cuDNN issues wgmma
        #        64x128x8, the candidate issues mma.m16n8k4.
        # Every metric already in this list looked unremarkable on that kernel
        # (sm__throughput 45%, dram 1.5%) or actively misleading (occupancy 24.7%
        # flagged as the limiter, while the kernel that beats it by 11x runs at
        # 18.2%). Derive MACs/instruction from these two and the bottleneck is
        # visible instead of inferred.
        "sm__inst_executed_pipe_tensor",
        "smsp__inst_executed",

])


# List of sections we want to collect. We'll expand these into multiple
# '--section=<name>' arguments because this ncu version does not support
# comma-separated lists in a single --section option.
SECTIONS = "Occupancy,WarpStateStats,MemoryWorkloadAnalysis,ComputeWorkloadAnalysis,SpeedOfLight"

# List version for convenient header selection
METRIC_COLUMNS: List[str] = [s.strip() for s in METRICS.split(",")]

# List version for convenient header selection
METRIC_COLUMNS_NEW: List[str] = [s.strip() for s in METRICS_NEW.split(",")]


def profile_bench(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    kernel_file: Optional[Union[str, Path]] = None,  # New: explicitly specify which kernel file to profile
    conda_bin: str = str(Path(sys.executable).parent),
    out_csv: Union[str, Path] = "ncu_temp.csv",
    repeat: int = 10,
    device_idx: Optional[int] = None,
    timeout_override: Optional[int] = None,  # New: override timeout for specific tasks (in seconds)
) -> Path:
    ncu_bin = os.environ.get("NCU_BIN") or shutil.which("ncu") or "ncu"
    csv_path = Path(out_csv).resolve()

    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    tmp_ncu_dir = Path.home() / "ncu-tmp"
    tmp_ncu_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_ncu_dir)
    
    # ========== FIX: reuse the existing PyTorch cache to avoid rebuilding under ncu ==========
    # Leave TORCH_EXTENSIONS_DIR unset so PyTorch falls back to ~/.cache/torch_extensions/
    # That way previously compiled .so files are reused, avoiding a rebuild under ncu that would hang
    # 
    # The original code forced a temp directory, so every ncu run recompiled from scratch:
    # tmp_ext = tempfile.mkdtemp(prefix="torch_ext_")
    # env["TORCH_EXTENSIONS_DIR"] = tmp_ext

    # Config file paths
    _repo_root = Path(__file__).resolve().parent
    config_metrics = str(_repo_root / 'config_metrics.ncu-cfg')
    config_section = str(_repo_root / 'config_section.ncu-cfg')
    
    # Resolve the kernel_file argument
    kernel_file_path = None
    if kernel_file:
        kernel_file_path = Path(kernel_file).resolve()
        if not kernel_file_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file_path}")
        print(f"[ncu] Profiling kernel from file: {kernel_file_path}", flush=True)

    def build_cmd(config_file: str, log_file: str, single_kernel_name: Optional[str] = None) -> List[str]:
        """Build the ncu command
        
        Args:
            config_file: path to the ncu config file
            log_file: path to the output CSV file
            single_kernel_name: if given, profile only this single kernel (used for the multi-kernel case)
        """
        cmd = [
            ncu_bin,
            "--config-file-path",
            config_file,
            f"--log-file={log_file}",
            sys.executable,
            bench_py,
        ]

        # Optional: respect caller-specified CUDA device index for the benchmark script
        if device_idx is not None:
            cmd.extend(["--device-idx", str(device_idx)])
        
        # If a kernel_file was given, add the --test argument
        if kernel_file_path:
            cmd.extend(["--test", str(kernel_file_path)])

        cmd.extend(["--repeat", str(repeat)])

        # Insert --kernel-name after --log-file, before sys.executable
        insert_pos = cmd.index(sys.executable)
        
        # If a single kernel name was given (multi-kernel case), use it
        if single_kernel_name:
            escaped_name = re.escape(single_kernel_name)
            pattern = f"\\b{escaped_name}\\b"
            cmd.insert(insert_pos, f"--kernel-name=::regex:{pattern}")
        elif kernel_names:
            names = sorted({k.strip() for k in kernel_names if k and k.strip()})
            if names:
                if len(names) == 1:
                    # Single name: use regex to match kernel name with optional parameter signature
                    kernel_name = names[0]
                    escaped_name = re.escape(kernel_name)
                    pattern = f"\\b{escaped_name}\\b"
                    cmd.insert(insert_pos, f"--kernel-name=::regex:{pattern}")
                else:
                    # Multiple names: merge into a single regex (legacy behavior, may miss kernels)
                    pattern = "|".join(re.escape(k) for k in names)
                    cmd.insert(insert_pos, f"--kernel-name=::regex:^({pattern})(\\(|$)")
        
        return cmd

    # ========== First run: use config_metrics.ncu-cfg ==========
    # Timeout selection: when timeout_override is given, both section and metrics use that value
    # Otherwise metrics gets 10 minutes and section gets 15 minutes
    metrics_timeout = timeout_override if timeout_override is not None else 600
    section_timeout = timeout_override if timeout_override is not None else 900
    
    def run_ncu_with_timeout(cmd: List[str], phase: str, timeout: int, output_csv: Optional[Path] = None) -> bool:
        """Run the ncu command with timeout protection (using the given timeout)
        
        Returns:
            bool: True if successful, False if timeout or error
        """
        print(f"[ncu] [{phase}] running:", " ".join(cmd), flush=True)
        
        try:
            # Use Popen rather than run so the whole process tree can be force-killed on timeout
            # NOTE: stderr is redirected to subprocess.STDOUT so error messages show up live
            # This matters a lot when debugging CUDA kernel crashes
            proc = subprocess.Popen(
                cmd, 
                env=env, 
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # merge stderr into stdout so errors are visible live
                preexec_fn=os.setsid  # start a new process group so the whole process tree can be killed
            )
            
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                # Since stderr=subprocess.STDOUT, stderr is None and all output lands in stdout
                if proc.returncode != 0:
                    # Dump the full stdout (which includes stderr) for debugging
                    if stdout:
                        sys.stderr.write("=" * 80 + "\n")
                        sys.stderr.write(f"[ncu] [{phase}] Process exited with return code {proc.returncode}\n")
                        sys.stderr.write("=" * 80 + "\n")
                        sys.stderr.write(stdout)
                        sys.stderr.write("=" * 80 + "\n")
                    raise SystemExit(proc.returncode)
                # Finished successfully
                print(f"[ncu] [{phase}] completed successfully", flush=True)
                return True
            except subprocess.TimeoutExpired:
                # Timed out: kill the entire process group
                print(f"[ncu] [{phase}] Timeout after {timeout} seconds, terminating process group...", flush=True)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    pass  # process already exited
                
                # On timeout, dump the last few lines of the CSV file (if it exists)
                if output_csv and output_csv.exists():
                    try:
                        with open(output_csv, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"[ncu] [{phase}] Last 10 lines of CSV file ({output_csv}):", flush=True)
                                for line in lines[-10:]:
                                    print(f"  {line.rstrip()}", flush=True)
                    except Exception as e:
                        print(f"[ncu] [{phase}] Failed to read CSV file: {e}", flush=True)
                
                error_msg = f"[ncu] [{phase}] Profiling timed out after {timeout} seconds"
                print(error_msg, flush=True)
                raise RuntimeError(error_msg)
                
        except RuntimeError:
            raise
        except Exception as e:
            # Make sure the process is cleaned up
            try:
                if 'proc' in locals() and proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            raise
    
    # ========== Strategy: single kernel vs multiple kernels ==========
    # With multiple kernels, run ncu once per kernel so that every kernel actually gets profiled
    # This avoids the --launch-count limit, which stops the later kernels from being captured
    if kernel_names and len(kernel_names) > 1:
        print(f"[ncu] Multiple kernels detected ({len(kernel_names)}), profiling each kernel separately to ensure all are captured")
        
        # Run ncu separately for each kernel
        all_metrics_csvs = []
        all_section_csvs = []
        kernel_name_map = {}  # Map kernel index to actual kernel name for markers
        
        for idx, kernel_name in enumerate(kernel_names):
            print(f"[ncu] Profiling kernel {idx+1}/{len(kernel_names)}: {kernel_name}")
            kernel_name_map[idx] = kernel_name  # Store mapping for use in merge function
            
            # Metrics phase for this kernel (profile this kernel on its own)
            kernel_metrics_csv = csv_path.parent / f"{csv_path.stem}_kernel{idx}_metrics{csv_path.suffix}"
            run_ncu_with_timeout(
                build_cmd(config_metrics, str(kernel_metrics_csv), single_kernel_name=kernel_name), 
                f"metrics[kernel{idx}]", 
                metrics_timeout,
                output_csv=kernel_metrics_csv
            )
            print(f"[ncu] [metrics] Completed for kernel {idx+1}/{len(kernel_names)}: {kernel_name}", flush=True)
            all_metrics_csvs.append(kernel_metrics_csv)
            
            # Section phase for this kernel (profile this kernel on its own)
            kernel_section_csv = csv_path.parent / f"{csv_path.stem}_kernel{idx}_section{csv_path.suffix}"
            try:
                run_ncu_with_timeout(
                    build_cmd(config_section, str(kernel_section_csv), single_kernel_name=kernel_name), 
                    f"section[kernel{idx}]", 
                    section_timeout,
                    output_csv=kernel_section_csv
                )
                print(f"[ncu] [section] Completed for kernel {idx+1}/{len(kernel_names)}: {kernel_name}", flush=True)
            except RuntimeError as e:
                # Section run failed, dump the last few lines of the CSV
                print(f"[ncu] [section] Failed for kernel {idx+1}/{len(kernel_names)}: {kernel_name}", flush=True)
                if kernel_section_csv.exists():
                    try:
                        with open(kernel_section_csv, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"[ncu] [section] Last 10 lines of section CSV ({kernel_section_csv}):", flush=True)
                                for line in lines[-10:]:
                                    print(f"  {line.rstrip()}", flush=True)
                    except Exception as read_err:
                        print(f"[ncu] [section] Failed to read section CSV: {read_err}", flush=True)
                raise
            all_section_csvs.append(kernel_section_csv)
        
        # Merge the results of all kernels (passing the kernel name map through)
        _merge_multiple_ncu_csvs(all_metrics_csvs, all_section_csvs, csv_path, kernel_name_map)
        
        # Clean up the temp files
        for f in all_metrics_csvs + all_section_csvs:
            if f.exists():
                f.unlink()
        
    else:
        # Single kernel or no kernel specified: use the original logic (a single run)
        run_ncu_with_timeout(build_cmd(config_metrics, str(csv_path)), "metrics", metrics_timeout, output_csv=csv_path)
        print(f"[ncu] [metrics] Completed", flush=True)

        # ========== Second run: use config_section.ncu-cfg, appended to the CSV ==========
        # Store the second run's output in a temp file
        temp_csv = csv_path.parent / f"{csv_path.stem}_section_temp{csv_path.suffix}"
        
        try:
            run_ncu_with_timeout(build_cmd(config_section, str(temp_csv)), "section", section_timeout, output_csv=temp_csv)
            print(f"[ncu] [section] Completed", flush=True)
            
            # Merge the two CSV files
            if temp_csv.exists() and csv_path.exists():
                with open(csv_path, 'r', encoding='utf-8') as f1:
                    first_lines = f1.readlines()
                
                with open(temp_csv, 'r', encoding='utf-8') as f2:
                    second_lines = f2.readlines()
                
                # Merge: the full contents of the first file + every non-comment line of the second
                merged_content = ''.join(first_lines).rstrip()
                if not merged_content.endswith('\n'):
                    merged_content += '\n'
                
                # Append all lines of the second file, skipping only comment lines (those starting with ==)
                for line in second_lines:
                    line_stripped = line.strip()
                    # Skip comment lines (those starting with ==) and blank lines
                    if line_stripped and not line_stripped.startswith('=='):
                        merged_content += line
                
                # Write the merged content back
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(merged_content)
                
                print(f"[ok] Merged CSV written: {csv_path}")
            
        except RuntimeError as e:
            # Section run failed, dump the last few lines of the CSV
            print(f"[ncu] [section] Failed: {e}", flush=True)
            if temp_csv.exists():
                try:
                    with open(temp_csv, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"[ncu] [section] Last 10 lines of section CSV ({temp_csv}):", flush=True)
                            for line in lines[-10:]:
                                print(f"  {line.rstrip()}", flush=True)
                except Exception as read_err:
                    print(f"[ncu] [section] Failed to read section CSV: {read_err}", flush=True)
            raise
        finally:
            # Clean up the temp file
            if temp_csv.exists():
                temp_csv.unlink()
    
    return csv_path


def _merge_multiple_ncu_csvs(metrics_csvs: List[Path], section_csvs: List[Path], out_csv: Path, kernel_name_map: Optional[Dict[int, str]] = None) -> None:
    """Merge the ncu profiling results of multiple kernels into a single CSV file
    
    Add explicit start/end markers around each kernel's metrics and section data so they can be read back and separated later.
    Marker format:
    - Metrics: ==METRICS_START:kernel_name== and ==METRICS_END:kernel_name==
    - Section: ==SECTION_START:kernel_name== and ==SECTION_END:kernel_name==
    
    Args:
        metrics_csvs: List of metrics CSV file paths for each kernel
        section_csvs: List of section CSV file paths for each kernel
        out_csv: Output merged CSV file path
        kernel_name_map: Optional dict mapping kernel index to actual kernel name (e.g., {0: "gn_reduce_kernel_shared", 1: "gn_apply_kernel"})
    """
    import csv
    from io import StringIO
    
    merged_content = ""
    header_written = False
    
    # Merge all metrics CSVs
    for idx, metrics_csv in enumerate(metrics_csvs):
        if not metrics_csv.exists():
            continue
        
        # Resolve the kernel name: prefer kernel_name_map, else parse it from the filename, and finally fall back to the default
        kernel_name = f"kernel_{idx}"  # default name
        if kernel_name_map and idx in kernel_name_map:
            kernel_name = kernel_name_map[idx]
        else:
            try:
                # Extract the kernel index from the filename
                if "_kernel" in metrics_csv.stem:
                    parts = metrics_csv.stem.split("_kernel")
                    if len(parts) > 1:
                        kernel_idx = parts[1].split("_")[0]
                        kernel_name = f"kernel_{kernel_idx}"
            except Exception:
                pass
        
        with open(metrics_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Add the metrics start marker
        merged_content += f"==METRICS_START:{kernel_name}==\n"
        
        for line in lines:
            line_stripped = line.strip()
            # Skip comment lines (but keep our own marker lines)
            if line_stripped.startswith('==') and not line_stripped.startswith('==METRICS_'):
                continue
            
            # For a header row, write it only once (metrics part only)
            if line.startswith('"ID"') or line.startswith('ID,'):
                if not header_written:
                    merged_content += line
                    header_written = True
                continue
            
            # Write the data row
            merged_content += line
        
        # Add the metrics end marker
        merged_content += f"==METRICS_END:{kernel_name}==\n"
    
    # Merge all section CSVs (appended at the end)
    # Section CSV data rows may have fewer columns than the metrics rows, so they need padding to the same column count
    for idx, section_csv in enumerate(section_csvs):
        if not section_csv.exists():
            continue
        
        # Resolve the kernel name: prefer kernel_name_map, else parse it from the filename, and finally fall back to the default
        kernel_name = f"kernel_{idx}"  # default name
        if kernel_name_map and idx in kernel_name_map:
            kernel_name = kernel_name_map[idx]
        else:
            try:
                # Extract the kernel index from the filename
                if "_kernel" in section_csv.stem:
                    parts = section_csv.stem.split("_kernel")
                    if len(parts) > 1:
                        kernel_idx = parts[1].split("_")[0]
                        kernel_name = f"kernel_{kernel_idx}"
            except Exception:
                pass
        
        # Add the section start marker
        merged_content += f"==SECTION_START:{kernel_name}==\n"
        
        with open(section_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # The section part has its own header (its column layout differs from metrics)
        # Keep the header from the section CSV file (each kernel's section part has its own header)
        section_header_written = False
        for line in lines:
            line_stripped = line.strip()
            # Skip comment lines (but keep our own marker lines)
            if line_stripped.startswith('==') and not line_stripped.startswith('==SECTION_'):
                continue
            
            # Section header: keep it (each kernel's section has its own header, whose columns differ from the metrics header)
            if (line.startswith('"ID"') or line.startswith('ID,')) and not section_header_written:
                merged_content += line
                section_header_written = True
                continue
            
            # Write the section data row as-is (no column padding needed, since metrics and section are handled separately on read)
            merged_content += line
        
        # Add the section end marker
        merged_content += f"==SECTION_END:{kernel_name}==\n"
    
    # Write out the merged content
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    print(f"[ok] Merged {len(metrics_csvs)} kernel profiles into: {out_csv}")



def _extract_core_kernel_name(kernel_str: str) -> str:
    """Extract core kernel name by removing suffixes and template params."""
    import re
    core = str(kernel_str)
    core = re.sub(r'_vec\d+', '', core)
    core = re.sub(r'_v\d+', '', core)
    core = re.sub(r'<.*?>', '', core)
    core = re.sub(r'\(.*\)', '', core)
    return core.strip()


def _match_kernel_name(marker_name: str, name_list: Optional[Sequence[str]]) -> str:
    """Match marker kernel name with name_list. Returns matched name or marker_name."""
    if not name_list:
        return marker_name
    if marker_name in name_list:
        return marker_name
    marker_core = _extract_core_kernel_name(marker_name)
    for name in name_list:
        if _extract_core_kernel_name(name) == marker_core:
            return name
    if marker_name.startswith("kernel_"):
        try:
            idx = int(marker_name.split('_')[-1])
            if idx < len(name_list):
                return name_list[idx]
        except (ValueError, IndexError):
            pass
    return marker_name


def _process_section_df(section_df: pd.DataFrame, kernel_name: str, name_list: Optional[Sequence[str]]) -> Optional[Tuple[str, str]]:
    """Process section DataFrame and return (matched_kernel_name, csv_string). Returns None if invalid."""
    if "Section Name" not in section_df.columns or "Kernel Name" not in section_df.columns:
        return None
    
    # Get first ID (ID=0)
    if "ID" in section_df.columns:
        section_df["ID_numeric"] = pd.to_numeric(section_df["ID"], errors='coerce')
        first_id = section_df["ID_numeric"].min()
        section_df = section_df[section_df["ID_numeric"] == first_id].copy()
        section_df = section_df.drop(columns=["ID_numeric"])
    
    # Remove unimportant columns
    columns_to_drop = ["Process ID", "Process Name", "Host Name", "Context", "Stream",
                       "Block Size", "Grid Size", "Device", "CC"]
    columns_to_drop = [c for c in columns_to_drop if c in section_df.columns]
    section_df = section_df.drop(columns=columns_to_drop)
    
    # Match kernel name
    matched_name = _match_kernel_name(kernel_name, name_list)
    
    # Convert to CSV string
    from io import StringIO
    csv_buffer = StringIO()
    section_df.to_csv(csv_buffer, index=False)
    return (matched_name, csv_buffer.getvalue())


def load_ncu_metrics(
    csv_path: Union[str, Path] = "ncu_temp.csv",
    columns: Optional[Sequence[str]] = None,
    extra_keep: Optional[Sequence[str]] = ("Kernel Name",),
    coerce_numeric: bool = True,
    name_list: Optional[Sequence[str]] = None,  # New: multiple kernel names
    select: str = "last",                       # Selection policy when multiple rows per name
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load NCU metrics from CSV file.
    
    Returns:
        tuple: (metrics_df, sections_dict)
        - metrics_df: DataFrame with CSV metrics data
        - sections_dict: Dictionary mapping kernel names to their section CSV data (as string)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read the entire file to separate CSV and text sections
    with open(csv_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # Check if file uses the new format with explicit markers (==METRICS_START==, ==SECTION_START==, etc.)
    has_markers = any('==METRICS_START:' in line or '==SECTION_START:' in line for line in all_lines)
    
    csv_end_line = len(all_lines)
    has_csv_section_format = False
    
    if has_markers:
        # New format with explicit markers: parse using markers
        # Metrics data is between ==METRICS_START:kernel_name== and ==METRICS_END:kernel_name==
        # Section data is between ==SECTION_START:kernel_name== and ==SECTION_END:kernel_name==
        # For now, collect all metrics lines (between METRICS_START and METRICS_END) and all section lines (between SECTION_START and SECTION_END)
        # Split into (kind, name, lines) sections first, so a single unusable
        # section can be dropped without taking the usable ones with it.
        sections: List[Tuple[str, str, List[str]]] = []
        cur: Optional[Tuple[str, str, List[str]]] = None

        for line in all_lines:
            stripped = line.strip()
            if stripped.startswith('==METRICS_START:'):
                cur = ('metrics', stripped[len('==METRICS_START:'):].rstrip('='), [])
                sections.append(cur)
                continue
            elif stripped.startswith('==SECTION_START:'):
                cur = ('section', stripped[len('==SECTION_START:'):].rstrip('='), [])
                sections.append(cur)
                continue
            elif stripped.startswith('==METRICS_END:') or stripped.startswith('==SECTION_END:'):
                cur = None
                continue

            if cur is not None:
                cur[2].append(line)

        # When --kernel-name matches nothing that actually launched (e.g. a kernel
        # symbol that is compiled but dead at this shape), ncu emits a diagnostic
        # instead of CSV:
        #     ==WARNING== No kernels were profiled.
        #     Available Kernels:
        #     1. gn_stats_kernel_h(const __half *, float *, int, int)
        # Those C++ signatures are full of commas, so the row widths are ragged and
        # pd.read_csv raises ParserError. Previously that exception propagated and
        # discarded every section that HAD profiled successfully, aborting the whole
        # optimization round. Keep the good sections and skip the empty ones.
        def _is_csv_payload(buf: List[str]) -> bool:
            text = ''.join(buf)
            if 'No kernels were profiled' in text or 'Available Kernels:' in text:
                return False
            return any(l.lstrip().startswith('"') for l in buf)

        metrics_lines = []
        section_lines = []
        skipped: List[str] = []
        for kind, name, buf in sections:
            if not buf:
                continue
            if not _is_csv_payload(buf):
                skipped.append(f"{name} [{kind}]")
                continue
            (metrics_lines if kind == 'metrics' else section_lines).extend(buf)

        if skipped:
            print(f"[ncu] No profile data for {len(skipped)} kernel name(s), skipping: "
                  f"{', '.join(skipped)}")
        if sections and not metrics_lines:
            raise RuntimeError(
                "ncu produced no profilable kernels for any requested name "
                f"({', '.join(n for _, n, _ in sections)}). The kernel names were "
                "likely extracted from source but never launched at this shape."
            )

        # Use metrics_lines for CSV parsing, section_lines for section parsing
        csv_content = ''.join(metrics_lines)
        csv_end_line = len(all_lines)  # All metrics are already extracted, section is separate
        has_csv_section_format = len(section_lines) > 0
    else:
        # Old format: find the boundary between metrics CSV and section data
        # Check if the file contains CSV-format section data (new format) or text-format section data (old format)
        # New format: section data is in CSV format with "Section Name" column (starts around line 11+)
        # Old format: section data is in text format starting with kernel name headers
        
        # First, try to detect if there's a CSV-format section header (new format)
        # Look for line containing "Section Name" column header (usually appears after metrics CSV)
        for i, line in enumerate(all_lines):
            stripped = line.strip()
            # Check for CSV section header: contains "Section Name" column
            # The header line typically looks like: "ID","Process ID",...,"Section Name","Metric Name",...
            if '"Section Name"' in stripped or (stripped.startswith('"') and 'Section Name' in stripped):
                csv_end_line = i  # Section CSV starts at this line (header)
                has_csv_section_format = True
                break
        
        # If not found, try old format detection (text section)
        if not has_csv_section_format:
            for i, line in enumerate(all_lines):
                # Look for pattern like "[number] python" or kernel name with launch info
                stripped = line.strip()
                if stripped and (
                    (stripped.startswith('[') and '@' in stripped) or
                    (stripped and not stripped.startswith('==') and 
                     not stripped.startswith('"') and 
                     '(' in stripped and 'x' in stripped and 'Context' in stripped)
                ):
                    csv_end_line = i
                    break
    
    # Extract metrics CSV section (first part)
    if has_markers:
        # Use the pre-extracted metrics_lines
        csv_content = ''.join(metrics_lines)
    else:
        csv_lines = all_lines[:csv_end_line]
        csv_content = ''.join(csv_lines)
    
    # Read CSV DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(csv_content), comment="=", low_memory=False)

    metric_cols = list(columns) if columns is not None else METRIC_COLUMNS_NEW
    keep_cols: List[str] = []
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns])
    
    # Match metrics: CSV columns may have suffixes like .avg, .max, .min, .sum, .max_rate, .pct, .ratio
    # Match any column that starts with the metric name followed by a dot or end of string
    # Keep ALL matching columns (not just one) to preserve all available metric variants
    for metric_name in metric_cols:
        # Try exact match first
        if metric_name in df.columns:
            keep_cols.append(metric_name)
        else:
            # Try matching columns that start with metric_name followed by a dot (e.g., metric_name.avg, metric_name.max_rate)
            matching_cols = [c for c in df.columns if c.startswith(metric_name + '.')]
            if matching_cols:
                # Keep ALL matching columns (not just .avg or first one)
                keep_cols.extend(matching_cols)
    
    # Remove duplicates while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if c not in seen and not seen.add(c)]
    
    if not keep_cols:
        raise ValueError("No requested columns found in the CSV header.")

    sub = df[keep_cols].copy()

    # Drop the units row
    if len(sub) > 0:
        # .str.lower() propagates missing values as NaN (a float) rather than a
        # string, so fill them before the membership test below.
        first_row_str = sub.iloc[0].astype(str).str.lower().fillna("")
        unit_tokens = ("%", "inst", "cycle", "block", "register", "register/thread")
        if first_row_str.apply(lambda x: any(tok in str(x) for tok in unit_tokens)).any():
            sub = sub.iloc[1:].reset_index(drop=True)
    
    # Filter out section data rows: they have fewer columns than metrics rows
    # Section data rows typically have only ~16 columns while metrics rows have 300+ columns
    # We can identify them by checking if key metric columns are empty/NaN
    if len(sub) > 0:
        # Check for rows where all key metrics are NaN/empty (indicating section data)
        key_metric_cols = [c for c in keep_cols if (c.startswith("gpu__time_duration") or c.startswith("sm__cycles_active")) and c not in ("Kernel Name", "Block Size", "Grid Size")]
        if key_metric_cols:
            # Before numeric conversion, check if values are non-empty strings
            # Section data rows will have empty strings in these columns
            has_valid_metrics = pd.Series([True] * len(sub), index=sub.index)
            for col in key_metric_cols:
                if col in sub.columns:
                    col_vals = sub[col].astype(str)
                    # Row is invalid if this key metric is empty or 'nan'
                    invalid_mask = (col_vals.str.strip() == '') | (col_vals.str.strip().str.lower() == 'nan')
                    has_valid_metrics = has_valid_metrics & ~invalid_mask
            
            # Only keep rows with at least one valid metric (exclude section data rows)
            sub = sub[has_valid_metrics].reset_index(drop=True)

    # Coerce metrics to numeric
    if coerce_numeric:
        # Use the actual columns in keep_cols (which may have suffixes like .avg)
        # Exclude non-metric columns (like "Kernel Name", "Block Size", "Grid Size")
        metric_in_sub = [c for c in keep_cols if c in sub.columns and c not in (extra_keep or [])]
        if metric_in_sub:
            sub[metric_in_sub] = (
                sub[metric_in_sub]
                .replace({",": "", "%": ""}, regex=True)
                .apply(pd.to_numeric, errors="coerce")
            )

    # ========== Extract by kernel name list ==========
    if name_list:
        results = []
        for name in name_list:
            # Use flexible matching: try multiple strategies
            kernel_name_col = sub["Kernel Name"].astype(str)
            
            # Strategy 1: Extract core name (remove common suffixes like _vec4, _v2, template/function params)
            csv_cores = kernel_name_col.apply(_extract_core_kernel_name)
            name_core = _extract_core_kernel_name(name)
            # Match if cores are equal or one contains the other
            core_match = (csv_cores == name_core) | csv_cores.str.contains(name_core, regex=False, na=False) | csv_cores.str.startswith(name_core, na=False)
            matched = sub[core_match]
            
            # Strategy 2: Contains match (original logic) - check if CSV name contains extracted name
            if matched.empty:
                matched = sub[kernel_name_col.str.contains(name, regex=False, na=False)]
            
            # Strategy 3: Prefix match - check if both start with common prefix (e.g., "swish_forward")
            if matched.empty:
                # Extract prefix (first part before _vec or similar)
                name_prefix = name.split('_vec')[0].split('_v')[0] if '_vec' in name or '_v' in name else name.split('_kernel')[0] if '_kernel' in name else name
                prefix_match = kernel_name_col.str.startswith(name_prefix, na=False)
                matched = sub[prefix_match]
            
            if matched.empty:
                # Log available kernels for debugging
                available_kernels = kernel_name_col.unique()
                print(f"[load_ncu_metrics] Warning: No match for kernel '{name}'. Available kernels: {list(available_kernels)[:3]}")
                continue
            
            # Filter out rows that are section data (have empty/null metrics)
            # Section data rows have empty metrics columns - they appear after the actual metrics rows
            # After numeric conversion, section data rows will have NaN in metric columns
            # We identify them by checking if key metrics are non-NaN
            if len(matched) > 1:
                # Check for rows with valid metrics data (non-NaN in at least one key metric)
                key_metric_cols = [c for c in matched.columns if (c.startswith("gpu__time_duration") or c.startswith("sm__cycles_active")) and c not in ("Kernel Name", "Block Size", "Grid Size")]
                if key_metric_cols:
                    # Filter: keep rows where at least one key metric is non-NaN
                    has_valid_metrics = pd.Series([False] * len(matched), index=matched.index)
                    for col in key_metric_cols:
                        if col in matched.columns:
                            col_vals = matched[col]
                            # Check: not NaN (after conversion) and not empty string (before conversion check)
                            # Since conversion already happened, check for NaN
                            valid_mask = col_vals.notna()
                            has_valid_metrics = has_valid_metrics | valid_mask
                    
                    # Only keep rows with valid metrics (exclude section data rows)
                    matched_with_data = matched[has_valid_metrics]
                    if len(matched_with_data) > 0:
                        matched = matched_with_data
                    # If all rows are invalid, log warning but proceed with original matched
                    elif len(matched) > 0:
                        import warnings
                        warnings.warn(f"All matched rows for kernel '{name}' have NaN metrics. This may indicate section data rows were matched. Using first row anyway.")
                
            if len(matched) > 1:
                if select == "first":
                    row = matched.iloc[[0]]
                elif select == "last":
                    row = matched.iloc[[-1]]
                elif select == "max_cycles" and "sm__cycles_active.avg" in matched.columns:
                    # Only consider rows with valid cycles data
                    valid_cycles = matched[matched["sm__cycles_active.avg"].notna()]
                    if len(valid_cycles) > 0:
                        row = valid_cycles.sort_values("sm__cycles_active.avg", ascending=False).head(1)
                    else:
                        row = matched.iloc[[-1]]  # fallback if no valid cycles
                else:
                    row = matched.iloc[[-1]]  # fallback
            else:
                row = matched
            results.append(row)

        if results:
            sub = pd.concat(results, ignore_index=True)
        else:
            # No matching kernels found - log warning and return empty DataFrame
            import warnings
            available_kernels = sub["Kernel Name"].unique() if "Kernel Name" in sub.columns else []
            warnings.warn(
                f"No matching kernels found for name_list={name_list}. "
                f"Available kernels in CSV: {list(available_kernels)[:5]}. "
                f"Returning empty DataFrame."
            )
            sub = pd.DataFrame(columns=keep_cols)

    # ========== Extract section CSV data for each kernel ==========
    sections_dict: Dict[str, str] = {}
    
    if has_markers:
        # New format with explicit markers: extract section data using markers
        # Parse section data for each kernel from section_lines
        current_kernel = None
        current_section_lines = []
        
        for line in all_lines:
            stripped = line.strip()
            if stripped.startswith('==SECTION_START:'):
                # Extract kernel name from marker
                kernel_name = stripped.replace('==SECTION_START:', '').replace('==', '').strip()
                current_kernel = kernel_name
                current_section_lines = []
                continue
            elif stripped.startswith('==SECTION_END:'):
                # Save section data for current kernel
                if current_kernel and current_section_lines:
                    section_csv_content = ''.join(current_section_lines)
                    try:
                        section_df = pd.read_csv(StringIO(section_csv_content), comment="=", low_memory=False)
                        
                        # Match kernel name with name_list if provided
                        matched_kernel_name = _match_kernel_name(current_kernel, name_list)
                        
                        # Process section data using helper function
                        result = _process_section_df(section_df, current_kernel, name_list)
                        if result:
                            matched_name, csv_str = result
                            sections_dict[matched_name] = csv_str
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Failed to parse section data for {current_kernel}: {e}")
                
                current_kernel = None
                current_section_lines = []
                continue
            
            if current_kernel is not None:
                current_section_lines.append(line)
    
    elif csv_end_line < len(all_lines):
        if has_csv_section_format:
            # Old format: section data is in CSV format (without markers)
            section_csv_content = ''.join(all_lines[csv_end_line:])
            section_df = pd.read_csv(StringIO(section_csv_content), comment="=", low_memory=False)
            
            # Check if required columns exist
            if "Section Name" in section_df.columns and "Kernel Name" in section_df.columns:
                # Group by kernel name and ID (to handle multiple runs)
                for kernel_name_full in section_df["Kernel Name"].unique():
                    kernel_df = section_df[section_df["Kernel Name"] == kernel_name_full].copy()
                    
                    # Process section data using helper function
                    result = _process_section_df(kernel_df, kernel_name_full, name_list)
                    if result:
                        matched_name, csv_str = result
                        sections_dict[matched_name] = csv_str
        else:
            # Old format: text-based section data
            text_section = ''.join(all_lines[csv_end_line:])
            
            # Parse sections by kernel name
            # Pattern: kernel_name (grid)x(block), Context X, Stream Y, Device Z, CC W.W
            # Note: Kernel name line may have leading spaces and may be preceded by process info line
            # Each kernel may have multiple groups (multiple profiling runs), we only need the last group
            import re
            # Find all kernel section headers - match lines that contain kernel name followed by launch info
            # The line may start with spaces and contain: kernel_name (grid)x(block), Context...
            kernel_pattern = re.compile(
                r'^\s*([^\n]+?)\s+\([^)]+\)x\([^)]+\),\s+Context\s+\d+,\s+Stream\s+\d+,\s+Device\s+\d+,\s+CC\s+[\d.]+',
                re.MULTILINE
            )
            
            matches = list(kernel_pattern.finditer(text_section))
            
            # Group matches by kernel name (extract kernel name from each match)
            kernel_groups: Dict[str, List[tuple]] = {}  # kernel_name -> [(match_index, kernel_header, start_pos, end_pos), ...]
            
            for i, match in enumerate(matches):
                kernel_header_line = match.group(0).strip()
                # Extract just the kernel name part (before the launch info)
                kernel_name_match = re.match(r'^(.+?)\s+\([^)]+\)x\([^)]+\)', kernel_header_line)
                if kernel_name_match:
                    kernel_header = kernel_name_match.group(1).strip()
                else:
                    kernel_header = kernel_header_line.split(',')[0].strip()
                
                start_pos = match.end()
                # Find the end of this kernel's section (next kernel or end of file)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text_section)
                
                # Group by kernel name (for matching with name_list)
                matched_kernel_name = None
                if name_list:
                    for name in name_list:
                        # Check if the name appears in the kernel header (use contains for partial matching)
                        if name in kernel_header:
                            matched_kernel_name = name
                            break
                
                # If no name_list or no match, use the full kernel header as key
                if matched_kernel_name is None:
                    matched_kernel_name = kernel_header
                
                if matched_kernel_name not in kernel_groups:
                    kernel_groups[matched_kernel_name] = []
                kernel_groups[matched_kernel_name].append((i, kernel_header, start_pos, end_pos))
            
            # For each kernel, extract only the last group
            for kernel_name, groups in kernel_groups.items():
                if not groups:
                    continue
                
                # Sort by match index to get the last one
                groups.sort(key=lambda x: x[0])
                last_group = groups[-1]
                _, kernel_header, start_pos, end_pos = last_group
                
                kernel_section_text = text_section[start_pos:end_pos].strip()
                sections_dict[kernel_name] = kernel_section_text

    return sub, sections_dict


def metrics_to_prompt(
    df: pd.DataFrame,
    sections_dict: Optional[Dict[str, str]] = None,
    title: str = "Here are the GPU profiling metrics:",  # Placeholder, not emitted
    key_by: str = "Kernel Name",
    round_digits: Optional[int] = 3,
    compact: bool = False,
    keep_cols: Optional[List[str]] = None,
) -> str:
    """
    Return metrics and sections as a formatted string.
    
    Args:
        df: DataFrame with metrics data
        sections_dict: Dictionary mapping kernel names to section CSV data (as string)
        title: Title for the metrics section (not used in output)
        key_by: Column name to use as key
        round_digits: Number of decimal places for rounding
        compact: Whether to use compact JSON format
        keep_cols: Columns to keep in output
    
    Returns:
        Formatted string containing metrics JSON and sections CSV data
    """

    def _safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            return str(v)
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    # Build metrics JSON
    metrics_json = "{}"
    if df is not None and not df.empty:
        cols = list(df.columns)

        # Round numeric columns
        if round_digits is not None:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                df = df.copy()
                df[num_cols] = df[num_cols].round(round_digits)

        # If key column is missing, return a list of rows
        if key_by not in cols:
            rows = [{k: _safe(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
            metrics_json = json.dumps(rows, ensure_ascii=False, indent=None if compact else 2)
        else:
            # Determine value columns
            value_cols = [c for c in cols if c != key_by]
            if keep_cols is not None:
                value_cols = [c for c in value_cols if c in keep_cols]

            data: Dict[str, Any] = {}
            for rec in df[[key_by] + value_cols].to_dict(orient="records"):
                k = str(rec.pop(key_by))
                val_obj = {ck: _safe(cv) for ck, cv in rec.items()}
                if k in data:
                    if isinstance(data[k], list):
                        data[k].append(val_obj)
                    else:
                        data[k] = [data[k], val_obj]
                else:
                    data[k] = val_obj

            # Ensure all values are properly cleaned before JSON serialization
            def _deep_clean(obj):
                """Recursively clean object to ensure JSON serializability."""
                if isinstance(obj, dict):
                    return {str(k): _deep_clean(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_deep_clean(item) for item in obj]
                elif obj is None:
                    return None
                elif isinstance(obj, float):
                    if math.isnan(obj):
                        return None
                    elif math.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Interval)):
                    return None
                elif obj is pd.NA or (hasattr(pd, 'NA') and obj is pd.NA):
                    return None
                elif isinstance(obj, np.generic):
                    val = obj.item()
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        return None
                    return val
                elif isinstance(obj, (int, str, bool)):
                    return obj
                else:
                    # Convert anything else to string
                    try:
                        return str(obj)
                    except:
                        return None
            
            cleaned_data = _deep_clean(data)
            # Validate JSON can be serialized and parsed
            try:
                metrics_json = json.dumps(cleaned_data, ensure_ascii=False, indent=None if compact else 2)
                # Verify it can be parsed back
                json.loads(metrics_json)
            except (TypeError, ValueError) as e:
                # If serialization fails, log error and use empty JSON
                import warnings
                warnings.warn(f"Failed to serialize metrics to JSON: {e}. Using empty JSON.")
                metrics_json = "{}"

    # Build sections CSV data
    sections_text = ""
    if sections_dict:
        sections_parts = []
        for kernel_name, section_csv in sections_dict.items():
            # Section data is in CSV format (string)
            # Wrap it in a code block for better readability
            sections_parts.append(f"## Kernel: {kernel_name}\n```csv\n{section_csv}```")
        sections_text = "\n\n".join(sections_parts)

    # Combine metrics and sections
    result_parts = []
    if metrics_json and metrics_json != "{}":
        # Wrap JSON in code block to make it clear it's valid JSON
        # This helps LLM distinguish JSON from the section analysis text
        result_parts.append(f"# Metrics (JSON)\n```json\n{metrics_json}\n```")
    else:
        # If metrics JSON is empty, add a warning message
        result_parts.append(f"# Metrics (JSON)\n```json\n{{}}\n```\n**WARNING**: No metrics data found. This may indicate that the kernel name in the code does not match the kernel name profiled by ncu.")
    if sections_text:
        result_parts.append(f"# Detailed Section Analysis\n{sections_text}")
    else:
        # If sections text is empty, add a note
        result_parts.append(f"# Detailed Section Analysis\n**WARNING**: No section analysis found. This may indicate that the kernel name in the code does not match the kernel name profiled by ncu.")
    
    return "\n\n".join(result_parts) if result_parts else ""



if __name__ == "__main__":
    # Simple self-check: doesn't force execution; only runs when this file is executed directly.
    # Note: `profile_bench` requires root privileges and an Nsight Compute environment.
    print("gpu_profile_utils module loaded. Import its functions in your main script.")
