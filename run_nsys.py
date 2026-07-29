#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module wraps nsys profiling to extract kernel launch counts.

1) Run nsys profile to generate .nsys-rep trace file
2) Use nsys stats to extract kernel information (especially launch counts)
3) Output CSV with kernel names and launch counts

Typical usage:
    from run_nsys import profile_bench, load_nsys_stats
    
    kernel_names = extract_cuda_kernel_names(kernel_file)
    rep_path = profile_bench(kernel_names=kernel_names)
    stats = load_nsys_stats(rep_path, kernel_names=kernel_names)
"""

import os
import re
import sys
import shutil
import subprocess
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence, Union, Dict, Tuple
import pandas as pd


__all__ = [
    "profile_bench",
    "load_nsys_stats",
    "extract_kernel_launch_counts",
]


def _find_qdstrm_importer() -> Optional[str]:
    """Locate QdstrmImporter, the host-side .qdstrm -> .nsys-rep converter.

    An nsys that comes from the Nsight Compute bundle collects into a .qdstrm
    but cannot convert it: it searches for the importer in a sibling directory
    named host-linux-x64, while the bundle ships it as
    linux-desktop-glibc_2_11_3-x64. Find it wherever it actually is.
    """
    explicit = os.environ.get("NSYS_QDSTRM_IMPORTER")
    if explicit and os.access(explicit, os.X_OK):
        return explicit

    roots = []
    nsys_path = shutil.which("nsys")
    if nsys_path:
        # <bundle>/host/target-linux-x64/nsys -> search <bundle>/host/*
        roots.append(Path(os.path.realpath(nsys_path)).parent.parent)
    roots.append(Path("/opt/nvidia/nsight-compute"))

    for root in roots:
        for pattern in ("*/QdstrmImporter", "*/*/QdstrmImporter"):
            try:
                for cand in sorted(root.glob(pattern)):
                    if os.access(cand, os.X_OK):
                        return str(cand)
            except OSError:
                continue
    return None


def _qdstrm_to_rep(rep_path: Path) -> bool:
    """Convert a collected .qdstrm into rep_path. Returns True on success.

    The importer additionally needs libdw.so.1 / libelf.so.1, which may not be
    installed system-wide; NSYS_IMPORTER_LIBS points at a directory holding
    them (locally extracted copies are fine).
    """
    qdstrm = rep_path.with_suffix(".qdstrm")
    if not qdstrm.exists():
        return False
    importer = _find_qdstrm_importer()
    if not importer:
        return False

    env = os.environ.copy()
    lib_dir = env.get("NSYS_IMPORTER_LIBS", "/home/elek/tools/nsys-fix/rootfs/usr/lib/x86_64-linux-gnu")
    if os.path.isdir(lib_dir):
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"[nsys] Converting {qdstrm.name} -> {rep_path.name}", flush=True)
    try:
        proc = subprocess.run(
            [importer, "-i", str(qdstrm)],
            env=env, text=True, capture_output=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[nsys] QdstrmImporter failed: {exc}", flush=True)
        return False

    if not rep_path.exists():
        print(f"[nsys] QdstrmImporter produced no report: "
              f"{(proc.stderr or proc.stdout or '').strip()[:300]}", flush=True)
        return False

    qdstrm.unlink(missing_ok=True)
    return True


def extract_cuda_kernel_names(py_path: Union[str, Path]) -> List[str]:
    """Extract CUDA kernel names from a Python file containing CUDA code."""
    try:
        src = Path(py_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    p1 = re.compile(r"""__global__\s+void\s+([A-Za-z_]\w*)\s*\(""", re.MULTILINE)
    p2 = re.compile(
        r"""__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(""",
        re.MULTILINE,
    )

    names = p1.findall(src) + p2.findall(src)
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def profile_bench(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    kernel_file: Optional[Union[str, Path]] = None,
    conda_bin: str = str(Path(sys.executable).parent),
    out_rep: Union[str, Path] = "nsys_temp.nsys-rep",
    device_idx: Optional[int] = None,
    timeout: int = 300,  # 5 minutes default timeout
) -> Path:
    """
    Run nsys profile to generate trace file.
    
    Args:
        bench_py: Path to benchmark script
        kernel_names: Optional list of kernel names to filter
        kernel_file: Optional path to kernel file (for --test parameter)
        conda_bin: Path to conda bin directory
        out_rep: Output .nsys-rep file path
        device_idx: Optional CUDA device index
        timeout: Timeout in seconds
    
    Returns:
        Path to generated .nsys-rep file
    """
    nsys_bin = "nsys"  # Assume nsys is in PATH
    rep_path = Path(out_rep).resolve()
    
    # Remove existing file if force-overwrite
    if rep_path.exists():
        rep_path.unlink()
    
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    
    # Use Python from conda_bin if available, otherwise use sys.executable.
    # Path.exists() re-raises PermissionError (it only swallows ENOENT/ENOTDIR/
    # EBADF/ELOOP), so a conda_bin under an unreadable directory such as /root
    # would abort profiling instead of falling back. Probe defensively.
    def _usable(p: str) -> bool:
        try:
            return os.access(p, os.X_OK)
        except OSError:
            return False

    if _usable(f"{conda_bin}/python3"):
        python_bin = f"{conda_bin}/python3"
    elif _usable(f"{conda_bin}/python"):
        python_bin = f"{conda_bin}/python"
    else:
        python_bin = sys.executable
    
    # Build nsys command
    cmd = [
        nsys_bin,
        "profile",
        # 'osrt' (OS-runtime syscall tracing) hangs indefinitely on this stack
        # (>150s vs 3s without it) and is unused — launch counts come from the
        # CUDA trace alone.
        "-t", "cuda,nvtx",
        "--sample=none",
        "--force-overwrite=true",
        "-o", str(rep_path.with_suffix("")),  # nsys adds .nsys-rep automatically
        python_bin,
        bench_py,
    ]
    
    # Optional: respect caller-specified CUDA device index
    if device_idx is not None:
        cmd.extend(["--device-idx", str(device_idx)])
    
    # If kernel_file is specified, add --test parameter
    if kernel_file:
        kernel_file_path = Path(kernel_file).resolve()
        if not kernel_file_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file_path}")
        cmd.extend(["--test", str(kernel_file_path)])
    
    print(f"[nsys] Running: {' '.join(cmd)}", flush=True)
    
    try:
        # Use Popen to handle timeout and process group
        proc = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            if proc.returncode != 0:
                sys.stderr.write(stderr or "")
                raise RuntimeError(f"nsys profile failed with return code {proc.returncode}")
            
            # Verify output file exists. A bundle-provided nsys may have
            # collected into a .qdstrm without converting it; do that here.
            if not rep_path.exists():
                _qdstrm_to_rep(rep_path)
            if not rep_path.exists():
                raise FileNotFoundError(f"nsys output file not found: {rep_path}")
            
            print(f"[nsys] Profile completed: {rep_path}", flush=True)
            return rep_path
            
        except subprocess.TimeoutExpired:
            print(f"[nsys] Timeout after {timeout} seconds, terminating...", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass
            raise RuntimeError(f"nsys profile timed out after {timeout} seconds")
            
    except RuntimeError:
        raise
    except Exception as e:
        # Ensure process is cleaned up
        try:
            if 'proc' in locals() and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        raise


_KERNEL_TIME_STATS: Dict[str, Dict[str, float]] = {}


def get_last_kernel_time_stats() -> Dict[str, Dict[str, float]]:
    """Per-kernel TIME from the most recent extract_kernel_launch_counts() call.

    Maps unfiltered kernel name -> {"time_pct", "total_ns", "instances"}.

    Why this exists: the launch-count table alone renders a convolution that is
    ~68% of the forward and a stats kernel that is ~4% as the same number (both
    "24 launches"). nsys already reports Time (%) and Total Time in the same
    table and we were discarding both, so the only time-attributed numbers
    reaching the judge came from NCU -- which profiles our own kernels only.
    That systematically pointed optimization rounds at the glue.
    """
    return dict(_KERNEL_TIME_STATS)


def extract_kernel_launch_counts(
    rep_path: Union[str, Path],
    kernel_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Extract kernel launch counts from nsys stats output.

    Args:
        rep_path: Path to .nsys-rep file
        kernel_names: Optional list of kernel names to filter (if None, extract all)

    Returns:
        Dictionary mapping kernel names to launch counts

    Side effect: records per-kernel time in the module-level cache readable via
    get_last_kernel_time_stats(), always unfiltered regardless of kernel_names.
    """
    rep_path = Path(rep_path)
    if not rep_path.exists():
        raise FileNotFoundError(f"nsys report file not found: {rep_path}")
    
    # Run nsys stats to get kernel summary
    # Note: Report name is 'cuda_gpu_kern_sum', not 'cuda_gpu_kern_summary'
    cmd = [
        "nsys",
        "stats",
        "--report", "cuda_gpu_kern_sum",
        "--force-export=true",  # Force re-export if SQLite file exists
        str(rep_path),
    ]
    
    print(f"[nsys] Extracting stats: {' '.join(cmd)}", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"nsys stats failed: {result.stderr}")
        
        # Parse output to extract kernel names and instance counts
        # The output format is:
        # Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
        # --------  ---------------  ---------  ---------  ...  ---
        #   84.2          148,357          1  148,357.0  ...  kernel_name
        launch_counts: Dict[str, int] = {}
        _KERNEL_TIME_STATS.clear()

        lines = result.stdout.split('\n')
        in_table = False
        header_found = False
        name_col_idx = None
        instances_col_idx = None
        time_pct_col_idx = None
        total_time_col_idx = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Look for table header - "CUDA GPU Kernel Summary" section
            if "CUDA GPU Kernel Summary" in line or "cuda_gpu_kern_sum" in line:
                in_table = False
                header_found = False
                continue
            
            # Look for header line with column names
            if "Name" in line and "Instances" in line:
                header_found = True
                in_table = True
                # Parse header to find column positions
                # Format: "Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name"
                # Use same splitting as data rows (2+ spaces) to get correct column alignment
                header_parts = re.split(r'\s{2,}', line_stripped)
                try:
                    # Find "Instances" in header - it should be one of the parts
                    for i, part in enumerate(header_parts):
                        if "Instances" in part:
                            instances_col_idx = i
                            break
                    else:
                        instances_col_idx = 2  # Fallback: typically 3rd column

                    # Time (%) and Total Time (ns) sit left of Instances; they are
                    # what makes the table a time breakdown rather than a call log.
                    for i, part in enumerate(header_parts):
                        if "Time (%)" in part or part.strip().startswith("Time"):
                            time_pct_col_idx = i
                            break
                    for i, part in enumerate(header_parts):
                        if "Total Time" in part:
                            total_time_col_idx = i
                            break

                    # Name is typically the last column
                    name_col_idx = len(header_parts) - 1
                except Exception:
                    # Fallback: use default positions
                    instances_col_idx = 2  # 3rd column (0-indexed)
                    name_col_idx = -1  # Last column
                continue
            
            # Skip separator lines
            if line_stripped.startswith('-') or line_stripped.startswith('=') or line_stripped.startswith('---'):
                continue
            
            if not in_table or not header_found:
                continue
            
            # Parse table rows - columns are separated by multiple spaces
            # Format: "84.2  148,357  1  148,357.0  148,357.0  148,357  148,357  0.0  kernel_name"
            # Columns: Time(%)  TotalTime(ns)  Instances  Avg(ns)  Med(ns)  Min(ns)  Max(ns)  StdDev(ns)  Name
            # Split on 2+ spaces, but be careful with kernel names that may have spaces
            parts = re.split(r'\s{2,}', line_stripped)  # Split on 2+ spaces
            if len(parts) < 3:  # Need at least Time%, TotalTime, Instances
                continue
            
            try:
                # Get instances count
                # Header: "Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name"
                # Data:   "84.2  148,357  1  148,357.0  ...  kernel_name"
                # When split by 2+ spaces: parts[0]=Time%, parts[1]=TotalTime, parts[2]=Instances, ..., parts[-1]=Name
                if instances_col_idx is not None and instances_col_idx < len(parts):
                    instances_str = parts[instances_col_idx].strip().replace(',', '')
                else:
                    # Fallback: Instances is typically the 3rd column (index 2)
                    # Format: Time(%)  TotalTime(ns)  Instances  Avg(ns)  ...
                    if len(parts) >= 3:
                        instances_str = parts[2].strip().replace(',', '')
                    else:
                        continue
                
                try:
                    instances = int(instances_str)
                except ValueError:
                    continue
                
                # Get kernel name (usually last column)
                if name_col_idx is not None and name_col_idx < len(parts):
                    kernel_name = parts[name_col_idx].strip()
                else:
                    # Fallback: name is typically the last non-empty part
                    kernel_name = parts[-1].strip() if parts else ""
                
                if not kernel_name:
                    continue

                # Record time for EVERY kernel before any name filtering: the
                # filtered path feeds machine_check's scalar count, but the time
                # breakdown must stay whole or the library kernels vanish again.
                def _num(idx):
                    if idx is None or idx >= len(parts):
                        return None
                    try:
                        return float(parts[idx].strip().replace(',', '').rstrip('%'))
                    except ValueError:
                        return None

                _tp, _tt = _num(time_pct_col_idx), _num(total_time_col_idx)
                if _tp is not None or _tt is not None:
                    _KERNEL_TIME_STATS[kernel_name] = {
                        "time_pct": _tp if _tp is not None else float("nan"),
                        "total_ns": _tt if _tt is not None else float("nan"),
                        "instances": float(instances),
                    }

                # Filter by kernel_names if provided
                if kernel_names:
                    # Try to match kernel name (flexible matching)
                    matched = False
                    for target_name in kernel_names:
                        # Exact match or contains match
                        if target_name in kernel_name or kernel_name in target_name:
                            # Use the target name as key for consistency
                            launch_counts[target_name] = launch_counts.get(target_name, 0) + instances
                            matched = True
                            break
                        # Try core name matching (remove suffixes)
                        target_core = re.sub(r'_vec\d+|_v\d+|<.*?>|\(.*\)', '', target_name)
                        kernel_core = re.sub(r'_vec\d+|_v\d+|<.*?>|\(.*\)', '', kernel_name)
                        if target_core == kernel_core or target_core in kernel_core:
                            launch_counts[target_name] = launch_counts.get(target_name, 0) + instances
                            matched = True
                            break
                    if not matched:
                        continue
                else:
                    # No filtering, add all kernels
                    launch_counts[kernel_name] = instances
                    
            except (IndexError, ValueError) as e:
                # Skip malformed lines
                continue
        
        print(f"[nsys] Extracted launch counts for {len(launch_counts)} kernel(s)", flush=True)
        return launch_counts
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("nsys stats timed out")
    except Exception as e:
        raise RuntimeError(f"Failed to extract kernel launch counts: {e}")


def load_nsys_stats(
    rep_path: Union[str, Path],
    kernel_names: Optional[List[str]] = None,
    out_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load kernel launch counts from nsys report and save to CSV.
    
    Args:
        rep_path: Path to .nsys-rep file
        kernel_names: Optional list of kernel names to filter
        out_csv: Optional output CSV path (default: rep_path with .csv extension)
    
    Returns:
        DataFrame with columns: Kernel Name, kernel_launch_count
    """
    rep_path = Path(rep_path)
    
    # Extract launch counts
    launch_counts = extract_kernel_launch_counts(rep_path, kernel_names)
    time_stats = get_last_kernel_time_stats()

    # Create DataFrame. Time columns are attached only for the UNFILTERED table
    # (kernel_names is None): the filtered CSV feeds machine_check's scalar
    # kernel_launch_count and its schema is left exactly as it was.
    data = []
    for kernel_name, count in launch_counts.items():
        row = {
            "Kernel Name": kernel_name,
            "kernel_launch_count": count,
        }
        if kernel_names is None:
            ts = time_stats.get(kernel_name, {})
            row["time_pct"] = ts.get("time_pct", float("nan"))
            row["total_time_ns"] = ts.get("total_ns", float("nan"))
        data.append(row)

    cols = ["Kernel Name", "kernel_launch_count"]
    if kernel_names is None:
        cols += ["time_pct", "total_time_ns"]
    if not data:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(data)
    
    # Save to CSV if requested
    if out_csv:
        csv_path = Path(out_csv)
        df.to_csv(csv_path, index=False)
        print(f"[nsys] Saved launch counts to: {csv_path}", flush=True)
    else:
        # Default: save next to .nsys-rep file
        csv_path = rep_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"[nsys] Saved launch counts to: {csv_path}", flush=True)
    
    return df


# =============================================================================
# CLI / main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run nsys profile and extract kernel launch counts")
    parser.add_argument("--bench-py", default="bench_ref_inputs.py", help="Benchmark script path")
    parser.add_argument("--kernel-file", help="Kernel file path (for --test parameter)")
    parser.add_argument("--out-rep", default="nsys_temp.nsys-rep", help="Output .nsys-rep file path")
    parser.add_argument("--out-csv", help="Output CSV file path (default: same as rep with .csv)")
    parser.add_argument("--device-idx", type=int, help="CUDA device index")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--kernel-names", nargs="+", help="Kernel names to filter")
    
    args = parser.parse_args()
    
    # Extract kernel names from kernel_file if provided
    kernel_names = args.kernel_names
    if not kernel_names and args.kernel_file:
        kernel_names = extract_cuda_kernel_names(args.kernel_file)
        print(f"[nsys] Extracted kernel names: {kernel_names}", flush=True)
    
    # Run profiling
    rep_path = profile_bench(
        bench_py=args.bench_py,
        kernel_names=kernel_names,
        kernel_file=args.kernel_file,
        out_rep=args.out_rep,
        device_idx=args.device_idx,
        timeout=args.timeout,
    )
    
    # Extract and save stats
    df = load_nsys_stats(
        rep_path=rep_path,
        kernel_names=kernel_names,
        out_csv=args.out_csv,
    )
    
    print(f"\n[nsys] Kernel launch counts:", flush=True)
    print(df.to_string(index=False), flush=True)
