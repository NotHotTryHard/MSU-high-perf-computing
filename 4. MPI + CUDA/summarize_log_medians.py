#!/usr/bin/env python3
import csv
import os
import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Metrics:
    solve: Optional[float] = None
    halo: Optional[float] = None
    h2d: Optional[float] = None
    d2h: Optional[float] = None
    matvec: Optional[float] = None
    vec: Optional[float] = None


FLOAT_RE = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"

PATTERNS = {
    "solve": re.compile(rf"solve time \(max over ranks\):\s*{FLOAT_RE}\s*s"),
    "halo": re.compile(rf"halo MPI comm time:\s*{FLOAT_RE}\s*s"),
    "h2d": re.compile(rf"H->D memcpy time:\s*{FLOAT_RE}\s*s"),
    "d2h": re.compile(rf"D->H memcpy time:\s*{FLOAT_RE}\s*s"),
    "matvec": re.compile(rf"matvec kernel time:\s*{FLOAT_RE}\s*s"),
    "vec": re.compile(rf"vector kernels time:\s*{FLOAT_RE}\s*s"),
}


def parse_grid_p_from_filename(path: str) -> Optional[Tuple[str, int]]:
    base = os.path.basename(path)
    # Supports:
    # - 4000x4000-p6.out
    # - part1-400x600-p1.out
    # - large-4000x4000-p4.out
    m = re.search(r"(\d+)x(\d+).*?-p(\d+)\.out$", base)
    if not m:
        return None
    n = int(m.group(1))
    m_ = int(m.group(2))
    p = int(m.group(3))
    return (f"{n}x{m_}", p)


def parse_metrics_from_text(text: str) -> Metrics:
    out = Metrics()
    for key, rx in PATTERNS.items():
        m = rx.search(text)
        if not m:
            continue
        val = float(m.group(1))
        setattr(out, key, val)
    return out


def median_or_blank(values: List[float]) -> str:
    if not values:
        return ""
    return f"{statistics.median(values):.10g}"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    base_dir = here

    log_dirs = [f"log3_{i}" for i in range(10)] + ["log3_xtra"]
    log_dirs = [os.path.join(base_dir, d) for d in log_dirs]
    log_dirs = [d for d in log_dirs if os.path.isdir(d)]

    # key: (grid, p) -> metric_name -> list[float]
    bucket: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    sources: Dict[Tuple[str, int], List[str]] = {}

    for d in log_dirs:
        for root, _, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".out"):
                    continue
                path = os.path.join(root, fn)
                key = parse_grid_p_from_filename(path)
                if key is None:
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue

                metrics = parse_metrics_from_text(text)
                if key not in bucket:
                    bucket[key] = {k: [] for k in PATTERNS.keys()}
                    sources[key] = []
                sources[key].append(os.path.relpath(path, base_dir))

                for k in PATTERNS.keys():
                    v = getattr(metrics, k)
                    if v is not None:
                        bucket[key][k].append(v)

    out_csv = os.path.join(base_dir, "median_times.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "grid",
            "p",
            "n_samples_solve",
            "n_samples_halo",
            "n_samples_h2d",
            "n_samples_d2h",
            "n_samples_matvec",
            "n_samples_vector",
            "solve_time_median_s",
            "halo_mpi_comm_time_median_s",
            "h2d_memcpy_time_median_s",
            "d2h_memcpy_time_median_s",
            "matvec_kernel_time_median_s",
            "vector_kernels_time_median_s",
        ])

        for (grid, p) in sorted(bucket.keys(), key=lambda x: (int(x[0].split("x")[0]), int(x[0].split("x")[1]), x[1])):
            b = bucket[(grid, p)]
            w.writerow([
                grid,
                p,
                len(b["solve"]),
                len(b["halo"]),
                len(b["h2d"]),
                len(b["d2h"]),
                len(b["matvec"]),
                len(b["vec"]),
                median_or_blank(b["solve"]),
                median_or_blank(b["halo"]),
                median_or_blank(b["h2d"]),
                median_or_blank(b["d2h"]),
                median_or_blank(b["matvec"]),
                median_or_blank(b["vec"]),
            ])

    # Also write a "long" format if needed for pivoting/plots
    out_csv_long = os.path.join(base_dir, "median_times_long.csv")
    with open(out_csv_long, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grid", "p", "metric", "median_s", "n_samples"])
        for (grid, p) in sorted(bucket.keys(), key=lambda x: (int(x[0].split("x")[0]), int(x[0].split("x")[1]), x[1])):
            b = bucket[(grid, p)]
            rows = [
                ("solve_time", b["solve"]),
                ("halo_mpi_comm_time", b["halo"]),
                ("h2d_memcpy_time", b["h2d"]),
                ("d2h_memcpy_time", b["d2h"]),
                ("matvec_kernel_time", b["matvec"]),
                ("vector_kernels_time", b["vec"]),
            ]
            for metric_name, vals in rows:
                if not vals:
                    continue
                w.writerow([grid, p, metric_name, f"{statistics.median(vals):.10g}", len(vals)])

    print(f"Wrote: {os.path.relpath(out_csv, base_dir)}")
    print(f"Wrote: {os.path.relpath(out_csv_long, base_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


