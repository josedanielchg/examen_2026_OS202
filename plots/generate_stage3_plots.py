#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGE2_END_TO_END = {
    1: 246.919,
    2: 138.853,
    4: 79.767,
}

STAGE3_RESULTS = [
    {"workers": 4, "threads": 1, "status": "ok", "end_to_end_ms": 125.816, "worker_step_mean_ms": 101.738, "worker_step_max_ms": 125.148, "migration_ms": 26.652, "halo_ms": 0.493, "reduce_ms": 0.255, "imbalance_ms": 23.410},
    {"workers": 4, "threads": 2, "status": "ok", "end_to_end_ms": 72.342, "worker_step_mean_ms": 58.431, "worker_step_max_ms": 71.717, "migration_ms": 12.227, "halo_ms": 0.488, "reduce_ms": 0.259, "imbalance_ms": 13.286},
    {"workers": 4, "threads": 4, "status": "ok", "end_to_end_ms": 52.114, "worker_step_mean_ms": 43.161, "worker_step_max_ms": 51.457, "migration_ms": 7.620, "halo_ms": 0.537, "reduce_ms": 0.276, "imbalance_ms": 8.295},
    {"workers": 6, "threads": 1, "status": "ok", "end_to_end_ms": 163.798, "worker_step_mean_ms": 108.963, "worker_step_max_ms": 162.997, "migration_ms": 52.802, "halo_ms": 0.596, "reduce_ms": 0.325, "imbalance_ms": 54.033},
    {"workers": 6, "threads": 2, "status": "ok", "end_to_end_ms": 100.596, "worker_step_mean_ms": 68.752, "worker_step_max_ms": 99.907, "migration_ms": 32.172, "halo_ms": 0.627, "reduce_ms": 0.347, "imbalance_ms": 31.154},
    {"workers": 6, "threads": 4, "status": "n/a"},
    {"workers": 8, "threads": 1, "status": "ok", "end_to_end_ms": 139.213, "worker_step_mean_ms": 95.797, "worker_step_max_ms": 138.461, "migration_ms": 47.931, "halo_ms": 0.658, "reduce_ms": 0.336, "imbalance_ms": 42.664},
    {"workers": 8, "threads": 2, "status": "unstable"},
    {"workers": 8, "threads": 4, "status": "n/a"},
]


def write_csv(path: Path, header: list[str], rows: list[list[str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_stage3_csv() -> None:
    rows = []
    baselines = {}
    for row in STAGE3_RESULTS:
        if row["status"] == "ok" and row["workers"] == 4:
            baselines[row["threads"]] = row["end_to_end_ms"]

    for row in STAGE3_RESULTS:
        if row["status"] != "ok":
            rows.append([row["workers"], row["threads"], row["status"], "", "", "", "", "", "", "", ""])
            continue
        speedup = baselines[row["threads"]] / row["end_to_end_ms"]
        comparison_stage2 = STAGE2_END_TO_END.get(row["threads"], 0.0) / row["end_to_end_ms"]
        rows.append(
            [
                row["workers"],
                row["threads"],
                row["status"],
                f"{row['end_to_end_ms']:.3f}",
                f"{row['worker_step_mean_ms']:.3f}",
                f"{row['worker_step_max_ms']:.3f}",
                f"{row['migration_ms']:.3f}",
                f"{row['halo_ms']:.3f}",
                f"{row['reduce_ms']:.3f}",
                f"{row['imbalance_ms']:.3f}",
                f"{speedup:.3f}",
                f"{comparison_stage2:.3f}",
            ]
        )

    write_csv(
        ROOT / "stage3_scaling.csv",
        [
            "workers",
            "threads",
            "status",
            "end_to_end_ms",
            "worker_step_mean_ms",
            "worker_step_max_ms",
            "migration_ms",
            "halo_ms",
            "reduce_ms",
            "imbalance_ms",
            "speedup_vs_4workers_same_threads",
            "speedup_vs_stage2_same_threads",
        ],
        rows,
    )


def _coords_for_threads(threads: int) -> str:
    baseline = None
    coords = []
    for row in STAGE3_RESULTS:
        if row["threads"] != threads or row["status"] != "ok":
            continue
        if row["workers"] == 4:
            baseline = row["end_to_end_ms"]
            break
    if baseline is None:
        return ""
    for row in STAGE3_RESULTS:
        if row["threads"] == threads and row["status"] == "ok":
            coords.append(f"        ({row['workers']}, {baseline / row['end_to_end_ms']:.3f})")
    return "\n".join(coords)


def write_stage3_speedup_plot() -> None:
    coords_t1 = _coords_for_threads(1)
    coords_t2 = _coords_for_threads(2)
    coords_t4 = _coords_for_threads(4)
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.5cm,
    xlabel={Nombre de workers MPI},
    ylabel={Speed-up relatif à 4 workers},
    xmin=4,
    xmax=8,
    ymin=0,
    xtick={4,6,8},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=3},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[color=enstaBleuFonce, mark=* , line width=1.1pt] coordinates {
"""
        + coords_t1
        + r"""
};
\addlegendentry{1 thread}

\addplot[color=enstaBleuClair, mark=square* , line width=1.1pt] coordinates {
"""
        + coords_t2
        + r"""
};
\addlegendentry{2 threads}

\addplot[color=black!70, mark=triangle* , line width=1.1pt] coordinates {
"""
        + coords_t4
        + r"""
};
\addlegendentry{4 threads}
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "stage3_speedup_plot.tex").write_text(content, encoding="utf-8")


def write_stage3_breakdown_plot() -> None:
    selected = [
        next(row for row in STAGE3_RESULTS if row["workers"] == 4 and row["threads"] == 1),
        next(row for row in STAGE3_RESULTS if row["workers"] == 4 and row["threads"] == 2),
        next(row for row in STAGE3_RESULTS if row["workers"] == 4 and row["threads"] == 4),
        next(row for row in STAGE3_RESULTS if row["workers"] == 6 and row["threads"] == 2),
        next(row for row in STAGE3_RESULTS if row["workers"] == 8 and row["threads"] == 1),
    ]
    labels = ",".join(f"{row['workers']}w-{row['threads']}t" for row in selected)
    other_compute = "\n".join(
        f"    ({row['workers']}w-{row['threads']}t, {row['worker_step_mean_ms'] - row['migration_ms'] - row['halo_ms'] - row['reduce_ms']:.3f})"
        for row in selected
    )
    migration = "\n".join(f"    ({row['workers']}w-{row['threads']}t, {row['migration_ms']:.3f})" for row in selected)
    halo = "\n".join(f"    ({row['workers']}w-{row['threads']}t, {row['halo_ms']:.3f})" for row in selected)
    reduce = "\n".join(f"    ({row['workers']}w-{row['threads']}t, {row['reduce_ms']:.3f})" for row in selected)
    imbalance = "\n".join(f"    ({row['workers']}w-{row['threads']}t, {row['imbalance_ms']:.3f})" for row in selected)
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    ybar stacked,
    width=\linewidth,
    height=7.0cm,
    ylabel={Temps (ms)},
    symbolic x coords={"""
        + labels
        + r"""},
    xtick=data,
    x tick label style={rotate=20, anchor=east},
    legend style={at={(0.5,-0.22)}, anchor=north, legend columns=2},
    enlarge x limits=0.15,
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[fill=enstaBleuFonce] coordinates {
"""
        + other_compute
        + r"""
};
\addlegendentry{Autre calcul}

\addplot[fill=enstaBleuClair] coordinates {
"""
        + migration
        + r"""
};
\addlegendentry{Migration}

\addplot[fill=black!20] coordinates {
"""
        + halo
        + r"""
};
\addlegendentry{Halo}

\addplot[fill=black!40] coordinates {
"""
        + reduce
        + r"""
};
\addlegendentry{Réduction}

\addplot[fill=black!65] coordinates {
"""
        + imbalance
        + r"""
};
\addlegendentry{Déséquilibre}
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "stage3_breakdown_plot.tex").write_text(content, encoding="utf-8")


def write_stage2_vs_stage3_plot() -> None:
    stage2 = "\n".join(f"    ({threads}, {STAGE2_END_TO_END[threads]:.3f})" for threads in (1, 2, 4))
    stage3 = "\n".join(
        f"    ({threads}, {next(row['end_to_end_ms'] for row in STAGE3_RESULTS if row['workers'] == 4 and row['threads'] == threads and row['status'] == 'ok'):.3f})"
        for threads in (1, 2, 4)
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.5cm,
    xlabel={Nombre de threads Numba},
    ylabel={Temps bout-en-bout (ms)},
    xmin=1,
    xmax=4,
    ymin=0,
    xtick={1,2,4},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=2},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[color=black!70, mark=square* , line width=1.1pt] coordinates {
"""
        + stage2
        + r"""
};
\addlegendentry{Étape 02}

\addplot[color=enstaBleuFonce, mark=* , line width=1.1pt] coordinates {
"""
        + stage3
        + r"""
};
\addlegendentry{Étape 03 avec 4 workers}
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "stage2_vs_stage3_plot.tex").write_text(content, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    write_stage3_csv()
    write_stage3_speedup_plot()
    write_stage3_breakdown_plot()
    write_stage2_vs_stage3_plot()
    print("Stage 3 plot data and LaTeX snippets generated in", ROOT)


if __name__ == "__main__":
    main()
