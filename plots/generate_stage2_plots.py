#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGE1_BENCHMARK = [
    (1, 227.970),
    (2, 124.934),
    (4, 69.480),
    (8, 48.834),
    (16, 33.233),
]

STAGE2_BENCHMARK = [
    (1, 246.802, 246.919, 0.117),
    (2, 138.732, 138.853, 0.121),
    (4, 79.649, 79.767, 0.118),
    (8, 52.546, 52.673, 0.127),
    (16, 42.668, 42.801, 0.132),
]

STAGE1_INTERACTIVE_SCALING = [
    (1, 4.500, 16.800, 21.300),
    (2, 5.025, 9.150, 14.175),
    (4, 4.750, 5.800, 10.550),
    (8, 4.475, 4.300, 8.775),
    (16, 4.688, 3.812, 8.500),
]

STAGE2_INTERACTIVE_SCALING = [
    (1, 5.450, 18.125, 23.575),
    (2, 4.350, 8.825, 13.175),
    (4, 4.425, 6.025, 10.450),
    (8, 4.825, 4.800, 9.625),
    (16, 4.500, 3.800, 8.300),
]


def write_csv(path: Path, header: list[str], rows: list[list[str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_stage2_csv() -> None:
    baseline = STAGE2_BENCHMARK[0][2]
    rows: list[list[str | int]] = []
    for threads, compute_ms, end_to_end_ms, overhead_ms in STAGE2_BENCHMARK:
        speedup = baseline / end_to_end_ms
        efficiency = speedup / threads
        rows.append(
            [
                threads,
                f"{compute_ms:.3f}",
                f"{end_to_end_ms:.3f}",
                f"{overhead_ms:.3f}",
                f"{speedup:.3f}",
                f"{efficiency:.3f}",
            ]
        )
    write_csv(
        ROOT / "stage2_scaling.csv",
        [
            "threads",
            "rank1_compute_ms",
            "end_to_end_ms",
            "mpi_overhead_ms",
            "speedup",
            "efficiency",
        ],
        rows,
    )


def write_stage2_interactive_csv() -> None:
    baseline = STAGE2_INTERACTIVE_SCALING[0][3]
    rows: list[list[str | int]] = []
    for threads, render_ms, wait_ms, total_ms in STAGE2_INTERACTIVE_SCALING:
        speedup = baseline / total_ms
        efficiency = speedup / threads
        rows.append(
            [
                threads,
                f"{render_ms:.3f}",
                f"{wait_ms:.3f}",
                f"{total_ms:.3f}",
                f"{speedup:.3f}",
                f"{efficiency:.3f}",
            ]
        )
    write_csv(
        ROOT / "stage2_interactive_scaling.csv",
        ["threads", "render_ms", "mpi_wait_ms", "total_ms", "speedup", "efficiency"],
        rows,
    )


def write_interactive_comparison_csv() -> None:
    stage1_render = STAGE1_INTERACTIVE_SCALING[2][1]
    stage1_update = STAGE1_INTERACTIVE_SCALING[2][2]
    stage2_render = STAGE2_INTERACTIVE_SCALING[2][1]
    stage2_wait = STAGE2_INTERACTIVE_SCALING[2][2]
    write_csv(
        ROOT / "interactive_stage1_stage2.csv",
        ["category", "stage1_ms", "stage2_ms"],
        [
            ["Render", f"{stage1_render:.3f}", f"{stage2_render:.3f}"],
            ["Update / attente", f"{stage1_update:.3f}", f"{stage2_wait:.3f}"],
        ],
    )


def write_compute_speedup_comparison_csv() -> None:
    stage1_baseline = STAGE1_BENCHMARK[0][1]
    stage2_baseline = STAGE2_BENCHMARK[0][2]
    rows: list[list[str | int]] = []
    for (threads, stage1_ms), (_, _compute_ms, stage2_end_ms, _overhead_ms) in zip(STAGE1_BENCHMARK, STAGE2_BENCHMARK):
        rows.append(
            [
                threads,
                f"{stage1_baseline / stage1_ms:.3f}",
                f"{stage2_baseline / stage2_end_ms:.3f}",
            ]
        )
    write_csv(
        ROOT / "compute_speedup_stage1_stage2.csv",
        ["threads", "stage1_speedup", "stage2_speedup"],
        rows,
    )


def write_interactive_speedup_comparison_csv() -> None:
    stage1_baseline = STAGE1_INTERACTIVE_SCALING[0][3]
    stage2_baseline = STAGE2_INTERACTIVE_SCALING[0][3]
    rows: list[list[str | int]] = []
    for (
        threads,
        _stage1_render,
        _stage1_update,
        stage1_total,
    ), (_, _stage2_render, _stage2_wait, stage2_total) in zip(STAGE1_INTERACTIVE_SCALING, STAGE2_INTERACTIVE_SCALING):
        rows.append(
            [
                threads,
                f"{stage1_baseline / stage1_total:.3f}",
                f"{stage2_baseline / stage2_total:.3f}",
            ]
        )
    write_csv(
        ROOT / "interactive_speedup_stage1_stage2.csv",
        ["threads", "stage1_speedup", "stage2_speedup"],
        rows,
    )


def write_stage2_speedup_plot() -> None:
    baseline = STAGE2_BENCHMARK[0][2]
    coords = "\n".join(
        f"        ({threads}, {baseline / end_to_end_ms:.3f})"
        for threads, _compute_ms, end_to_end_ms, _overhead_ms in STAGE2_BENCHMARK
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.1cm,
    xlabel={Nombre de threads sur le rank 1},
    ylabel={Speed-up},
    xmin=1,
    xmax=16,
    ymin=0,
    xtick={1,2,4,8,16},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[
    color=enstaBleuFonce,
    mark=*,
    line width=1.1pt,
]
coordinates {
"""
        + coords
        + r"""
};
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "stage2_speedup_plot.tex").write_text(content, encoding="utf-8")


def write_stage2_efficiency_plot() -> None:
    baseline = STAGE2_BENCHMARK[0][2]
    coords = "\n".join(
        f"        ({threads}, {(baseline / end_to_end_ms) / threads:.3f})"
        for threads, _compute_ms, end_to_end_ms, _overhead_ms in STAGE2_BENCHMARK
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.1cm,
    xlabel={Nombre de threads sur le rank 1},
    ylabel={Efficacité},
    xmin=1,
    xmax=16,
    ymin=0,
    ymax=1.05,
    xtick={1,2,4,8,16},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[
    color=enstaBleuClair,
    mark=square*,
    line width=1.1pt,
]
coordinates {
"""
        + coords
        + r"""
};
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "stage2_efficiency_plot.tex").write_text(content, encoding="utf-8")


def write_stage1_vs_stage2_plot() -> None:
    stage1_coords = "\n".join(
        f"        ({threads}, {mean_ms:.3f})"
        for threads, mean_ms in STAGE1_BENCHMARK
    )
    stage2_compute_coords = "\n".join(
        f"        ({threads}, {compute_ms:.3f})"
        for threads, compute_ms, _end_to_end_ms, _overhead_ms in STAGE2_BENCHMARK
    )
    stage2_end_coords = "\n".join(
        f"        ({threads}, {end_to_end_ms:.3f})"
        for threads, _compute_ms, end_to_end_ms, _overhead_ms in STAGE2_BENCHMARK
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=7.0cm,
    xlabel={Nombre de threads},
    ylabel={Temps moyen par pas (ms)},
    xmin=1,
    xmax=16,
    ymin=0,
    xtick={1,2,4,8,16},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=1},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[
    color=enstaBleuFonce,
    mark=*,
    line width=1.1pt,
]
coordinates {
"""
        + stage1_coords
        + r"""
};
\addlegendentry{Étape 1 compute-only}

\addplot[
    color=enstaBleuClair,
    mark=square*,
    line width=1.1pt,
]
coordinates {
"""
        + stage2_compute_coords
        + r"""
};
\addlegendentry{Étape 2 rank 1 compute}

\addplot[
    color=black!70,
    mark=triangle*,
    line width=1.1pt,
]
coordinates {
"""
        + stage2_end_coords
        + r"""
};
\addlegendentry{Étape 2 end-to-end}
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "stage1_vs_stage2_plot.tex").write_text(content, encoding="utf-8")


def write_compute_speedup_stage1_stage2_plot() -> None:
    stage1_baseline = STAGE1_BENCHMARK[0][1]
    stage2_baseline = STAGE2_BENCHMARK[0][2]
    stage1_coords = "\n".join(
        f"        ({threads}, {stage1_baseline / mean_ms:.3f})"
        for threads, mean_ms in STAGE1_BENCHMARK
    )
    stage2_coords = "\n".join(
        f"        ({threads}, {stage2_baseline / end_to_end_ms:.3f})"
        for threads, _compute_ms, end_to_end_ms, _overhead_ms in STAGE2_BENCHMARK
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.5cm,
    xlabel={Nombre de threads},
    ylabel={Speed-up},
    xmin=1,
    xmax=16,
    ymin=0,
    xtick={1,2,4,8,16},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=2},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[
    color=enstaBleuFonce,
    mark=*,
    line width=1.1pt,
]
coordinates {
"""
        + stage1_coords
        + r"""
};
\addlegendentry{Étape 1 compute-only}

\addplot[
    color=black!70,
    mark=triangle*,
    line width=1.1pt,
]
coordinates {
"""
        + stage2_coords
        + r"""
};
\addlegendentry{Étape 2 end-to-end}
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "compute_speedup_stage1_stage2_plot.tex").write_text(content, encoding="utf-8")


def write_interactive_comparison_plot() -> None:
    stage1_render = STAGE1_INTERACTIVE_SCALING[2][1]
    stage1_update = STAGE1_INTERACTIVE_SCALING[2][2]
    stage2_render = STAGE2_INTERACTIVE_SCALING[2][1]
    stage2_wait = STAGE2_INTERACTIVE_SCALING[2][2]
    content = f"""
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    bar width=14pt,
    width=\\linewidth,
    height=6.8cm,
    ymin=0,
    ylabel={{Temps moyen (ms)}},
    symbolic x coords={{Render, Update / attente}},
    xtick=data,
    enlarge x limits=0.3,
    grid=both,
    minor y tick num=1,
    legend style={{at={{(0.5,-0.18)}}, anchor=north, legend columns=2}},
    axis line style={{draw=black!60}},
]
\\addplot[fill=enstaBleuClair, draw=enstaBleuFonce] coordinates {{
    (Render, {stage1_render:.3f})
    (Update / attente, {stage1_update:.3f})
}};
\\addlegendentry{{Étape 1 interactive}}

\\addplot[fill=black!25, draw=black!70] coordinates {{
    (Render, {stage2_render:.3f})
    (Update / attente, {stage2_wait:.3f})
}};
\\addlegendentry{{Étape 2 interactive}}
\\end{{axis}}
\\end{{tikzpicture}}
""".strip() + "\n"
    (ROOT / "interactive_stage1_stage2_plot.tex").write_text(content, encoding="utf-8")


def write_interactive_speedup_stage1_stage2_plot() -> None:
    stage1_baseline = STAGE1_INTERACTIVE_SCALING[0][3]
    stage2_baseline = STAGE2_INTERACTIVE_SCALING[0][3]
    stage1_coords = "\n".join(
        f"        ({threads}, {stage1_baseline / total_ms:.3f})"
        for threads, _render_ms, _update_ms, total_ms in STAGE1_INTERACTIVE_SCALING
    )
    stage2_coords = "\n".join(
        f"        ({threads}, {stage2_baseline / total_ms:.3f})"
        for threads, _render_ms, _wait_ms, total_ms in STAGE2_INTERACTIVE_SCALING
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.5cm,
    xlabel={Nombre de threads},
    ylabel={Speed-up total},
    xmin=1,
    xmax=16,
    ymin=0,
    xtick={1,2,4,8,16},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=2},
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[
    color=enstaBleuFonce,
    mark=*,
    line width=1.1pt,
]
coordinates {
"""
        + stage1_coords
        + r"""
};
\addlegendentry{Étape 1 interactive}

\addplot[
    color=black!70,
    mark=triangle*,
    line width=1.1pt,
]
coordinates {
"""
        + stage2_coords
        + r"""
};
\addlegendentry{Étape 2 interactive}
\end{axis}
\end{tikzpicture}
"""
    ).strip() + "\n"
    (ROOT / "interactive_speedup_stage1_stage2_plot.tex").write_text(content, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    write_stage2_csv()
    write_stage2_interactive_csv()
    write_interactive_comparison_csv()
    write_compute_speedup_comparison_csv()
    write_interactive_speedup_comparison_csv()
    write_stage2_speedup_plot()
    write_stage2_efficiency_plot()
    write_stage1_vs_stage2_plot()
    write_compute_speedup_stage1_stage2_plot()
    write_interactive_comparison_plot()
    write_interactive_speedup_stage1_stage2_plot()
    print("Stage 2 plot data and LaTeX snippets generated in", ROOT)


if __name__ == "__main__":
    main()
