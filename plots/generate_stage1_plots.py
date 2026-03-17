#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INITIAL_MEASURES = [
    ("Render + events", 5.0),
    ("Update", 52.0),
]

STAGE1_BENCHMARK = [
    (1, 227.970),
    (2, 124.934),
    (4, 69.480),
    (8, 48.834),
    (16, 33.233),
]


def write_initial_csv() -> None:
    path = ROOT / "initial_render_update.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "time_ms"])
        writer.writerows(INITIAL_MEASURES)


def write_scaling_csv() -> None:
    baseline = STAGE1_BENCHMARK[0][1]
    path = ROOT / "stage1_scaling.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "mean_step_ms", "speedup", "efficiency"])
        for threads, mean_ms in STAGE1_BENCHMARK:
            speedup = baseline / mean_ms
            efficiency = speedup / threads
            writer.writerow(
                [
                    threads,
                    f"{mean_ms:.3f}",
                    f"{speedup:.3f}",
                    f"{efficiency:.3f}",
                ]
            )


def write_initial_plot() -> None:
    content = r"""
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=22pt,
    width=0.82\linewidth,
    height=6.2cm,
    ymin=0,
    ylabel={Temps (ms)},
    symbolic x coords={Render + events, Update},
    xtick=data,
    xticklabel style={font=\small, align=center},
    nodes near coords,
    every node near coord/.append style={font=\footnotesize},
    enlarge x limits=0.35,
    grid=both,
    minor y tick num=1,
    axis line style={draw=black!60},
]
\addplot[fill=enstaBleuClair, draw=enstaBleuFonce] coordinates {
    (Render + events, 5.0)
    (Update, 52.0)
};
\end{axis}
\end{tikzpicture}
""".strip() + "\n"
    (ROOT / "initial_render_update_plot.tex").write_text(content, encoding="utf-8")


def write_speedup_plot() -> None:
    coords = "\n".join(
        f"        ({threads}, {STAGE1_BENCHMARK[0][1] / mean_ms:.3f})"
        for threads, mean_ms in STAGE1_BENCHMARK
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.0cm,
    xlabel={Nombre de threads},
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
    (ROOT / "stage1_speedup_plot.tex").write_text(content, encoding="utf-8")


def write_efficiency_plot() -> None:
    coords = "\n".join(
        f"        ({threads}, {(STAGE1_BENCHMARK[0][1] / mean_ms) / threads:.3f})"
        for threads, mean_ms in STAGE1_BENCHMARK
    )
    content = (
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=6.0cm,
    xlabel={Nombre de threads},
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
    (ROOT / "stage1_efficiency_plot.tex").write_text(content, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    write_initial_csv()
    write_scaling_csv()
    write_initial_plot()
    write_speedup_plot()
    write_efficiency_plot()
    print("Stage 1 plot data and LaTeX snippets generated in", ROOT)


if __name__ == "__main__":
    main()
