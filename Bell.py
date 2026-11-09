import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    receptor_total: float = 100.0
    kd: float = 1.0
    antigen_clearance: float = 1.0
    T1: float = 1.0
    T2: float = 1.0
    T2_death: float = 8.0
    T3: float = 5.0
    T4: float = 30.0
    initial_S: float = 1.0
    t_end: float = 40.0
    points: int = 1601
    rtol: float = 1e-8
    atol: float = 1e-10


SCENARIOS = (
    ("Низкая доза", 1e-3),
    ("Промежуточная доза", 1e-1),
    ("Высокая доза", 1e1),
)

POPULATIONS = ("S", "L", "P", "M")
POPULATION_LABELS = {
    "S": "Малые покоящиеся лимфоциты S",
    "L": "Пролиферирующие лимфоциты L",
    "P": "Плазматические клетки P",
    "M": "Клетки памяти M",
}


def antigen_profile(t, dose_ratio, p):
    return dose_ratio * p.kd * np.exp(-p.antigen_clearance * np.asarray(t))


def receptor_occupancy(t, dose_ratio, p):
    antigen = antigen_profile(t, dose_ratio, p)
    fraction = antigen / (p.kd + antigen)
    fraction = np.clip(fraction, 0.0, 1.0)
    return p.receptor_total * fraction


def bell_functions(R_prime, p):
    R = np.maximum(np.asarray(R_prime, dtype=float), 0.0)
    fraction = np.clip(R / p.receptor_total, 0.0, 1.0)
    F = R / (1.0 + R)
    G = (1.0 - fraction) * F
    H = (R - 1.0) / (R + 1.0)
    return F, G, H


def bell_rhs(t, y, dose_ratio, p):
    S, L, P, M = np.maximum(y, 0.0)
    R_prime = receptor_occupancy(t, dose_ratio, p)
    F, G, H = bell_functions(R_prime, p)
    dS = -F * S / p.T1
    dL = G * S / p.T1 + H * L / p.T2 - L / p.T2_death
    dP = (1.0 - H) * L / (2.0 * p.T2) - P / p.T3
    dM = (1.0 - H) * L / (2.0 * p.T2) - M / p.T4
    return np.array([dS, dL, dP, dM], dtype=float)


def simulate(dose_ratio, p):
    if not math.isfinite(dose_ratio) or dose_ratio <= 0:
        raise ValueError("Отношение A0/KD должно быть положительным и конечным.")
    times = np.linspace(0.0, p.t_end, p.points)
    initial = np.array([p.initial_S, 0.0, 0.0, 0.0], dtype=float)
    solution = solve_ivp(
        lambda t, y: bell_rhs(t, y, dose_ratio, p),
        (0.0, p.t_end),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=p.rtol,
        atol=p.atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    values = np.maximum(solution.y, 0.0)
    return {
        "t": times,
        "S": values[0],
        "L": values[1],
        "P": values[2],
        "M": values[3],
        "R_prime": receptor_occupancy(times, dose_ratio, p),
        "dose_ratio": float(dose_ratio),
    }


def peak_metrics(times, values):
    index = int(np.argmax(values))
    return float(values[index]), float(times[index])


def summarize_run(run):
    times = run["t"]
    row = {
        "scenario": run.get("scenario", ""),
        "A0_over_KD": run["dose_ratio"],
    }
    for name in ("L", "P", "M"):
        peak, peak_time = peak_metrics(times, run[name])
        row[f"{name}_peak"] = peak
        row[f"{name}_t_peak_days"] = peak_time
    for name in POPULATIONS:
        row[f"{name}_final"] = float(run[name][-1])
    row["P_integral_population_days"] = float(np.trapezoid(run["P"], times))
    return row


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_functions(p, outdir):
    R = np.geomspace(1e-4, p.receptor_total * (1.0 - 1e-6), 800)
    F, G, H = bell_functions(R, p)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, values, name in zip(axes, (F, G, H), ("F", "G", "H")):
        axis.plot(R, values)
        axis.set_xscale("log")
        axis.set_xlabel(r"$R'$, занятых участков на клетку")
        axis.set_ylabel("Значение функции")
        axis.set_title(rf"Функция ${name}(R')$")
        axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(outdir / "01_functions.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_population_dynamics(runs, outdir):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, name in zip(axes.flat, POPULATIONS):
        for label, run in runs:
            values = np.where(run[name] > 1e-10, run[name], np.nan)
            axis.plot(run["t"], values, label=label)
        axis.set_title(POPULATION_LABELS[name])
        axis.set_xlabel("Время, дни")
        axis.set_ylabel("Нормированная численность")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / "02_population_dynamics.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def dose_sweep(p, outdir, count):
    doses = np.geomspace(1e-4, 1e5, count)
    rows = []
    for dose in doses:
        run = simulate(float(dose), p)
        row = summarize_run(run)
        row["F_initial"], row["G_initial"], row["H_initial"] = (
            float(value) for value in bell_functions(run["R_prime"][0], p)
        )
        rows.append(row)
    write_csv(outdir / "dose_sweep.csv", rows)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    chart_specs = (
        ("L_peak", "Максимум пролиферирующих лимфоцитов L"),
        ("P_peak", "Максимум плазматических клеток P"),
        ("M_peak", "Максимум клеток памяти M"),
        ("P_integral_population_days", r"Интеграл $I_P=\int P(t)dt$"),
    )
    x = np.array([row["A0_over_KD"] for row in rows])
    for axis, (column, title) in zip(axes.flat, chart_specs):
        y = np.array([row[column] for row in rows])
        axis.plot(x, y, marker="o", markersize=3)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel(r"Начальная доза $A_0/K_D$")
        axis.set_ylabel("Нормированная величина")
        axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(outdir / "03_dose_response.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="bell_results")
    parser.add_argument("--t-end", type=float, default=40.0)
    parser.add_argument("--points", type=int, default=1601)
    parser.add_argument("--rt", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--clearance", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--t2", type=float, default=1.0)
    parser.add_argument("--t2-death", type=float, default=8.0)
    parser.add_argument("--t3", type=float, default=5.0)
    parser.add_argument("--t4", type=float, default=30.0)
    parser.add_argument("--dose-count", type=int, default=45)
    args = parser.parse_args()
    if args.t_end <= 0 or args.points < 100 or args.dose_count < 5:
        parser.error("Требуются t-end > 0, points >= 100 и dose-count >= 5.")
    for name in ("rt", "kd", "clearance", "t1", "t2", "t2_death", "t3", "t4"):
        if getattr(args, name) <= 0:
            parser.error(f"Параметр --{name.replace('_', '-')} должен быть положительным.")
    return args


def main():
    args = parse_args()
    p = Parameters(
        receptor_total=args.rt,
        kd=args.kd,
        antigen_clearance=args.clearance,
        T1=args.t1,
        T2=args.t2,
        T2_death=args.t2_death,
        T3=args.t3,
        T4=args.t4,
        t_end=args.t_end,
        points=args.points,
    )
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        runs = []
        for label, dose in SCENARIOS:
            run = simulate(dose, p)
            run["scenario"] = label
            runs.append((label, run))
        plot_functions(p, outdir)
        plot_population_dynamics(runs, outdir)
        summary_rows = [summarize_run(run) for _, run in runs]
        write_csv(outdir / "scenario_summary.csv", summary_rows)
        sweep_rows = dose_sweep(p, outdir, args.dose_count)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        print(f"Ошибка расчёта: {error}", file=sys.stderr)
        return 1

    best = max(sweep_rows, key=lambda row: row["P_peak"])
    print("Расчёт завершён.")
    print(f"Результаты сохранены: {outdir}")
    print("PNG: 01_functions.png, 02_population_dynamics.png, 03_dose_response.png")
    print("CSV: scenario_summary.csv, dose_sweep.csv")
    print(f"Максимум P_peak в параметрическом проходе: A0/KD={best['A0_over_KD']:.4g}, P_peak={best['P_peak']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
