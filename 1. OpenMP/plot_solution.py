#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot u(x,y) from solution.csv produced by your C++ code.
Usage examples:
  python plot_solution.py                 # ищет solution.csv, сохранит sol_MxN.png
  python plot_solution.py solustion.csv   # опечатка тоже поддерживается
  python plot_solution.py -o sol_40x40.png
  python plot_solution.py --kind contour --levels 20 --dpi 200
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # поддержка частой опечатки в названии файла
        alt = "solustion.csv" if os.path.basename(path) == "solution.csv" else "solution.csv"
        if os.path.exists(alt):
            path = alt
        else:
            sys.exit(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    # сгладим возможные двоичные хвосты, чтобы pivot не споткнулся о почти-равные координаты
    df["x"] = df["x"].round(12)
    df["y"] = df["y"].round(12)
    return df, path

def grid_from_df(df: pd.DataFrame):
    # pivot: строки=y, столбцы=x, значения=u  -> массив формы (Ny, Nx)
    table = df.pivot(index="y", columns="x", values="u").sort_index().sort_index(axis=1)
    y = table.index.values
    x = table.columns.values
    U = table.values
    Ny, Nx = U.shape
    return x, y, U, Nx, Ny

def main():
    ap = argparse.ArgumentParser(description="Plot u(x,y) from solution.csv")
    ap.add_argument("csv", nargs="?", default="solution.csv", help="входной CSV (по умолчанию solution.csv)")
    ap.add_argument("-o", "--output", default=None, help="имя PNG (по умолчанию sol_MxN.png)")
    ap.add_argument("--kind", choices=["heat", "contour", "pcolormesh"], default="heat",
                    help="тип графика: heat (imshow), contour (contourf), pcolormesh")
    ap.add_argument("--levels", type=int, default=30, help="число уровней для contour")
    ap.add_argument("--dpi", type=int, default=150, help="dpi сохранения")
    ap.add_argument("--vmin", type=float, default=None, help="минимум цветовой шкалы (фиксировать масштаб)")
    ap.add_argument("--vmax", type=float, default=None, help="максимум цветовой шкалы (фиксировать масштаб)")
    ap.add_argument("--title", default=None, help="заголовок графика (по умолчанию авто)")
    args = ap.parse_args()

    df, used = load_csv(args.csv)
    x, y, U, Nx, Ny = grid_from_df(df)

    if args.output is None:
        base = f"sol_{Nx}x{Ny}.png"
        args.output = base

    plt.figure(figsize=(7, 6), dpi=args.dpi)
    extent = [x.min(), x.max(), y.min(), y.max()]
    ttl = args.title or f"u(x,y), сетка {Nx}×{Ny}"

    if args.kind == "heat":
        im = plt.imshow(U, extent=extent, origin="lower", aspect="equal",
                        vmin=args.vmin, vmax=args.vmax, interpolation="nearest", cmap="viridis")
        plt.colorbar(im, shrink=0.85, label="u")
    elif args.kind == "contour":
        # координатные сетки для контуров
        X, Y = np.meshgrid(x, y)
        cf = plt.contourf(X, Y, U, levels=args.levels, vmin=args.vmin, vmax=args.vmax, cmap="viridis")
        plt.colorbar(cf, shrink=0.85, label="u")
    else:  # pcolormesh
        X, Y = np.meshgrid(x, y)
        pm = plt.pcolormesh(X, Y, U, shading="auto", vmin=args.vmin, vmax=args.vmax, cmap="viridis")
        plt.colorbar(pm, shrink=0.85, label="u")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"[ok] Прочитал: {used}")
    print(f"[ok] Размер сетки: {Nx}x{Ny}")
    print(f"[ok] Сохранил: {args.output}")

if __name__ == "__main__":
    main()