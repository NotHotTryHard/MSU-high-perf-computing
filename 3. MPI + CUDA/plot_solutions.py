#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot solutions from CSV files for hw2
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    # сгладим возможные двоичные хвосты
    df["x"] = df["x"].round(12)
    df["y"] = df["y"].round(12)
    return df

def grid_from_df(df: pd.DataFrame):
    # pivot: строки=y, столбцы=x, значения=u  -> массив формы (Ny, Nx)
    table = df.pivot(index="y", columns="x", values="u").sort_index().sort_index(axis=1)
    y = table.index.values
    x = table.columns.values
    U = table.values
    Ny, Nx = U.shape
    return x, y, U, Nx, Ny

def plot_solution(csv_file, output_file=None, kind="contour", levels=30, dpi=150):
    """Plot a single solution from CSV file"""
    df = load_csv(csv_file)
    x, y, U, Nx, Ny = grid_from_df(df)
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = f"{base_name}.png"
    
    plt.figure(figsize=(10, 8), dpi=dpi)
    extent = [x.min(), x.max(), y.min(), y.max()]
    ttl = f"u(x,y), сетка {Nx}×{Ny}"
    
    if kind == "heat":
        im = plt.imshow(U, extent=extent, origin="lower", aspect="equal",
                        interpolation="nearest", cmap="viridis")
        plt.colorbar(im, shrink=0.85, label="u")
    elif kind == "contour":
        X, Y = np.meshgrid(x, y)
        cf = plt.contourf(X, Y, U, levels=levels, cmap="viridis")
        plt.colorbar(cf, shrink=0.85, label="u")
    else:  # pcolormesh
        X, Y = np.meshgrid(x, y)
        pm = plt.pcolormesh(X, Y, U, shading="auto", cmap="viridis")
        plt.colorbar(pm, shrink=0.85, label="u")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    print(f"[ok] Прочитал: {csv_file}")
    print(f"[ok] Размер сетки: {Nx}×{Ny}")
    print(f"[ok] Сохранил: {output_file}")
    plt.close()

def main():
    # Список файлов для визуализации
    csv_files = [
        "sol2_40x40.csv",
        "sol2_1200x800.csv"
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            # Для маленькой сетки используем contour, для большой - pcolormesh
            if "40x40" in csv_file:
                plot_solution(csv_file, kind="contour", levels=30, dpi=200)
            else:
                plot_solution(csv_file, kind="pcolormesh", dpi=200)
        else:
            print(f"[warning] Файл не найден: {csv_file}")

if __name__ == "__main__":
    main()

