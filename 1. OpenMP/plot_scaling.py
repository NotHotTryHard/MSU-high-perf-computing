#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение графиков ускорения (strong scaling) для сеток 400×600 и 800×1200
по данным из отчёта (табл. \ref{tab:scaling4060} и \ref{tab:scaling8012}).
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

def plot_scaling(p, t, title, output):
    """
    Построить график ускорения по временам t[p], где t[0] соответствует p[0]=1.
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)

    # Ускорение S_p = T_1 / T_p
    s = t[0] / t

    # Идеальное линейное ускорение: s = p
    p_ideal = np.linspace(p.min(), p.max(), 200)
    s_ideal = p_ideal

    plt.figure(figsize=(8, 6))

    # Идеальное ускорение
    plt.plot(p_ideal, s_ideal, 'k--', linewidth=1.5,
             label='Идеальное линейное ускорение', alpha=0.7)

    # Фактическое ускорение
    plt.plot(p, s, 'o-', linewidth=2, markersize=8,
             label='Фактическое ускорение', color='steelblue')

    plt.xlabel('Число процессов $p$', fontsize=12)
    plt.ylabel('Ускорение $S_p$', fontsize=12)
    plt.title(title, fontsize=13, pad=10)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Небольшой запас по осям
    plt.xlim(0, p.max() * 1.1)
    plt.ylim(0, s.max() * 1.1)

    plt.xticks(p)
    plt.yticks(np.arange(0, max(2, int(s.max()) + 2)))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"[ok] Сохранил: {output}")


def main():
    # Данные из report.tex (табл. scaling4060 и scaling8012)

    # 1) Сетка 400×600
    p_4060 = np.array([1, 2, 4, 8, 16, 32])
    t_4060 = np.array([9.475, 4.740, 3.544, 2.388, 1.413, 1.270])  # секунды
    plot_scaling(
        p_4060,
        t_4060,
        'График ускорения для сетки $400\\times600$',
        'scaling_400x600.png',
    )

    # 2) Сетка 800×1200
    p_8012 = np.array([1, 4, 8, 16, 32])
    t_8012 = np.array([87.900, 22.404, 18.915, 16.916, 14.616])  # секунды
    plot_scaling(
        p_8012,
        t_8012,
        'График ускорения для сетки $800\\times1200$',
        'scaling_800x1200.png',
    )

if __name__ == "__main__":
    main()

