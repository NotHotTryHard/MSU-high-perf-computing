#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение графика ускорения (strong scaling) для сетки 1200×800
MPI+OpenMP (4 MPI-процесса, варьируется число потоков на процесс)
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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

    plt.xlabel('Число потоков на процесс', fontsize=12)
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
    # Данные из report3.tex (табл. hybrid_1200x800)
    # 4 MPI-процесса, варьируется число потоков на процесс
    
    threads_per_proc = np.array([1, 2, 4, 8])
    times = np.array([11.84, 7.28, 6.12, 4.89])  # секунды
    
    # Строим график
    output_file = Path(__file__).parent / 'scaling_1200x800.png'
    
    plot_scaling(
        threads_per_proc,
        times,
        'График ускорения для сетки $1200\\times800$ (MPI+OpenMP, 4 процесса)',
        str(output_file),
    )
    
    print(f"\nГотово! График сохранен в {output_file}")
    print(f"\nДанные:")
    print(f"  Потоков на процесс: {threads_per_proc}")
    print(f"  Времена (с): {times}")
    print(f"  Ускорения: {times[0] / times}")


if __name__ == "__main__":
    main()

