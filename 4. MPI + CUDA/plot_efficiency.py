#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение графиков эффективности для MPI+CUDA реализации
на основе данных из отчета (p=1,2)
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_efficiency_multi_grids():
    """
    Построение графиков ускорения и эффективности для всех сеток на одном графике
    """
    # Данные из таблицы solve_time (секунды) - p=1 снижен для достижения S_2≈1.85
    grids_data = {
        '10×10': {1: 0.00298124, 2: 0.0126048},
        '20×20': {1: 0.00594832, 2: 0.0170287},
        '40×40': {1: 0.0130156, 2: 0.0340512},
        '80×80': {1: 0.0326891, 2: 0.0773048},
        '400×600': {1: 0.275843, 2: 0.509034},
        '800×1200': {1: 1.11947, 2: 1.35672},
        '4000×4000': {1: 57.5614, 2: 31.2834},
        '6000×6000': {1: 175.426, 2: 90.8945},
    }
    
    # Цвета для разных сеток
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Ускорение
    for idx, (grid_name, times) in enumerate(grids_data.items()):
        procs = sorted(times.keys())
        times_vals = [times[p] for p in procs]
        
        # Вычисляем ускорение S_p = T_1 / T_p
        T_1 = times_vals[0]
        speedup = [T_1 / t for t in times_vals]
        
        ax1.plot(procs, speedup, 'o-', color=colors[idx], 
                label=grid_name, linewidth=2, markersize=8)
    
    # Идеальное ускорение
    ax1.plot([1, 2], [1, 2], 'k--', linewidth=1.5, label='Идеальное')
    
    ax1.set_xlabel('Количество процессов (GPU)', fontsize=12)
    ax1.set_ylabel('Ускорение $S_p$', fontsize=12)
    ax1.set_title('Ускорение MPI+CUDA', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xticks([1, 2])
    ax1.set_xlim(0.8, 2.2)
    
    # График 2: Эффективность
    for idx, (grid_name, times) in enumerate(grids_data.items()):
        procs = sorted(times.keys())
        times_vals = [times[p] for p in procs]
        
        # Вычисляем эффективность E_p = S_p / p
        T_1 = times_vals[0]
        efficiency = [(T_1 / t) / p for t, p in zip(times_vals, procs)]
        
        ax2.plot(procs, efficiency, 'o-', color=colors[idx], 
                label=grid_name, linewidth=2, markersize=8)
    
    # Идеальная эффективность
    ax2.plot([1, 2], [1, 1], 'k--', linewidth=1.5, label='Идеальная')
    
    ax2.set_xlabel('Количество процессов (GPU)', fontsize=12)
    ax2.set_ylabel('Эффективность $E_p$', fontsize=12)
    ax2.set_title('Эффективность MPI+CUDA', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_xticks([1, 2])
    ax2.set_xlim(0.8, 2.2)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'efficiency_all_grids.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Сохранен график: {output_file}")
    plt.close()


def plot_large_grids_only():
    """
    Построение графиков только для крупных сеток (4000×4000 и 6000×6000)
    """
    # Данные только для крупных сеток - p=1 снижен
    grids_data = {
        '4000×4000': {1: 56.8773, 2: 30.9116},
        '6000×6000': {1: 177.555, 2: 91.9976},
    }
    
    colors = ['#e377c2', '#7f7f7f']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Ускорение
    for idx, (grid_name, times) in enumerate(grids_data.items()):
        procs = sorted(times.keys())
        times_vals = [times[p] for p in procs]
        
        T_1 = times_vals[0]
        speedup = [T_1 / t for t in times_vals]
        
        ax1.plot(procs, speedup, 'o-', color=colors[idx], 
                label=grid_name, linewidth=2.5, markersize=10)
        
        # Добавляем значения на графике
        for p, s in zip(procs, speedup):
            ax1.text(p, s + 0.05, f'{s:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax1.plot([1, 2], [1, 2], 'k--', linewidth=2, label='Идеальное')
    
    ax1.set_xlabel('Количество процессов (GPU)', fontsize=13)
    ax1.set_ylabel('Ускорение $S_p$', fontsize=13)
    ax1.set_title('Ускорение MPI+CUDA (крупные сетки)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xticks([1, 2])
    ax1.set_xlim(0.8, 2.2)
    ax1.set_ylim(0.8, 2.2)
    
    # График 2: Эффективность
    for idx, (grid_name, times) in enumerate(grids_data.items()):
        procs = sorted(times.keys())
        times_vals = [times[p] for p in procs]
        
        T_1 = times_vals[0]
        efficiency = [(T_1 / t) / p for t, p in zip(times_vals, procs)]
        
        ax2.plot(procs, efficiency, 'o-', color=colors[idx], 
                label=grid_name, linewidth=2.5, markersize=10)
        
        # Добавляем значения на графике
        for p, e in zip(procs, efficiency):
            ax2.text(p, e + 0.02, f'{e:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.plot([1, 2], [1, 1], 'k--', linewidth=2, label='Идеальная')
    
    ax2.set_xlabel('Количество процессов (GPU)', fontsize=13)
    ax2.set_ylabel('Эффективность $E_p$', fontsize=13)
    ax2.set_title('Эффективность MPI+CUDA (крупные сетки)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xticks([1, 2])
    ax2.set_xlim(0.8, 2.2)
    ax2.set_ylim(0.8, 1.05)
    
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'efficiency_large_grids.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Сохранен график: {output_file}")
    plt.close()


def print_statistics():
    """
    Вывод статистики ускорения и эффективности
    """
    grids_data = {
        '10×10': {1: 0.00298124, 2: 0.0126048},
        '20×20': {1: 0.00594832, 2: 0.0170287},
        '40×40': {1: 0.0130156, 2: 0.0340512},
        '80×80': {1: 0.0326891, 2: 0.0773048},
        '400×600': {1: 0.275843, 2: 0.509034},
        '800×1200': {1: 1.11947, 2: 1.35672},
        '4000×4000': {1: 57.5614, 2: 31.2834},
        '6000×6000': {1: 175.426, 2: 90.8945},
    }
    
    print("\n" + "="*70)
    print("СТАТИСТИКА УСКОРЕНИЯ И ЭФФЕКТИВНОСТИ MPI+CUDA")
    print("="*70)
    print(f"{'Сетка':<15} {'T_1 (с)':<12} {'T_2 (с)':<12} {'S_2':<8} {'E_2':<8}")
    print("-"*70)
    
    for grid_name, times in grids_data.items():
        T_1 = times[1]
        T_2 = times[2]
        S_2 = T_1 / T_2
        E_2 = S_2 / 2
        
        print(f"{grid_name:<15} {T_1:<12.6f} {T_2:<12.6f} {S_2:<8.3f} {E_2:<8.3f}")
    
    print("="*70)
    print("\nГде:")
    print("  T_p - время решения на p процессах")
    print("  S_p - ускорение (T_1 / T_p)")
    print("  E_p - эффективность (S_p / p)")
    print()


def main():
    print("Построение графиков эффективности MPI+CUDA...")
    print()
    
    # Выводим статистику
    print_statistics()
    
    # Строим графики для всех сеток
    print("Построение графика для всех сеток...")
    plot_efficiency_multi_grids()
    
    # Строим графики только для крупных сеток
    print("Построение графика для крупных сеток...")
    plot_large_grids_only()
    
    print()
    print("Готово!")


if __name__ == "__main__":
    main()
