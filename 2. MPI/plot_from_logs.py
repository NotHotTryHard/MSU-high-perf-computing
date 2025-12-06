#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсинг логов из log2 и построение графиков ускорения (scaling)
используя функции из hw1/plot_scaling.py
"""
import os
import re
import sys
from pathlib import Path

# Добавляем путь к hw1 для импорта функций
sys.path.insert(0, str(Path(__file__).parent.parent / 'hw1'))

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

# Импортируем функции построения графиков
from plot_scaling import plot_scaling
from plot_solution import load_csv, grid_from_df


def parse_log_file(log_path):
    """
    Парсит лог-файл и извлекает время решения (solve time).
    Возвращает None, если файл не найден или время не найдено.
    """
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Ищем строку вида "solve time: X.XXXXX s"
    match = re.search(r'solve time:\s*([\d.]+)\s*s', content)
    if match:
        return float(match.group(1))
    
    return None


def extract_grid_and_procs(filename):
    """
    Извлекает размер сетки и количество процессов из имени файла.
    Например: part1-400x600-p2.out -> (400, 600, 2)
    """
    match = re.match(r'part\d+-(\d+)x(\d+)-p(\d+)\.out', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def collect_scaling_data(log_dir):
    """
    Собирает данные о времени решения для разных конфигураций.
    Возвращает словарь: {(grid_x, grid_y): {p: time, ...}, ...}
    """
    data = {}
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Ошибка: директория {log_dir} не найдена")
        return data
    
    # Обрабатываем только part1 файлы (для scaling)
    for log_file in sorted(log_path.glob('part1-*.out')):
        grid_x, grid_y, procs = extract_grid_and_procs(log_file.name)
        if grid_x is None:
            continue
        
        grid_key = (grid_x, grid_y)
        if grid_key not in data:
            data[grid_key] = {}
        
        time = parse_log_file(log_file)
        if time is not None:
            data[grid_key][procs] = time
            print(f"  {log_file.name}: {time:.4f} s (p={procs})")
        else:
            print(f"  Предупреждение: не удалось извлечь время из {log_file.name}")
    
    return data


def collect_part3_data(log_dir):
    """
    Собирает данные для part3 (40x40 с разным количеством процессов).
    """
    data = {}
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return data
    
    for log_file in sorted(log_path.glob('part3-40x40-p*.out')):
        grid_x, grid_y, procs = extract_grid_and_procs(log_file.name)
        if grid_x is None:
            continue
        
        grid_key = (grid_x, grid_y)
        if grid_key not in data:
            data[grid_key] = {}
        
        time = parse_log_file(log_file)
        if time is not None:
            data[grid_key][procs] = time
            print(f"  {log_file.name}: {time:.4f} s (p={procs})")
    
    return data


def main():
    log_dir = Path(__file__).parent / 'log2'
    output_dir = Path(__file__).parent
    
    print("Парсинг логов из:", log_dir)
    print()
    
    # Собираем данные для part1 (scaling)
    print("=== Part1: Scaling данные ===")
    scaling_data = collect_scaling_data(log_dir)
    print()
    
    # Строим графики для каждой сетки
    for (grid_x, grid_y), times_dict in scaling_data.items():
        if not times_dict:
            continue
        
        # Сортируем по количеству процессов
        procs_list = sorted(times_dict.keys())
        times_list = [times_dict[p] for p in procs_list]
        
        if len(procs_list) < 2:
            print(f"Пропускаем {grid_x}x{grid_y}: недостаточно данных")
            continue
        
        print(f"Построение графика для сетки {grid_x}x{grid_y}...")
        print(f"  Процессы: {procs_list}")
        print(f"  Времена: {times_list}")
        
        output_file = output_dir / f'scaling_{grid_x}x{grid_y}.png'
        title = f'График ускорения для сетки ${grid_x}\\times{grid_y}$'
        
        plot_scaling(
            np.array(procs_list),
            np.array(times_list),
            title,
            str(output_file)
        )
        print()
    
    # Собираем данные для part3 (40x40)
    print("=== Part3: 40x40 scaling ===")
    part3_data = collect_part3_data(log_dir)
    print()
    
    for (grid_x, grid_y), times_dict in part3_data.items():
        if not times_dict:
            continue
        
        procs_list = sorted(times_dict.keys())
        times_list = [times_dict[p] for p in procs_list]
        
        if len(procs_list) < 2:
            print(f"Пропускаем {grid_x}x{grid_y}: недостаточно данных")
            continue
        
        print(f"Построение графика для сетки {grid_x}x{grid_y} (part3)...")
        print(f"  Процессы: {procs_list}")
        print(f"  Времена: {times_list}")
        
        output_file = output_dir / f'scaling_part3_{grid_x}x{grid_y}.png'
        title = f'График ускорения для сетки ${grid_x}\\times{grid_y}$ (part3)'
        
        plot_scaling(
            np.array(procs_list),
            np.array(times_list),
            title,
            str(output_file)
        )
        print()
    
    # Ищем CSV файлы для построения графиков решения
    print("=== Поиск CSV файлов для графиков решения ===")
    csv_files = list(output_dir.glob('solution*.csv'))
    if csv_files:
        print(f"Найдено CSV файлов: {len(csv_files)}")
        for csv_file in csv_files:
            try:
                df, used = load_csv(str(csv_file))
                x, y, U, Nx, Ny = grid_from_df(df)
                
                output_png = output_dir / f'sol_{Nx}x{Ny}.png'
                print(f"  Построение графика из {csv_file.name} -> {output_png.name}")
                
                plt.figure(figsize=(7, 6), dpi=150)
                extent = [x.min(), x.max(), y.min(), y.max()]
                im = plt.imshow(U, extent=extent, origin="lower", aspect="equal",
                              interpolation="nearest", cmap="viridis")
                plt.colorbar(im, shrink=0.85, label="u")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"u(x,y), сетка {Nx}×{Ny}")
                plt.tight_layout()
                plt.savefig(output_png, dpi=150)
                plt.close()
                print(f"    [ok] Сохранено: {output_png.name}")
            except Exception as e:
                print(f"    [ошибка] Не удалось построить график из {csv_file.name}: {e}")
    else:
        print("CSV файлы не найдены. Если программа создает solution.csv, убедитесь, что он находится в текущей директории.")
    
    print()
    print("Готово!")


if __name__ == "__main__":
    main()

