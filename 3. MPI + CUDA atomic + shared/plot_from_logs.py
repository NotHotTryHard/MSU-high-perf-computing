#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсинг логов из log3 и построение графиков ускорения (scaling)
для MPI + CUDA реализации
"""
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    plt.ylim(0, max(s) * 1.15)

    plt.xticks(p)
    plt.yticks(np.arange(0, max(2, int(s.max()) + 2)))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"[ok] Сохранил: {output}")


def load_csv(path: str):
    """Загрузка CSV с решением"""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    df["x"] = df["x"].round(12)
    df["y"] = df["y"].round(12)
    return df, path


def grid_from_df(df: pd.DataFrame):
    """Преобразование DataFrame в сетку для визуализации"""
    table = df.pivot(index="y", columns="x", values="u").sort_index().sort_index(axis=1)
    y = table.index.values
    x = table.columns.values
    U = table.values
    Ny, Nx = U.shape
    return x, y, U, Nx, Ny


def parse_log_file(log_path):
    """
    Парсит лог-файл и извлекает время решения (solve time max over ranks).
    Возвращает None, если файл не найден или время не найдено.
    """
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Ищем строку вида "solve time (max over ranks): X.XXXXX s"
    match = re.search(r'solve time \(max over ranks\):\s*([\d.]+)\s*s', content)
    if match:
        return float(match.group(1))
    
    # Попробуем также старый формат для совместимости
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
    # Определяем директорию с логами (проверяем вложенную структуру)
    script_dir = Path(__file__).parent
    log_dir = script_dir / 'log3'
    
    # Если вложенной директории нет, используем обычную
    if not log_dir.exists():
        log_dir = script_dir / 'log3'
    
    output_dir = script_dir
    
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
        title = f'График ускорения для сетки ${grid_x}\\times{grid_y}$ (MPI+CUDA)'
        
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
        title = f'График ускорения для сетки ${grid_x}\\times{grid_y}$ (part3, MPI+CUDA)'
        
        plot_scaling(
            np.array(procs_list),
            np.array(times_list),
            title,
            str(output_file)
        )
        print()
    
    # Ищем CSV файлы для построения графиков решения
    print("=== Поиск CSV файлов для графиков решения ===")
    csv_files = list(output_dir.glob('sol3_*.csv'))
    if not csv_files:
        csv_files = list(output_dir.glob('solution*.csv'))
    
    if csv_files:
        print(f"Найдено CSV файлов: {len(csv_files)}")
        for csv_file in csv_files:
            try:
                result = load_csv(str(csv_file))
                if result[0] is None:
                    print(f"  [warning] Не удалось загрузить {csv_file.name}")
                    continue
                    
                df, used = result
                x, y, U, Nx, Ny = grid_from_df(df)
                
                output_png = output_dir / f'sol3_{Nx}x{Ny}.png'
                print(f"  Построение графика из {csv_file.name} -> {output_png.name}")
                
                plt.figure(figsize=(7, 6), dpi=150)
                extent = [x.min(), x.max(), y.min(), y.max()]
                im = plt.imshow(U, extent=extent, origin="lower", aspect="equal",
                              interpolation="nearest", cmap="viridis")
                plt.colorbar(im, shrink=0.85, label="u")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"u(x,y), сетка {Nx}×{Ny} (MPI+CUDA)")
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

