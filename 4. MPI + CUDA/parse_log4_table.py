#!/usr/bin/env python3
"""
Скрипт для создания таблицы времен processes x grid из log4
"""
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple


def parse_grid_p_from_filename(path: str) -> Optional[Tuple[str, int]]:
    """Извлекает размер сетки и количество процессов из имени файла"""
    base = os.path.basename(path)
    # Формат: 400x600-p1.out
    m = re.search(r"(\d+)x(\d+).*?-p(\d+)\.out$", base)
    if not m:
        return None
    n = int(m.group(1))
    m_ = int(m.group(2))
    p = int(m.group(3))
    return (f"{n}x{m_}", p)


def parse_time_from_log(text: str) -> Optional[float]:
    """Извлекает время выполнения из лога"""
    # Ищем "solve time (max over ranks): X.XX s"
    solve_match = re.search(r"solve time \(max over ranks\):\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*s", text)
    if solve_match:
        return float(solve_match.group(1))
    
    # Если нет solve time, используем Run time из LSF
    run_time_match = re.search(r"Run time\s*:\s*([0-9]+)\s*sec", text)
    if run_time_match:
        return float(run_time_match.group(1))
    
    # Или CPU time
    cpu_time_match = re.search(r"CPU time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*sec", text)
    if cpu_time_match:
        return float(cpu_time_match.group(1))
    
    return None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "log4")
    
    # Словарь: grid -> p -> list of times
    times_by_grid_p: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    
    # Парсим все .out файлы
    if not os.path.isdir(log_dir):
        print(f"Директория {log_dir} не найдена")
        return 1
    
    for filename in os.listdir(log_dir):
        if not filename.endswith(".out"):
            continue
        
        filepath = os.path.join(log_dir, filename)
        key = parse_grid_p_from_filename(filepath)
        if key is None:
            continue
        
        grid, p = key
        
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as e:
            print(f"Ошибка чтения {filepath}: {e}")
            continue
        
        time = parse_time_from_log(text)
        if time is not None:
            times_by_grid_p[grid][p].append(time)
    
    # Сортируем сетки по размеру
    def grid_sort_key(grid_str):
        parts = grid_str.split("x")
        return (int(parts[0]), int(parts[1]))
    
    grids = sorted(times_by_grid_p.keys(), key=grid_sort_key)
    
    # Находим все уникальные значения p
    all_ps = set()
    for grid in grids:
        all_ps.update(times_by_grid_p[grid].keys())
    all_ps = sorted(all_ps)
    
    if not grids or not all_ps:
        print("Не найдено данных для построения таблицы")
        return 1
    
    # Создаем CSV файл
    csv_path = os.path.join(base_dir, "log4_times_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Заголовок
        header = ["Grid"] + [f"p={p}" for p in all_ps]
        writer.writerow(header)
        
        # Данные
        for grid in grids:
            row = [grid]
            for p in all_ps:
                times = times_by_grid_p[grid].get(p, [])
                if times:
                    # Используем среднее значение, если несколько измерений
                    avg_time = sum(times) / len(times)
                    row.append(f"{avg_time:.6f}")
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"Таблица сохранена в: {csv_path}")
    
    # Выводим таблицу в консоль
    print("\nТаблица времен (processes x grid):")
    print("=" * (15 + 12 * len(all_ps)))
    
    # Заголовок
    header_str = f"{'Grid':<15}"
    for p in all_ps:
        header_str += f"{'p=' + str(p):>12}"
    print(header_str)
    print("-" * (15 + 12 * len(all_ps)))
    
    # Данные
    for grid in grids:
        row_str = f"{grid:<15}"
        for p in all_ps:
            times = times_by_grid_p[grid].get(p, [])
            if times:
                avg_time = sum(times) / len(times)
                row_str += f"{avg_time:>12.6f}"
            else:
                row_str += f"{'N/A':>12}"
        print(row_str)
    
    return 0


if __name__ == "__main__":
    exit(main())

