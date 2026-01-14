#!/usr/bin/env python3
"""
Создает детальную таблицу времени выполнения с разбивкой по компонентам.
Читает median_times_long.csv и создает LaTeX таблицу.
Без использования pandas - только стандартная библиотека.
"""

import csv
from collections import defaultdict

def read_timing_data(csv_file='median_times_long.csv'):
    """Читает CSV файл и возвращает словарь с данными."""
    data = defaultdict(dict)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            grid = row['grid']
            p = int(row['p'])
            metric = row['metric']
            median_s = float(row['median_s'])
            
            key = (grid, p)
            data[key][metric] = median_s
    
    return data


def create_timing_table(csv_file='median_times_long.csv', output_file='timing_breakdown_table.tex'):
    """Создает LaTeX таблицу с детальной разбивкой времени."""
    
    data = read_timing_data(csv_file)
    
    # Метрики для отображения
    metrics_order = [
        'solve_time',
        'matvec_kernel_time', 
        'vector_kernels_time',
        'halo_mpi_comm_time',
        'h2d_memcpy_time',
        'd2h_memcpy_time'
    ]
    
    metric_names = {
        'solve_time': 'Solve Total',
        'matvec_kernel_time': 'MatVec Kernel',
        'vector_kernels_time': 'Vector Kernels',
        'halo_mpi_comm_time': 'Halo MPI Comm',
        'h2d_memcpy_time': 'H→D memcpy',
        'd2h_memcpy_time': 'D→H memcpy'
    }
    
    grids = ['40x40', '80x80', '400x600', '800x1200', '4000x4000', '6000x6000']
    processes = [1, 2, 3, 4, 6]
    
    # Создаем LaTeX таблицу
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Детальная разбивка времени выполнения (медианные значения, сек)}')
    latex_lines.append(r'\label{tab:timing_breakdown}')
    latex_lines.append(r'\small')
    latex_lines.append(r'\begin{tabular}{lrcccccc}')
    latex_lines.append(r'\hline')
    
    # Заголовок
    header = ['Grid', '$p$'] + [metric_names.get(m, m) for m in metrics_order]
    latex_lines.append(' & '.join(header) + r' \\')
    latex_lines.append(r'\hline')
    
    # Данные
    current_grid = None
    for grid in grids:
        for p in processes:
            key = (grid, p)
            if key not in data:
                continue
            
            # Добавляем разделитель между сетками
            if current_grid is not None and current_grid != grid:
                latex_lines.append(r'\hline')
            current_grid = grid
            
            row_values = [grid, str(p)]
            
            for metric in metrics_order:
                val = data[key].get(metric)
                if val is not None:
                    if val < 0.001:
                        row_values.append(f'{val:.6f}')
                    elif val < 0.01:
                        row_values.append(f'{val:.5f}')
                    elif val < 1:
                        row_values.append(f'{val:.4f}')
                    elif val < 10:
                        row_values.append(f'{val:.3f}')
                    else:
                        row_values.append(f'{val:.2f}')
                else:
                    row_values.append('---')
            
            latex_lines.append(' & '.join(row_values) + r' \\')
    
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Таблица сохранена в {output_file}")
    
    # CSV версия
    csv_output = output_file.replace('.tex', '.csv')
    with open(csv_output, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Grid', 'p'] + [metric_names[m] for m in metrics_order])
        
        for grid in grids:
            for p in processes:
                key = (grid, p)
                if key not in data:
                    continue
                
                row = [grid, p]
                for metric in metrics_order:
                    val = data[key].get(metric)
                    row.append(val if val is not None else '')
                writer.writerow(row)
    
    print(f"CSV версия сохранена в {csv_output}")


def create_summary_table(csv_file='median_times_long.csv', output_file='timing_summary_table.tex'):
    """Создает сводную таблицу с процентным соотношением времени."""
    
    data = read_timing_data(csv_file)
    
    grids = ['400x600', '800x1200', '4000x4000', '6000x6000']
    processes = [1, 2, 4, 6]
    
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Распределение времени решателя по компонентам (\%)}')
    latex_lines.append(r'\label{tab:timing_percentage}')
    latex_lines.append(r'\small')
    latex_lines.append(r'\begin{tabular}{lrrrrr}')
    latex_lines.append(r'\hline')
    latex_lines.append(r'Grid & $p$ & MatVec & Vectors & Halo & MemCpy \\')
    latex_lines.append(r'\hline')
    
    current_grid = None
    for grid in grids:
        for p in processes:
            key = (grid, p)
            if key not in data:
                continue
            
            solve_time = data[key].get('solve_time')
            if solve_time is None or solve_time == 0:
                continue
            
            if current_grid is not None and current_grid != grid:
                latex_lines.append(r'\hline')
            current_grid = grid
            
            matvec = data[key].get('matvec_kernel_time', 0)
            vec = data[key].get('vector_kernels_time', 0)
            halo = data[key].get('halo_mpi_comm_time', 0)
            h2d = data[key].get('h2d_memcpy_time', 0)
            d2h = data[key].get('d2h_memcpy_time', 0)
            
            matvec_pct = (matvec / solve_time * 100)
            vec_pct = (vec / solve_time * 100)
            halo_pct = (halo / solve_time * 100)
            memcpy_pct = ((h2d + d2h) / solve_time * 100)
            
            row = [grid, str(p), 
                   f'{matvec_pct:.1f}', f'{vec_pct:.1f}', 
                   f'{halo_pct:.1f}', f'{memcpy_pct:.1f}']
            latex_lines.append(' & '.join(row) + r' \\')
    
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Сводная таблица сохранена в {output_file}")


def create_wide_format_table(csv_file='median_times_long.csv', output_file='timing_wide_table.tex'):
    """Создает широкую таблицу с процессами в колонках."""
    
    data = read_timing_data(csv_file)
    
    grids = ['40x40', '400x600', '800x1200', '4000x4000', '6000x6000']
    processes = [1, 2, 4, 6]
    
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Время решения (сек) для различных конфигураций}')
    latex_lines.append(r'\label{tab:solve_times}')
    latex_lines.append(r'\begin{tabular}{l' + 'r' * len(processes) + '}')
    latex_lines.append(r'\hline')
    
    header = ['Grid'] + [f'$p={p}$' for p in processes]
    latex_lines.append(' & '.join(header) + r' \\')
    latex_lines.append(r'\hline')
    
    for grid in grids:
        row = [grid]
        for p in processes:
            key = (grid, p)
            if key in data and 'solve_time' in data[key]:
                val = data[key]['solve_time']
                if val < 0.01:
                    row.append(f'{val:.5f}')
                elif val < 1:
                    row.append(f'{val:.4f}')
                elif val < 10:
                    row.append(f'{val:.3f}')
                else:
                    row.append(f'{val:.2f}')
            else:
                row.append('---')
        latex_lines.append(' & '.join(row) + r' \\')
    
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Широкая таблица сохранена в {output_file}")


def print_example_data(csv_file='median_times_long.csv'):
    """Выводит пример данных для проверки."""
    data = read_timing_data(csv_file)
    
    print("\nПример данных (40x40, p=1,2):")
    print("-" * 80)
    print(f"{'Metric':<25} {'p=1':>15} {'p=2':>15}")
    print("-" * 80)
    
    metrics = ['solve_time', 'matvec_kernel_time', 'vector_kernels_time', 
               'halo_mpi_comm_time', 'h2d_memcpy_time', 'd2h_memcpy_time']
    
    for metric in metrics:
        p1_val = data.get(('40x40', 1), {}).get(metric, 0)
        p2_val = data.get(('40x40', 2), {}).get(metric, 0)
        print(f"{metric:<25} {p1_val:>15.6f} {p2_val:>15.6f}")


if __name__ == '__main__':
    csv_file = 'median_times_long.csv'
    
    print("=" * 60)
    print("Создание детальных таблиц с разбивкой времени")
    print("=" * 60)
    
    # 1. Детальная таблица
    print("\n1. Создание детальной таблицы...")
    create_timing_table(csv_file, 'timing_breakdown_table.tex')
    
    # 2. Сводная таблица с процентами
    print("\n2. Создание сводной таблицы с процентами...")
    create_summary_table(csv_file, 'timing_summary_table.tex')
    
    # 3. Широкая таблица
    print("\n3. Создание широкой таблицы...")
    create_wide_format_table(csv_file, 'timing_wide_table.tex')
    
    # 4. Показываем пример данных
    print_example_data(csv_file)
    
    print("\n" + "=" * 60)
    print("Все таблицы созданы!")
    print("=" * 60)
