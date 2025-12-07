#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

static int M_in = 80, N_in = 80;
inline int ID(int i, int j) { return (i - 1) * N_in + (j - 1); }

// Локальный индекс для процесса (i_local относительно локального диапазона)
inline int ID_local(int i_local, int j, int local_M) { 
    return (i_local - 1) * N_in + (j - 1); 
}

struct Box { double x0, x1, y0, y1; };

// 10 варик: D = { x in (1,3), |y| < 0.5*sqrt(x^2-1) }
inline double y_cap(double x) {
    double s = x * x - 1.0;
    return (s > 0.0) ? 0.5 * std::sqrt(s) : 0.0;
}
inline bool in_D(double x, double y) {
    if (x <= 1.0 || x >= 3.0) return false;
    return std::fabs(y) < y_cap(x);
}

inline int IX_X(int i_face, int j) { return i_face * N_in + (j - 1); }
inline int IX_Y(int i, int j_face) { return (i - 1) * (N_in + 1) + j_face; }

// Локальные версии индексации
inline int IX_X_local(int i_face_local, int j, int local_M) { 
    return i_face_local * N_in + (j - 1); 
}
inline int IX_Y_local(int i_local, int j_face, int local_M) { 
    return (i_local - 1) * (N_in + 1) + j_face; 
}

// CUDA-ядро матрично-векторного умножения с учетом halo-строк
__global__
void matvec_kernel(const double* __restrict__ v,
                   double* __restrict__ Av,
                   const double* __restrict__ ax,
                   const double* __restrict__ by,
                   const double* __restrict__ A_diag,
                   const double* __restrict__ recv_left,
                   const double* __restrict__ recv_right,
                   int local_M, int N_in, int M_in, int i_start,
                   double hx, double hy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;      // 1..N_in
    int i_local = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1..local_M
    if (j > N_in || i_local > local_M) return;

    int i_global = i_start + i_local - 1;
    int k = (i_local - 1) * N_in + (j - 1);

    // Индексы для коэффициентов
    int idx_aL = (i_local - 1) * N_in + (j - 1);     // (i_local-1, j)
    int idx_aR = i_local * N_in + (j - 1);           // (i_local,   j)
    int idx_bD = (i_local - 1) * (N_in + 1) + (j - 1); // (i_local, j-1)
    int idx_bU = (i_local - 1) * (N_in + 1) + j;       // (i_local, j)

    double aL = ax[idx_aL];
    double aR = ax[idx_aR];
    double bD = by[idx_bD];
    double bU = by[idx_bU];

    double s = A_diag[k] * v[k];

    // Левый сосед по i
    if (i_local > 1) {
        int kL = (i_local - 2) * N_in + (j - 1);
        s += (-aL / (hx * hx)) * v[kL];
    } else if (i_global > 1) {
        s += (-aL / (hx * hx)) * recv_left[j - 1];
    }

    // Правый сосед по i
    if (i_local < local_M) {
        int kR = i_local * N_in + (j - 1);
        s += (-aR / (hx * hx)) * v[kR];
    } else if (i_global < M_in) {
        s += (-aR / (hx * hx)) * recv_right[j - 1];
    }

    // Нижний сосед по j (локальный)
    if (j > 1) {
        int kD = (i_local - 1) * N_in + (j - 2);
        s += (-bD / (hy * hy)) * v[kD];
    }

    // Верхний сосед по j (локальный)
    if (j < N_in) {
        int kU = (i_local - 1) * N_in + j;
        s += (-bU / (hy * hy)) * v[kU];
    }

    Av[k] = s;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Инициализация CUDA-устройств: один GPU на процесс, если доступен
    bool use_gpu = false;
    int dev_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&dev_count);
    if (cerr != cudaSuccess || dev_count == 0) {
        if (rank == 0) {
            std::cerr << "Warning: no CUDA devices visible, running as pure MPI.\n";
        }
    } else {
        int dev_id = rank % dev_count;
        cudaError_t serr = cudaSetDevice(dev_id);
        if (serr != cudaSuccess) {
            if (rank == 0) {
                std::cerr << "Warning: cudaSetDevice failed, running as pure MPI.\n";
            }
        } else {
            use_gpu = true;
            if (rank == 0) {
                cudaDeviceProp prop{};
                cudaGetDeviceProperties(&prop, dev_id);
                std::cout << "Using CUDA device " << dev_id << " (" << prop.name << ")\n";
            }
        }
    }

    if (argc >= 3) {
        M_in = std::atoi(argv[1]);
        N_in = std::atoi(argv[2]);
    }

    // Декомпозиция по строкам (i-направление)
    // Каждый процесс получает диапазон строк [i_start, i_end]
    int base_rows = M_in / size;
    int remainder = M_in % size;
    
    int i_start = rank * base_rows + std::min(rank, remainder) + 1;
    int local_M = base_rows + (rank < remainder ? 1 : 0);
    int i_end = i_start + local_M - 1;

    if (rank == 0) {
        std::cout << "Grid: " << M_in << " x " << N_in << "\n";
        std::cout << "MPI processes: " << size << "\n";
    }

    Box R;
    R.x0 = 1.0; R.x1 = 3.0;
    const double Ymax = 0.5 * std::sqrt(9.0 - 1.0);
    R.y0 = -Ymax; R.y1 = Ymax;

    const double hx = (R.x1 - R.x0) / (M_in + 1.0);
    const double hy = (R.y1 - R.y0) / (N_in + 1.0);
    const double h  = std::max(hx, hy);
    const double eps = h * h;

    const int local_NN = local_M * N_in;

    // Указатели на данные на устройстве (GPU)
    double *d_ax = nullptr, *d_by = nullptr, *d_A_diag = nullptr, *d_F = nullptr;
    double *d_u = nullptr, *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    double *d_recv_left = nullptr, *d_recv_right = nullptr;

    // Локальные коэффициенты ax: для граней от i_start-1 до i_end (local_M + 1 граней)
    std::vector<double> ax((local_M + 1) * N_in, 0.0);
    
    for (int i_local = 0; i_local <= local_M; ++i_local) {
        int i_global = i_start - 1 + i_local;  // глобальный индекс грани
        for (int j = 1; j <= N_in; ++j) {
            const double x_face = R.x0 + (i_global + 0.5) * hx;
            const double y_low  = R.y0 + (j - 0.5) * hy;
            const double y_high = y_low + hy;
            const double yc = y_cap(x_face);

            const double inter_low  = std::max(y_low,  -yc);
            const double inter_high = std::min(y_high,  yc);
            const double L = std::max(0.0, inter_high - inter_low);
            const double frac = L / hy;

            ax[IX_X_local(i_local, j, local_M)] = frac * 1.0 + (1.0 - frac) * (1.0 / eps);
        }
    }

    // Локальные коэффициенты by: для строк от i_start до i_end
    std::vector<double> by(local_M * (N_in + 1), 0.0);
    
    for (int i_local = 1; i_local <= local_M; ++i_local) {
        int i_global = i_start + i_local - 1;
        for (int j = 0; j <= N_in; ++j) {
            const double y_face = R.y0 + (j + 0.5) * hy;
            const double x_low  = R.x0 + (i_global - 0.5) * hx;
            const double x_high = x_low + hx;

            const double x_min_in = std::max(1.0, std::sqrt(1.0 + 4.0 * y_face * y_face));
            const double inter_low  = std::max(x_low,  x_min_in);
            const double inter_high = std::min(x_high, R.x1);
            const double L = std::max(0.0, inter_high - inter_low);
            const double frac = L / hx;

            by[IX_Y_local(i_local, j, local_M)] = frac * 1.0 + (1.0 - frac) * (1.0 / eps);
        }
    }

    // Локальная правая часть F
    std::vector<double> F(local_NN, 0.0);
    const int SS = 4;
    
    for (int i_local = 1; i_local <= local_M; ++i_local) {
        int i_global = i_start + i_local - 1;
        for (int j = 1; j <= N_in; ++j) {
            const double xc = R.x0 + i_global * hx;
            const double yc = R.y0 + j * hy;

            const double xl = xc - 0.5 * hx, xr = xc + 0.5 * hx;
            const double yb = yc - 0.5 * hy, yt = yc + 0.5 * hy;

            int inside = 0;
            for (int sx = 0; sx < SS; ++sx) {
                for (int sy = 0; sy < SS; ++sy) {
                    const double xs = xl + (sx + 0.5) * (hx / SS);
                    const double ys = yb + (sy + 0.5) * (hy / SS);
                    if (in_D(xs, ys)) ++inside;
                }
            }
            const double frac_area = static_cast<double>(inside) / (SS * SS);
            F[ID_local(i_local, j, local_M)] = frac_area;
        }
    }

    // Локальная диагональ A_diag
    std::vector<double> A_diag(local_NN, 0.0);
    
    for (int i_local = 1; i_local <= local_M; ++i_local) {
        for (int j = 1; j <= N_in; ++j) {
            const double aL = ax[IX_X_local(i_local - 1, j, local_M)];
            const double aR = ax[IX_X_local(i_local, j, local_M)];
            const double bD = by[IX_Y_local(i_local, j - 1, local_M)];
            const double bU = by[IX_Y_local(i_local, j, local_M)];
            A_diag[ID_local(i_local, j, local_M)] =
                (aL + aR) / (hx * hx) + (bD + bU) / (hy * hy);
        }
    }

    // Копирование коэффициентов и правой части на GPU (шаги 3–4 подготовки CUDA)
    if (use_gpu) {
        std::size_t ax_bytes = ax.size() * sizeof(double);
        std::size_t by_bytes = by.size() * sizeof(double);
        std::size_t diag_bytes = A_diag.size() * sizeof(double);
        std::size_t f_bytes = F.size() * sizeof(double);
        std::size_t vec_bytes = static_cast<std::size_t>(local_NN) * sizeof(double);

        cudaMalloc(&d_ax, ax_bytes);
        cudaMalloc(&d_by, by_bytes);
        cudaMalloc(&d_A_diag, diag_bytes);
        cudaMalloc(&d_F, f_bytes);

        cudaMemcpy(d_ax, ax.data(), ax_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_by, by.data(), by_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_diag, A_diag.data(), diag_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_F, F.data(), f_bytes, cudaMemcpyHostToDevice);

        // Векторы PCG на устройстве (пока используем d_p/d_Ap как рабочие буферы для matvec)
        cudaMalloc(&d_u, vec_bytes);
        cudaMalloc(&d_r, vec_bytes);
        cudaMalloc(&d_z, vec_bytes);
        cudaMalloc(&d_p, vec_bytes);
        cudaMalloc(&d_Ap, vec_bytes);

        cudaMemset(d_u, 0, vec_bytes);
        cudaMemcpy(d_r, F.data(), vec_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_z, 0, vec_bytes);
        cudaMemset(d_p, 0, vec_bytes);
        cudaMemset(d_Ap, 0, vec_bytes);

        // Halo-буферы на устройстве
        cudaMalloc(&d_recv_left,  N_in * sizeof(double));
        cudaMalloc(&d_recv_right, N_in * sizeof(double));
    }

    // Определяем соседей
    int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Буферы для halo-обмена (по одной строке = N_in элементов)
    std::vector<double> recv_left(N_in, 0.0);   // от левого соседа (строка i_start-1)
    std::vector<double> recv_right(N_in, 0.0);  // от правого соседа (строка i_end+1)
    std::vector<double> send_left(N_in, 0.0);   // буферы для отправки на CPU
    std::vector<double> send_right(N_in, 0.0);

    // Функция обмена halo
    auto exchange_halo_cpu = [&](const std::vector<double>& v) {
        // Отправляем первую строку левому соседу, получаем от левого соседа
        // Отправляем последнюю строку правому соседу, получаем от правого соседа
        MPI_Request reqs[4];
        int req_count = 0;

        // Отправка левому соседу (наша первая строка)
        if (left_rank != MPI_PROC_NULL) {
            MPI_Isend(&v[ID_local(1, 1, local_M)], N_in, MPI_DOUBLE, 
                      left_rank, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Отправка правому соседу (наша последняя строка)
        if (right_rank != MPI_PROC_NULL) {
            MPI_Isend(&v[ID_local(local_M, 1, local_M)], N_in, MPI_DOUBLE, 
                      right_rank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Прием от левого соседа
        if (left_rank != MPI_PROC_NULL) {
            MPI_Irecv(recv_left.data(), N_in, MPI_DOUBLE, 
                      left_rank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Прием от правого соседа
        if (right_rank != MPI_PROC_NULL) {
            MPI_Irecv(recv_right.data(), N_in, MPI_DOUBLE, 
                      right_rank, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    };

    // Функция обмена halo для вектора на GPU (d_v)
    auto exchange_halo_gpu = [&](double* d_v) {
        if (!use_gpu) return;

        MPI_Request reqs[4];
        int req_count = 0;

        // Скопировать граничные строки с GPU на хост и отправить соседям
        if (left_rank != MPI_PROC_NULL) {
            cudaMemcpy(send_left.data(),
                       d_v + ID_local(1, 1, local_M),
                       N_in * sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Isend(send_left.data(), N_in, MPI_DOUBLE,
                      left_rank, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (right_rank != MPI_PROC_NULL) {
            cudaMemcpy(send_right.data(),
                       d_v + ID_local(local_M, 1, local_M),
                       N_in * sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Isend(send_right.data(), N_in, MPI_DOUBLE,
                      right_rank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // Прием граничных строк от соседей на хост
        if (left_rank != MPI_PROC_NULL) {
            MPI_Irecv(recv_left.data(), N_in, MPI_DOUBLE,
                      left_rank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (right_rank != MPI_PROC_NULL) {
            MPI_Irecv(recv_right.data(), N_in, MPI_DOUBLE,
                      right_rank, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        // Копируем полученные halo-строки на устройство
        if (left_rank != MPI_PROC_NULL) {
            cudaMemcpy(d_recv_left, recv_left.data(),
                       N_in * sizeof(double), cudaMemcpyHostToDevice);
        }
        if (right_rank != MPI_PROC_NULL) {
            cudaMemcpy(d_recv_right, recv_right.data(),
                       N_in * sizeof(double), cudaMemcpyHostToDevice);
        }
    };

    // Коэффициенты для halo-обмена (ax на границах) – пока не используются в CUDA-ветке,
    // но могут пригодиться для дальнейшей оптимизации
    double aL_left_boundary = ax[IX_X_local(0, 1, local_M)];  // грань слева от первой строки
    double aR_right_boundary = ax[IX_X_local(local_M, 1, local_M)]; // грань справа от последней

    // Матрично-векторное умножение с обменом halo
    auto matvec = [&](const std::vector<double>& v, std::vector<double>& Av) {
        if (!use_gpu) {
            // CPU-вариант (как в исходном MPI-коде)
            exchange_halo_cpu(v);
            
            for (int i_local = 1; i_local <= local_M; ++i_local) {
                int i_global = i_start + i_local - 1;
                for (int j = 1; j <= N_in; ++j) {
                    const int k = ID_local(i_local, j, local_M);
                    const double aL = ax[IX_X_local(i_local - 1, j, local_M)];
                    const double aR = ax[IX_X_local(i_local, j, local_M)];
                    const double bD =
                        by[IX_Y_local(i_local, j - 1, local_M)];
                    const double bU =
                        by[IX_Y_local(i_local, j, local_M)];
                    
                    double s = A_diag[k] * v[k];
                    
                    // Левый сосед по i
                    if (i_local > 1) {
                        s += (-aL / (hx * hx)) *
                             v[ID_local(i_local - 1, j, local_M)];
                    } else if (i_global > 1) {
                        // Берем из halo от левого процесса
                        s += (-aL / (hx * hx)) * recv_left[j - 1];
                    }
                    
                    // Правый сосед по i
                    if (i_local < local_M) {
                        s += (-aR / (hx * hx)) *
                             v[ID_local(i_local + 1, j, local_M)];
                    } else if (i_global < M_in) {
                        // Берем из halo от правого процесса
                        s += (-aR / (hx * hx)) * recv_right[j - 1];
                    }
                    
                    // Нижний сосед по j (локальный)
                    if (j > 1) {
                        s += (-bD / (hy * hy)) *
                             v[ID_local(i_local, j - 1, local_M)];
                    }
                    
                    // Верхний сосед по j (локальный)
                    if (j < N_in) {
                        s += (-bU / (hy * hy)) *
                             v[ID_local(i_local, j + 1, local_M)];
                    }
                    
                    Av[k] = s;
                }
            }
        } else {
            // GPU-вариант: копируем вектор v на устройство, выполняем halo-обмен и ядро
            std::size_t vec_bytes = static_cast<std::size_t>(local_NN) * sizeof(double);
            cudaMemcpy(d_p, v.data(), vec_bytes, cudaMemcpyHostToDevice);

            // halo-обмен через CPU-буферы и копирование на GPU
            exchange_halo_gpu(d_p);

            dim3 block(16, 16);
            dim3 grid(
                (N_in + block.x - 1) / block.x,
                (local_M + block.y - 1) / block.y
            );

            matvec_kernel<<<grid, block>>>(
                d_p, d_Ap,
                d_ax, d_by, d_A_diag,
                d_recv_left, d_recv_right,
                local_M, N_in, M_in, i_start,
                hx, hy
            );
            cudaDeviceSynchronize();

            // Копируем результат обратно на хостовый вектор Av
            cudaMemcpy(Av.data(), d_Ap, vec_bytes, cudaMemcpyDeviceToHost);
        }
    };

    // Скалярное произведение с MPI_Allreduce
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double local_s = 0.0;
        for (int k = 0; k < local_NN; ++k) {
            local_s += a[k] * b[k];
        }
        double global_s = 0.0;
        MPI_Allreduce(&local_s, &global_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global_s;
    };

    // Векторы PCG
    std::vector<double> u(local_NN, 0.0);
    std::vector<double> r = F;
    std::vector<double> z(local_NN, 0.0);
    std::vector<double> p(local_NN, 0.0);
    std::vector<double> Ap(local_NN, 0.0);

    // Начальное предобуславливание
    for (int k = 0; k < local_NN; ++k) {
        z[k] = (A_diag[k] > 0.0 ? r[k] / A_diag[k] : r[k]);
    }
    p = z;

    const int maxit = 100000;
    const double bnorm = std::sqrt(std::max(1e-30, dot(F, F)));
    const double tol = 1e-8;
    const double atol = tol * bnorm;

    double rz_old = dot(r, z);

    double t0 = MPI_Wtime();
    int it;
    for (it = 0; it < maxit; ++it) {
        matvec(p, Ap);
        const double pAp = dot(p, Ap);
        if (pAp <= 0.0) {
            if (rank == 0) std::cerr << "Breakdown in PCG\n";
            break;
        }

        const double alpha = rz_old / pAp;

        for (int k = 0; k < local_NN; ++k) {
            u[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }

        const double rnorm = std::sqrt(dot(r, r));
        if (rank == 0 && it % 50 == 0) {
            std::cout << "iter=" << std::setw(6) << it
                      << "  |r|/|b|=" << std::setprecision(10) << (rnorm / bnorm)
                      << "  |r|=" << rnorm << "\n";
        }
        if (rnorm <= atol) {
            if (rank == 0) {
                std::cout << "final |r|/|b|=" << (rnorm / bnorm) << "\n";
            }
            break;
        }

        for (int k = 0; k < local_NN; ++k) {
            z[k] = (A_diag[k] > 0.0 ? r[k] / A_diag[k] : r[k]);
        }

        const double rz_new = dot(r, z);
        const double beta = rz_new / rz_old;

        for (int k = 0; k < local_NN; ++k) {
            p[k] = z[k] + beta * p[k];
        }

        rz_old = rz_new;
    }
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "solve time: " << (t1 - t0) << " s\n";
        std::cout << "iterations: " << it << "\n";
    }

    // Сбор решения на процессе 0 и запись в файл
    {
        // Собираем размеры и смещения
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_count = local_NN;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int p = 1; p < size; ++p) {
                displs[p] = displs[p-1] + recvcounts[p-1];
            }
        }

        std::vector<double> u_global;
        if (rank == 0) {
            u_global.resize(M_in * N_in);
        }

        MPI_Gatherv(u.data(), local_NN, MPI_DOUBLE,
                    u_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::ofstream out("solution.csv");
            out << "x,y,u\n";
            
            // Восстанавливаем порядок: процессы расположены последовательно по i
            int global_idx = 0;
            for (int p = 0; p < size; ++p) {
                int p_base_rows = M_in / size;
                int p_remainder = M_in % size;
                int p_i_start = p * p_base_rows + std::min(p, p_remainder) + 1;
                int p_local_M = p_base_rows + (p < p_remainder ? 1 : 0);
                
                for (int i_local = 1; i_local <= p_local_M; ++i_local) {
                    int i_global = p_i_start + i_local - 1;
                    for (int j = 1; j <= N_in; ++j) {
                        const double x = R.x0 + i_global * hx;
                        const double y = R.y0 + j * hy;
                        out << std::fixed << std::setprecision(10) 
                            << x << "," << y << "," << u_global[global_idx++] << "\n";
                    }
                }
            }
            out.close();
        }
    }

    MPI_Finalize();
    return 0;
}


