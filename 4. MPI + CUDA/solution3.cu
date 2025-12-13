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

__device__ __forceinline__ double warp_reduce_sum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__
void dot_partials_kernel(const double* __restrict__ a,
                         const double* __restrict__ b,
                         double* __restrict__ out_partials,
                         int n) {
    double sum = 0.0;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += blockDim.x * gridDim.x) {
        sum += a[k] * b[k];
    }

    sum = warp_reduce_sum(sum);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5; // 0..(blockDim/32-1)
    const int warps_per_block = (blockDim.x >> 5);
    if (lane == 0) {
        out_partials[blockIdx.x * warps_per_block + warp] = sum;
    }
}

// Один warp суммирует partials и пишет out[0]
__global__
void reduce_final_warp_kernel(const double* __restrict__ in,
                              double* __restrict__ out,
                              int n) {
    double sum = 0.0;
    for (int k = threadIdx.x; k < n; k += 32) {
        sum += in[k];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) out[0] = sum;
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

// CUDA-ядро обновления u и r
__global__
void update_ur_kernel(double* __restrict__ u,
                      double* __restrict__ r,
                      const double* __restrict__ p,
                      const double* __restrict__ Ap,
                      double alpha,
                      int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    u[k] += alpha * p[k];
    r[k] -= alpha * Ap[k];
}

// CUDA-ядро предобуславливания: z = r / diag(A)
__global__
void precond_kernel(double* __restrict__ z,
                    const double* __restrict__ r,
                    const double* __restrict__ A_diag,
                    int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double d = A_diag[k];
    z[k] = (d > 0.0 ? r[k] / d : r[k]);
}

// CUDA-ядро обновления направления
__global__
void update_p_kernel(double* __restrict__ p,
                     const double* __restrict__ z,
                     double beta,
                     int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    p[k] = z[k] + beta * p[k];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dev_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&dev_count);
    if (cerr != cudaSuccess || dev_count == 0) {
        if (rank == 0) {
            std::cerr << "Error: no CUDA devices visible.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dev_id = rank % dev_count;
    cudaError_t serr = cudaSetDevice(dev_id);
    if (serr != cudaSuccess) {
        if (rank == 0) {
            std::cerr << "Error: cudaSetDevice(" << dev_id << ") failed.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (rank == 0) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev_id);
        std::cout << "Using CUDA device " << dev_id << " (" << prop.name << ")\n";
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
    // int i_end = i_start + local_M - 1; // не используется

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

    // Аккумуляторы для измерения времени (MPI_Wtime)
    double time_init = 0.0;
    double time_cpu_build = 0.0;
    double time_gpu_alloc = 0.0;
    double time_gpu_upload = 0.0;
    double time_mpi_collective = 0.0;
    double time_gather_io = 0.0;
    double time_finalize = 0.0;

    double time_comm_halo = 0.0;
    double time_h2d = 0.0;
    double time_d2h = 0.0;
    double time_kernel_matvec = 0.0;
    double time_kernel_vec = 0.0;

    // Указатели на данные на устройстве (GPU)
    double *d_ax = nullptr, *d_by = nullptr, *d_A_diag = nullptr;
    double *d_u = nullptr, *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    double *d_recv_left = nullptr, *d_recv_right = nullptr;
    double *d_dot_partials = nullptr;
    double *d_dot_tmp = nullptr;

    double t_init0 = MPI_Wtime();

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

            const double xl = xc - 0.5 * hx;
            const double yb = yc - 0.5 * hy;

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

    // CPU precompute done
    double t_cpu1 = MPI_Wtime();
    time_cpu_build += (t_cpu1 - t_init0);

    // Копирование коэффициентов и правой части на GPU (подготовка CUDA)
    const std::size_t ax_bytes   = ax.size() * sizeof(double);
    const std::size_t by_bytes   = by.size() * sizeof(double);
    const std::size_t diag_bytes = A_diag.size() * sizeof(double);
    const std::size_t vec_bytes  = static_cast<std::size_t>(local_NN) * sizeof(double);

    double t_alloc0 = MPI_Wtime();
    cudaMalloc(&d_ax, ax_bytes);
    cudaMalloc(&d_by, by_bytes);
    cudaMalloc(&d_A_diag, diag_bytes);
    double t_alloc1 = MPI_Wtime();
    time_gpu_alloc += (t_alloc1 - t_alloc0);

    double t_up0 = MPI_Wtime();
    cudaMemcpy(d_ax, ax.data(), ax_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_by, by.data(), by_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_diag, A_diag.data(), diag_bytes, cudaMemcpyHostToDevice);
    double t_up1 = MPI_Wtime();
    time_gpu_upload += (t_up1 - t_up0);
    time_h2d += (t_up1 - t_up0); // считаем как H->D копирование

    // Векторы PCG на устройстве
    t_alloc0 = MPI_Wtime();
    cudaMalloc(&d_u, vec_bytes);
    cudaMalloc(&d_r, vec_bytes);
    cudaMalloc(&d_z, vec_bytes);
    cudaMalloc(&d_p, vec_bytes);
    cudaMalloc(&d_Ap, vec_bytes);
    t_alloc1 = MPI_Wtime();
    time_gpu_alloc += (t_alloc1 - t_alloc0);

    t_alloc0 = MPI_Wtime();
    cudaMemset(d_u, 0, vec_bytes);
    t_alloc1 = MPI_Wtime();
    time_gpu_alloc += (t_alloc1 - t_alloc0);

    t_up0 = MPI_Wtime();
    cudaMemcpy(d_r, F.data(), vec_bytes, cudaMemcpyHostToDevice); // r0 = b (u0=0)
    t_up1 = MPI_Wtime();
    time_gpu_upload += (t_up1 - t_up0);
    time_h2d += (t_up1 - t_up0);

    t_alloc0 = MPI_Wtime();
    cudaMemset(d_z, 0, vec_bytes);
    cudaMemset(d_p, 0, vec_bytes);
    cudaMemset(d_Ap, 0, vec_bytes);
    t_alloc1 = MPI_Wtime();
    time_gpu_alloc += (t_alloc1 - t_alloc0);

    // Halo-буферы на устройстве
    t_alloc0 = MPI_Wtime();
    cudaMalloc(&d_recv_left,  N_in * sizeof(double));
    cudaMalloc(&d_recv_right, N_in * sizeof(double));
    t_alloc1 = MPI_Wtime();
    time_gpu_alloc += (t_alloc1 - t_alloc0);

    // Буферы для скалярных произведений на GPU (без atomics/shared)
    t_alloc0 = MPI_Wtime();
    cudaMalloc(&d_dot_partials, 8192 * sizeof(double));
    cudaMalloc(&d_dot_tmp, sizeof(double));
    t_alloc1 = MPI_Wtime();
    time_gpu_alloc += (t_alloc1 - t_alloc0);

    // Определяем соседей
    int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    std::vector<double> recv_left(N_in, 0.0);
    std::vector<double> recv_right(N_in, 0.0);
    std::vector<double> send_left(N_in, 0.0);
    std::vector<double> send_right(N_in, 0.0);

    // Функция обмена halo для вектора на GPU (d_v)
    auto exchange_halo_gpu = [&](const double* d_v) {
        double t_copy0, t_copy1;
        // Скопировать граничные строки с GPU на хост и отправить соседям
        if (left_rank != MPI_PROC_NULL) {
            t_copy0 = MPI_Wtime();
            cudaMemcpy(send_left.data(),
                       d_v + ID_local(1, 1, local_M),
                       N_in * sizeof(double), cudaMemcpyDeviceToHost);
            t_copy1 = MPI_Wtime();
            time_d2h += (t_copy1 - t_copy0);
            double tc0 = MPI_Wtime();
            MPI_Sendrecv(send_left.data(), N_in, MPI_DOUBLE, left_rank, 0,
                         recv_left.data(), N_in, MPI_DOUBLE, left_rank, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double tc1 = MPI_Wtime();
            time_comm_halo += (tc1 - tc0);
        }
        if (right_rank != MPI_PROC_NULL) {
            t_copy0 = MPI_Wtime();
            cudaMemcpy(send_right.data(),
                       d_v + ID_local(local_M, 1, local_M),
                       N_in * sizeof(double), cudaMemcpyDeviceToHost);
            t_copy1 = MPI_Wtime();
            time_d2h += (t_copy1 - t_copy0);
            double tc0 = MPI_Wtime();
            MPI_Sendrecv(send_right.data(), N_in, MPI_DOUBLE, right_rank, 1,
                         recv_right.data(), N_in, MPI_DOUBLE, right_rank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double tc1 = MPI_Wtime();
            time_comm_halo += (tc1 - tc0);
        }

        // Копируем полученные halo-строки на устройство
        if (left_rank != MPI_PROC_NULL) {
            t_copy0 = MPI_Wtime();
            cudaMemcpy(d_recv_left, recv_left.data(),
                       N_in * sizeof(double), cudaMemcpyHostToDevice);
            t_copy1 = MPI_Wtime();
            time_h2d += (t_copy1 - t_copy0);
        }
        if (right_rank != MPI_PROC_NULL) {
            t_copy0 = MPI_Wtime();
            cudaMemcpy(d_recv_right, recv_right.data(),
                       N_in * sizeof(double), cudaMemcpyHostToDevice);
            t_copy1 = MPI_Wtime();
            time_h2d += (t_copy1 - t_copy0);
        }
    };

    // Матрично-векторное умножение на GPU
    dim3 mv_block(16, 16);
    dim3 mv_grid(
        (N_in + mv_block.x - 1) / mv_block.x,
        (local_M + mv_block.y - 1) / mv_block.y
    );
    auto matvec_gpu = [&](const double* d_v, double* d_Av) {
        // halo-обмен через CPU-буферы и копирование на GPU
        exchange_halo_gpu(d_v);

        double tk0 = MPI_Wtime();
        matvec_kernel<<<mv_grid, mv_block>>>(
            d_v, d_Av,
            d_ax, d_by, d_A_diag,
            d_recv_left, d_recv_right,
            local_M, N_in, M_in, i_start,
            hx, hy
        );
        cudaDeviceSynchronize();
        double tk1 = MPI_Wtime();
        time_kernel_matvec += (tk1 - tk0);
    };

    // Скалярное произведение на GPU (локально на GPU -> 1 число на CPU -> MPI_Allreduce)
    auto dot_gpu = [&](const double* d_a, const double* d_b) {
        const int threads = 256;
        int blocks = (local_NN + threads - 1) / threads;
        blocks = std::max(1, std::min(1024, blocks));
        const int warps_per_block = threads / 32;
        const int n_partials = blocks * warps_per_block;

        double tk0 = MPI_Wtime();
        dot_partials_kernel<<<blocks, threads>>>(d_a, d_b, d_dot_partials, local_NN);
        // финальная редукция partials -> 1 double (один warp)
        reduce_final_warp_kernel<<<1, 32>>>(d_dot_partials, d_dot_tmp, n_partials);
        cudaDeviceSynchronize();
        double tk1 = MPI_Wtime();
        time_kernel_vec += (tk1 - tk0); // считаем dot как часть "vector kernels"

        double local_s = 0.0;
        double t0 = MPI_Wtime();
        cudaMemcpy(&local_s, d_dot_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        double t1 = MPI_Wtime();
        time_d2h += (t1 - t0);

        double global_s = 0.0;
        double tc0 = MPI_Wtime();
        MPI_Allreduce(&local_s, &global_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tc1 = MPI_Wtime();
        time_mpi_collective += (tc1 - tc0);
        return global_s;
    };

    // Буфер на host для финального решения (копируем один раз после PCG)
    std::vector<double> u(local_NN, 0.0);

    const int maxit = 100000;

    // ||b|| считаем на CPU один раз (само решение и итерации — на GPU)
    double local_b2 = 0.0;
    for (int k = 0; k < local_NN; ++k) local_b2 += F[k] * F[k];
    double global_b2 = 0.0;
    {
        double tc0 = MPI_Wtime();
        MPI_Allreduce(&local_b2, &global_b2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tc1 = MPI_Wtime();
        time_mpi_collective += (tc1 - tc0);
    }
    const double bnorm = std::sqrt(std::max(1e-30, global_b2));

    const double tol = 1e-8;
    const double atol = tol * bnorm;

    // PCG на GPU: вектора u/r/z/p/Ap постоянно живут на устройстве.
    const int threads = 256;
    const int blocks = (local_NN + threads - 1) / threads;

    double t0 = MPI_Wtime();

    // Инициализация PCG: z = M^{-1} r, p = z
    {
        double tk0 = MPI_Wtime();
        precond_kernel<<<blocks, threads>>>(d_z, d_r, d_A_diag, local_NN);
        cudaDeviceSynchronize();
        double tk1 = MPI_Wtime();
        time_kernel_vec += (tk1 - tk0);

        cudaMemcpy(d_p, d_z, vec_bytes, cudaMemcpyDeviceToDevice);
    }

    double rz_old = dot_gpu(d_r, d_z);

    int it;
    for (it = 0; it < maxit; ++it) {
        // Ap = A*p (полностью на GPU)
        matvec_gpu(d_p, d_Ap);

        const double pAp = dot_gpu(d_p, d_Ap);
        if (pAp <= 0.0) {
            if (rank == 0) std::cerr << "Breakdown in PCG\n";
            break;
        }

        const double alpha = rz_old / pAp;

        double tk0 = MPI_Wtime();
        update_ur_kernel<<<blocks, threads>>>(d_u, d_r, d_p, d_Ap, alpha, local_NN);
        cudaDeviceSynchronize();
        double tk1 = MPI_Wtime();
        time_kernel_vec += (tk1 - tk0);

        const double rnorm = std::sqrt(dot_gpu(d_r, d_r));
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

        // Предобуславливание на GPU: z = r / A_diag
        tk0 = MPI_Wtime();
        precond_kernel<<<blocks, threads>>>(d_z, d_r, d_A_diag, local_NN);
        cudaDeviceSynchronize();
        tk1 = MPI_Wtime();
        time_kernel_vec += (tk1 - tk0);

        const double rz_new = dot_gpu(d_r, d_z);
        const double beta = rz_new / rz_old;

        // Обновление направления p на GPU
        tk0 = MPI_Wtime();
        update_p_kernel<<<blocks, threads>>>(d_p, d_z, beta, local_NN);
        cudaDeviceSynchronize();
        tk1 = MPI_Wtime();
        time_kernel_vec += (tk1 - tk0);

        rz_old = rz_new;
    }

    // После завершения PCG копируем решение u с GPU на CPU один раз (для MPI_Gatherv и записи)
    double tc0 = MPI_Wtime();
    cudaMemcpy(u.data(), d_u, vec_bytes, cudaMemcpyDeviceToHost);
    double tc1 = MPI_Wtime();
    time_d2h += (tc1 - tc0);
    double t1 = MPI_Wtime();

    double t_init1 = t0;
    time_init += (t_init1 - t_init0);

    double solve_time = t1 - t0;
    double max_solve_time = 0.0;
    {
        double tc0 = MPI_Wtime();
        MPI_Reduce(&solve_time, &max_solve_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double tc1 = MPI_Wtime();
        time_mpi_collective += (tc1 - tc0);
    }

    if (rank == 0) {
        std::cout << "solve time (max over ranks): " << max_solve_time << " s\n";
        std::cout << "iterations: " << it << "\n";
    }

    {
        double tg0 = MPI_Wtime();
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_count = local_NN;
        {
            double tc0 = MPI_Wtime();
            MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            double tc1 = MPI_Wtime();
            time_mpi_collective += (tc1 - tc0);
        }
        
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

        {
            double tc0 = MPI_Wtime();
            MPI_Gatherv(u.data(), local_NN, MPI_DOUBLE,
                        u_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            double tc1 = MPI_Wtime();
            time_mpi_collective += (tc1 - tc0);
        }

        if (rank == 0) {
            std::ofstream out("solution.csv");
            out << "x,y,u\n";
            
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
        double tg1 = MPI_Wtime();
        time_gather_io += (tg1 - tg0);
    }

    // Освобождение ресурсов GPU
    double tf0 = MPI_Wtime();
    cudaFree(d_ax);
    cudaFree(d_by);
    cudaFree(d_A_diag);
    cudaFree(d_u);
    cudaFree(d_r);
    cudaFree(d_z);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_recv_left);
    cudaFree(d_recv_right);
    cudaFree(d_dot_partials);
    cudaFree(d_dot_tmp);
    double tf1 = MPI_Wtime();
    time_finalize += (tf1 - tf0);

    // Сводная печать таймингов (max over ranks)
    {
        double max_init = 0.0, max_cpu = 0.0, max_alloc = 0.0, max_upload = 0.0;
        double max_comm = 0.0, max_h2d = 0.0, max_d2h = 0.0;
        double max_k_matvec = 0.0, max_k_vec = 0.0;
        double max_collect = 0.0, max_gio = 0.0, max_fin = 0.0;

        MPI_Reduce(&time_init, &max_init, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_cpu_build, &max_cpu, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_gpu_alloc, &max_alloc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_gpu_upload, &max_upload, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_comm_halo, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_h2d, &max_h2d, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_d2h, &max_d2h, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_kernel_matvec, &max_k_matvec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_kernel_vec, &max_k_vec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_mpi_collective, &max_collect, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_gather_io, &max_gio, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_finalize, &max_fin, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "---- FULL timing breakdown (max over ranks) ----\n";
            std::cout << "init (up to solver start): " << max_init << " s\n";
            std::cout << "  cpu build (ax/by/F/diag): " << max_cpu << " s\n";
            std::cout << "  gpu alloc/memset: " << max_alloc << " s\n";
            std::cout << "  gpu upload (coeff+init vec): " << max_upload << " s\n";
            std::cout << "solver total (max): " << max_solve_time << " s\n";
            std::cout << "  halo MPI comm time: " << max_comm << " s\n";
            std::cout << "  H->D memcpy time: " << max_h2d << " s\n";
            std::cout << "  D->H memcpy time: " << max_d2h << " s\n";
            std::cout << "  matvec kernel time: " << max_k_matvec << " s\n";
            std::cout << "  vector/dot kernels time: " << max_k_vec << " s\n";
            std::cout << "mpi collectives (Allreduce/Gather/Gatherv): " << max_collect << " s\n";
            std::cout << "gather+io: " << max_gio << " s\n";
            std::cout << "finalize (cudaFree+etc): " << max_fin << " s\n";
        }
    }

    MPI_Finalize();
    return 0;
}


