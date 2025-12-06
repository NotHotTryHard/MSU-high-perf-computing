#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

static int M_in = 80, N_in = 80;
inline int ID(int i, int j) { return (i - 1) * N_in + (j - 1); }

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

int main(int argc, char** argv) {
    if (argc >= 3) {
        M_in = std::atoi(argv[1]);
        N_in = std::atoi(argv[2]);
    }

    Box R;
    R.x0 = 1.0; R.x1 = 3.0;
    const double Ymax = 0.5 * std::sqrt(9.0 - 1.0);
    R.y0 = -Ymax; R.y1 =  Ymax;

    const double hx = (R.x1 - R.x0) / (M_in + 1.0);
    const double hy = (R.y1 - R.y0) / (N_in + 1.0);
    const double h  = std::max(hx, hy);
    const double eps = h * h;

    const int NN = M_in * N_in;

    std::vector<double> ax((M_in + 1) * N_in, 0.0);
    std::vector<double> by(M_in * (N_in + 1), 0.0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= M_in; ++i) {
        for (int j = 1; j <= N_in; ++j) {
            const double x_face = R.x0 + (i + 0.5) * hx;
            const double y_low  = R.y0 + (j - 0.5) * hy;
            const double y_high = y_low + hy;
            const double yc = y_cap(x_face);

            const double inter_low  = std::max(y_low,  -yc);
            const double inter_high = std::min(y_high,  yc);
            const double L = std::max(0.0, inter_high - inter_low);
            const double frac = L / hy;

            ax[IX_X(i, j)] = frac * 1.0 + (1.0 - frac) * (1.0 / eps);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M_in; ++i) {
        for (int j = 0; j <= N_in; ++j) {
            const double y_face = R.y0 + (j + 0.5) * hy;
            const double x_low  = R.x0 + (i - 0.5) * hx;
            const double x_high = x_low + hx;

            const double x_min_in = std::max(1.0, std::sqrt(1.0 + 4.0 * y_face * y_face));
            const double inter_low  = std::max(x_low,  x_min_in);
            const double inter_high = std::min(x_high, R.x1);
            const double L = std::max(0.0, inter_high - inter_low);
            const double frac = L / hx;

            by[IX_Y(i, j)] = frac * 1.0 + (1.0 - frac) * (1.0 / eps);
        }
    }

    std::vector<double> F(NN, 0.0);
    const int SS = 4;
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M_in; ++i) {
        for (int j = 1; j <= N_in; ++j) {
            const double xc = R.x0 + i * hx;
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
            F[ID(i, j)] = frac_area;
        }
    }

    std::vector<double> A_diag(NN, 0.0);
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M_in; ++i) {
        for (int j = 1; j <= N_in; ++j) {
            const double aL = ax[IX_X(i - 1, j)];
            const double aR = ax[IX_X(i, j)];
            const double bD = by[IX_Y(i, j - 1)];
            const double bU = by[IX_Y(i, j)];
            A_diag[ID(i, j)] = (aL + aR) / (hx * hx) + (bD + bU) / (hy * hy);
        }
    }

    auto matvec = [&](const std::vector<double>& v, std::vector<double>& Av) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= M_in; ++i) {
            for (int j = 1; j <= N_in; ++j) {
                const int k = ID(i, j);
                const double aL = ax[IX_X(i - 1, j)];
                const double aR = ax[IX_X(i, j)];
                const double bD = by[IX_Y(i, j - 1)];
                const double bU = by[IX_Y(i, j)];
                double s = A_diag[k] * v[k];
                if (i > 1)     s += (-aL / (hx * hx)) * v[ID(i - 1, j)];
                if (i < M_in)  s += (-aR / (hx * hx)) * v[ID(i + 1, j)];
                if (j > 1)     s += (-bD / (hy * hy)) * v[ID(i, j - 1)];
                if (j < N_in)  s += (-bU / (hy * hy)) * v[ID(i, j + 1)];
                Av[k] = s;
            }
        }
    };

    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        #pragma omp parallel for reduction(+:s)
        for (int k = 0; k < NN; ++k) s += a[k] * b[k];
        return s;
    };

    std::vector<double> u(NN, 0.0), r = F, z(NN, 0.0), p(NN, 0.0), Ap(NN, 0.0);

    #pragma omp parallel for
    for (int k = 0; k < NN; ++k) z[k] = (A_diag[k] > 0.0 ? r[k] / A_diag[k] : r[k]);
    p = z;

    const int maxit = 100000;
    const double bnorm = std::sqrt(std::max(1e-30, dot(F, F)));
    const double tol = 1e-8;
    const double atol = tol * bnorm;

    double rz_old = dot(r, z);

    double t0 = omp_get_wtime();
    int it;
    for (it = 0; it < maxit; ++it) {
        matvec(p, Ap);
        const double pAp = dot(p, Ap);
        if (pAp <= 0.0) { std::cerr << "Breakdown in PCG\n"; break; }

        const double alpha = rz_old / pAp;

        #pragma omp parallel for
        for (int k = 0; k < NN; ++k) {
            u[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }

        const double rnorm = std::sqrt(dot(r, r));
        if (it % 50 == 0) {
            std::cout << "iter=" << std::setw(6) << it
                      << "  |r|/|b|=" << std::setprecision(10) << (rnorm / bnorm)
                      << "  |r|=" << rnorm << "\n";
        }
        if (rnorm <= atol) {
            std::cout << "final |r|/|b|=" << (rnorm / bnorm) << "\n";
            break;
        }

        #pragma omp parallel for
        for (int k = 0; k < NN; ++k) z[k] = (A_diag[k] > 0.0 ? r[k] / A_diag[k] : r[k]);

        const double rz_new = dot(r, z);
        const double beta = rz_new / rz_old;

        #pragma omp parallel for
        for (int k = 0; k < NN; ++k) p[k] = z[k] + beta * p[k];

        rz_old = rz_new;
    }
    double t1 = omp_get_wtime();

    std::cout << "solve time: " << (t1 - t0) << " s\n";
    std::cout << "iterations: " << it << "\n";

    {
        std::ofstream out("solution.csv");
        out << "x,y,u\n";
        for (int i = 1; i <= M_in; ++i) {
            for (int j = 1; j <= N_in; ++j) {
                const int k = ID(i, j);
                const double x = R.x0 + i * hx;
                const double y = R.y0 + j * hy;
                out << std::fixed << std::setprecision(10) << x << "," << y << "," << u[k] << "\n";
            }
        }
        out.close();
    }

    return 0;
}