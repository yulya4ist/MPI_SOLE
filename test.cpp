#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

const double eps = 1.e-5;
const int N = 2070;
const double h = 1.0 / (N - 1);
const double k = 1.0 / h;
const double coef = (4.0 + h * h * k * k);

double f(double x, double y) {
    return 2 * sin(M_PI * y) + k * k * (1 - x) * x * sin(M_PI * y) +
           M_PI * M_PI * (1 - x) * x * sin(M_PI * y);
}

double norm(std::vector<double>& A, std::vector<double>& B, int i_beg,
            int i_end) {
    double norma = 0.0;
    for (int i = i_beg; i < i_end; ++i)
        if (norma < fabs(A[i] - B[i])) norma = fabs(A[i] - B[i]);
    return norma;
}

void main_y(std::vector<int>& el_num, std::vector<double>& y_n,
            std::vector<double>& y, std::vector<int>& displs, int np,
            int my_id) {
    int size;
    if ((my_id == 0 || my_id == np - 1) && np != 1)
        size = el_num[my_id] - N;
    else if (np != 1)
        size = el_num[my_id] - 2 * N;
    else
        size = el_num[my_id];

    MPI_Gatherv((my_id == 0) ? y_n.data() : y_n.data() + N, size, MPI_DOUBLE,
                y.data(), el_num.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
}

void analitic_solution(std::vector<double>& u) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            u[i * N + j] = (1 - i * h) * i * h * sin(M_PI * j * h);
}

double Jacobi(std::vector<double>& y, std::vector<double>& y_n,
              std::vector<int>& el_num, int my_id, int np, int& iterations,
              int send_type) {
    double norm_f;

    if (np == 1) {
        // тут последовательный
        iterations = 0;
        do {
            ++iterations;
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f(i * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                     y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                   coef;

            norm_f = norm(y, y_n, 0, N * N);
            y_n.swap(y);
        } while (norm_f > eps);
    }

    if (np > 1) {
        double norma;

        int shift = 0;
        for (int i = 0; i < my_id; ++i) shift += el_num[i] / N;
        shift -= (my_id == 0) ? 0 : my_id * 2;

        iterations = 0;
        do {
            if (send_type == 1) {
                MPI_Send(y_n.data() + el_num[my_id] - 2 * N, (my_id != np - 1) ? N : 0,
                         MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 1,
                         MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (my_id != 0) ? N : 0, MPI_DOUBLE,
                         (my_id != 0) ? my_id - 1 : np - 1, 1, MPI_COMM_WORLD,
                         MPI_STATUSES_IGNORE);

                MPI_Send(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                         (my_id != 0) ? my_id - 1 : np - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                         MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 2, MPI_COMM_WORLD,
                         MPI_STATUSES_IGNORE);
            }
            if (send_type == 2) {
                MPI_Sendrecv(
                        y_n.data() + el_num[my_id] - 2 * N, (my_id != np - 1) ? N : 0,
                        MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 3, y_n.data(),
                        (my_id != 0) ? N : 0, MPI_DOUBLE, (my_id != 0) ? my_id - 1 : np - 1, 3,
                        MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Sendrecv(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                             (my_id != 0) ? my_id - 1 : np - 1, 4,
                             y_n.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                             MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 4,
                             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (send_type == 3) {
                if (my_id != np - 1) {
                    MPI_Isend(y_n.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE, my_id + 1,
                              5, MPI_COMM_WORLD, &req_send_up);

                    MPI_Irecv(y_n.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1, 6,
                              MPI_COMM_WORLD, &req_recv_up);
                }
                if (my_id != 0) {
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                              &req_recv_down);

                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, my_id - 1, 6, MPI_COMM_WORLD,
                              &req_send_down);
                }
            }

            ++iterations;

            if (send_type == 1 || send_type == 2) {
                for (int i = 1; i < el_num[my_id] / N - 1; ++i)
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                         y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                       coef;
            }
            if (send_type == 3) {
                for (int i = 2; i < el_num[my_id] / N - 2; ++i)
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                         y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                       coef;

                if (my_id != 0) MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (my_id != np - 1) MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                int i = 1;
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                     y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                   coef;

                i = el_num[my_id] / N - 2;
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                     y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                   coef;
            }

            norma = norm(y, y_n, (my_id == 0) ? 0 : N,
                         (my_id == np) ? el_num[my_id] : el_num[my_id] - N);
            MPI_Allreduce(&norma, &norm_f, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
        } while (norm_f > eps);
    }
    if (my_id == 0) {
        if (send_type == 1) {
            std::cout << "Jacobi "
                      << " (MPI_Send + MPI_Recv)\n";
        } else if (send_type == 2) {
            std::cout << "Jacobi "
                      << " (MPI_SendRecv)\n";
        } else if (send_type == 3) {
            std::cout << "Jacobi "
                      << " (MPI_ISend + MPI_IRecv)\n";
        }
    }
    return norm_f;
}

double Red_Black(std::vector<double>& y, std::vector<double>& y_n,
                 std::vector<int> el_num, int my_id, int np, int& iterations,
                 int send_type) {
    double norm_f;
    if (np == 1) {
        iterations = 0;
        do {
            ++iterations;
            for (int i = 1; i < N - 1; ++i)
                for (int j = (i % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f(i * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                     y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                   coef;

            for (int i = 1; i < N - 1; ++i)
                for (int j = ((i + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f(i * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] +
                                     y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   coef;

            norm_f = norm(y, y_n, 0, N * N);
            y_n.swap(y);
        } while (norm_f > eps);
    }

    if (np > 1) {
        double norma;

        int shift = 0;
        for (int i = 0; i < my_id; ++i) shift += el_num[i] / N;
        shift -= (my_id == 0) ? 0 : my_id * 2;

        iterations = 0;
        do {
            if (send_type == 1) {
                MPI_Send(y_n.data() + el_num[my_id] - 2 * N, (my_id != np - 1) ? N : 0,
                         MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 1,
                         MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (my_id != 0) ? N : 0, MPI_DOUBLE,
                         (my_id != 0) ? my_id - 1 : np - 1, 1, MPI_COMM_WORLD,
                         MPI_STATUSES_IGNORE);

                MPI_Send(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                         (my_id != 0) ? my_id - 1 : np - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                         MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 2, MPI_COMM_WORLD,
                         MPI_STATUSES_IGNORE);
            }
            if (send_type == 2) {
                MPI_Sendrecv(
                        y_n.data() + el_num[my_id] - 2 * N, (my_id != np - 1) ? N : 0,
                        MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 3, y_n.data(),
                        (my_id != 0) ? N : 0, MPI_DOUBLE, (my_id != 0) ? my_id - 1 : np - 1, 3,
                        MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Sendrecv(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                             (my_id != 0) ? my_id - 1 : np - 1, 4,
                             y_n.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                             MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 4,
                             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (send_type == 3) {
                if (my_id != np - 1) {
                    MPI_Isend(y_n.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE, my_id + 1,
                              5, MPI_COMM_WORLD, &req_send_up);
                    MPI_Irecv(y_n.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1, 6,
                              MPI_COMM_WORLD, &req_recv_up);
                }
                if (my_id != 0) {
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                              &req_recv_down);
                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, my_id - 1, 6, MPI_COMM_WORLD,
                              &req_send_down);
                }
            }

            ++iterations;

            if (send_type == 1 || send_type == 2) {
                for (int i = 1; i < el_num[my_id] / N - 1; ++i)
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                         y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                       coef;
            }
            if (send_type == 3) {
                for (int i = 2; i < el_num[my_id] / N - 2; ++i)
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                         y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                       coef;

                if (my_id != 0) MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (my_id != np - 1) MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                int i = 1;
                for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                     y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                   coef;

                i = el_num[my_id] / N - 2;
                for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                                     y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                                   coef;
            }

            if (send_type == 1) {
                MPI_Send(y.data() + el_num[my_id] - 2 * N, (my_id != np - 1) ? N : 0,
                         MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 1,
                         MPI_COMM_WORLD);
                MPI_Recv(y.data(), (my_id != 0) ? N : 0, MPI_DOUBLE,
                         (my_id != 0) ? my_id - 1 : np - 1, 1, MPI_COMM_WORLD,
                         MPI_STATUSES_IGNORE);

                MPI_Send(y.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                         (my_id != 0) ? my_id - 1 : np - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(y.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                         MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 2, MPI_COMM_WORLD,
                         MPI_STATUSES_IGNORE);
            }
            if (send_type == 2) {
                MPI_Sendrecv(y.data() + el_num[my_id] - 2 * N, (my_id != np - 1) ? N : 0,
                             MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 3, y.data(),
                             (my_id != 0) ? N : 0, MPI_DOUBLE,
                             (my_id != 0) ? my_id - 1 : np - 1, 3, MPI_COMM_WORLD,
                             MPI_STATUSES_IGNORE);

                MPI_Sendrecv(y.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                             (my_id != 0) ? my_id - 1 : np - 1, 4,
                             y.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                             MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 4,
                             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 3) {
                if (my_id != np - 1) {
                    MPI_Isend(y.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE, my_id + 1, 5,
                              MPI_COMM_WORLD, &req_send_up);
                    MPI_Irecv(y.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1, 6,
                              MPI_COMM_WORLD, &req_recv_up);
                }
                if (my_id != 0) {
                    MPI_Irecv(y.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                              &req_recv_down);
                    MPI_Isend(y.data() + N, N, MPI_DOUBLE, my_id - 1, 6, MPI_COMM_WORLD,
                              &req_send_down);
                }
            }

            if (send_type == 1 || send_type == 2) {
                for (int i = 1; i < el_num[my_id] / N - 1; ++i)
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y[i * N + j - 1] + y[i * N + j + 1] +
                                         y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                       coef;
            }
            if (send_type == 3) {
                for (int i = 2; i < el_num[my_id] / N - 2; ++i)
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y[i * N + j - 1] + y[i * N + j + 1] +
                                         y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                       coef;

                if (my_id != 0) MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (my_id != np - 1) MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                int i = 1;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] +
                                     y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   coef;

                i = el_num[my_id] / N - 2;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] +
                                     y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   coef;
            }

            norma = norm(y, y_n, (my_id == 0) ? 0 : N,
                         (my_id == np) ? el_num[my_id] : el_num[my_id] - N);
            MPI_Allreduce(&norma, &norm_f, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
        } while (norm_f > eps);
    }
    if (my_id == 0) {
        if (send_type == 1) {
            std::cout << "R-B"
                      << " (MPI_Send + MPI_Recv)\n";
        } else if (send_type == 2) {
            std::cout << "R-B"
                      << " (MPI_SendRecv)\n";
        } else if (send_type == 3) {
            std::cout << "R-B"
                      << " (MPI_ISend + MPI_IRecv)\n";
        }
    }
    return norm_f;
}


void zero(std::vector<double>& A) {
    for (double & i : A)
        i = 0.0;
}

int main(int argc, char** argv) {
    std::cout << "N = " << N << std::endl;

    int my_id, np, iterations;
    double t1, t2, t3, t4, norm_f;
    double time_J, time_RB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    std::vector<double> y, y_n, y_gen, u;
    std::vector<int> el_num(np), displs(np);

    if (my_id == 0) {
        if (N % np == 0) {
            for (int i = 0; i < np; ++i) el_num[i] = (N / np) * N;
        } else {
            int temp = 0;
            for (int i = 0; i < np - 1; ++i) {
                el_num[i] = round(((double)N / (double)np)) * N;
                temp += el_num[i] / N;
            }
            el_num[np - 1] = (N - temp) * N;
        }

        displs[0] = 0;
        for (int i = 1; i < np; ++i) displs[i] = displs[i - 1] + el_num[i - 1];

        for (int i = 0; i < np; ++i) el_num[i] += 2 * N;
        el_num[0] -= N;
        el_num[np - 1] -= N;
    }
    MPI_Bcast(el_num.data(), np, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), np, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_id == 0) {
        std::cout << "np: " << np << std::endl << std::endl;
        y_gen.resize(N * N, 0);
        u.resize(N * N);
        analitic_solution(u);
    }

    // if (np == 1) {
    if (my_id == 0) {
        y.resize(el_num[my_id], 0);
        zero(y);
        y_n.resize(el_num[my_id], 0);
        zero(y_n);

        y.resize(N * N, 0);
        zero(y);
        y_n.resize(N * N, 0);
        zero(y_n);

        t1 = MPI_Wtime();
        norm_f = Jacobi(y, y_n, el_num, my_id, 1, iterations, 0);
        t2 = MPI_Wtime();
        std::cout << std::endl << "Jacobi seq" << std::endl;
        std::cout << "Time = " << t2 - t1 << std::endl;
        std::cout << "Iter = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;

        time_J = t2 - t1;
        zero(y);
        zero(y_n);

        t3 = MPI_Wtime();
        norm_f = Red_Black(y, y_n, el_num, my_id, 1, iterations, 0);
        t4 = MPI_Wtime();
        std::cout << std::endl << "R-B seq" << std::endl;
        std::cout << "Time = " << t4 - t3 << std::endl;
        time_RB = t4 - t3;
        std::cout << "Iter = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl << std::endl;

    }
    // time_J = 1.50193;
    // time_RB = 0.689987;
    for (int send_type = 1; send_type <= 3; send_type++) {

        if (np > 1) {
            y.resize(el_num[my_id], 0);
            zero(y);
            y_n.resize(el_num[my_id], 0);
            zero(y_n);

            t1 = MPI_Wtime();
            norm_f = Jacobi(y, y_n, el_num, my_id, np, iterations, send_type);
            t2 = MPI_Wtime();
            if (my_id == 0) {
                std::cout << "Time = " << t2 - t1 << std::endl;
                std::cout << "Iter = " << iterations << std::endl;
                std::cout << "Error = " << norm_f << std::endl;
                std::cout << "speedup = " << time_J / (t2 - t1) << std::endl;
            }
            main_y(el_num, y, y_gen, displs, np, my_id);

            MPI_Barrier(MPI_COMM_WORLD);

            y.resize(el_num[my_id], 0);
            zero(y);
            y_n.resize(el_num[my_id], 0);
            zero(y_n);

            t1 = MPI_Wtime();
            norm_f = Red_Black(y, y_n, el_num, my_id, np, iterations, send_type);
            t2 = MPI_Wtime();
            if (my_id == 0) {
                std::cout << "Time = " << t2 - t1 << std::endl;
                std::cout << "Iter = " << iterations << std::endl;
                std::cout << "Error = " << norm_f << std::endl;
                std::cout << "speedup = " << time_RB / (t2 - t1) << std::endl;
            }
            main_y(el_num, y, y_gen, displs, np, my_id);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}
