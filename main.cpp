#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

const double eps = 1.e-8;
const int N = 18*30*10;              //количество точек
const double h = 1.0 / (N - 1);  //шаг
const double k = 1.0 / h;
const double coef = (4.0 + h * h * k * k);

//заполняем вектор нулями
void zero(std::vector<double>& A) {
  for (std::size_t i = 0; i < A.size(); ++i) {
    A[i] = 0.;
  }
}

//правая часть
double f(double x, double y) {
  return 2 * sin(M_PI * y) + k * k * (1 - x) * x * sin(M_PI * y) +
         M_PI * M_PI * (1 - x) * x * sin(M_PI * y);
}

double norm(std::vector<double>& A, std::vector<double>& B, int i_start,
            int i_end) {
  double norma = 0.0;
  for (int i = i_start; i < i_end; ++i)
    if (norma < fabs(A[i] - B[i])) norma = fabs(A[i] - B[i]);
  return norma;
}

void y_main(std::vector<int>& el_num, std::vector<double>& y_n,
            std::vector<double>& y, std::vector<int>& displs, int np,
            int my_id) {
  int size;
  if ((my_id == 0 || my_id == np - 1) && np != 1)
    size = el_num[my_id] - N;
  else if (np != 1)
    size = el_num[my_id] - 2 * N;
  else
    size = el_num[my_id];

  //объединение частей матрицы
  MPI_Gatherv((my_id == 0) ? y_n.data() : y_n.data() + N, size, MPI_DOUBLE,
              y.data(), el_num.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
}

void analytic_solution(std::vector<double>& u) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      u[i * N + j] = (1 - i * h) * i * h * sin(M_PI * j * h);
}

double Jacobi(std::vector<double>& y, std::vector<double>& y_n,
              const std::vector<int>& el_num, int my_id, int np,
              int& iterations, int send_type) {
  double norm_f;
  //последовательный
  if (np == 1) {
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

  //параллельный
  if (np > 1) {
    double norma;

    int shift = 0;
    for (int i = 0; i < my_id; ++i) shift += el_num[i] / N;

    shift -= (my_id == 0) ? 0 : my_id * 2;

    iterations = 0;


    MPI_Request* req_send = new MPI_Request[2];
    MPI_Request* req_recv = new MPI_Request[2];

    MPI_Request* req_send1 = new MPI_Request[2];
    MPI_Request* req_recv1 = new MPI_Request[2];

    if (my_id != np - 1) {
      //передача вниз
      MPI_Send_init(y_n.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE,
                    my_id + 1, 5, MPI_COMM_WORLD, req_send);

      //прием снизу
      MPI_Recv_init(y_n.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1, 6,
                    MPI_COMM_WORLD, req_recv);

      //передача вниз
      MPI_Send_init(y.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE, my_id + 1,
                    5, MPI_COMM_WORLD, req_send1);

      //прием снизу
      MPI_Recv_init(y.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1, 6,
                    MPI_COMM_WORLD, req_recv1);
    } else {
      MPI_Send_init(y_n.data(), 0, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, req_send);

      MPI_Recv_init(y_n.data(), 0, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, req_recv);

      MPI_Send_init(y.data(), 0, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, req_send1);

      MPI_Recv_init(y.data(), 0, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, req_recv1);
    }
    if (my_id != 0) {
      //прием сверху
      MPI_Recv_init(y_n.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                    req_recv + 1);

      //передача вверх
      MPI_Send_init(y_n.data() + N, N, MPI_DOUBLE, my_id - 1, 6, MPI_COMM_WORLD,
                    req_send + 1);

      //прием сверху
      MPI_Recv_init(y.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                    req_recv1 + 1);

      //передача вверх
      MPI_Send_init(y.data() + N, N, MPI_DOUBLE, my_id - 1, 6, MPI_COMM_WORLD,
                    req_send1 + 1);
    } else {
      MPI_Recv_init(y_n.data(), 0, MPI_DOUBLE, np - 1, 5, MPI_COMM_WORLD,
                    req_recv + 1);

      MPI_Send_init(y_n.data(), 0, MPI_DOUBLE, np - 1, 6, MPI_COMM_WORLD,
                    req_send + 1);

      MPI_Recv_init(y.data(), 0, MPI_DOUBLE, np - 1, 5, MPI_COMM_WORLD,
                    req_recv1 + 1);

      MPI_Send_init(y.data(), 0, MPI_DOUBLE, np - 1, 6, MPI_COMM_WORLD,
                    req_send1 + 1);
    }

    do {
      //Для isend
      MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
      switch (send_type) {
        case 1:
          //передача вниз
          MPI_Send(y_n.data() + el_num[my_id] - 2 * N,
                   (my_id != np - 1) ? N : 0, MPI_DOUBLE,
                   (my_id != np - 1) ? my_id + 1 : 0, 1, MPI_COMM_WORLD);
          MPI_Recv(y_n.data(), (my_id != 0) ? N : 0, MPI_DOUBLE,
                   (my_id != 0) ? my_id - 1 : np - 1, 1, MPI_COMM_WORLD,
                   MPI_STATUSES_IGNORE);

          //передача вверх
          MPI_Send(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                   (my_id != 0) ? my_id - 1 : np - 1, 2, MPI_COMM_WORLD);
          MPI_Recv(y_n.data() + el_num[my_id] - N, (my_id != np - 1) ? N : 0,
                   MPI_DOUBLE, (my_id != np - 1) ? my_id + 1 : 0, 2,
                   MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
          break;

        case 2:
          //передать вниз и принять сверху
          MPI_Sendrecv(y_n.data() + el_num[my_id] - 2 * N,
                       (my_id != np - 1) ? N : 0, MPI_DOUBLE,
                       (my_id != np - 1) ? my_id + 1 : 0, 3, y_n.data(),
                       (my_id != 0) ? N : 0, MPI_DOUBLE,
                       (my_id != 0) ? my_id - 1 : np - 1, 3, MPI_COMM_WORLD,
                       MPI_STATUSES_IGNORE);

          //передать вверх и принять снизу
          MPI_Sendrecv(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                       (my_id != 0) ? my_id - 1 : np - 1, 4,
                       y_n.data() + el_num[my_id] - N,
                       (my_id != np - 1) ? N : 0, MPI_DOUBLE,
                       (my_id != np - 1) ? my_id + 1 : 0, 4, MPI_COMM_WORLD,
                       MPI_STATUSES_IGNORE);
          break;

        case 3:
          if (my_id != np - 1) {
            //передача вниз
            MPI_Isend(y_n.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE,
                      my_id + 1, 5, MPI_COMM_WORLD, &req_send_up);

            //прием снизу
            MPI_Irecv(y_n.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1,
                      6, MPI_COMM_WORLD, &req_recv_up);
          }
          if (my_id != 0) {
            //прием сверху
            MPI_Irecv(y_n.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                      &req_recv_down);

            //передача вверх
            MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, my_id - 1, 6,
                      MPI_COMM_WORLD, &req_send_down);
          }
          break;
        case 4:
          if (iterations % 2 == 0) {
            MPI_Startall(2, req_send);
            MPI_Startall(2, req_recv);
          } else {
            MPI_Startall(2, req_send1);
            MPI_Startall(2, req_recv1);
          }
          break;
        default:
        {}
      }

      ++iterations;

      if ((send_type == 1) || (send_type == 2)) {
        for (int i = 1; i < el_num[my_id] / N - 1; ++i)
          for (int j = 1; j < N - 1; ++j)
            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                             y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                           coef;
      } else if (send_type == 4) {
        if (iterations % 2 == 1) {
          for (int i = 2; i < el_num[my_id] / N - 2; ++i)
            for (int j = 1; j < N - 1; ++j)
              y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                              (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                               y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                             coef;
        } else {
          for (int i = 2; i < el_num[my_id] / N - 2; ++i)
            for (int j = 1; j < N - 1; ++j)
              y_n[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                (y[i * N + j - 1] + y[i * N + j + 1] +
                                 y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                               coef;
        }
        if (iterations % 2 == 1) {
          MPI_Waitall(2, req_send, MPI_STATUSES_IGNORE);
          MPI_Waitall(2, req_recv, MPI_STATUSES_IGNORE);
        } else {
          MPI_Waitall(2, req_send1, MPI_STATUSES_IGNORE);
          MPI_Waitall(2, req_recv1, MPI_STATUSES_IGNORE);
        }
        if (iterations % 2 == 1) {
          int i = 1;
          for (int j = 1; j < N - 1; ++j)
            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                             y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                           coef;

          //нижняя строка
          i = el_num[my_id] / N - 2;
          for (int j = 1; j < N - 1; ++j)
            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                             y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                           coef;
        } else {
          int i = 1;
          for (int j = 1; j < N - 1; ++j)
            y_n[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                              (y[i * N + j - 1] + y[i * N + j + 1] +
                               y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                             coef;

          //нижняя строка
          i = el_num[my_id] / N - 2;
          for (int j = 1; j < N - 1; ++j)
            y_n[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                              (y[i * N + j - 1] + y[i * N + j + 1] +
                               y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                             coef;
        }
      } else {
        //все строки, кроме верхней и нижней
        for (int i = 2; i < el_num[my_id] / N - 2; ++i)
          for (int j = 1; j < N - 1; ++j)
            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                             y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                           coef;

        if (my_id != 0) MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
        if (my_id != np - 1) MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

        //верхняя строка
        int i = 1;
        for (int j = 1; j < N - 1; ++j)
          y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                          (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                           y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                         coef;

        //нижняя строка
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
      if (send_type != 4) {
        y_n.swap(y);
      }
    } while ((norm_f > eps) && (iterations < 100));

    delete[] req_send;
    delete[] req_recv;
    delete[] req_send1;
    delete[] req_recv1;
  }
  if (my_id == 0) {
    switch (send_type) {
      case 1:
        std::cout << "Jacobi"
                  << " (MPI_Send + MPI_Recv)\n";
        break;
      case 2:
        std::cout << "Jacobi"
                  << " (MPI_SendRecv)\n";
        break;
      case 3:
        std::cout << "Jacobi"
                  << " (MPI_ISend + MPI_IRecv)\n";
        break;
      case 4:
        std::cout << "Jacobi"
                  << " new method\n";
        break;
      default:
      {}
    }
  }
  return norm_f;
}

double Red_Black(std::vector<double>& y, std::vector<double>& y_n,
                 std::vector<int>& el_num, int my_id, int n_th, int& iterations,
                 int send_type) {
  double norm_f;
  //последовательный
  if (n_th == 1) {
    if (my_id == 0) {
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
      return norm_f;
    }
  }
  //параллельный
  if (n_th > 1) {
    double norma;

    int shift = 0;
    for (int i = 0; i < my_id; ++i) shift += el_num[i] / N;

    shift -= (my_id == 0) ? 0 : my_id * 2;

    iterations = 0;

    do {
      MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
      switch (send_type) {
        case 1:

          //передача вниз
          MPI_Send(y_n.data() + el_num[my_id] - 2 * N,
                   (my_id != n_th - 1) ? N : 0, MPI_DOUBLE,
                   (my_id != n_th - 1) ? my_id + 1 : 0, 1, MPI_COMM_WORLD);
          MPI_Recv(y_n.data(), (my_id != 0) ? N : 0, MPI_DOUBLE,
                   (my_id != 0) ? my_id - 1 : n_th - 1, 1, MPI_COMM_WORLD,
                   MPI_STATUSES_IGNORE);

          //передача вверх
          MPI_Send(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                   (my_id != 0) ? my_id - 1 : n_th - 1, 2, MPI_COMM_WORLD);
          MPI_Recv(y_n.data() + el_num[my_id] - N, (my_id != n_th - 1) ? N : 0,
                   MPI_DOUBLE, (my_id != n_th - 1) ? my_id + 1 : 0, 2,
                   MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
          break;
        case 2:
          //передать вниз и принять сверху
          MPI_Sendrecv(y_n.data() + el_num[my_id] - 2 * N,
                       (my_id != n_th - 1) ? N : 0, MPI_DOUBLE,
                       (my_id != n_th - 1) ? my_id + 1 : 0, 3, y_n.data(),
                       (my_id != 0) ? N : 0, MPI_DOUBLE,
                       (my_id != 0) ? my_id - 1 : n_th - 1, 3, MPI_COMM_WORLD,
                       MPI_STATUSES_IGNORE);

          //передать вверх и принять снизу
          MPI_Sendrecv(y_n.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                       (my_id != 0) ? my_id - 1 : n_th - 1, 4,
                       y_n.data() + el_num[my_id] - N,
                       (my_id != n_th - 1) ? N : 0, MPI_DOUBLE,
                       (my_id != n_th - 1) ? my_id + 1 : 0, 4, MPI_COMM_WORLD,
                       MPI_STATUSES_IGNORE);
          break;
        case 3:

          if (my_id != n_th - 1) {
            //передача вниз
            MPI_Isend(y_n.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE,
                      my_id + 1, 5, MPI_COMM_WORLD, &req_send_up);
            MPI_Irecv(y_n.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1,
                      6, MPI_COMM_WORLD, &req_recv_up);
          }
          if (my_id != 0) {
            //передача вверх
            MPI_Irecv(y_n.data(), N, MPI_DOUBLE, my_id - 1, 5, MPI_COMM_WORLD,
                      &req_recv_down);
            MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, my_id - 1, 6,
                      MPI_COMM_WORLD, &req_send_down);
          }
          break;

          default:
          {}
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
        //все строки, кроме верхней и нижней
        for (int i = 2; i < el_num[my_id] / N - 2; ++i)
          for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                             y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                           coef;

        if (send_type == 3) {
          if (my_id != 0) MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
          if (my_id != n_th - 1) MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);
        }

        //верхняя строка
        int i = 1;
        for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
          y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                          (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                           y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                         coef;

        //нижняя строка
        i = el_num[my_id] / N - 2;
        for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
          y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                          (y_n[i * N + j - 1] + y_n[i * N + j + 1] +
                           y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) /
                         coef;
      }

      if (send_type == 1) {
        //передача вниз
        MPI_Send(y.data() + el_num[my_id] - 2 * N, (my_id != n_th - 1) ? N : 0,
                 MPI_DOUBLE, (my_id != n_th - 1) ? my_id + 1 : 0, 1,
                 MPI_COMM_WORLD);
        MPI_Recv(y.data(), (my_id != 0) ? N : 0, MPI_DOUBLE,
                 (my_id != 0) ? my_id - 1 : n_th - 1, 1, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);

        //передача вверх
        MPI_Send(y.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                 (my_id != 0) ? my_id - 1 : n_th - 1, 2, MPI_COMM_WORLD);
        MPI_Recv(y.data() + el_num[my_id] - N, (my_id != n_th - 1) ? N : 0,
                 MPI_DOUBLE, (my_id != n_th - 1) ? my_id + 1 : 0, 2,
                 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }
      if (send_type == 2) {
        //передать вниз и принять сверху
        MPI_Sendrecv(y.data() + el_num[my_id] - 2 * N,
                     (my_id != n_th - 1) ? N : 0, MPI_DOUBLE,
                     (my_id != n_th - 1) ? my_id + 1 : 0, 3, y.data(),
                     (my_id != 0) ? N : 0, MPI_DOUBLE,
                     (my_id != 0) ? my_id - 1 : n_th - 1, 3, MPI_COMM_WORLD,
                     MPI_STATUSES_IGNORE);

        //передать вверх и принять снизу
        MPI_Sendrecv(y.data() + N, (my_id != 0) ? N : 0, MPI_DOUBLE,
                     (my_id != 0) ? my_id - 1 : n_th - 1, 4,
                     y.data() + el_num[my_id] - N, (my_id != n_th - 1) ? N : 0,
                     MPI_DOUBLE, (my_id != n_th - 1) ? my_id + 1 : 0, 4,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }
      if (send_type == 3) {
        if (my_id != n_th - 1) {
          //передача вниз
          MPI_Isend(y.data() + el_num[my_id] - 2 * N, N, MPI_DOUBLE, my_id + 1,
                    5, MPI_COMM_WORLD, &req_send_up);
          MPI_Irecv(y.data() + el_num[my_id] - N, N, MPI_DOUBLE, my_id + 1, 6,
                    MPI_COMM_WORLD, &req_recv_up);
        }
        if (my_id != 0) {
          //передача вверх
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
        //все строки, кроме верхней и нижней
        for (int i = 2; i < el_num[my_id] / N - 2; ++i)
          for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                            (y[i * N + j - 1] + y[i * N + j + 1] +
                             y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                           coef;
        // if (send_type == 3) {
        if (my_id != 0) MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
        if (my_id != n_th - 1) MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);
        // }

        //верхняя строка
        int i = 1;
        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
          y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                          (y[i * N + j - 1] + y[i * N + j + 1] +
                           y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                         coef;

        //нижняя строка
        i = el_num[my_id] / N - 2;
        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
          y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                          (y[i * N + j - 1] + y[i * N + j + 1] +
                           y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                         coef;
      }

      norma = norm(y, y_n, (my_id == 0) ? 0 : N,
                   (my_id == n_th) ? el_num[my_id] : el_num[my_id] - N);
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

int main(int argc, char** argv) {
  int my_id, n_th, iterations;
  double t1, t2, t3, t4, norm_f;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_th);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  std::vector<double> y, y_n, y_gen, u;
  // el_num - количество элементов в блокее
  // displs - смещение для вектора
  std::vector<int> el_num(n_th), displs(n_th);
  double time_seq_J, time_seq_RB;
  if (my_id == 0) {
    std::vector<double> y_seq(N * N, 0);
    std::vector<double> y_n_seq(N * N, 0);
    std::vector<int> el_num_seq(0);
    std::vector<double> u_seq(N * N);
    analytic_solution(u_seq);

    t1 = omp_get_wtime();
    norm_f = Jacobi(y_seq, y_n_seq, el_num_seq, my_id, 1, iterations, 0);
    t2 = omp_get_wtime();
    std::cout << "N=" << N << std::endl;
    std::cout << std::endl << "Jacobi n = 1" << std::endl;
    std::cout << "Time = " << t2 - t1 << std::endl;
    time_seq_J = t2 - t1;
    std::cout << "Iter = " << iterations << std::endl;
    std::cout << "|y - u| = " << norm(y_seq, u_seq, 0, N * N) << std::endl
              << std::endl;

    zero(y_seq);
    zero(y_n_seq);

    t3 = omp_get_wtime();
    norm_f = Red_Black(y_seq, y_n_seq, el_num_seq, my_id, 1, iterations, 0);
    t4 = omp_get_wtime();
    time_seq_RB = t4 - t3;
    std::cout << std::endl << "R-B n = 1" << std::endl;
    std::cout << "Time = " << t4 - t3 << std::endl;
    std::cout << "Iter = " << iterations << std::endl;
    std::cout << "|y - u| = " << norm(y_seq, u_seq, 0, N * N) << std::endl
              << std::endl;
  }

  //распределение размеров блоков
  if (my_id == 0) {
    if (N % n_th == 0) {
      for (int i = 0; i < n_th; ++i)
        el_num[i] = (N / n_th) * N;  //число элементов в i-м блоке
    } else {
      int temp = 0;
      for (int i = 0; i < n_th - 1; ++i) {
        el_num[i] = (int)(round(((double)N / (double)n_th)) * N);
        temp += el_num[i] / N;
      }
      el_num[n_th - 1] = (N - temp) * N;
    }

    displs[0] = 0;
    for (int i = 1; i < n_th; ++i) displs[i] = displs[i - 1] + el_num[i - 1];

    for (int i = 0; i < n_th; ++i) el_num[i] += 2 * N;

    el_num[0] -= N;
    el_num[n_th - 1] -= N;
  }
  //рассылка всем процессам
  MPI_Bcast(el_num.data(), n_th, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), n_th, MPI_INT, 0, MPI_COMM_WORLD);

  if (my_id == 0) {
    std::cout << "n_th: " << n_th << std::endl << std::endl;
    y_gen.resize(N * N, 0);
    u.resize(N * N);
    analytic_solution(u);
  }

  if (n_th == 1) {
    return 0;
  }
  for (int send_type = 1; send_type <= 4; ++send_type) {
    if (n_th > 1) {
        y.resize(el_num[my_id]);
        zero(y);
        y_n.resize(el_num[my_id]);
        zero(y_n);

        t1 = MPI_Wtime();
        norm_f = Jacobi(y, y_n, el_num, my_id, n_th, iterations, send_type);
        t2 = MPI_Wtime();
        if (my_id == 0) {
            std::cout << "Time = " << t2 - t1 << std::endl;
            std::cout << "Iter = " << iterations << std::endl;
            std::cout << "Speedup = " << time_seq_J / (t2 - t1) << std::endl;
        }
        y_main(el_num, y, y_gen, displs, n_th, my_id);
        if (my_id == 0)
            std::cout << "|y - u| = " << norm(y_gen, u, 0, N * N) << std::endl
                      << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        y.resize(el_num[my_id]);
        zero(y);
        y_n.resize(el_num[my_id]);
        zero(y_n);

//        if (send_type != 4) {
//            t1 = MPI_Wtime();
//            norm_f = Red_Black(y, y_n, el_num, my_id, n_th, iterations, send_type);
//            t2 = MPI_Wtime();
//            if (my_id == 0) {
//                std::cout << "Time = " << t2 - t1 << std::endl;
//                std::cout << "Iter = " << iterations << std::endl;
//                std::cout << "Speedup = " << time_seq_RB / (t2 - t1) << std::endl;
//            }
//            y_main(el_num, y, y_gen, displs, n_th, my_id);
//            if (my_id == 0)
//                std::cout << "|y - u| = " << norm(y_gen, u, 0, N * N) << std::endl
//                          << std::endl;
//            MPI_Barrier(MPI_COMM_WORLD);
//        }
    }
  }
  MPI_Finalize();
  return 0;
}