// common.h - Infraestrutura compartilhada, bordas e exportação CSV
#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <random>
#include <string>

// Cabeçalhos do OpenCV
#include <opencv2/opencv.hpp>

using namespace std;
using Clock = chrono::system_clock;

#define NOVEC __attribute__((noinline, optimize("no-tree-vectorize","no-tree-slp-vectorize")))
#define VEC   __attribute__((noinline))

// Mascara 3x3 (Blur Gaussiano)
static const float MASCARA[9] = {1/16.f, 2/16.f, 1/16.f,
                                 2/16.f, 4/16.f, 2/16.f,
                                 1/16.f, 2/16.f, 1/16.f};

// BORDER_REPLICATE real (clampa índices para dentro da imagem)
static inline float viz(const float* in, int N, int i, int j, int di, int dj) {
    int ii = std::min(std::max(i + di, 0), N - 1);
    int jj = std::min(std::max(j + dj, 0), N - 1);
    return in[(size_t)ii * N + jj];
}

static inline float pixel_borda(const float* in, int N, int i, int j, const float* m) {
    float s = 0.f;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            s += m[(di + 1) * 3 + (dj + 1)] * viz(in, N, i, j, di, dj);
    return s;
}

static void bordas(const float* in, float* out, int N, const float* m) {
    for (int j = 0; j < N; j++) {
        out[j] = pixel_borda(in, N, 0, j, m);
        out[(size_t)(N - 1) * N + j] = pixel_borda(in, N, N - 1, j, m);
    }
    for (int i = 1; i < N - 1; i++) {
        out[(size_t)i * N] = pixel_borda(in, N, i, 0, m);
        out[(size_t)i * N + N - 1] = pixel_borda(in, N, i, N - 1, m);
    }
}

#define STENCIL(in, N, i, j, m)                                             \
    (m[0]*in[(size_t)((i)-1)*N+(j)-1] + m[1]*in[(size_t)((i)-1)*N+(j)] + m[2]*in[(size_t)((i)-1)*N+(j)+1] + \
     m[3]*in[(size_t)(i)*N+(j)-1]     + m[4]*in[(size_t)(i)*N+(j)]     + m[5]*in[(size_t)(i)*N+(j)+1] +     \
     m[6]*in[(size_t)((i)+1)*N+(j)-1] + m[7]*in[(size_t)((i)+1)*N+(j)] + m[8]*in[(size_t)((i)+1)*N+(j)+1])

using Func = void (*)(const float*, float*, int, const float*);

static double agora() { return chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

static float maxdiff(const float* a, const float* b, size_t n) {
    float d = 0; for (size_t k = 0; k < n; k++) d = max(d, fabsf(a[k] - b[k])); return d;
}

static void testar(const char* nome, Func f, const float* in, float* out, int N, ofstream& log) {
    // 1. Matrizes para validação do OpenCV
    cv::Mat imagem_in(N, N, CV_32FC1, const_cast<float*>(in));
    cv::Mat imagem_out_opencv(N, N, CV_32FC1);
    cv::Mat mascara(3, 3, CV_32FC1, const_cast<float*>(MASCARA));

    // Ref OpenCV fora do tempo cronometrado
    cv::filter2D(imagem_in, imagem_out_opencv, -1, mascara, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

    // 2. Cronometragem da execução
    double t0 = agora();
    f(in, out, N, MASCARA);
    double t1 = agora();
    double t = t1 - t0;

    // 3. Comparação com OpenCV
    float dif = maxdiff(reinterpret_cast<const float*>(imagem_out_opencv.data), out, (size_t)N * N);

    // 4. Saída formatada no terminal
    printf("%-10s N=%-5d tempo=%.6f s  maxdiff_vs_opencv=%g\n", nome, N, t, dif);

    // 5. Gravação da linha no arquivo CSV
    log.precision(6); log << fixed;
    log << N << ";" << t0 << ";" << t1 << ";" << t << ";" << dif << "\n";
}

#endif // COMMON_H