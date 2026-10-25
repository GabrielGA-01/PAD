// PPAD - Convolucao 2D 3x3 (stencil) - Exercicios 1 e 2
// Compilar: g++ -O3 -mavx2 -fopenmp-simd convolucao.cpp -o convolucao
//  (as funcoes do Ex.1 desligam a vetorizacao via atributo; so o Ex.2 vetoriza)
// Uso: ./convolucao [reps] [tamanhos...]   ex: ./convolucao 10 1024 2048 4096
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <thread>

using namespace std;
using Clock = chrono::system_clock;  // epoch (comparavel com timestamps do HWiNFO)

#define NOVEC __attribute__((noinline, optimize("no-tree-vectorize","no-tree-slp-vectorize")))
#define VEC   __attribute__((noinline))

// Mascara 3x3 (Blur Gaussiano) - mesma para todas as versoes
static const float MASCARA[9] = {1/16.f, 2/16.f, 1/16.f,
                                 2/16.f, 4/16.f, 2/16.f,
                                 1/16.f, 2/16.f, 1/16.f};

// ---------- Borda: vizinho inexistente = valor do pixel atual ----------
static inline float viz(const float* in, int N, int i, int j, int di, int dj) {
    int ii = i + di, jj = j + dj;
    if (ii < 0 || ii >= N || jj < 0 || jj >= N) return in[(size_t)i * N + j];
    return in[(size_t)ii * N + jj];
}
static inline float pixel_borda(const float* in, int N, int i, int j, const float* m) {
    float s = 0.f;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            s += m[(di + 1) * 3 + (dj + 1)] * viz(in, N, i, j, di, dj);
    return s;
}
// Processa somente as bordas (linhas 0 e N-1, colunas 0 e N-1)
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

// Miolo (pixel interior), 9 mult + 8 soma = 17 FLOPs
#define STENCIL(in, N, i, j, m)                                              \
    (m[0]*in[(size_t)((i)-1)*N+(j)-1] + m[1]*in[(size_t)((i)-1)*N+(j)] + m[2]*in[(size_t)((i)-1)*N+(j)+1] + \
     m[3]*in[(size_t)(i)*N+(j)-1]     + m[4]*in[(size_t)(i)*N+(j)]     + m[5]*in[(size_t)(i)*N+(j)+1] +     \
     m[6]*in[(size_t)((i)+1)*N+(j)-1] + m[7]*in[(size_t)((i)+1)*N+(j)] + m[8]*in[(size_t)((i)+1)*N+(j)+1])

// ---------- Ex.1a: row-major (acesso contiguo), SEM vetorizacao ----------
NOVEC void conv_linhas(const float* in, float* out, int N, const float* m) {
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            out[(size_t)i * N + j] = STENCIL(in, N, i, j, m);
    bordas(in, out, N, m);
}
// ---------- Ex.1b: column-major (acesso com stride N), SEM vetorizacao ----------
NOVEC void conv_colunas(const float* in, float* out, int N, const float* m) {
    for (int j = 1; j < N - 1; j++)
        for (int i = 1; i < N - 1; i++)
            out[(size_t)i * N + j] = STENCIL(in, N, i, j, m);
    bordas(in, out, N, m);
}
// ---------- Ex.2: melhor ordem (linhas) + SIMD via pragma + alinhamento ----------
VEC void conv_simd(const float* __restrict in_, float* __restrict out_, int N, const float* m_) {
    const float* in  = (const float*)__builtin_assume_aligned(in_, 64);
    float*       out = (float*)__builtin_assume_aligned(out_, 64);
    const float m0=m_[0],m1=m_[1],m2=m_[2],m3=m_[3],m4=m_[4],m5=m_[5],m6=m_[6],m7=m_[7],m8=m_[8];
    for (int i = 1; i < N - 1; i++) {
        const float* a = in + (size_t)(i - 1) * N;
        const float* b = in + (size_t)i * N;
        const float* c = in + (size_t)(i + 1) * N;
        float* o = out + (size_t)i * N;
        #pragma omp simd
        for (int j = 1; j < N - 1; j++)
            o[j] = m0*a[j-1] + m1*a[j] + m2*a[j+1]
                 + m3*b[j-1] + m4*b[j] + m5*b[j+1]
                 + m6*c[j-1] + m7*c[j] + m8*c[j+1];
    }
    bordas(in, out, N, m_);
}

// ---------- Infraestrutura de medicao ----------
using Func = void (*)(const float*, float*, int, const float*);
static double agora() { return chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

static float maxdiff(const float* a, const float* b, size_t n) {
    float d = 0; for (size_t k = 0; k < n; k++) d = max(d, fabsf(a[k] - b[k])); return d;
}

// Executa 'reps' vezes (para durar o suficiente p/ o HWiNFO amostrar) e registra inicio/fim
static void testar(const char* nome, Func f, const float* in, float* out, int N, int reps,
                   const float* ref, ofstream& log) {
    double t0 = agora();
    for (int r = 0; r < reps; r++) f(in, out, N, MASCARA);
    double t1 = agora();
    double t = (t1 - t0) / reps;
    double mflops = 17.0 * N * N / t / 1e6;
    float dif = ref ? maxdiff(ref, out, (size_t)N * N) : 0.f;
    printf("%-10s N=%-5d tempo/exec=%.6f s  MFLOPS=%.1f  maxdiff_vs_ref=%g\n", nome, N, t, mflops, dif);
    log.precision(6); log << fixed;
    log << nome << ";" << N << ";" << reps << ";" << t0 << ";" << t1 << ";" << t << ";" << mflops << "\n";
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    vector<int> tams;
    for (int a = 2; a < argc; a++) tams.push_back(atoi(argv[a]));
    if (tams.empty()) tams = {1024, 2048, 4096};

    ofstream log("exec_tempo.txt");
    log << "versao;N;reps;inicio_epoch;fim_epoch;tempo_medio_s;mflops\n";

    for (int N : tams) {
        size_t bytes = ((size_t)N * N * sizeof(float) + 63) / 64 * 64;
        float* in  = (float*)aligned_alloc(64, bytes);
        float* out = (float*)aligned_alloc(64, bytes);
        float* ref = (float*)aligned_alloc(64, bytes);
        srand(42);
        for (size_t k = 0; k < (size_t)N * N; k++) in[k] = 255.f * rand() / RAND_MAX;

        conv_linhas(in, ref, N, MASCARA);  // referencia p/ validacao
        this_thread::sleep_for(chrono::seconds(3));  // pausa p/ separar testes no log de energia
        testar("linhas",  conv_linhas,  in, out, N, reps, ref, log);
        this_thread::sleep_for(chrono::seconds(3));
        testar("colunas", conv_colunas, in, out, N, reps, ref, log);
        this_thread::sleep_for(chrono::seconds(3));
        testar("simd",    conv_simd,    in, out, N, reps, ref, log);
        this_thread::sleep_for(chrono::seconds(3));
        free(in); free(out); free(ref);
    }
    return 0;
}
