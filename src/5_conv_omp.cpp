// 5_conv_omp.cpp - Convolução 3x3 (Blur Gaussiano) paralela: OpenMP (threads) + SIMD (AVX2)
// Base: 2_conv_simd.cpp (melhor versão da Atividade 1)
//
// Uso:  OMP_NUM_THREADS=8 ./5_conv_omp [N] [iters] [rep]
//   N     : lado da imagem NxN            (padrão 4096)
//   iters : execuções do kernel na região medida (padrão 5) - melhora a leitura de energia
//   rep   : número da repetição (apenas registrado no CSV)
//
// Medições feitas SOMENTE em volta do kernel:
//   - tempo  : omp_get_wtime()
//   - energia: RAPL via /sys/class/powercap (pacote da CPU)
//   - cache  : perf stat controlado por --control fd (variáveis PERF_CTL_FD / PERF_ACK_FD)
#include "common.h"
#include <omp.h>
#include <unistd.h>
#include <cstring>

// ---------------------------------------------------------------- RAPL (energia)
static const char* RAPL_E   = "/sys/class/powercap/intel-rapl:0/energy_uj";
static const char* RAPL_MAX = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

static double ler_uj(const char* p) {
    ifstream f(p);
    double v = -1;
    if (f) f >> v;
    return v;
}

// Energia em Joules entre duas leituras (trata overflow do contador). -1 se indisponível
static double delta_joules(double e0, double e1) {
    if (e0 < 0 || e1 < 0) return -1;
    if (e1 < e0) e1 += ler_uj(RAPL_MAX);
    return (e1 - e0) / 1e6;
}

// ---------------------------------------------------------------- perf (liga/desliga contadores)
static int CTL = -1, ACK = -1;
static void perf_cmd(const char* cmd) {
    if (CTL < 0) return;
    ssize_t r = write(CTL, cmd, strlen(cmd)); (void)r;
    char buf[16];
    if (ACK >= 0) { r = read(ACK, buf, sizeof buf); (void)r; }
}

// ---------------------------------------------------------------- kernel paralelo
// Paralelismo por LINHAS (cada thread processa um bloco contíguo de linhas)
// + SIMD no laço interno (colunas, acesso contíguo).
VEC void conv_omp(const float* __restrict in_, float* __restrict out_, int N, const float* m_) {
    const float* in  = (const float*)__builtin_assume_aligned(in_, 64);
    float*       out = (float*)__builtin_assume_aligned(out_, 64);
    const float m0=m_[0],m1=m_[1],m2=m_[2],m3=m_[3],m4=m_[4],m5=m_[5],m6=m_[6],m7=m_[7],m8=m_[8];

    #pragma omp parallel
    {
        // Interior da imagem
        #pragma omp for schedule(static) nowait
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
        // Bordas: linha 0 e N-1 (não se sobrepõem com o interior)
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < N; j++) {
            out[j] = pixel_borda(in, N, 0, j, m_);
            out[(size_t)(N - 1) * N + j] = pixel_borda(in, N, N - 1, j, m_);
        }
        // Bordas: coluna 0 e N-1
        #pragma omp for schedule(static)
        for (int i = 1; i < N - 1; i++) {
            out[(size_t)i * N]         = pixel_borda(in, N, i, 0, m_);
            out[(size_t)i * N + N - 1] = pixel_borda(in, N, i, N - 1, m_);
        }
    }
}

// Referência SERIAL (mesma conta, sem OpenMP) para validar o resultado paralelo
static void conv_ref(const float* in, float* out, int N, const float* m) {
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            out[(size_t)i * N + j] = STENCIL(in, N, i, j, m);
    bordas(in, out, N, m);
}

int main(int argc, char** argv) {
    int N     = argc > 1 ? atoi(argv[1]) : 4096;
    int iters = argc > 2 ? atoi(argv[2]) : 5;
    int rep   = argc > 3 ? atoi(argv[3]) : 1;
    if (getenv("PERF_CTL_FD")) CTL = atoi(getenv("PERF_CTL_FD"));
    if (getenv("PERF_ACK_FD")) ACK = atoi(getenv("PERF_ACK_FD"));
    int T = omp_get_max_threads();

    size_t bytes = ((size_t)N * N * sizeof(float) + 63) / 64 * 64;
    float* in  = (float*)aligned_alloc(64, bytes);
    float* out = (float*)aligned_alloc(64, bytes);

    // Inicialização paralela com o MESMO escalonamento do kernel (first-touch NUMA).
    // Gerador por linha => resultado independente do nº de threads.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        mt19937 gen(42 + i);
        uniform_real_distribution<float> dis(0.0f, 255.0f);
        for (int j = 0; j < N; j++) {
            in[(size_t)i * N + j]  = dis(gen);
            out[(size_t)i * N + j] = 0.f;
        }
    }

    conv_omp(in, out, N, MASCARA);           // aquecimento (fora da medição)

    // ------------------ REGIÃO MEDIDA ------------------
    perf_cmd("enable\n");
    double e0 = ler_uj(RAPL_E);
    double t0 = omp_get_wtime();
    for (int it = 0; it < iters; it++) conv_omp(in, out, N, MASCARA);
    double t1 = omp_get_wtime();
    double e1 = ler_uj(RAPL_E);
    perf_cmd("disable\n");
    // ---------------------------------------------------

    double total   = t1 - t0;
    double tempo   = total / iters;                         // s por execução
    double ej      = delta_joules(e0, e1);
    double energia = ej < 0 ? -1 : ej / iters;              // J por execução
    double potencia = ej < 0 ? -1 : ej / total;             // W médios
    double mflops  = 17.0 * (double)(N - 2) * (N - 2) / tempo / 1e6;   // 9 mul + 8 add por pixel

    // Validação contra referência serial (fora da medição)
    float* ref = (float*)aligned_alloc(64, bytes);
    conv_ref(in, ref, N, MASCARA);
    float dif = maxdiff(ref, out, (size_t)N * N);

    printf("omp        N=%-5d T=%-3d tempo=%.6f s  energia=%.3f J  pot=%.2f W  MFLOPS=%.0f  maxdiff=%g\n",
           N, T, tempo, energia, potencia, mflops, dif);

    string caminho = "csv_time/5_conv_omp.csv";
    ifstream chk(caminho);
    bool existe = chk.is_open() && chk.peek() != ifstream::traits_type::eof();
    chk.close();
    ofstream log(caminho, ios::app);
    if (!existe) log << "N;threads;rep;iters;tempo_s;energia_J;potencia_W;mflops;maxdiff_vs_serial\n";
    log.precision(6); log << fixed;
    log << N << ";" << T << ";" << rep << ";" << iters << ";" << tempo << ";" << energia << ";"
        << potencia << ";" << mflops << ";" << dif << "\n";

    free(in); free(out); free(ref);
    return 0;
}
