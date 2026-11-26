// 6_conv_mpi_omp.cpp - Convolução 3x3 (Blur Gaussiano) HÍBRIDA: MPI (processos) + OpenMP (threads) + SIMD (AVX2)
// Base: 5_conv_omp.cpp (Atividade 2), que por sua vez parte de 2_conv_simd.cpp (Atividade 1)
//
// Estratégia
//   MPI    : decomposição de domínio em FAIXAS de linhas contíguas (1 faixa por processo).
//            Cada processo guarda (nr + 2) linhas: nr linhas suas + 1 linha "fantasma" (halo) acima e abaixo.
//            A cada passo, vizinhos trocam as linhas de fronteira (halo exchange).
//   OpenMP : dentro de cada processo, as linhas da faixa são divididas entre as threads (schedule static).
//   SIMD   : laço interno de colunas vetorizado (#pragma omp simd).
//   Borda  : BORDER_REPLICATE (igual ao OpenCV). Nas bordas globais o halo é uma cópia da própria linha;
//            nas colunas 0 e N-1 os índices são "clampados".
//
// Cada execução aplica a convolução "iters" vezes em sequência (a saída de um passo é a entrada do
// próximo, como num filtro iterativo/stencil). Assim há comunicação MPI a cada passo.
//
// Uso:  OMP_NUM_THREADS=T mpirun -np P ./6_conv_mpi_omp [N] [iters] [rep] [modo]
//   N     : lado da imagem NxN                      (padrão 4096)
//   iters : passos de convolução na região medida   (padrão 5)
//   rep   : nº da repetição (só registrado no CSV)
//   modo  : 0 = troca de halo bloqueante, depois cálculo
//           1 = sobreposição: MPI_Isend/Irecv + cálculo das linhas internas enquanto a comunicação ocorre
//
// Medições (somente em volta dos passos medidos): tempo (MPI_Wtime), tempo de comunicação,
// energia (RAPL, somando 1 leitor por nó) e contadores de cache (perf, via --control fd).
#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <algorithm>
#include <unistd.h>

using namespace std;
#define VEC __attribute__((noinline))

static const float MASCARA[9] = {1/16.f, 2/16.f, 1/16.f,
                                 2/16.f, 4/16.f, 2/16.f,
                                 1/16.f, 2/16.f, 1/16.f};

// ================================================================ RAPL (energia, por nó)
struct Rapl { int n = 0; string path[8]; double mx[8]; };

static double ler_uj(const string& p) {
    ifstream f(p);
    double v = -1;
    if (f) f >> v;
    return v;
}
static Rapl rapl_init() {
    Rapl r;
    for (int k = 0; k < 8; k++) {
        string base = "/sys/class/powercap/intel-rapl:" + to_string(k);
        if (ler_uj(base + "/energy_uj") >= 0) {
            r.path[r.n] = base + "/energy_uj";
            r.mx[r.n]   = ler_uj(base + "/max_energy_range_uj");
            r.n++;
        }
    }
    return r;
}
static void rapl_ler(const Rapl& r, double* v) { for (int k = 0; k < r.n; k++) v[k] = ler_uj(r.path[k]); }
static double rapl_joules(const Rapl& r, const double* a, const double* b) {   // soma dos pacotes; -1 se indisponível
    if (r.n == 0) return -1;
    double s = 0;
    for (int k = 0; k < r.n; k++) { double d = b[k] - a[k]; if (d < 0) d += r.mx[k]; s += d; }
    return s / 1e6;
}

// ================================================================ perf (liga/desliga contadores por processo)
static int CTL = -1, ACK = -1;
static void perf_cmd(const char* cmd) {
    if (CTL < 0) return;
    ssize_t r = write(CTL, cmd, strlen(cmd)); (void)r;
    char buf[16];
    if (ACK >= 0) { r = read(ACK, buf, sizeof buf); (void)r; }
}

// ================================================================ kernel
// Calcula as linhas locais li = lo..hi (1..nr) da faixa. 'in' tem linhas 0..nr+1 (0 e nr+1 = halo).
VEC static void conv_rows(const float* __restrict in_, float* __restrict out_, int N, int lo, int hi, const float* m_) {
    const float* in  = (const float*)__builtin_assume_aligned(in_, 64);
    float*       out = (float*)__builtin_assume_aligned(out_, 64);
    const float m0=m_[0],m1=m_[1],m2=m_[2],m3=m_[3],m4=m_[4],m5=m_[5],m6=m_[6],m7=m_[7],m8=m_[8];

    #pragma omp parallel for schedule(static)
    for (int li = lo; li <= hi; li++) {
        const float* a = in + (size_t)(li - 1) * N;
        const float* b = in + (size_t)li * N;
        const float* c = in + (size_t)(li + 1) * N;
        float* o = out + (size_t)li * N;
        // colunas de borda (replicate)
        o[0]     = m0*a[0]   + m1*a[0]   + m2*a[1] + m3*b[0]   + m4*b[0]   + m5*b[1] + m6*c[0]   + m7*c[0]   + m8*c[1];
        o[N - 1] = m0*a[N-2] + m1*a[N-1] + m2*a[N-1] + m3*b[N-2] + m4*b[N-1] + m5*b[N-1] + m6*c[N-2] + m7*c[N-1] + m8*c[N-1];
        // interior
        #pragma omp simd
        for (int j = 1; j < N - 1; j++)
            o[j] = m0*a[j-1] + m1*a[j] + m2*a[j+1]
                 + m3*b[j-1] + m4*b[j] + m5*b[j+1]
                 + m6*c[j-1] + m7*c[j] + m8*c[j+1];
    }
}

// Referência SERIAL (mesma conta, sem OpenMP) para a faixa local; escreve em ref[(li-1)*N + j]
static void ref_faixa(const float* in, float* ref, int N, int nr, const float* m) {
    for (int li = 1; li <= nr; li++)
        for (int j = 0; j < N; j++) {
            int jm = max(j - 1, 0), jp = min(j + 1, N - 1);
            const float* a = in + (size_t)(li - 1) * N;
            const float* b = in + (size_t)li * N;
            const float* c = in + (size_t)(li + 1) * N;
            ref[(size_t)(li - 1) * N + j] =
                m[0]*a[jm] + m[1]*a[j] + m[2]*a[jp] + m[3]*b[jm] + m[4]*b[j] + m[5]*b[jp] + m[6]*c[jm] + m[7]*c[j] + m[8]*c[jp];
        }
}

// ================================================================ halo exchange
// Posta as recepções/envios das linhas de fronteira. Tag 0: mensagens que descem; tag 1: que sobem.
static void halo_post(float* cur, int N, int nr, int rank, int P, MPI_Request req[4]) {
    int up   = rank > 0     ? rank - 1 : MPI_PROC_NULL;
    int down = rank < P - 1 ? rank + 1 : MPI_PROC_NULL;
    MPI_Irecv(cur,                         N, MPI_FLOAT, up,   0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(cur + (size_t)(nr + 1) * N,  N, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(cur + (size_t)nr * N,        N, MPI_FLOAT, down, 0, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(cur + (size_t)1 * N,         N, MPI_FLOAT, up,   1, MPI_COMM_WORLD, &req[3]);
    // bordas globais: halo = cópia da própria linha (BORDER_REPLICATE)
    if (rank == 0)     memcpy(cur,                        cur + (size_t)1 * N,  N * sizeof(float));
    if (rank == P - 1) memcpy(cur + (size_t)(nr + 1) * N, cur + (size_t)nr * N, N * sizeof(float));
}

// Um passo: cur -> nxt. Retorna o tempo gasto em comunicação (s) por este processo.
static double passo(float* cur, float* nxt, int N, int nr, int rank, int P, int modo) {
    MPI_Request req[4];
    double tc = 0;
    double t = MPI_Wtime();
    halo_post(cur, N, nr, rank, P, req);
    if (modo == 0) {
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        tc += MPI_Wtime() - t;
        conv_rows(cur, nxt, N, 1, nr, MASCARA);
    } else {
        tc += MPI_Wtime() - t;
        conv_rows(cur, nxt, N, 2, nr - 1, MASCARA);         // linhas internas enquanto o halo "viaja"
        t = MPI_Wtime();
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        tc += MPI_Wtime() - t;
        conv_rows(cur, nxt, N, 1, 1, MASCARA);              // linhas que dependem do halo
        if (nr > 1) conv_rows(cur, nxt, N, nr, nr, MASCARA);
    }
    return tc;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int N     = argc > 1 ? atoi(argv[1]) : 4096;
    int iters = argc > 2 ? atoi(argv[2]) : 5;
    int rep   = argc > 3 ? atoi(argv[3]) : 1;
    int modo  = argc > 4 ? atoi(argv[4]) : 0;
    if (getenv("PERF_CTL_FD")) CTL = atoi(getenv("PERF_CTL_FD"));
    if (getenv("PERF_ACK_FD")) ACK = atoi(getenv("PERF_ACK_FD"));
    int T = omp_get_max_threads();

    // Leitor de energia: 1 processo por nó (o de menor rank local)
    MPI_Comm nodecomm; int lrank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodecomm);
    MPI_Comm_rank(nodecomm, &lrank);
    bool leitor = (lrank == 0);
    Rapl rapl = rapl_init();
    double ea[8], eb[8];

    // Decomposição em faixas de linhas
    int base = N / P, resto = N % P;
    int nr    = base + (rank < resto ? 1 : 0);
    int start = rank * base + min(rank, resto);          // 1ª linha global desta faixa
    if (nr < 2) { if (rank == 0) fprintf(stderr, "N muito pequeno para %d processos\n", P); MPI_Abort(MPI_COMM_WORLD, 1); }

    size_t bytes = (((size_t)(nr + 2) * N * sizeof(float)) + 63) / 64 * 64;
    float* cur = (float*)aligned_alloc(64, bytes);
    float* nxt = (float*)aligned_alloc(64, bytes);

    // Inicialização paralela (first-touch); gerador por linha GLOBAL => imagem independente de P e T
    #pragma omp parallel for schedule(static)
    for (int li = 1; li <= nr; li++) {
        mt19937 gen(42 + start + li - 1);
        uniform_real_distribution<float> dis(0.0f, 255.0f);
        float* r1 = cur + (size_t)li * N;
        float* r2 = nxt + (size_t)li * N;
        for (int j = 0; j < N; j++) { r1[j] = dis(gen); r2[j] = 0.f; }
    }
    memset(cur, 0, N * sizeof(float));  memset(cur + (size_t)(nr + 1) * N, 0, N * sizeof(float));
    memset(nxt, 0, N * sizeof(float));  memset(nxt + (size_t)(nr + 1) * N, 0, N * sizeof(float));

    // ---- Passo de aquecimento + VALIDAÇÃO (fora da medição) ----
    passo(cur, nxt, N, nr, rank, P, modo);
    float* ref = (float*)aligned_alloc(64, (((size_t)nr * N * sizeof(float)) + 63) / 64 * 64);
    ref_faixa(cur, ref, N, nr, MASCARA);
    double dif_local = 0;
    for (int li = 1; li <= nr; li++)
        for (int j = 0; j < N; j++)
            dif_local = max(dif_local, (double)fabsf(ref[(size_t)(li - 1) * N + j] - nxt[(size_t)li * N + j]));
    double dif;
    MPI_Reduce(&dif_local, &dif, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    free(ref);
    swap(cur, nxt);

    // ---------------- REGIÃO MEDIDA ----------------
    double comm = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    perf_cmd("enable\n");
    if (leitor) rapl_ler(rapl, ea);
    double t0 = MPI_Wtime();
    for (int it = 0; it < iters; it++) {
        comm += passo(cur, nxt, N, nr, rank, P, modo);
        swap(cur, nxt);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (leitor) rapl_ler(rapl, eb);
    perf_cmd("disable\n");
    // ------------------------------------------------

    // Energia: soma dos leitores (1 por nó). Se algum não tiver RAPL => -1
    double ej = leitor ? rapl_joules(rapl, ea, eb) : 0.0;
    double v[3] = { ej > 0 ? ej : 0.0, leitor ? 1.0 : 0.0, (leitor && ej >= 0) ? 1.0 : 0.0 };
    double vs[3];
    MPI_Reduce(v, vs, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double comm_max;
    MPI_Reduce(&comm, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Checksum global da imagem final (para conferir que todas as configurações P x T dão o mesmo resultado)
    double cs_local = 0;
    for (int li = 1; li <= nr; li++) {
        const float* r = cur + (size_t)li * N;
        double s = 0; for (int j = 0; j < N; j++) s += r[j];
        cs_local += s;
    }
    double cs;
    MPI_Reduce(&cs_local, &cs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total   = t1 - t0;
        double tempo   = total / iters;
        bool   okE     = (vs[1] > 0 && vs[2] == vs[1]);
        double energia = okE ? vs[0] / iters : -1;
        double potencia = okE ? vs[0] / total : -1;
        double mflops  = 17.0 * (double)N * N / tempo / 1e6;
        double comm_s  = comm_max / iters;

        printf("hibrido    N=%-5d P=%-2d T=%-2d modo=%d tempo=%.6f s  comm=%.6f s  energia=%.3f J  pot=%.2f W  MFLOPS=%.0f  maxdiff=%g  checksum=%.6e\n",
               N, P, T, modo, tempo, comm_s, energia, potencia, mflops, dif, cs);

        string caminho = "csv_time/6_conv_mpi_omp.csv";
        ifstream chk(caminho);
        bool existe = chk.is_open() && chk.peek() != ifstream::traits_type::eof();
        chk.close();
        ofstream log(caminho, ios::app);
        if (!existe) log << "N;P;T;modo;rep;iters;tempo_s;comm_s;energia_J;potencia_W;mflops;maxdiff_vs_serial;checksum\n";
        log.precision(6); log << fixed;
        log << N << ";" << P << ";" << T << ";" << modo << ";" << rep << ";" << iters << ";" << tempo << ";" << comm_s << ";"
            << energia << ";" << potencia << ";" << mflops << ";" << dif << ";" << cs << "\n";
    }

    free(cur); free(nxt);
    MPI_Comm_free(&nodecomm);
    MPI_Finalize();
    return 0;
}
