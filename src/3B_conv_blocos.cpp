#include "common.h"
#include <vector>

// Submatriz de um bloco. Os buffers são alocados com alinhamento de 64 bytes
// (em vez de std::vector) para que o __builtin_assume_aligned continue válido.
struct Bloco {
    int bi, bj;     // origem do bloco na matriz global
    int h, w;       // altura e largura úteis (sem halo)
    float* ent;     // (h+2) x (w+2), com halo
    float* sai;     // h x w
};

static float* alocar_alinhado(size_t n_floats) {
    size_t bytes = (n_floats * sizeof(float) + 63) / 64 * 64;
    float* p = (float*)aligned_alloc(64, bytes);
    memset(p, 0, bytes);
    return p;
}

// ---- Divisão da matriz em blocos ----
std::vector<Bloco> dividir_blocos(const float* in, int N, int BLOCK_SIZE) {
    std::vector<Bloco> blocos;
    for (int bi = 1; bi < N - 1; bi += BLOCK_SIZE) {
        for (int bj = 1; bj < N - 1; bj += BLOCK_SIZE) {
            Bloco b;
            b.bi = bi;
            b.bj = bj;
            b.h = std::min(bi + BLOCK_SIZE, N - 1) - bi;
            b.w = std::min(bj + BLOCK_SIZE, N - 1) - bj;
            int W = b.w + 2;
            b.ent = alocar_alinhado((size_t)(b.h + 2) * W);
            b.sai = alocar_alinhado((size_t)b.h * b.w);

            // copia bloco + halo (linhas bi-1..bi+h, colunas bj-1..bj+w)
            for (int li = 0; li < b.h + 2; li++) {
                memcpy(b.ent + (size_t)li * W,
                       in + (size_t)(bi - 1 + li) * N + (bj - 1),
                       (size_t)W * sizeof(float));
            }
            blocos.push_back(b);
        }
    }
    return blocos;
}

// ---- Convolução vetorizada sobre as submatrizes ----
VEC void conv_blocos_vec(const Bloco* blocos, size_t nb,
                         const float* in_, float* out_, int N, const float* m_) {
    const float m0=m_[0], m1=m_[1], m2=m_[2],
                m3=m_[3], m4=m_[4], m5=m_[5],
                m6=m_[6], m7=m_[7], m8=m_[8];

    for (size_t k = 0; k < nb; k++) {
        const Bloco& blk = blocos[k];
        const int h = blk.h, w = blk.w, W = w + 2;

        const float* __restrict ent = (const float*)__builtin_assume_aligned(blk.ent, 64);
        float*       __restrict sai = (float*)__builtin_assume_aligned(blk.sai, 64);

        for (int i = 0; i < h; i++) {
            const float* a = ent + (size_t)i * W;        // linha acima
            const float* b = ent + (size_t)(i + 1) * W;  // linha atual
            const float* c = ent + (size_t)(i + 2) * W;  // linha abaixo
            float* o = sai + (size_t)i * w;

            #pragma omp simd
            for (int j = 0; j < w; j++) {
                o[j] = m0*a[j] + m1*a[j+1] + m2*a[j+2]
                     + m3*b[j] + m4*b[j+1] + m5*b[j+2]
                     + m6*c[j] + m7*c[j+1] + m8*c[j+2];
            }
        }
    }
    bordas(in_, out_, N, m_);
}

void liberar_blocos(std::vector<Bloco>& blocos) {
    for (size_t k = 0; k < blocos.size(); k++) {
        free(blocos[k].ent);
        free(blocos[k].sai);
    }
    blocos.clear();
}

int main(int argc, char** argv) {
    int iteracao = iteracao_atual();
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int B = (argc > 2) ? atoi(argv[2]) : 64;

    string caminho_csv = nome_arquivo_csv("3B_conv_blocos", iteracao);

    ifstream checar_arquivo(caminho_csv);
    bool arquivo_existe = checar_arquivo.is_open() && checar_arquivo.peek() != ifstream::traits_type::eof();
    checar_arquivo.close();

    ofstream log(caminho_csv, ios::app);
    if (!arquivo_existe) {
        log << "iteracao;N;bloco;t0_epoch;t1_epoch;t2_epoch;t3_epoch;"
               "tempo_divisao_s;tempo_stencil_s;tempo_juncao_s;tempo_total_s;"
               "maxdiff_vs_opencv\n";
    }

    size_t bytes = ((size_t)N * N * sizeof(float) + 63) / 64 * 64;
    float* in  = (float*)aligned_alloc(64, bytes);
    float* out = (float*)aligned_alloc(64, bytes);

    // Lê a matriz pré-gerada pelo script .sh
    string filename = "matriz_in_N" + to_string(N) + ".bin";
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Erro: Arquivo " << filename << " não encontrado!" << endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(in), (size_t)N * N * sizeof(float));
    file.close();

    this_thread::sleep_for(chrono::seconds(1));

    // ---- Quatro instantes de medição ----
    double t0 = agora();                                   // 1) antes da quebra da matriz
    std::vector<Bloco> blocos = dividir_blocos(in, N, B);

    double t1 = agora();                                   // 2) antes da aplicação do stencil
    conv_blocos_vec(blocos.data(), blocos.size(), in, out, N, MASCARA);

    double t2 = agora();                                   // 3) antes da junção em uma só
    montar_saida_blocos(blocos, out, N);

    double t3 = agora();                                   // 4) ao final

    double t_divisao = t1 - t0;
    double t_stencil = t2 - t1;
    double t_juncao  = t3 - t2;
    double t_total   = t3 - t0;

    // Validação contra OpenCV (fora de qualquer medição)
    cv::Mat imagem_in(N, N, CV_32FC1, const_cast<float*>(in));
    cv::Mat imagem_out_opencv(N, N, CV_32FC1);
    cv::Mat mascara(3, 3, CV_32FC1, const_cast<float*>(MASCARA));
    cv::filter2D(imagem_in, imagem_out_opencv, -1, mascara, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
    float dif = maxdiff(reinterpret_cast<const float*>(imagem_out_opencv.data), out, (size_t)N * N);

    printf("blocos_vec (B=%-2d) N=%-5d divisao=%.6f s stencil=%.6f s juncao=%.6f s total=%.6f s maxdiff_vs_opencv=%g\n",
           B, N, t_divisao, t_stencil, t_juncao, t_total, dif);

    log.precision(6); log << fixed;
    log << iteracao << ";" << N << ";" << B << ";"
        << t0 << ";" << t1 << ";" << t2 << ";" << t3 << ";"
        << t_divisao << ";" << t_stencil << ";" << t_juncao << ";" << t_total << ";"
        << dif << "\n";

    liberar_blocos(blocos);
    free(in); free(out);
    return 0;
}