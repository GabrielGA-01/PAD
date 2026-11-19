#include "common.h"
#include <vector>

// Submatriz de um bloco: entrada com halo de 1 elemento e saída sem halo.
struct Bloco {
    int bi, bj;               // origem do bloco na matriz global
    int h, w;                 // altura e largura úteis (sem halo)
    std::vector<float> ent;   // (h+2) x (w+2)
    std::vector<float> sai;   // h x w
};

// ---- Divisão da matriz em blocos (FORA da medição de tempo) ----
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
            b.ent.assign((size_t)(b.h + 2) * W, 0.0f);
            b.sai.assign((size_t)b.h * b.w, 0.0f);

            // copia bloco + halo (linhas bi-1..bi+h, colunas bj-1..bj+w)
            for (int li = 0; li < b.h + 2; li++) {
                for (int lj = 0; lj < W; lj++) {
                    b.ent[(size_t)li * W + lj] =
                        in[(size_t)(bi - 1 + li) * N + (bj - 1 + lj)];
                }
            }
            blocos.push_back(std::move(b));
        }
    }
    return blocos;
}

// ---- Convolução sobre as submatrizes (ÚNICA parte cronometrada) ----
#pragma GCC push_options
#pragma GCC optimize ("O0")
__attribute__((optimize("O0", "no-tree-vectorize")))
NOVEC void conv_blocos(std::vector<Bloco>& blocos, const float* in, float* out,
                       int N, const float* m) {
    for (size_t k = 0; k < blocos.size(); k++) {
        Bloco& b = blocos[k];
        int W = b.w + 2;
        for (int li = 1; li <= b.h; li++) {
            for (int lj = 1; lj <= b.w; lj++) {
                b.sai[(size_t)(li - 1) * b.w + (lj - 1)] =
                    STENCIL(b.ent.data(), W, li, lj, m);
            }
        }
    }
    bordas(in, out, N, m);
}
#pragma GCC pop_options

int main(int argc, char** argv) {
    int iteracao = iteracao_atual();
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int B = (argc > 2) ? atoi(argv[2]) : 64;

    string caminho_csv = nome_arquivo_csv("3A_conv_blocos", iteracao);
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
    conv_blocos(blocos, in, out, N, MASCARA);

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

    printf("blocos (B=%-2d) N=%-5d divisao=%.6f s stencil=%.6f s juncao=%.6f s total=%.6f s maxdiff_vs_opencv=%g\n",
           B, N, t_divisao, t_stencil, t_juncao, t_total, dif);

    log.precision(6); log << fixed;
    log << iteracao << ";" << N << ";" << B << ";"
        << t0 << ";" << t1 << ";" << t2 << ";" << t3 << ";"
        << t_divisao << ";" << t_stencil << ";" << t_juncao << ";" << t_total << ";"
        << dif << "\n";

    free(in); free(out);
    return 0;
}