#include "common.h"

NOVEC void conv_blocos(const float* in, float* out, int N, const float* m, int BLOCK_SIZE) {
    for (int bi = 1; bi < N - 1; bi += BLOCK_SIZE) {
        for (int bj = 1; bj < N - 1; bj += BLOCK_SIZE) {
            int i_fim = std::min(bi + BLOCK_SIZE, N - 1);
            int j_fim = std::min(bj + BLOCK_SIZE, N - 1);

            for (int i = bi; i < i_fim; i++) {
                for (int j = bj; j < j_fim; j++) {
                    out[(size_t)i * N + j] = STENCIL(in, N, i, j, m);
                }
            }
        }
    }
    bordas(in, out, N, m);
}

int main(int argc, char** argv) {
    // Recebe N e B via argumento (com fallbacks se executado manualmente)
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int B = (argc > 2) ? atoi(argv[2]) : 64;

    string caminho_csv = "csv_time/3A_conv_blocos.csv";

    ifstream checar_arquivo(caminho_csv);
    bool arquivo_existe = checar_arquivo.is_open() && checar_arquivo.peek() != ifstream::traits_type::eof();
    checar_arquivo.close();

    ofstream log(caminho_csv, ios::app);
    if (!arquivo_existe) {
        log << "N;bloco;inicio_epoch;fim_epoch;tempo_s;maxdiff_vs_opencv\n";
    }

    size_t bytes = ((size_t)N * N * sizeof(float) + 63) / 64 * 64;
    float* in  = (float*)aligned_alloc(64, bytes);
    float* out = (float*)aligned_alloc(64, bytes);
    
    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 255.0f);
    for (size_t k = 0; k < (size_t)N * N; k++) in[k] = dis(gen);

    this_thread::sleep_for(chrono::seconds(1));
    
    double t0 = agora();
    conv_blocos(in, out, N, MASCARA, B);
    double t1 = agora();
    double t = t1 - t0;

    cv::Mat imagem_in(N, N, CV_32FC1, const_cast<float*>(in));
    cv::Mat imagem_out_opencv(N, N, CV_32FC1);
    cv::Mat mascara(3, 3, CV_32FC1, const_cast<float*>(MASCARA));
    cv::filter2D(imagem_in, imagem_out_opencv, -1, mascara, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
    float dif = maxdiff(reinterpret_cast<const float*>(imagem_out_opencv.data), out, (size_t)N * N);

    printf("blocos (B=%-2d) N=%-5d tempo=%.6f s  maxdiff_vs_opencv=%g\n", B, N, t, dif);
    log.precision(6); log << fixed;
    log << N << ";" << B << ";" << t0 << ";" << t1 << ";" << t << ";" << dif << "\n";

    free(in); free(out);
    return 0;
}