#include "common.h"

int main(int argc, char** argv) {
    vector<int> tams;
    for (int a = 1; a < argc; a++) tams.push_back(atoi(argv[a]));
    if (tams.empty()) tams = {1024, 2048, 4096};

    string caminho_csv = "csv_time/4_conv_opencv.csv";

    // Verifica se o arquivo CSV já existe e tem conteúdo
    ifstream checar_arquivo(caminho_csv);
    bool arquivo_existe = checar_arquivo.is_open() && checar_arquivo.peek() != ifstream::traits_type::eof();
    checar_arquivo.close();

    // Abre em modo append para acumular os resultados
    ofstream log(caminho_csv, ios::app);
    if (!arquivo_existe) {
        log << "N;inicio_epoch;fim_epoch;tempo_s;maxdiff_vs_opencv\n";
    }

    for (int N : tams) {
        size_t bytes = ((size_t)N * N * sizeof(float) + 63) / 64 * 64;
        float* in  = (float*)aligned_alloc(64, bytes);
        float* out = (float*)aligned_alloc(64, bytes);
        
        mt19937 gen(42);
        uniform_real_distribution<float> dis(0.0f, 255.0f);
        for (size_t k = 0; k < (size_t)N * N; k++) in[k] = dis(gen);

        this_thread::sleep_for(chrono::seconds(1));

        cv::Mat imagem_in(N, N, CV_32FC1, in);
        cv::Mat imagem_out(N, N, CV_32FC1, out);
        cv::Mat mascara(3, 3, CV_32FC1, const_cast<float*>(MASCARA));

        double t0 = agora();
        cv::filter2D(imagem_in, imagem_out, -1, mascara, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
        double t1 = agora();
        double t = t1 - t0;

        // Como o OpenCV é a própria referência, a diferença dele para ele mesmo é 0.0
        float dif = maxdiff(out, out, (size_t)N * N);

        printf("opencv N=%-5d tempo=%.6f s  maxdiff_vs_opencv=%g\n", N, t, dif);
        log.precision(6); log << fixed;
        log << N << ";" << t0 << ";" << t1 << ";" << t << ";" << dif << "\n";

        free(in); free(out);
    }
    return 0;
}