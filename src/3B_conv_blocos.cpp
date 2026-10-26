#include "common.h"

VEC void conv_blocos_vec(const float* __restrict in_, float* __restrict out_, int N, const float* m_, int BLOCK_SIZE) {
    const float* in  = (const float*)__builtin_assume_aligned(in_, 64);
    float*       out = (float*)__builtin_assume_aligned(out_, 64);
    
    const float m0=m_[0], m1=m_[1], m2=m_[2],
                m3=m_[3], m4=m_[4], m5=m_[5],
                m6=m_[6], m7=m_[7], m8=m_[8];

    for (int bi = 1; bi < N - 1; bi += BLOCK_SIZE) {
        for (int bj = 1; bj < N - 1; bj += BLOCK_SIZE) {
            int i_fim = std::min(bi + BLOCK_SIZE, N - 1);
            int j_fim = std::min(bj + BLOCK_SIZE, N - 1);

            for (int i = bi; i < i_fim; i++) {
                const float* a = in + (size_t)(i - 1) * N;
                const float* b = in + (size_t)i * N;
                const float* c = in + (size_t)(i + 1) * N;
                float* o = out + (size_t)i * N;

                #pragma omp simd
                for (int j = bj; j < j_fim; j++) {
                    o[j] = m0*a[j-1] + m1*a[j] + m2*a[j+1]
                         + m3*b[j-1] + m4*b[j] + m5*b[j+1]
                         + m6*c[j-1] + m7*c[j] + m8*c[j+1];
                }
            }
        }
    }
    bordas(in, out, N, m_);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int B = (argc > 2) ? atoi(argv[2]) : 64;

    string caminho_csv = "csv_time/3B_conv_blocos.csv";

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
    conv_blocos_vec(in, out, N, MASCARA, B);
    double t1 = agora();
    double t = t1 - t0;

    cv::Mat imagem_in(N, N, CV_32FC1, const_cast<float*>(in));
    cv::Mat imagem_out_opencv(N, N, CV_32FC1);
    cv::Mat mascara(3, 3, CV_32FC1, const_cast<float*>(MASCARA));
    cv::filter2D(imagem_in, imagem_out_opencv, -1, mascara, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
    float dif = maxdiff(reinterpret_cast<const float*>(imagem_out_opencv.data), out, (size_t)N * N);

    printf("blocos_vec (B=%-2d) N=%-5d tempo=%.6f s  maxdiff_vs_opencv=%g\n", B, N, t, dif);
    log.precision(6); log << fixed;
    log << N << ";" << B << ";" << t0 << ";" << t1 << ";" << t << ";" << dif << "\n";

    free(in); free(out);
    return 0;
}