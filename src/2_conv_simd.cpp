#include "common.h"

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

int main(int argc, char** argv) {
    vector<int> tams;
    for (int a = 1; a < argc; a++) tams.push_back(atoi(argv[a]));
    if (tams.empty()) tams = {1024, 2048, 4096};

    // Abre o arquivo CSV único fora do loop para armazenar todos os tamanhos
    string caminho_csv = "csv_time/2_conv_simd.csv";

    // Verifica se o arquivo já existe e tem tamanho > 0
    ifstream checar_arquivo(caminho_csv);
    bool arquivo_existe = checar_arquivo.is_open() && checar_arquivo.peek() != ifstream::traits_type::eof();
    checar_arquivo.close();

    // Abre em modo append
    ofstream log(caminho_csv, ios::app);

    // Escreve o cabeçalho SOMENTE se o arquivo for novo/vazio
    if (!arquivo_existe) {
        log << "N;inicio_epoch;fim_epoch;tempo_s;maxdiff_vs_opencv\n";
    }

    for (int N : tams) {
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

        this_thread::sleep_for(chrono::seconds(2));
        testar("simd", conv_simd, in, out, N, log);
        
        free(in); free(out);
    }
    return 0;
}