#include "common.h"

NOVEC void conv_linhas(const float* in, float* out, int N, const float* m) {
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            out[(size_t)i * N + j] = STENCIL(in, N, i, j, m);
    bordas(in, out, N, m);
}

int main(int argc, char** argv) {
    vector<int> tams;
    for (int a = 1; a < argc; a++) tams.push_back(atoi(argv[a]));
    if (tams.empty()) tams = {1024, 2048, 4096};

    // Abre o arquivo CSV único fora do loop para armazenar todos os tamanhos
    string caminho_csv = "csv_time/1A_conv_linhas.csv";

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
        
        mt19937 gen(42);
        uniform_real_distribution<float> dis(0.0f, 255.0f);
        for (size_t k = 0; k < (size_t)N * N; k++) in[k] = dis(gen);

        this_thread::sleep_for(chrono::seconds(2));
        testar("linhas", conv_linhas, in, out, N, log);
        
        free(in); free(out);
    }
    return 0;
}