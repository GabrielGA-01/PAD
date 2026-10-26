#include "common.h"

#pragma GCC push_options
#pragma GCC optimize ("O0")
__attribute__((optimize("O0", "no-tree-vectorize")))
NOVEC void conv_colunas(const float* in, float* out, int N, const float* m) {
    for (int j = 1; j < N - 1; j++)
        for (int i = 1; i < N - 1; i++)
            out[(size_t)i * N + j] = STENCIL(in, N, i, j, m);
    bordas(in, out, N, m);
}
#pragma GCC pop_options

int main(int argc, char** argv) {
    vector<int> tams;
    for (int a = 1; a < argc; a++) tams.push_back(atoi(argv[a]));
    if (tams.empty()) tams = {1024, 2048, 4096};

    // Abre o arquivo CSV único fora do loop para armazenar todos os tamanhos
    string caminho_csv = "csv_time/1B_conv_colunas.csv";

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
        testar("colunas", conv_colunas, in, out, N, log);
        
        free(in); free(out);
    }
    return 0;
}