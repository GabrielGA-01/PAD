#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cstdlib>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    string filename = "matriz_in_N" + to_string(N) + ".bin";
    size_t total_elementos = (size_t)N * N;

    vector<float> in(total_elementos);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk_size = total_elementos / nthreads;
        size_t start = tid * chunk_size;
        size_t end = (tid == nthreads - 1) ? total_elementos : start + chunk_size;

        mt19937 gen(42);
        gen.discard(start);
        uniform_real_distribution<float> dis(0.0f, 255.0f);

        for (size_t k = start; k < end; k++) {
            in[k] = dis(gen);
        }
    }

    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<char*>(in.data()), total_elementos * sizeof(float));
    file.close();
    return 0;
}
