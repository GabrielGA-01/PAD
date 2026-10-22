#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <thread>

int main() {
    int largura = 16384, altura = 16384;

    // 1. Alocação de matrizes em ponto flutuante (CV_32F = float)
    cv::Mat imagem_in(altura, largura, CV_32F);
    cv::Mat imagem_out(altura, largura, CV_32F);
    cv::randu(imagem_in, cv::Scalar(0.0f), cv::Scalar(255.0f));

    // 2. Definir a máscara de convolução 3x3 (Blur Gaussiano)
    cv::Mat mascara = (cv::Mat_<float>(3, 3) <<
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    );

    // 3. Processamento com a biblioteca otimizada (Medição simples original)
    
    // Pausa inicial de 5 segundos para isolar o consumo base no log do HWiNFO
    std::cout << "A preparar... (Inicia em 5 segundos)\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Captura o timestamp exato do início do teste (sincronizado com o relógio do sistema/HWiNFO)
    auto inicio_sys = std::chrono::system_clock::now();
    auto tempo_inicio_t = std::chrono::system_clock::to_time_t(inicio_sys);
    auto ms_inicio = std::chrono::duration_cast<std::chrono::milliseconds>(
                        inicio_sys.time_since_epoch()) % 1000;

    auto inicio = std::chrono::high_resolution_clock::now();
    cv::filter2D(imagem_in, imagem_out, -1, mascara);
    auto fim = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> tempo = fim - inicio;
    std::cout << "Tempo OpenCV (único): " << tempo.count() << " segundos.\n";
    
    // Captura o timestamp exato do fim do teste
    auto fim_sys = std::chrono::system_clock::now();
    auto tempo_fim_t = std::chrono::system_clock::to_time_t(fim_sys);
    auto ms_fim = std::chrono::duration_cast<std::chrono::milliseconds>(
                    fim_sys.time_since_epoch()) % 1000;

    // Salva os horários de início e fim em um arquivo de texto para o script Python ler
    std::ofstream arquivo_tempo("exec_tempo.txt");
    
    auto tm_ini = *std::localtime(&tempo_inicio_t);
    arquivo_tempo << std::put_time(&tm_ini, "%d.%m.%Y %H:%M:%S") 
                  << '.' << std::setfill('0') << std::setw(3) << ms_inicio.count() << "\n";

    auto tm_fim = *std::localtime(&tempo_fim_t);
    arquivo_tempo << std::put_time(&tm_fim, "%d.%m.%Y %H:%M:%S") 
                  << '.' << std::setfill('0') << std::setw(3) << ms_fim.count() << "\n";
    
    arquivo_tempo.close();

    // Pausa final de 5 segundos para isolar o retorno ao consumo base
    std::cout << "Teste concluído. Tempos gravados em exec_tempo.txt. A finalizar...\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));
    // Fim

    return 0;
}