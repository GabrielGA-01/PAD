# Execução do código
Entre em /src usando 'cd src'
Execute o arquivo './executar.sh' para criar limpar as pastas de resultados, compilar e executar os códigos
A pasta objects em os arquivos executáveis gerados de cada um dos códigos
A pasta csv_time contém os tempos de execução para cada tamanho de bloco em cada um dos códigos
A pasta perfs contém informações sobre a cache, número de ciclos e instruções referentes às execuções de cada um dos códigos em cada umd dos cenários

# Coleta dos dados de energia
É preciso utilizar o programa HWiNFO (Windows) para coletar o log de consumo de energia do computado. Esse log dever ser salvo com o nome 'energia.CSV'.

Após ter o log é preciso copiar ele para o diretório 'src/.

# Interpretação dos resultados

Instale as dependências do Python utilizando 'pip intall -r requirements.txt'

Após isso, é preciso executar o código de 'script_aux.py', que irá ler o log gerado e cruzar informações com o tempo de início e de fim, para coletar informações de energia e utilizar os perfs gerados para gerar as métricas de cada um dos cenários.