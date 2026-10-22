#!/bin/bash

# Para o script imediatamente se ocorrer algum erro
set -e

# Nomes dos arquivos
SRC_FILE="script_opencv.cpp"
OUT_FILE="script_opencv"

echo "==> Verificando dependências do OpenCV..."
if pkg-config --exists opencv4; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)
elif pkg-config --exists opencv; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
else
    echo "Erro: OpenCV não encontrado via pkg-config. Certifique-se de que o OpenCV está instalado."
    exit 1
fi

echo "==> Compilando $SRC_FILE..."
g++ -std=c++17 "$SRC_FILE" -o "$OUT_FILE" $OPENCV_FLAGS

echo "==> Compilação concluída com sucesso!"
echo "==> Executando $OUT_FILE..."
echo "----------------------------------------"

# Executa o programa gerado
./"$OUT_FILE"