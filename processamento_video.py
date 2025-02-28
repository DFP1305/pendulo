from functools import partial
import cv2 as cv
import numpy as np
import os

# Relação pixel para cm (ajuste com base nos experimentos)
PIXEL_POR_CM = 8 / 260  # cm/pixel
VIDEO_FILE = "videopendulo1.mp4"  # Nome do vídeo
OUTPUT_DIR = "./output/"
FRAME_RATE = 1 / 10  # 10 FPS

# Comando para criar a pasta de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função para deixar frame em preto e branco
def binariza(image):
    escala_cinza = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Converte para escala de cinza
    remove_ruido = cv.GaussianBlur(escala_cinza, (5, 5), 0)  # Suaviza para remover ruído
    _, preto_branco = cv.threshold(remove_ruido, 100, 255, cv.THRESH_BINARY_INV)  # Binarização
    return preto_branco

# Função para calcular o centro de massa
def centro_de_massa(image):
    # Calcula os momentos da imagem já tratada
    momentos = cv.moments(image)
    
    # Evita divisão por zero se a soma dos momentos for zero
    if momentos["m00"] == 0:
        raise ValueError("Não foi possível identificar o centro de massa. A imagem está vazia ou sem objeto.")
    
    # Calcula o centro de massa (cX, cY)
    cX = int(momentos["m10"] / momentos["m00"])

    return cX  # Retorna apenas a coordenada horizontal (1D)

# Função para processar um frame e extrair informações de tempo e posição
def processa_frame(frame, seg, tempos, posicoes):
    try:
        binariza_frame = binariza(frame)
        cX = centro_de_massa(binariza_frame)
        tempos.append(f"{seg}\n")
        posicoes.append(f"{cX * PIXEL_POR_CM}\n")
    except ValueError as e:
        print(f"Erro ao processar frame no tempo {seg}s: {e}")

# Função principal otimizada para processar o vídeo inteiro
def main():
    # Tenta abrir o vídeo usando o OpenCV
    vidcap = cv.VideoCapture(VIDEO_FILE)
    
    if not vidcap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {VIDEO_FILE}")
        return

    seg = 0  # Inicia o tempo
    contador_frame = 0  # Contador de frames processados
    intervalo_frame = int(vidcap.get(cv.CAP_PROP_FPS) * FRAME_RATE)  # Define o intervalo entre frames

    # Listas para armazenar os dados de tempo e posição
    tempos = []
    posicoes = []

    # Laço principal para processar o vídeo frame a frame
    ret = True  # Variável de controle do laço

    while ret:  # Continua enquanto o frame for lido corretamente
        # Tenta ler o próximo frame
        ret, frame = vidcap.read()

        # Processa o frame a cada intervalo de tempo definido
        if ret and contador_frame % intervalo_frame == 0:
            processa_frame(frame, seg, tempos, posicoes)
            seg = round(seg + FRAME_RATE, 2)  # Atualiza o tempo

        contador_frame += 1  # Incrementa o contador de frames

    vidcap.release()

    # Salva os dados extraídos de tempo e posição em arquivos de texto
    with open(os.path.join(OUTPUT_DIR, "tempos.txt"), "w") as f_time:
        f_time.writelines(tempos)

    with open(os.path.join(OUTPUT_DIR, "espacos.txt"), "w") as f_space:
        f_space.writelines(posicoes)

if __name__ == "__main__":
    main()