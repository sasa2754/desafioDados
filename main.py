import cv2 as cv
import numpy as np

# Carregar a imagem do labirinto em escala de cinza
labirinto = cv.imread('labirinto.jpeg', cv.IMREAD_GRAYSCALE)

# Converter a imagem para colorida para que as cores dos contornos apareçam
labirinto_colorido = cv.cvtColor(labirinto, cv.COLOR_GRAY2BGR)

# Aplicar um filtro GaussianBlur para suavizar a imagem
blur = cv.medianBlur(labirinto, 5)

# Aplicar thresholding para binarizar a imagem (invertida para que os pontos sejam brancos)
_, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Encontrar todos os contornos na imagem
contornos, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Iterar sobre os contornos para detectar pontos
for contorno in contornos:
    area = cv.contourArea(contorno)
    # Filtrar contornos pequenos demais ou grandes demais que não sejam pontos
    if 300 < area < 500:  # Ajuste esses valores conforme necessário
        # Obter o centro do contorno
        (x, y), radius = cv.minEnclosingCircle(contorno)
        centro = (int(x), int(y))
        raio = int(radius)
        
        # Desenhar um círculo ao redor dos pontos detectados em vermelho
        cv.circle(labirinto_colorido, centro, raio, (0, 0, 255), 2)

# Mostrar a imagem com os pontos marcados em vermelho
cv.imshow('Pontos detectados', labirinto_colorido)
cv.waitKey(0)
cv.destroyAllWindows()
