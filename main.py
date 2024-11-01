import cv2 as cv
import numpy as np

# Carregar a imagem do labirinto com os contornos em vermelho
labirinto = cv.imread('labirintoContornado2.jpg')

# Verificar se a imagem foi carregada corretamente
if labirinto is None:
    print("Erro ao carregar a imagem.")
    exit()


# Converter a imagem para escala de cinza
labirinto_gray = cv.cvtColor(labirinto, cv.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.uint8)
labMorph = cv.morphologyEx(labirinto_gray, cv.MORPH_OPEN, kernel)

# cv.imshow('Labirinto com morfologia', labMorph)

# Aplicar um filtro para suavizar a imagem
blur = cv.medianBlur(labMorph, 5)

# Usar Canny para detectar bordas
bordas = cv.Canny(blur, 50, 150)

# Criar a máscara - ajuste os limites conforme necessário
mask = cv.inRange(labMorph, 100, 200)  # Ajuste esses valores conforme a iluminação da imagem

# Encontrar todos os contornos na máscara
contornos, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# cv.imshow('Bordas', bordas)

# Imprimir a quantidade de contornos encontrados
print(f"Total de contornos encontrados: {len(contornos)}")

# Iterar sobre os contornos para detectar dados
for contorno in contornos:
    area = cv.contourArea(contorno)
    
    # Limite para tentar pegar todos os dados
    if 200 < area < 9000:  # Aumentei o limite mínimo da área para evitar detecção de pequenos contornos
        # Obter o retângulo delimitador do dado
        x, y, w, h = cv.boundingRect(contorno)

        # Verificar a proporção para considerar se é um dado
        aspect_ratio = float(w) / h
        if (0.5 < aspect_ratio < 2):  # Filtrar por proporção
            # Desenhar o retângulo delimitador em verde
            cv.rectangle(labirinto, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calcular o centro do dado
            cx = int(x + w // 2)
            cy = int(y + h // 2)
            
            # Desenhar o centro do dado em amarelo
            cv.circle(labirinto, (cx, cy), 5, (0, 255, 255), -1)

            # Cortar a região do dado da imagem original para detectar bolinhas
            roi = labirinto_gray[y:y+h, x:x+w]
            
            # Usar HoughCircles para detectar as bolinhas dentro da ROI
            bolinhas = cv.HoughCircles(roi, cv.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=30, minRadius=1, maxRadius=15)
            
            # Verificar se alguma bolinha foi encontrada
            if bolinhas is not None:
                bolinhas = np.uint16(np.around(bolinhas))
                for i in bolinhas[0, :]:
                    # Desenhar uma bolinha azul em cima da bolinha do dado
                    cv.circle(labirinto, (x + i[0], y + i[1]), 5, (255, 0, 0), -1)

# Mostrar a imagem com os dados detectados
cv.imshow('Dados detectados', labirinto)
cv.waitKey(0)
cv.destroyAllWindows()
