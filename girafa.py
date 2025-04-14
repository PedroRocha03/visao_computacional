# pip install opencv-python

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importa e converte para RGB
img = cv2.imread('./GIRAFA.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a / 2 * 1.95, a, cv2.THRESH_BINARY_INV)

# Kernel para operações morfológicas
tamanhoKernel = 3
kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)
thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Filtro de ruído (blurring)
img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel, tamanhoKernel))

# Detecção de bordas com Canny (sem e com blur)
edges_gray = cv2.Canny(image=img_gray, threshold1=a/2, threshold2=a/2)
edges_blur = cv2.Canny(image=img_blur, threshold1=a/2, threshold2=a/2)

# Contornos
contours, hierarchy = cv2.findContours(
    image=thresh,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_SIMPLE
)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                         color=(255, 0, 0), thickness=2)

# Lista de imagens intermediárias
imagens = [img, img_blur, img_gray, edges_gray, edges_blur, thresh, thresh_open]

# Tamanho do grid para as imagens pequenas
formatoX = math.ceil(len(imagens) ** 0.5)
formatoY = math.ceil(len(imagens) / formatoX)

# Figura 1: imagens intermediárias
plt.figure(figsize=(15, 10))
for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i], 'gray')
    plt.xticks([]), plt.yticks([])
plt.suptitle("Imagens Intermediárias", fontsize=16)
plt.tight_layout()
plt.show()

# Figura 2: imagem final em destaque
plt.figure(figsize=(10, 10))
plt.imshow(final)
plt.title("Imagem Final com Contornos Destacados", fontsize=18)
plt.xticks([]), plt.yticks([])
plt.show()