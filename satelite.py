import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def processar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a/2*1.3, a, cv2.THRESH_BINARY_INV)

    tamanhoKernel = 7
    kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_close = cv2.morphologyEx(thresh_open, cv2.MORPH_CLOSE, kernel)

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 3)

    lower_threshold = 30
    upper_threshold = 40
    edges = cv2.Canny(image=img_blur, threshold1=lower_threshold, threshold2=upper_threshold)

    kernel_edges = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel_edges, iterations=1)

    contours, _ = cv2.findContours(image=edges.copy(),
                                   mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, contourIdx=-1,
                     color=(255, 0, 0), thickness=3)

    return img_contours, [img, img_gray, thresh, thresh_close, edges]

imagens = ['SATELITE.jpeg']

for imagem in imagens:
    img_final, imagens_intermediarias = processar_imagem(imagem)
    
    if img_final is None:
        continue
    
    plt.figure(figsize=(15, 10))
    
    for i, img in enumerate(imagens_intermediarias):
        plt.subplot(2, 3, i+1)
        plt.imshow(img, cmap='gray' if len(img.shape)==2 else None)
        plt.title(['Original', 'Escala Cinza', 'Binarizada', 'Morfologia', 'Bordas Canny'][i])
        plt.axis('off')
    
    plt.suptitle(f'Processamento - {imagem}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_final)
    plt.title(f'Contornos Detectados - {imagem}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()