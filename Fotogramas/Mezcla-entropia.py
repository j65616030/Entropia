import cv2
import numpy as np

# Leer los videos
cap_base = cv2.VideoCapture('video_base.mp4')
cap_high_entropy = cv2.VideoCapture('video_high_entropy.mp4')

# Inicializar el video resultante
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap_base.get(cv2.CAP_PROP_FPS)
size = (int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('video_resultante.mp4', fourcc, fps, size)

while True:
    ret_base, frame_base = cap_base.read()
    ret_high_entropy, frame_high_entropy = cap_high_entropy.read()
    if not ret_base or not ret_high_entropy:
        break

    # Extraer la alta entropía
    gray_high_entropy = cv2.cvtColor(frame_high_entropy, cv2.COLOR_BGR2GRAY)
    entropy_high_entropy = np.zeros(gray_high_entropy.shape)
    for i in range(gray_high_entropy.shape[0]):
        for j in range(gray_high_entropy.shape[1]):
            entropy_high_entropy[i, j] = np.log2(gray_high_entropy[i, j] + 1)

    # Sumar la alta entropía al video base
    frame_resultante = frame_base.copy()
    for i in range(frame_resultante.shape[0]):
        for j in range(frame_resultante.shape[1]):
            if entropy_high_entropy[i, j] > np.mean(entropy_high_entropy) + np.std(entropy_high_entropy):
                frame_resultante[i, j, :] = np.clip(frame_resultante[i, j, :] + 10, 0, 255)

    # Escribir el frame resultante en el video resultante
    out.write(frame_resultante)

# Liberar recursos
cap_base.release()
cap_high_entropy.release()
out.release()
cv2.destroyAllWindows()
