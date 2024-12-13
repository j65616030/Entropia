import cv2
import numpy as np

# Leer el video
cap = cv2.VideoCapture('video.mp4')

# Inicializar las variables
frames = []
entropias = []

# Leer 3 frames consecutivos
for i in range(3):
    ret, frame = cap.read()
    frames.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    entropy = np.zeros(gray.shape)
    for j in range(gray.shape[0]):
        for k in range(gray.shape[1]):
            entropy[j, k] = np.log2(gray[j, k] + 1)
    entropias.append(entropy)

# Calcular la entropía promedio en los 3 frames
entropy_promedio = np.mean(entropias, axis=0)

# Seleccionar los píxeles con mayor entropía como centroides
centroides = []
for i in range(entropy_promedio.shape[0]):
    for j in range(entropy_promedio.shape[1]):
        if entropy_promedio[i, j] > np.mean(entropy_promedio) + np.std(entropy_promedio):
            centroides.append((i, j))

# Mostrar los centroides
for frame in frames:
    for centroide in centroides:
        cv2.circle(frame, centroide, 5, (0, 255, 0), -1)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
