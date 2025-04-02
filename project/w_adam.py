import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Fonksiyon: w parametrelerini bir dosyadan oku
def read_w_values(file_path):
    w_values = []
    with open(file_path, 'r') as file:
        for line in file:
            w_epoch = list(map(float, line.split()))
            w_values.append(w_epoch)
    return np.array(w_values)

# 1. Dosyalardan w değerlerini oku (Adam için)
w_adam_0 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_adam_0.txt')
w_adam_1 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_adam_1.txt')
w_adam_2 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_adam_2.txt')
w_adam_3 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_adam_3.txt')
w_adam_4 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_adam_4.txt')

# 2. T-SNE ile 2D'ye indirgeme
tsne = TSNE(n_components=2, random_state=42)
w_adam_0_2d = tsne.fit_transform(w_adam_0)
w_adam_1_2d = tsne.fit_transform(w_adam_1)
w_adam_2_2d = tsne.fit_transform(w_adam_2)
w_adam_3_2d = tsne.fit_transform(w_adam_3)
w_adam_4_2d = tsne.fit_transform(w_adam_4)

# 3. 2D verileri görselleştirme
plt.figure(figsize=(10, 8))

# Adam 0'dan başladığı halinin çizimi
plt.plot(w_adam_0_2d[:, 0], w_adam_0_2d[:, 1], label="Adam 0", color='black', marker='o', markersize=2)
for i in range(len(w_adam_0_2d)):
    if i % 10 == 0:
        plt.text(w_adam_0_2d[i, 0], w_adam_0_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Adam 1'den başlayarak çizim
plt.plot(w_adam_1_2d[:, 0], w_adam_1_2d[:, 1], label="Adam 1", color='darkred', marker='o', markersize=2)
for i in range(len(w_adam_1_2d)):
    if i % 10 == 0:
        plt.text(w_adam_1_2d[i, 0], w_adam_1_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Adam 2'den başlayarak çizim
plt.plot(w_adam_2_2d[:, 0], w_adam_2_2d[:, 1], label="Adam 2", color='red', marker='o', markersize=2)
for i in range(len(w_adam_2_2d)):
    if i % 10 == 0:
        plt.text(w_adam_2_2d[i, 0], w_adam_2_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Adam 3'den başlayarak çizim
plt.plot(w_adam_3_2d[:, 0], w_adam_3_2d[:, 1], label="Adam 3", color='tomato', marker='o', markersize=2)
for i in range(len(w_adam_3_2d)):
    if i % 10 == 0:
        plt.text(w_adam_3_2d[i, 0], w_adam_3_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Adam 4'den başlayarak çizim
plt.plot(w_adam_4_2d[:, 0], w_adam_4_2d[:, 1], label="Adam 4", color='darkorange', marker='o', markersize=2)
for i in range(len(w_adam_4_2d)):
    if i % 10 == 0:
        plt.text(w_adam_4_2d[i, 0], w_adam_4_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Grafiğin genel ayarları
plt.title('T-SNE Visualization of Adam Optimization Paths')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
