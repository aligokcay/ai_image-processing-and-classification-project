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

# 1. Dosyalardan w değerlerini oku
w_sgd_0 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_sgd_0.txt')
w_sgd_1 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_sgd_1.txt')
w_sgd_2 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_sgd_2.txt')
w_sgd_3 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_sgd_3.txt')
w_sgd_4 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_sgd_4.txt')

# 2. T-SNE ile 2D'ye indirgeme
tsne = TSNE(n_components=2, random_state=42)
w_sgd_0_2d = tsne.fit_transform(w_sgd_0)
w_sgd_1_2d = tsne.fit_transform(w_sgd_1)
w_sgd_2_2d = tsne.fit_transform(w_sgd_2)
w_sgd_3_2d = tsne.fit_transform(w_sgd_3)
w_sgd_4_2d = tsne.fit_transform(w_sgd_4)

# 3. 2D verileri görselleştirme
plt.figure(figsize=(10, 8))

# SGD 0'dan başladığı halinin çizimi
plt.plot(w_sgd_0_2d[:, 0], w_sgd_0_2d[:, 1], label="SGD 0", color='black', marker='o', markersize=2)
for i in range(len(w_sgd_0_2d)):
    if i % 100 == 0:
        plt.text(w_sgd_0_2d[i, 0], w_sgd_0_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# SGD 1'den başlayarak çizim
plt.plot(w_sgd_1_2d[:, 0], w_sgd_1_2d[:, 1], label="SGD 1", color='darkolivegreen', marker='o', markersize=2)
for i in range(len(w_sgd_1_2d)):
    if i % 100 == 0:
        plt.text(w_sgd_1_2d[i, 0], w_sgd_1_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# SGD 2'den başlayarak çizim
plt.plot(w_sgd_2_2d[:, 0], w_sgd_2_2d[:, 1], label="SGD 2", color='green', marker='o', markersize=2)
for i in range(len(w_sgd_2_2d)):
    if i % 100 == 0:
        plt.text(w_sgd_2_2d[i, 0], w_sgd_2_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# SGD 3'den başlayarak çizim
plt.plot(w_sgd_3_2d[:, 0], w_sgd_3_2d[:, 1], label="SGD 3", color='lime', marker='o', markersize=2)
for i in range(len(w_sgd_3_2d)):
    if i % 100 == 0:
        plt.text(w_sgd_3_2d[i, 0], w_sgd_3_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# SGD 4'den başlayarak çizim
plt.plot(w_sgd_4_2d[:, 0], w_sgd_4_2d[:, 1], label="SGD 4", color='lightgreen', marker='o', markersize=2)
for i in range(len(w_sgd_4_2d)):
    if i % 100 == 0:
        plt.text(w_sgd_4_2d[i, 0], w_sgd_4_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Grafiğin genel ayarları
plt.title('T-SNE Visualization of Stochastic Gradient Descent Methods')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
