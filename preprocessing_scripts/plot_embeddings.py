import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

np.random.seed(1337)

data_dir = "/media/user/data/complete_synth/"

npy_files = []
counter = 0
for path, subdirs, files in os.walk(data_dir):
    internal_counter = 0
    for name in files:
        extension = name[-3:]
        if extension == "npy":
            internal_counter += 1
            full_file_path = path + "/" + name
            npy_files.append((full_file_path, counter))

    # Check if any matches, this means speaker
    if internal_counter > 0:
        counter += 1

print("All np files found")
print(len(npy_files))
print(counter)

new_counter = 0

array_list = []
speakers = []

for (fpath, speaker) in npy_files:
    embed = np.load(fpath)
    array_list.append(embed)
    speakers.append(speaker)
    new_counter += 1

    if new_counter == 1200:
        break

X = np.array(array_list)
X_lowdim = PCA(n_components=2).fit_transform(X)

plt.figure(1, figsize=(4, 3))
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.scatter(X_lowdim[:, 0], X_lowdim[:, 1], c=speakers, cmap='tab20c')
plt.show()

X_lowdim = TSNE(n_components=2).fit_transform(X)

plt.figure(1, figsize=(4, 3))
plt.xlabel("1")
plt.ylabel("2")
plt.scatter(X_lowdim[:, 0], X_lowdim[:, 1], c=speakers, cmap='tab20c')
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
new_X = np.asarray([X_lowdim[:, 0], X_lowdim[:, 1]]).T
kmeans.fit(new_X)
y_kmeans = kmeans.predict(new_X)
plt.scatter(new_X[:, 0], new_X[:, 1], c=y_kmeans, s=50, cmap='tab20c')
centers = kmeans.cluster_centers_
plt.scatter(centers[0, 0], centers[0, 1], c='black', s=200, alpha=0.5)
plt.scatter(centers[1, 0], centers[1, 1], c='red', s=200, alpha=0.5)
plt.scatter(centers[2, 0], centers[2, 1], c='blue', s=200, alpha=0.5)
plt.scatter(centers[3, 0], centers[3, 1], c='yellow', s=200, alpha=0.5)
plt.scatter(centers[4, 0], centers[4, 1], c='green', s=200, alpha=0.5)
plt.show()


from collections import Counter

for i in range(5):
    indices = np.where(y_kmeans == i)[0]
    all_items = []
    for index in indices:
        all_items.append(npy_files[index])
    speakers = map(lambda x: x[1], all_items)
    majority_speaker = Counter(speakers).most_common(1)[0][0]
    print(majority_speaker)
    for item in all_items:
        if item[1] != majority_speaker:
            print(item)

# 366, 387, 385 386