# samples
print(samples)

from sklearn.cluster import KMeans
models = KMeans(n_clusters=3)

model.fit(samples)
KMeans(algorithm='auto',...)

labels = model.predict(samples)

# new_samples
print(new_samples)

new_labels = model.predict(new_samples)

print(new_labels)

# Scatter plots
import matplotlib.pyplot as plt

xs = samples[:,0] # x-coordinates // Sepal.Length is 0th column

ys = samples[:,2] # y-coordinates // Petal.Length is in the 2nd column

plt.scatter(xs, ys, c=labels)

plt.show()
