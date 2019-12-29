######### KMeans #########
# k-means attempts to minimize the inertia when choosing clusters

##############################################################################

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)     # n_clusters: choose an "elbow: in the inertia plot

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

##############################################################################

## Visualization ##

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0] #  column 0 of points
ys = new_points[:,1] #  column 1 of points

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker = 'D', s=50) #D as a diamond

plt.show()

##############################################################################

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

##############################################################################

# Scatter plots_1
import matplotlib.pyplot as plt

xs = samples[:,0] # x-coordinates // Sepal.Length is 0th column
ys = samples[:,2] # y-coordinates // Petal.Length is in the 2nd column
plt.scatter(xs, ys, c=labels)
plt.show()

# Scatter plots_2
import matplotlib.pyplot as plt

xs = points[:,0]      #  column 0 of points
ys = points[:,1]      #  column 1 of points
plt.scatter(xs, ys)
plt.show()

##############################################################################

# Cross tabulation with pandas // provides great insights into which sort of samples are in which cluster.
print(species)

import pandas as pd
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)

# Measuring clustering quality = measuring how spread out the clusters are (lower is better)
# how far samples are from their centroids // Inertia measures clustering quality

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)       # after fit(), available as attribute inertia_



