from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

magic = pd.read_csv('twoGaussians.csv', header=None)

# Get data in a format that fits sklearn
magic[11] = pd.Categorical(magic[11])
magic[11] = magic[11].cat.codes

# Get data as arrays, shuffle, and separate features from labels
X_raw = magic.values

#np.random.shuffle(X_raw)

y = X_raw[:, -1]
X = X_raw[:, :-1]

X_embed = TSNE(n_components=2).fit_transform(X)

X_embed.shape
plt.scatter(X_embed[:,0], X_embed[:,1], c=y, alpha=0.5)
plt.show()