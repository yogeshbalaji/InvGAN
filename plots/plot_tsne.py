import cPickle
from sklearn import manifold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

filename = '/scratch0/defenseganv2/data/cache/cifar-10_recon/ours_feats.pkl'

with open(filename) as f:
    data_dict = cPickle.load(f)

latents = data_dict['latents']


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=10)
Y = tsne.fit_transform(latents[:1000, :])


plt.figure()
plt.scatter(Y[:, 0], Y[:, 1])
plt.savefig('/scratch0/defenseganv2/data/cache/cifar-10_recon/age_tsne.png')