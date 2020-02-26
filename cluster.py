import pandas as pd
import os
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import seaborn as sb

from sklearn import metrics

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()


def load_data(path):
  data = pd.read_csv(path)
  data.columns = data.columns[1:].insert(0,'cell_no')
  data.set_index('cell_no', inplace=True)
  return data

filtered14 = load_data('/x/filtered14')
filtered15 = load_data('/x/filtered15')
filtered206 = load_data('/x/filtered207')
filtered207 = load_data('/x/filtered206')

# Get AnnData object from pandas dataframe
def get_adata(data):
  adata = sc.AnnData(data)
  adata.obs['n_transcripts'] = adata.X.sum(1)
  adata.obs['n_genes'] = (adata.X > 0).sum(1)
  # Mitochondrial mRNA can suggest bad cells or elevated respiration
  mt_genes_mask = ['mt' in gene for gene in adata.var_names]
  adata.obs['mt_frac'] = adata.X[:, mt_genes_mask].sum(1)/adata.obs['n_transcripts']

  return adata

adata_14 = get_adata(filtered14)
adata_15 = get_adata(filtered15)
adata_206 = get_adata(filtered206)
adata_207 = get_adata(filtered207)

def exploratory_plot_mt_frac(adata_list):
  
  for adata in adata_list:
    sc.pl.violin(adata, 'mt_frac')

exploratory_plot_mt_frac([adata_14, adata_15, adata_206, adata_207])

"""Primecujemo da se u prvoj grupi javlja dosta mitohondrijske RNK, sto moze sugerisati da su celije pod stresom ili da postoji povecan nivo kiseonika u njima. Nasuprot tome, u drugoj grupi su nivou mitohondrijske RNK znacajno nizi (ispod 30% u 206 i ispod 20% u 207) Kako bismo izbacili celije sa visokim nivoima mtRNK, uradicemo dodatne analize podataka. Iscrtacemo odnos broja izrazenih gena i kolicine informacione RNK, a procenat mitohondrijske RNK ce biti predstavljen uz pomoc gradijenta boja."""

sc.pl.scatter(adata_14, 'n_counts', 'n_genes', color='mt_frac')

sc.pl.scatter(adata_15, 'n_counts', 'n_genes', color='mt_frac')

"""# Visualisation"""

# Perform PCA on data
sc.tl.pca(adata_14)
sc.tl.pca(adata_15)
sc.tl.pca(adata_206)
sc.tl.pca(adata_207)

"""Vidimo projekciju na dve ose u kojima je varijacija najveca u originalnom skupu. Uz svaku glavnu komponentu je naznaceno koliki procenat varijanse objasnjava. Bitno je naglasiti da su razlike na x osi vaznije od razlika na y osi, odnosno tacke koje su po x osi udaljenije su razlicitije od tacaka iste udaljenosti na y osi. Sa slike se vidi da je u prvoj grupi sa dve glavne komponente dosta dobro pokrivena varijansa pocetnog skupa (oko 84% u obe datoteke), dok su podaci u drugoj grupi varijabilni u vise smerova, te slika za njih deluje manje usmereno. PCA dekompozicija nam ne pruza dovoljno informacija o izdvajanju bilo kakvih klastera, ali moze biti korisna prilikom izracunavanja grafa susedstva u narednim koracima."""

sc.pl.pca(adata_14, return_fig=True)
plt.xlabel('PC1 %.2f %% of variance' % (adata_14.uns['pca']['variance_ratio'][0] * 100))
plt.ylabel('PC2 %.2f %% of variance' % (adata_14.uns['pca']['variance_ratio'][1] * 100))
plt.title('%.2f %% of total variance' % (adata_14.uns['pca']['variance_ratio'][:2].sum() * 100))
plt.show()

sc.pl.pca(adata_15, return_fig=True)
plt.xlabel('PC1 %.2f %% of variance' % (adata_15.uns['pca']['variance_ratio'][0] * 100))
plt.ylabel('PC2 %.2f %% of variance' % (adata_15.uns['pca']['variance_ratio'][1] * 100))
plt.title('%.2f %% of total variance' % (adata_15.uns['pca']['variance_ratio'][:2].sum() * 100))
plt.show()

sc.pl.pca(adata_206, return_fig=True)
plt.xlabel('PC1 %.2f %% of variance' % (adata_206.uns['pca']['variance_ratio'][0] * 100))
plt.ylabel('PC2 %.2f %% of variance' % (adata_206.uns['pca']['variance_ratio'][1] * 100))
plt.title('%.2f %% of total variance' % (adata_206.uns['pca']['variance_ratio'][:2].sum() * 100))
plt.show()

sc.pl.pca(adata_207, return_fig=True)
plt.xlabel('PC1 %.2f %% of variance' % (adata_207.uns['pca']['variance_ratio'][0] * 100))
plt.ylabel('PC2 %.2f %% of variance' % (adata_207.uns['pca']['variance_ratio'][1] * 100))
plt.title('%.2f %% of total variance' % (adata_207.uns['pca']['variance_ratio'][:2].sum() * 100))
plt.show()

sc.pl.pca_variance_ratio(adata_14)
sc.pl.pca_variance_ratio(adata_15)
sc.pl.pca_variance_ratio(adata_206)
sc.pl.pca_variance_ratio(adata_207)

print(adata_14.uns['pca']['variance_ratio'][:4].sum())
print(adata_15.uns['pca']['variance_ratio'][:4].sum())

print(adata_206.uns['pca']['variance_ratio'][:42].sum())
print(adata_207.uns['pca']['variance_ratio'][:21].sum())

"""Vidimo da je za pokrivenost od 90% potrebno 4 glavne komponente u prvog grupi. U drugoj grupi je potrebno vise glavnih komponenti za istu pokrivenost i to 42 za datoteku 206 i 21 za datoteku 207. Koristicemo ove vrednosti za konstrukciju grafa povezanosti:

Pravljenje grafa povezanosti je neophodno za tsne i umap vizualizacije.
"""

def embed_neighbors(adata, n_pcs, metrics=['euclidean', 'cosine']):
  graphs = []
  for m in metrics:
    neighbors = sc.pp.neighbors(adata, n_pcs = n_pcs, metric=m, copy=True)
    graphs.append(neighbors)
  return graphs

neighbors14 = embed_neighbors(adata_14, n_pcs=4)
neighbors15 = embed_neighbors(adata_15, n_pcs=4)
neighbors206 = embed_neighbors(adata_206, n_pcs=42)
neighbors207 = embed_neighbors(adata_207, n_pcs=21)

# Calculate tSNE embeddings 
sc.tl.tsne(neighbors14[0])
sc.tl.tsne(neighbors14[1])

sc.tl.tsne(neighbors15[0])
sc.tl.tsne(neighbors15[1])

sc.tl.tsne(neighbors206[0])
sc.tl.tsne(neighbors206[1])

sc.tl.tsne(neighbors207[0])
sc.tl.tsne(neighbors207[1])

sc.pl.tsne(neighbors14[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.tsne(neighbors14[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

sc.pl.tsne(neighbors15[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.tsne(neighbors15[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

sc.pl.tsne(neighbors206[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.tsne(neighbors206[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

sc.pl.tsne(neighbors207[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.tsne(neighbors207[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

# Calculate UMAP embeddings
sc.tl.umap(neighbors14[0])
sc.tl.umap(neighbors14[1])

sc.tl.umap(neighbors15[0])
sc.tl.umap(neighbors15[1])

sc.tl.umap(neighbors206[0])
sc.tl.umap(neighbors206[1])

sc.tl.umap(neighbors207[0])
sc.tl.umap(neighbors207[1])

sc.pl.umap(neighbors14[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.umap(neighbors14[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

sc.pl.umap(neighbors15[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.umap(neighbors15[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

sc.pl.umap(neighbors206[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.umap(neighbors206[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

sc.pl.umap(neighbors207[0], return_fig=True)
plt.title('Euclidean distance')
plt.show()
sc.pl.umap(neighbors207[1], return_fig=True)
plt.title('Cosine distance')
plt.show()

"""# scanpy clustering"""

def replicate_louvain(adata, data_key , resolutions, metric):
  for r in resolutions:
    key_ = 'louvain_' + data_key + '_' + str(r) + '_' + metric
    sc.tl.louvain(adata, key_added=key_, resolution=r)

def replicate_leiden(adata, data_key , resolutions, metric):
  for r in resolutions:
    key_ = 'leiden_' + data_key + '_' + str(r) + '_' + metric
    sc.tl.leiden(adata, key_added=key_, resolution=r)

def log_cluster(adata, clustering_method, title):
  if('Euclidean' in title):
    path = os.path.join('./drive/My Drive/x', clustering_method + '_euclidean.txt')
  else:
    path = os.path.join('./drive/My Drive/x', clustering_method + '_cosine.txt')
  with open(path, 'w') as f:

    f.write(title + '\n')
    for l in adata.obs.loc[:,[clustering_method in x for x in adata.obs]]:
      if 'louvain' in clustering_method or 'leiden' in clustering_method:  
        f.write('Resolution: ' + l.split('_')[-2] + '\n')
      elif 'kmeans' in clustering_method:
        f.write('No of clusters: ' + l.split('_')[-2] + '\n')
      elif 'birch' in clustering_method:
        f.write('Threshold : ' + l.split('_')[2] + '\n')
        f.write('Branching factor : ' + l.split('_')[3] + '\n')
        f.write('Number of clusters : ' + l.split('_')[4] + '\n')
      elif 'ward' in clustering_method:
        f.write('No of clusters: ' + l.split('_')[-2] + '\n')
      elif 'dbscan' in clustering_method:
        f.write('Epsilon: ' + l.split('_')[2] + '\n')
        f.write('Min samples: ' + l.split('_')[3] + '\n')
      grouped = adata.obs[l].groupby(adata.obs[l])
      groups = grouped.groups
      f.write('%d clusters found.\n' % len(groups))
      #NOTE: Run evaluations on dimensionally reduced data!
      adata_reduced = adata.obsm['X_umap']
      try:
        f.write('Silhouette score %.4f\n' % metrics.silhouette_score(adata_reduced, adata.obs[l]))
        f.write('Calinski-Harabasz score %.4f\n' % metrics.calinski_harabasz_score(adata_reduced, adata.obs[l]))
        f.write('Davies-Bouldin score %.4f\n' % metrics.davies_bouldin_score(adata_reduced, adata.obs[l]))
      except ValueError as e:
        print(e)
        pass
      for g in groups:
        if 'louvain' in clustering_method or 'leiden' in clustering_method:
          key = str(g)
        else:
          key = g
        f.write('\nCluster %s with %d cells\n' % (g, len(groups[key])))
        f.write('\n'.join(groups[key]))
        f.write('\n')

def plot_clustering_results(adata, clustering_algorithms):
  colors = [x for x in adata.obs.columns if (x.split('_')[0] in clustering_algorithms)]
  sc.pl.umap(adata, color=colors)

"""## Louvain clustering"""

#TODO: automatize clustering
adata_list = [adata_14]
g_clustering_algorithms = ['louvain', 'leiden']

resolutions = [1.0, 0.7, 0.5, 0.3, 0.15]
replicate_louvain(neighbors14[0], '14', resolutions, 'euclidean')
replicate_louvain(neighbors14[1], '14', resolutions, 'cosine')

replicate_louvain(neighbors15[0], '15', resolutions, 'euclidean')
replicate_louvain(neighbors15[1], '15', resolutions, 'cosine')

replicate_louvain(neighbors206[0], '206', resolutions, 'euclidean')
replicate_louvain(neighbors206[1], '206', resolutions, 'cosine')

replicate_louvain(neighbors207[0], '207', resolutions, 'euclidean')
replicate_louvain(neighbors207[1], '207', resolutions, 'cosine')

log_cluster(neighbors14[0], 'louvain_14', 'Euclidean distance used')
log_cluster(neighbors14[1], 'louvain_14', 'Cosine distance used')

log_cluster(neighbors15[0], 'louvain_15', 'Euclidean distance used')
log_cluster(neighbors15[1], 'louvain_15', 'Cosine distance used')

log_cluster(neighbors206[0], 'louvain_206', 'Euclidean distance used')
log_cluster(neighbors206[1], 'louvain_206', 'Cosine distance used')

log_cluster(neighbors207[0], 'louvain_207', 'Euclidean distance used')
log_cluster(neighbors207[1], 'louvain_207', 'Cosine distance used')

"""## Leiden clustering"""

resolutions = [1.0, 0.7, 0.5, 0.3, 0.15]
replicate_leiden(neighbors14[0], '14', resolutions, 'euclidean')
replicate_leiden(neighbors14[1], '14', resolutions, 'cosine')

replicate_leiden(neighbors15[0], '15', resolutions, 'euclidean')
replicate_leiden(neighbors15[1], '15', resolutions, 'cosine')

replicate_leiden(neighbors206[0], '206', resolutions, 'euclidean')
replicate_leiden(neighbors206[1], '206', resolutions, 'cosine')

replicate_leiden(neighbors207[0], '207', resolutions, 'euclidean')
replicate_leiden(neighbors207[1], '207', resolutions, 'cosine')

log_cluster(neighbors14[0], 'leiden_14', 'Euclidean distance used')
log_cluster(neighbors14[1], 'leiden_14', 'Cosine distance used')

log_cluster(neighbors15[0], 'leiden_15', 'Euclidean distance used')
log_cluster(neighbors15[1], 'leiden_15', 'Cosine distance used')

log_cluster(neighbors206[0], 'leiden_206', 'Euclidean distance used')
log_cluster(neighbors206[1], 'leiden_206', 'Cosine distance used')

log_cluster(neighbors207[0], 'leiden_207', 'Euclidean distance used')
log_cluster(neighbors207[1], 'leiden_207', 'Cosine distance used')

plot_clustering_results(neighbors14[0], ['louvain', 'leiden'])
plot_clustering_results(neighbors14[1], ['louvain', 'leiden'])

plot_clustering_results(neighbors15[0], ['louvain', 'leiden'])
plot_clustering_results(neighbors15[1], ['louvain', 'leiden'])

plot_clustering_results(neighbors206[0], ['louvain', 'leiden'])
plot_clustering_results(neighbors206[1], ['louvain', 'leiden'])

plot_clustering_results(neighbors207[0], ['louvain', 'leiden'])
plot_clustering_results(neighbors207[1], ['louvain', 'leiden'])


from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

def replicate_birch(adata, data_key, thresholds=[0.5], branching_factors=[50], n_clusters=[None], metric=''):
  for t in thresholds:
    for b in branching_factors:
      for n in n_clusters:
        print('Running birch for ' + str(t) + ' ' + str(b) + ' ' + str(n))
        brc = Birch(t, b, n)
        brc.fit(adata.X)
        column_name = 'birch_' + data_key + '_'  + str(brc.threshold) + '_' + str(brc.branching_factor) + '_' + str(brc.n_clusters) + '_' + metric
        adata.obs[column_name] = brc.predict(adata.X)

"""## Birch clustering"""

thresholds = [0.3,0.5]
branching_factors =  [50]
n_clusters = [5, 10]
replicate_birch(neighbors14[0], '14', thresholds, branching_factors, n_clusters, 'euclidean')
replicate_birch(neighbors15[0], '15', thresholds, branching_factors, n_clusters, 'euclidean')
replicate_birch(neighbors206[0], '206', thresholds, branching_factors, n_clusters, 'euclidean')
replicate_birch(neighbors207[0], '207', thresholds, branching_factors, n_clusters, 'euclidean')

replicate_birch(neighbors14[1], '14', thresholds, branching_factors, n_clusters, 'cosine')
replicate_birch(neighbors15[1], '15', thresholds, branching_factors, n_clusters, 'cosine')
replicate_birch(neighbors206[1], '206', thresholds, branching_factors, n_clusters, 'cosine')
replicate_birch(neighbors207[1], '207', thresholds, branching_factors, n_clusters, 'cosine')

log_cluster(neighbors14[0], 'birch_14', 'Euclidean distance used')
log_cluster(neighbors14[1], 'birch_14', 'Cosine distance used')

log_cluster(neighbors15[0], 'birch_15', 'Euclidean distance used')
log_cluster(neighbors15[1], 'birch_15', 'Cosine distance used')

log_cluster(neighbors206[0], 'birch_206', 'Euclidean distance used')
log_cluster(neighbors206[1], 'birch_206', 'Cosine distance used')

log_cluster(neighbors207[0], 'birch_207', 'Euclidean distance used')
log_cluster(neighbors207[1], 'birch_207', 'Cosine distance used')

plot_clustering_results(neighbors14[0], ['birch'])
plot_clustering_results(neighbors15[0], ['birch'])
plot_clustering_results(neighbors206[0], ['birch'])
plot_clustering_results(neighbors207[0], ['birch'])

plot_clustering_results(neighbors14[1], ['birch'])
plot_clustering_results(neighbors15[1], ['birch'])
plot_clustering_results(neighbors206[1], ['birch'])
plot_clustering_results(neighbors207[1], ['birch'])

"""## KMeans clustering"""

def replicate_kmeans(adata, data_key, n_clusters=[5], metric=''):
  for n in n_clusters:
    print('Running kmeans for k = ' + str(n))
    kmeans = KMeans(n_clusters=n).fit(adata.X)
    column_name = 'kmeans_' + data_key + '_' + str(n) + '_' + metric
    adata.obs[column_name] = kmeans.predict(adata.X)

replicate_kmeans(neighbors14[0], data_key = '14', n_clusters=range(2,11), metric = 'euclidean')
replicate_kmeans(neighbors15[0], data_key = '15', n_clusters=range(2,11), metric = 'euclidean')
replicate_kmeans(neighbors206[0], data_key = '206', n_clusters=range(2,11), metric = 'euclidean')
replicate_kmeans(neighbors207[0], data_key = '207', n_clusters=range(2,11), metric = 'euclidean')

replicate_kmeans(neighbors14[1], data_key = '14', n_clusters=range(2,11), metric = 'cosine')
replicate_kmeans(neighbors15[1], data_key = '15', n_clusters=range(2,11), metric = 'cosine')
replicate_kmeans(neighbors206[1], data_key = '206', n_clusters=range(2,11), metric = 'cosine')
replicate_kmeans(neighbors207[1], data_key = '207', n_clusters=range(2,11), metric = 'cosine')

log_cluster(neighbors14[0], 'kmeans_14', 'Euclidean distance used')
log_cluster(neighbors14[1], 'kmeans_14', 'Cosine distance used')

log_cluster(neighbors15[0], 'kmeans_15', 'Euclidean distance used')
log_cluster(neighbors15[1], 'kmeans_15', 'Cosine distance used')

log_cluster(neighbors206[0], 'kmeans_206', 'Euclidean distance used')
log_cluster(neighbors206[1], 'kmeans_206', 'Cosine distance used')

log_cluster(neighbors207[0], 'kmeans_207', 'Euclidean distance used')
log_cluster(neighbors207[1], 'kmeans_207', 'Cosine distance used')

neighbors14[0].write('/content/drive/My Drive/x/n14_0')
neighbors14[1].write('/content/drive/My Drive/x/n14_1')

neighbors15[0].write('/content/drive/My Drive/x/n15_0')
neighbors15[1].write('/content/drive/My Drive/x/n15_1')

neighbors206[0].write('/content/drive/My Drive/x/n206_0')
neighbors206[1].write('/content/drive/My Drive/x/n206_1')

neighbors207[0].write('/content/drive/My Drive/x/n207_0')
neighbors207[1].write('/content/drive/My Drive/x/n207_1')

neighbors206[0].obs

plot_clustering_results(neighbors14[0], ['kmeans'])
plot_clustering_results(neighbors14[1], ['kmeans'])

plot_clustering_results(neighbors15[0], ['kmeans'])
plot_clustering_results(neighbors15[1], ['kmeans'])

plot_clustering_results(neighbors206[0], ['kmeans'])
plot_clustering_results(neighbors206[1], ['kmeans'])

plot_clustering_results(neighbors207[0], ['kmeans'])
plot_clustering_results(neighbors207[1], ['kmeans'])

"""## Ward clustering"""

def replicate_ward(adata, data_key, n_clusters=[5], metric=''):
  for n in n_clusters:
    print('Running ward for n_clusters = ' + str(n))
    ward = AgglomerativeClustering(n_clusters=n, linkage='ward', connectivity=adata.uns['neighbors']['connectivities'])
    column_name = 'ward_' + data_key + '_' + str(n) + '_' + metric
    adata.obs[column_name] = ward.fit_predict(adata.X)

replicate_ward(neighbors14[0], data_key = '14', n_clusters=range(2,11), metric = 'euclidean')
replicate_ward(neighbors15[0], data_key = '15', n_clusters=range(2,11), metric = 'euclidean')
replicate_ward(neighbors206[0], data_key = '206', n_clusters=range(2,11), metric = 'euclidean')
replicate_ward(neighbors207[0], data_key = '207', n_clusters=range(2,11), metric = 'euclidean')

replicate_ward(neighbors14[1], data_key = '14', n_clusters=range(2,11), metric = 'cosine')
replicate_ward(neighbors15[1], data_key = '15', n_clusters=range(2,11), metric = 'cosine')
replicate_ward(neighbors206[1], data_key = '206', n_clusters=range(2,11), metric = 'cosine')
replicate_ward(neighbors207[1], data_key = '207', n_clusters=range(2,11), metric = 'cosine')

log_cluster(neighbors14[0], 'ward_14', 'Euclidean distance used')
log_cluster(neighbors14[1], 'ward_14', 'Cosine distance used')

log_cluster(neighbors15[0], 'ward_15', 'Euclidean distance used')
log_cluster(neighbors15[1], 'ward_15', 'Cosine distance used')

log_cluster(neighbors206[0], 'ward_206', 'Euclidean distance used')
log_cluster(neighbors206[1], 'ward_206', 'Cosine distance used')

log_cluster(neighbors207[0], 'ward_207', 'Euclidean distance used')
log_cluster(neighbors207[1], 'ward_207', 'Cosine distance used')

plot_clustering_results(neighbors14[0], ['ward'])
plot_clustering_results(neighbors14[1], ['ward'])

plot_clustering_results(neighbors15[0], ['ward'])
plot_clustering_results(neighbors15[1], ['ward'])

plot_clustering_results(neighbors206[0], ['ward'])
plot_clustering_results(neighbors206[1], ['ward'])

plot_clustering_results(neighbors207[0], ['ward'])
plot_clustering_results(neighbors207[1], ['ward'])

"""## DBSCAN clustering"""

def replicate_dbscan(adata, data_key, epsilons=[3], min_samples=[2], metric=''):
  for e in epsilons:
    for s in min_samples:
      print('Running DBSCAN for eps %.2f, min_samples %d' % (e, s))
      dbscan = DBSCAN(eps=e, min_samples=s)
      column_name = 'dbscan_' + data_key + '_' + str(e) + '_' + str(s) + '_' + metric
      adata.obs[column_name] = dbscan.fit_predict(adata.X)

epsilons_=[0.1,0.05,0.025]
min_samples_=[2,5,10,30]
replicate_dbscan(adata_14, '14', epsilons=epsilons_, min_samples=min_samples_, metric = 'euclidean')
replicate_dbscan(adata_14, '14', epsilons=epsilons_, min_samples=min_samples_, metric = 'cosine')

replicate_dbscan(adata_15, '15', epsilons=epsilons_, min_samples=min_samples_, metric = 'euclidean')
replicate_dbscan(adata_15, '15', epsilons=epsilons_, min_samples=min_samples_, metric = 'cosine')

replicate_dbscan(adata_206, '206', epsilons=epsilons_, min_samples=min_samples_, metric = 'euclidean')
replicate_dbscan(adata_206, '206', epsilons=epsilons_, min_samples=min_samples_, metric = 'cosine')

replicate_dbscan(adata_207, '207', epsilons=epsilons_, min_samples=min_samples_, metric = 'euclidean')
replicate_dbscan(adata_207, '207', epsilons=epsilons_, min_samples=min_samples_, metric = 'cosine')

plot_clustering_results(neighbors14[0], ['dbscan'])
plot_clustering_results(neighbors14[1], ['dbscan'])

plot_clustering_results(neighbors15[0], ['dbscan'])
plot_clustering_results(neighbors15[1], ['dbscan'])

plot_clustering_results(neighbors206[0], ['dbscan'])
plot_clustering_results(neighbors206[1], ['dbscan'])

plot_clustering_results(neighbors207[0], ['dbscan'])
plot_clustering_results(neighbors207[1], ['dbscan'])

log_cluster(neighbors14[0], 'dbscan_14', 'Euclidean distance used')
log_cluster(neighbors14[1], 'dbscan_14', 'Cosine distance used')

log_cluster(neighbors15[0], 'dbscan_15', 'Euclidean distance used')
log_cluster(neighbors15[1], 'dbscan_15', 'Cosine distance used')

log_cluster(neighbors206[0], 'dbscan_206', 'Euclidean distance used')
log_cluster(neighbors206[1], 'dbscan_206', 'Cosine distance used')

log_cluster(neighbors207[0], 'dbscan_207', 'Euclidean distance used')
log_cluster(neighbors207[1], 'dbscan_207', 'Cosine distance used')
