import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Data
data = pd.read_csv('gene_expression_data.csv')
data.rename({'Unnamed: 0': 'ct concentration'}, axis=1, inplace=True)
ct = data['ct concentration'].copy()
data.drop(['ct concentration'], axis=1, inplace=True)

# Scale Data
scaler = StandardScaler()
data_scaled = pd.DataFrame(data=scaler.fit_transform(data), columns=data.columns)

# Plot correlation clustermap (Fig 1 A)
cm = sns.clustermap(data_scaled.loc[:, data_scaled.columns != 'ct concentration'].cov(),
               metric='correlation',
               cmap='RdBu_r', vmin=-1., vmax=1.,
               cbar_kws={'label': 'correlation',
                         'ticks': [-1., -0.5, 0, 0.5, 1.],
                         'boundaries': np.linspace(-1.01, 1.01, 1000)})
plt.show()
plt.close()

# order columns in dataframe to match order of clustermap
ordered_idx = np.array(cm.dendrogram_row.reordered_ind)
cols = data_scaled.columns[ordered_idx]
data_scaled = data_scaled.reindex(cols, axis=1)

# plot gene expression heatmap (Fig 1 B)
plt.figure(figsize=(10,20))
ax = sns.heatmap(data_scaled.transpose(), cmap='PuOr', linewidths=0,
            vmin=-3, vmax=3)
ax.set_xticklabels(ct)
plt.xlabel('ct concentration [uM]')
plt.show()