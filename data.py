import pandas as pd
import numpy as np 

data = pd.read_csv('CalMod_06222009.csv')
data = data.dropna()
data = data.reset_index()

features = data.iloc[:, 2:44]

# compute mean, min, max, std, var for all bands
mean = features.mean(axis=0).to_frame(name="mean").T
stats = pd.DataFrame(mean)
min = features.min(axis=0).to_frame(name="min").T
max = features.max(axis=0).to_frame(name="max").T
std = features.std(axis=0).to_frame(name="std").T
var = features.var(axis=0).to_frame(name="var").T

stats = pd.concat([stats, min])
stats = pd.concat([stats, max])
stats = pd.concat([stats, std])
stats = pd.concat([stats, var])

# find band that has the worst spread using numpy array
features_stats = stats.to_numpy()
worst_spread = features_stats.max(axis=1)[stats.index.get_loc("var")]
print(stats.loc[:, stats.T['var'] == worst_spread])



