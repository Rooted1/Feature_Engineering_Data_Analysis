import pandas as pd
import numpy as np 

data = pd.read_csv('CalMod_06222009.csv')
data = data.dropna()
data = data.reset_index()

# print first seven bands in DataFrame using list comprehension
print(data[["Band{}".format(x) for x in range(1, 8)]])

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

# print first seven bands in stats using .loc[]
print(stats.loc[:, "Band1" : "Band7"])

# find band that has the worst spread using numpy array
features_stats = stats.to_numpy()
worst_spread = features_stats.max(axis=1)[stats.index.get_loc("var")]
print(stats.loc[:, stats.T['var'] == worst_spread])


######## class 'dust' 0, 1, 2, 3

# Print the number of records for each class using numpy.unique
dust_class = data['dust']
classes, counts = np.unique(dust_class, return_counts=True)
for cls, count in zip(classes, counts):
    print("Dust ", int(cls), ": ", count)


# compute mean, min, max, std, var for dust class 0
dust_class0 = features.loc[dust_class == 0]
feature0_stats = pd.DataFrame({})

for name in ["mean", "min", "max", "std", "var"]:
    feature0_stats = pd.concat([feature0_stats, getattr(dust_class0, name)(axis=0).to_frame(name=name).T])
print(feature0_stats)

worst_spread = feature0_stats.max(axis=1)[feature0_stats.index.get_loc("var")]
print(feature0_stats.loc[:, feature0_stats.T['var'] == worst_spread])

# compute mean, min, max, std, var for dust class 1
dust_class1 = features.loc[dust_class == 1]
feature1_stats = pd.DataFrame({})

for name in ["mean", "min", "max", "std", "var"]:
    feature1_stats = pd.concat([feature1_stats, getattr(dust_class1, name)(axis=0).to_frame(name=name).T])
print(feature1_stats)

worst_spread = feature1_stats.max(axis=1)[feature1_stats.index.get_loc("var")]
print(feature1_stats.loc[:, feature1_stats.T['var'] == worst_spread])


