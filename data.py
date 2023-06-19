import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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

# compute mean, min, max, std, var for dust class 2
dust_class2 = features.loc[dust_class == 2]
feature2_stats = pd.DataFrame({})

for name in ["mean", "min", "max", "std", "var"]:
    feature2_stats = pd.concat([feature2_stats, getattr(dust_class2, name)(axis=0).to_frame(name=name).T])
print(feature2_stats)

worst_spread = feature2_stats.max(axis=1)[feature2_stats.index.get_loc("var")]
print(feature2_stats.loc[:, feature2_stats.T['var'] == worst_spread])

# compute mean, min, max, std, var for dust class 3
dust_class3 = features.loc[dust_class == 3]
feature3_stats = pd.DataFrame({})

for name in ["mean", "min", "max", "std", "var"]:
    feature3_stats = pd.concat([feature3_stats, getattr(dust_class3, name)(axis=0).to_frame(name=name).T])
print(feature3_stats)

worst_spread = feature3_stats.max(axis=1)[feature3_stats.index.get_loc("var")]
print(feature3_stats.loc[:, feature3_stats.T['var'] == worst_spread])

###### SKEW

# compute skew for each band
features_skew = features.skew(axis=0, skipna=True)
print(features_skew.to_frame(name="skew").T)

# four largest absolute skew
four_largest_skews = features_skew.abs().nlargest(4)
print(four_largest_skews.to_frame(name="skew").T)
index = four_largest_skews.index

# print the name and skew value for the 4 most skewed bands and the class breakdowns
class_skews = pd.DataFrame({})
for dust_class in range(4):
    class_skews = pd.concat([class_skews, data.loc[data['dust'] == dust_class, index]
                            .skew(axis=0)
                            .to_frame(name="dust{:1}".format(dust_class)).T])
print(class_skews)


########### VISUALIZING DATA USING MATPLOTLIB

# for the 4 most skewed bands create a box and whisker plot using matplotlib and its subplot. Use horizontal plots.

h, w = (2, 2)
fig, axs = plt.subplots(h, w, figsize=(20,10), tight_layout=True)
getBand = lambda x, y: index[x+2*y]
for i in range(h):
    for j in range(w):
        toplot = data.loc[:, getBand(i,j)]
        print(getBand(i, j), data.loc[:, getBand(i, j)].shape)
        axs[i][j].set_title(getBand(i, j))
        axs[i][j].boxplot(toplot.to_numpy(), vert=False)
# plt.show()

# for the 4 most skewed bands create a histogram using matplotlib and its subplotting utilities. Use at least 100 bins. Add a title and axis labels
h, w = (2, 2)
fig, axs = plt.subplots(h, w, figsize=(20,10), tight_layout=True)
getBand = lambda x, y: index[x+2*y]
for i in range(h):
    for j in range(w):
        toplot = data.loc[:, getBand(i,j)]
        print(getBand(i, j), data.loc[:, getBand(i, j)].shape)
        axs[i][j].set_title(getBand(i, j))
        axs[i][j].hist(toplot.to_numpy(), bins=100)
plt.show()