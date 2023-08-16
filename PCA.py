import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%% Data

# import data
with open('path to file','r') as f:
    lines = f.readlines()
del lines[0]

# format data
perms = []
invs = [[],[],[],[],[],[],[],[],[]]
for i in range(len(lines)):
    data = [item for item in lines[i].split('[') if item != '']
    
    perm = data[0].replace('(','').replace(')','').split(',')
    perm = [int(item) for item in perm if item != ' ']
    perms.append(perm)
    
    inv = []
    for j in range(1,len(data)):
        line1 = data[j].replace(']','').split(',')
        line1 = [item.replace(' ','') for item in line1 if item != ' ']
        line2 = []
        for k in range(len(line1)):
            if '/' in line1[k]:
                num = line1[k].split('/')
                line2.append(int(num[0])/int(num[1]))
            else:
                line2.append(int(line1[k]))
        invs[j-1].append(line2)

# extract the unique invariants
unique_invs = [[] for i in range(9)]
for i in range(9):
    for j in range(len(invs[i])):
        if not invs[i][j] in unique_invs[i]:
            unique_invs[i].append(invs[i][j])


#%% PCA - Indvidual

# create a pca instance and fit to the scaled data 
pca = PCA(n_components=10)
principal_components = pca.fit_transform(invs[8])
pca.explained_variance_ratio_

# save the pca components to a dataframe
PCA_components = pd.DataFrame(principal_components)

# plot the first two pca components of the data 
scatter = plt.scatter(PCA_components[0], PCA_components[1])
plt.xlabel('PCA 1') 
plt.ylabel('PCA 2')
plt.title('Invariant ...')


#%% PCA - All

flat_invs = []
classes = []
for i in range(len(invs)):
    for j in range(len(invs[i])):
        flat_invs.append(invs[i][j])
        classes.append(i)

# create a pca instance and fit to the scaled data 
pca = PCA(n_components=200)
principal_components = pca.fit_transform(flat_invs)
ratiosE = pca.explained_variance_ratio_

# save the pca components to a dataframe
PCA_components = pd.DataFrame(principal_components)

# plot the first two pca components of the data colour coded to the corresponding web class
palette = np.array(sns.color_palette("hls", 9))
fig = plt.figure()
ax = plt.subplot(111)
ax.scatter(PCA_components[0][:len(invs[0])], PCA_components[1][:len(invs[0])], c=palette[np.array([0 for i in range(len(invs[0]))])], alpha=0.5, label='0')
ax.scatter(PCA_components[0][len(invs[0]):2*len(invs[0])], PCA_components[1][len(invs[0]):2*len(invs[0])], c=palette[np.array([1 for i in range(len(invs[0]))])], alpha=0.5, label='1')
ax.scatter(PCA_components[0][2*len(invs[0]):3*len(invs[0])], PCA_components[1][2*len(invs[0]):3*len(invs[0])], c=palette[np.array([2 for i in range(len(invs[0]))])], alpha=0.5, label='2')
ax.scatter(PCA_components[0][3*len(invs[0]):4*len(invs[0])], PCA_components[1][3*len(invs[0]):4*len(invs[0])], c=palette[np.array([3 for i in range(len(invs[0]))])], alpha=0.5, label='3')
ax.scatter(PCA_components[0][4*len(invs[0]):5*len(invs[0])], PCA_components[1][4*len(invs[0]):5*len(invs[0])], c=palette[np.array([4 for i in range(len(invs[0]))])], alpha=0.5, label='4')
ax.scatter(PCA_components[0][5*len(invs[0]):6*len(invs[0])], PCA_components[1][5*len(invs[0]):6*len(invs[0])], c=palette[np.array([5 for i in range(len(invs[0]))])], alpha=0.5, label='5')
ax.scatter(PCA_components[0][6*len(invs[0]):7*len(invs[0])], PCA_components[1][6*len(invs[0]):7*len(invs[0])], c=palette[np.array([6 for i in range(len(invs[0]))])], alpha=0.5, label='6')
ax.scatter(PCA_components[0][7*len(invs[0]):8*len(invs[0])], PCA_components[1][7*len(invs[0]):8*len(invs[0])], c=palette[np.array([7 for i in range(len(invs[0]))])], alpha=0.5, label='7')
ax.scatter(PCA_components[0][8*len(invs[0]):9*len(invs[0])], PCA_components[1][8*len(invs[0]):9*len(invs[0])], c=palette[np.array([8 for i in range(len(invs[0]))])], alpha=0.5, label='8')
plt.xlabel('PCA 1') 
plt.ylabel('PCA 2')
plt.title('All Invariants')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(loc='right', bbox_to_anchor=(1.135, 0.5), fancybox=True, shadow=True, ncol=1)
plt.show()

