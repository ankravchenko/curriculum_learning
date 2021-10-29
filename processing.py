import numpy as np
import argparse
import seaborn as sns
import matplotlib.pylab as plt





#basefile="goldilocks_vectorfield_logfile_fulltraining_"
#n=6

#basefile="goldilocks_vectorfield_logfile_early_inc"
#n=5

runs = []




basefile="goldilocks_vectorfield_logfile_slow"
n=5

for i in range(1,n+1):
	arr=np.genfromtxt(basefile+str(i)+".log", delimiter='\t')[:,:-1]
	runs.append(arr)

a=np.array(runs)


b=np.median(a,axis=0)
print("!!!!!!!!!")
print(b)


y=np.flipud(b)

x_axis_labels = np.arange(start=1250, stop=30000, step=1250) # labels for x-axis

#x_axis_labels = np.arange(start=30000, stop=1250, step=-1250)
y_axis_labels = np.arange(start=30000, stop=0, step=-1250) #np.arange(start=0, stop=30000, step=1250) # labels for y-axis


#ax = sns.heatmap(x, linewidth=0, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="YlGnBu")

ax = sns.heatmap(y, linewidth=0, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="YlGnBu")


#ax = sns.heatmap(y, linewidth=0, xticklabels=y_axis_labels, yticklabels=x_axis_labels, cmap="YlGnBu")
ax.set(xlabel='Task 2', ylabel='Task 1')


plt.show()
plt.savefig("heatmap_slow_correct.png")
