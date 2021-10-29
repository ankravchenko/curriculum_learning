import numpy as np
import argparse
import seaborn as sns
import matplotlib.pylab as plt


basefile="goldilocks_FI_logfile_slow"
n=5


#basefile="goldilocks_FI_logfile_early_inc_"
#n=5

#basefile="goldilocks_FI_logfile_early_correct"
#n=6

#basefile="testfile"
#n=2
#basefile="goldilocks_vectorfield_logfile_early_correct"
#n=4

#basefile="goldilocks_vectorfield_logfile_early_inc"
#n=5

runs = []

for i in range(1,n+1):
	arr=np.genfromtxt(basefile+str(i)+".log", delimiter='\t',dtype=None)#[:10]
	runs.append(arr)


a=np.array(runs)


b=np.median(a,axis=0)
c=np.std(a,axis=0)
print("!!!!!!!!!")
print(b)
print(c)

b = np.insert(b, 0, 0, axis=0)
x=np.arange(start=0, stop=30000, step=1250)
x=x[0:24]


y=b[0:24]
print(":)")
###########
plt.plot(x, y, linewidth=1,  color='orange', label='performance on task 1, stationary pre-training on task 1')

y_error=c[0:24]


plt.fill_between(x, y-y_error, y+y_error,
    alpha=1, edgecolor='orange', facecolor='#F1AC9D',
    linewidth=0.5, label='_nolegend_')



plt.xlabel('number of examples')
plt.ylabel("accuracy on task 1")  
plt.show()


