#python 3.5, Tensorflow 1.14
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
from os import path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
mnist = tf.keras.datasets.mnist

from copy import copy

#from __future__ import absolute_import, division, print_function, unicode_literals

import random
from datetime import datetime
from tensorflow import keras

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, MaxPooling2D


import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import pickle

#tensorboard logs
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

import logging



from random import randrange
import sys
import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	

def load_data(): # use as a global variable maybe? less hassle
	#check for pickled and load pickled
	#load MNIST set
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	
	#pickle
	return train_images, train_labels, test_images, test_labels
	
#generate1digit_dataset(n)
#generate2digit(n, digitlist)

def generate_cnn_model():
#create model
	model = tf.keras.models.Sequential([
		keras.layers.Conv2D(64, kernel_size=3, name='conv1', activation='relu', input_shape=(84,84,1)),
		keras.layers.AveragePooling2D(name='pool1'),
		keras.layers.Conv2D(32, kernel_size=3, name='conv2', activation='relu'),
		keras.layers.AveragePooling2D(name='pool2'),
		keras.layers.Flatten(name='flatten'),
		keras.layers.Dense(10, activation='softmax', name='dense1')
		])


	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
      	        metrics=['accuracy'])

	return model

def ten_random_2digits():
	digitlist=[];
	while len(digitlist)<10:
		x=random.randint(10,99)
		reversex=int(str(x)[::-1])
		if (x not in digitlist)&(reversex not in digitlist):
			digitlist.append(x);
	return digitlist

def random_pos():
	pos1x=random.randint(0,2)
	pos1y=random.randint(0,2)
	while True:
		pos2x=random.randint(0,2)
		pos2y=random.randint(0,2)
		if (pos1x,pos1y)!=(pos2x,pos2y):
			break	
	return pos1x,pos2x,pos1y,pos2y

def generate_1digit_set(n):
	train_images_1digit=numpy.zeros(shape=(n,84,84)) #FIXME change array name, they are not necessarily training images
	train_labels_1digit=numpy.zeros(n,dtype=int)
	for d in range(0,n):
		pos0x=random.randint(0,2)
		pos0y=random.randint(0,2)
		train_images_1digit[d,pos0y*28:pos0y*28+28,pos0x*28:pos0x*28+28] = train_images[d]
		train_labels_1digit[d]=train_labels[d]		
	return train_images_1digit, train_labels_1digit

def generate_2digit_set(n, twodigit_labels):
	train_images_2digit=numpy.zeros(shape=(n,84,84))
	train_labels_2digit=numpy.zeros(n,dtype=int)
	for l in twodigit_labels:
			l_count=twodigit_labels.index(l);
			j_label= l//10 #1st digit label
			i_label= l%10 #2nd digit label
			i_list = numpy.where(train_labels == i_label)[0]
			j_list = numpy.where(train_labels == j_label)[0]
			for d in range(0,int(n/10)): #FIXME this is only for 10 categories
				i = i_list[random.randint(0,i_list.size-1)]
				j = j_list[random.randint(0,j_list.size-1)]
				pos1x,pos2x,pos1y,pos2y=random_pos()
				train_images_2digit[l_count*int(n/10)+d,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = train_images[i]
				train_images_2digit[l_count*int(n/10)+d,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = train_images[j]
				train_labels_2digit[l_count*int(n/10)+d]=l
		

	number_of_sets=numpy.unique(train_labels_2digit).size
	return train_images_2digit, train_labels_2digit

def generate_2digit_set_50categories(n): #FIXME incorporate into a previous one with changeable number of catgories
	train_images_2digit=numpy.zeros(shape=(n,84,84))
	train_labels_2digit=numpy.zeros(n,dtype=int)
	

	for i in range(0,n):
		j=random.randint(0,30000)
		pos1x=random.randint(0,2)
		pos1y=random.randint(0,2)
		while True:
			pos2x=random.randint(0,2)
			pos2y=random.randint(0,2)
			if (pos1x,pos1y)!=(pos2x,pos2y):
				break	
		train_images_2digit[i,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = train_images[i]
		train_images_2digit[i,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = train_images[j]
		train_labels_2digit[i]=train_labels[i]*10+train_labels[j]

	return train_images_2digit, train_labels_2digit



def copyModel2Model(model_source,model_target,certain_layer=""):        
	for l_tg,l_sr in zip(model_target.layers,model_source.layers):
			wk0=l_sr.get_weights()
			l_tg.set_weights(wk0)
			if (l_tg.name==certain_layer):
				break
	print("model source was copied into model target") 

#suppress tensorlow output FIXME currently still doesn't remove everything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#command line arguments: 
n1 = 30000 #number of examples in the 1st training phase (2n total after full training)
n2 = 30000 #number of examples in a training phase (2n total after full training)
task1_type="digit1" #phase1
task2_type="oddeven50" #phase2
architecture="add" #same/add/replace - how do we handle ANN's output layers

#FIXME should be fixed after proper tensorflow output supression
if path.exists("FIM_1250.log"):
	os.remove("FIM_1250.log")
if path.exists("FIM_2500.log"):
	os.remove("FIM_2500.log")
if path.exists("FIM_5000.log"):
	os.remove("FIM_5000.log")
if path.exists("acc.log"):
	os.remove("acc.log")


twodigit_labels=ten_random_2digits() #FIXME only works for 10 output categories
(test_images_1digit, test_labels_1digit) = generate_1digit_set(10000)
(test_images_2digit, test_labels_2digit) = generate_2digit_set(10000,twodigit_labels)

if task1_type == "digit1":  
		(goldilocks_phase1_train_images, goldilocks_phase1_train_labels) = generate_1digit_set(n1)
elif task1_type == "digit2":  
		(goldilocks_phase1_train_images, goldilocks_phase1_train_labels) = generate_2digit_set(n1,twodigit_labels)
		#dothething

if task2_type == "digit1":  #not really going to use this one, but just in case
		(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_1digit_set(n2)
		(regular_phase1_train_images, regular_phase1_train_labels) = generate_1digit_set(n1)
		(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
elif task2_type == "digit2": #normal scattered 2-digit set
		(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set(n2,twodigit_labels)
		(regular_phase1_train_images, regular_phase1_train_labels) = generate_2digit_set(n1,twodigit_labels)
		(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
elif task2_type == "digit2_1stfixed":  
		#FIXME write this part
		(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set(n2,twodigit_labels)
elif task2_type == "oddeven50": 
		(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set_50categories(n2) 
		(regular_phase1_train_images, regular_phase1_train_labels) = generate_2digit_set_50categories(n1)
		(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
		for l in range(0,10000):
			test_labels_2digit[l] = (test_labels_2digit[l]//10+test_labels_2digit[l]%10)%2
elif task2_type == "oddeven":  
		(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set(n2,twodigit_labels)
		(regular_phase1_train_images, regular_phase1_train_labels) = generate_2digit_set(n1,twodigit_labels)
		(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
		#label modification, FIXME change it into a lambda operator or a function for clarity	
		for l in range(0,n1):
			regular_phase1_train_labels[l] = (regular_phase1_train_labels[l]//10+regular_phase1_train_labels[l]%10)%2
		for l in range(0,n2):
			regular_phase2_train_labels[l] = (regular_phase2_train_labels[l]//10+regular_phase2_train_labels[l]%10)%2
			goldilocks_phase2_train_labels[l] = (goldilocks_phase2_train_labels[l]//10+goldilocks_phase2_train_labels[l]%10)%2
		for l in range(0,10000):
			test_labels_2digit[l] = (test_labels_2digit[l]//10+test_labels_2digit[l]%10)%2



#normalize training set
regular_phase1_train_images = regular_phase1_train_images / 255.0
regular_phase2_train_images = regular_phase2_train_images  / 255.0
goldilocks_phase1_train_images = goldilocks_phase1_train_images / 255.0
goldilocks_phase2_train_images = goldilocks_phase2_train_images / 255.0
test_images_2digit = test_images_2digit / 255.0
test_images_1digit = test_images_1digit / 255.0



########################################DEBUG###############################################################
# reshape
regular_phase1_train_images = regular_phase1_train_images.reshape(n1,84,84,1)
regular_phase2_train_images = regular_phase2_train_images.reshape(n2,84,84,1)
goldilocks_phase1_train_images = goldilocks_phase1_train_images.reshape(n1,84,84,1)
goldilocks_phase2_train_images = goldilocks_phase2_train_images.reshape(n2,84,84,1)
test_images_2digit = test_images_2digit.reshape(10000,84,84,1)
test_images_1digit = test_images_1digit.reshape(10000,84,84,1)

#nn training


x_train = goldilocks_phase2_train_images#.astype('float32')#do we need the astype?
x_test = test_images_2digit#.astype('float32')

y_train = goldilocks_phase2_train_labels#.astype('float32')
y_test = test_labels_2digit#.astype('float32')

# Prepare the training dataset.
batch_size = 1250
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 3#!FIXME move this inside the cycle
#checkpoints
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


(doubledigit_images, doubledigit_labels) = generate_2digit_set_50categories(30000) 
doubledigit_images= doubledigit_images / 255.0
doubledigit_images= doubledigit_images.reshape(30000,84,84,1)
#goldilocks_phase1_train_images



#numpy.take(a, indices, axis=None, out=None, mode='raise')[source]¶
#np.argwhere(x>1)

model = generate_cnn_model()
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])






print('No pretraining')



extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

layer_name = 'dense1'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model(data)



ind1_list = [1, 3, 8]
ind2_list=[11, 13, 18, 33, 88, 38]

actlist = []
labellist = []
for i in ind1_list:
	ind=numpy.argwhere(goldilocks_phase1_train_labels==i)
	img=numpy.take(goldilocks_phase1_train_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("1 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#
	#print("1 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist+activations
	labellist=labellist+indices


for i in ind2_list:
	ind=numpy.argwhere(doubledigit_labels==i) #or (doubledigit_labels==int(str(i)[::-1])))
	img=numpy.take(doubledigit_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("2 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#.flatten()
	#print("2 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist + activations
	labellist=labellist+indices

#mat = numpy.array(mylist)



pickle.dump(actlist, open( "actlist.p", "wb" ) )
pickle.dump(labellist, open( "labellist.p", "wb" ) )

#find all indices of 3 in goldilocks_phase1_train_images
#take all images of 3
#find all indices of 11 doubledigit_images

#foreach index, take array of images, turn in into an array of activations, store in X and add indices to y
#X = mnist.data / 255.0
#y = mnist.target
#print(X.shape, y.shape)
#
# pickle.dump( favorite_color, open( "save.p", "wb" ) )

# favorite_color = pickle.load( open( "save.p", "rb" ) )

X=numpy.array(actlist)
y=numpy.array(labellist)
print("X type: ", type(X))
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
'''
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((82,82,64)).astype(float))

plt.show()
'''

print("started pca")
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.savefig("pca_0_50examples.png")

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3
)

pca_50 = PCA(n_components=10)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

plt.savefig("tsne_0_all.png")


#train on 2500 examples of single digit
train_images = goldilocks_phase1_train_images[0:2500]
train_labels = goldilocks_phase1_train_labels[0:2500]
print(len(train_images))

model.fit(train_images, 
		train_labels,  
		epochs=3)#, callbacks=[cp_callback])#,

print('Pre-training finished on 2500 examples')




ind1_list = [1, 3, 8]
ind2_list=[11, 13, 18, 33, 88, 38]

actlist = []
labellist = []
for i in ind1_list:
	ind=numpy.argwhere(goldilocks_phase1_train_labels==i)
	img=numpy.take(goldilocks_phase1_train_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("1 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#
	#print("1 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist+activations
	labellist=labellist+indices


for i in ind2_list:
	ind=numpy.argwhere(doubledigit_labels==i) #or (doubledigit_labels==int(str(i)[::-1])))
	img=numpy.take(doubledigit_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("2 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#.flatten()
	#print("2 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist + activations
	labellist=labellist+indices

#mat = numpy.array(mylist)



pickle.dump(actlist, open( "actlist.p", "wb" ) )
pickle.dump(labellist, open( "labellist.p", "wb" ) )

#find all indices of 3 in goldilocks_phase1_train_images
#take all images of 3
#find all indices of 11 doubledigit_images

#foreach index, take array of images, turn in into an array of activations, store in X and add indices to y
#X = mnist.data / 255.0
#y = mnist.target
#print(X.shape, y.shape)
#
# pickle.dump( favorite_color, open( "save.p", "wb" ) )

# favorite_color = pickle.load( open( "save.p", "rb" ) )

X=numpy.array(actlist)
y=numpy.array(labellist)
print("X type: ", type(X))
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
'''
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((82,82,64)).astype(float))

plt.show()
'''

print("started pca")
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.savefig("pca_2500_50examples.png")

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3
)

pca_50 = PCA(n_components=10)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

plt.savefig("tsne_2500_all.png")


#############################################################



#train on 5000 examples of single digit
train_images = goldilocks_phase1_train_images[2500:5000]
train_labels = goldilocks_phase1_train_labels[2500:5000]
print(len(train_images))

model.fit(train_images, 
		train_labels,  
		epochs=3)#, callbacks=[cp_callback])#,

print('Pre-training finished on 5000 examples')





ind1_list = [1, 3, 8]
ind2_list=[11, 13, 18, 33, 88, 38]

actlist = []
labellist = []
for i in ind1_list:
	ind=numpy.argwhere(goldilocks_phase1_train_labels==i)
	img=numpy.take(goldilocks_phase1_train_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("1 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#
	#print("1 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist+activations
	labellist=labellist+indices


for i in ind2_list:
	ind=numpy.argwhere(doubledigit_labels==i) #or (doubledigit_labels==int(str(i)[::-1])))
	img=numpy.take(doubledigit_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("2 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#.flatten()
	#print("2 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist + activations
	labellist=labellist+indices

#mat = numpy.array(mylist)



pickle.dump(actlist, open( "actlist.p", "wb" ) )
pickle.dump(labellist, open( "labellist.p", "wb" ) )

#find all indices of 3 in goldilocks_phase1_train_images
#take all images of 3
#find all indices of 11 doubledigit_images

#foreach index, take array of images, turn in into an array of activations, store in X and add indices to y
#X = mnist.data / 255.0
#y = mnist.target
#print(X.shape, y.shape)
#
# pickle.dump( favorite_color, open( "save.p", "wb" ) )

# favorite_color = pickle.load( open( "save.p", "rb" ) )

X=numpy.array(actlist)
y=numpy.array(labellist)
print("X type: ", type(X))
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
'''
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((82,82,64)).astype(float))

plt.show()
'''

print("started pca")
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.savefig("pca_5000_50examples.png")

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3
)

pca_50 = PCA(n_components=10)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

plt.savefig("tsne_5000_all.png")

#########################################################################


#train on 10000 examples of single digit
train_images = goldilocks_phase1_train_images[5000:10000]
train_labels = goldilocks_phase1_train_labels[5000:10000]
print(len(train_images))

model.fit(train_images, 
		train_labels,  
		epochs=3)#, callbacks=[cp_callback])#,

print('Pre-training finished on 10000 examples')



ind1_list = [1, 3, 8]
ind2_list=[11, 13, 18, 33, 88, 38]

actlist = []
labellist = []
for i in ind1_list:
	ind=numpy.argwhere(goldilocks_phase1_train_labels==i)
	img=numpy.take(goldilocks_phase1_train_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("1 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#
	#print("1 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist+activations
	labellist=labellist+indices


for i in ind2_list:
	ind=numpy.argwhere(doubledigit_labels==i) #or (doubledigit_labels==int(str(i)[::-1])))
	img=numpy.take(doubledigit_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("2 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#.flatten()
	#print("2 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist + activations
	labellist=labellist+indices

#mat = numpy.array(mylist)



pickle.dump(actlist, open( "actlist.p", "wb" ) )
pickle.dump(labellist, open( "labellist.p", "wb" ) )

#find all indices of 3 in goldilocks_phase1_train_images
#take all images of 3
#find all indices of 11 doubledigit_images

#foreach index, take array of images, turn in into an array of activations, store in X and add indices to y
#X = mnist.data / 255.0
#y = mnist.target
#print(X.shape, y.shape)
#
# pickle.dump( favorite_color, open( "save.p", "wb" ) )

# favorite_color = pickle.load( open( "save.p", "rb" ) )

X=numpy.array(actlist)
y=numpy.array(labellist)
print("X type: ", type(X))
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
'''
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((82,82,64)).astype(float))

plt.show()
'''

print("started pca")
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.savefig("pca_10000_50examples.png")

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3
)

pca_50 = PCA(n_components=10)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

plt.savefig("tsne_10000_all.png")

########################################
#train on 15000 examples of single digit
train_images = goldilocks_phase1_train_images[10000:15000]
train_labels = goldilocks_phase1_train_labels[10000:15000]
print(len(train_images))

model.fit(train_images, 
		train_labels,  
		epochs=3)#, callbacks=[cp_callback])#,

print('Pre-training finished on 15000 examples')




ind1_list = [1, 3, 8]
ind2_list=[11, 13, 18, 33, 88, 38]

actlist = []
labellist = []
for i in ind1_list:
	ind=numpy.argwhere(goldilocks_phase1_train_labels==i)
	img=numpy.take(goldilocks_phase1_train_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("1 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#
	#print("1 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist+activations
	labellist=labellist+indices


for i in ind2_list:
	ind=numpy.argwhere(doubledigit_labels==i) #or (doubledigit_labels==int(str(i)[::-1])))
	img=numpy.take(doubledigit_images, ind, axis=0)
	img_cropped=img[0:100]
	#print("2 digit, img_cropped len:", len(img_cropped))
	activations = [intermediate_layer_model(np.reshape(v, (-1, 84, 84, 1)))[0].numpy().flatten() for v in img_cropped]#.flatten()
	#print("2 digit, activations len:", len(activations))
	indices = [i]*100
	actlist=actlist + activations
	labellist=labellist+indices

#mat = numpy.array(mylist)



pickle.dump(actlist, open( "actlist.p", "wb" ) )
pickle.dump(labellist, open( "labellist.p", "wb" ) )

#find all indices of 3 in goldilocks_phase1_train_images
#take all images of 3
#find all indices of 11 doubledigit_images

#foreach index, take array of images, turn in into an array of activations, store in X and add indices to y
#X = mnist.data / 255.0
#y = mnist.target
#print(X.shape, y.shape)
#
# pickle.dump( favorite_color, open( "save.p", "wb" ) )

# favorite_color = pickle.load( open( "save.p", "rb" ) )

X=numpy.array(actlist)
y=numpy.array(labellist)
print("X type: ", type(X))
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
'''
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((82,82,64)).astype(float))

plt.show()
'''

print("started pca")
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.savefig("pca_15000_50examples.png")

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3
)

pca_50 = PCA(n_components=10)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("bright", 9),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

plt.savefig("tsne_15000_all.png")


'''
img = np.reshape(x_train[1], (-1, 28, 28, 1))
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
features = extractor(img)
features_np=features[0].numpy()'''

'''
################################
#train on 10000 examples of single digit
train_images = goldilocks_phase1_train_images[2500:10000]
train_labels = goldilocks_phase1_train_labels[2500:10000]
model.fit(train_images, 
		train_labels,  
		epochs=3)#, callbacks=[cp_callback])#,

print('Pre-training finished on 10000 examples')

'''



