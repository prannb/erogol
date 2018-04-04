from __future__ import absolute_import
from __future__ import print_function

import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
df = pd.read_csv("quora.tsv",delimiter='\t')

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

print("done with loading training data ...")

stoplist = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(df['question1']) + list(df['question2'])

count_vect = CountVectorizer(stop_words = stoplist)
q_counts = count_vect.fit_transform(questions)
q1_counts = q_counts[0:len(list(df['question1']))]
q2_counts = q_counts [len(list(df['question1'])):] 
	
#tf normalizes
tf_transformer = TfidfTransformer(use_idf=False).fit(q1_counts)
q1_tf = tf_transformer.transform(q1_counts)
tf_transformer= TfidfTransformer(use_idf=False).fit(q2_counts)
q2_tf = tf_transformer.transform(q2_counts)
# df['q1_feats'] = list(q1_tf)
# df['q2_feats'] = list(q2_tf)
print("Tf vectors done")

df = df.reindex(np.random.permutation(df.index))

# set number of train and test instances
num_train = int(df.shape[0] * 0.75)
num_val = int(df.shape[0] * 0.10)
num_test = df.shape[0] - num_train - num_val				 
print("Number of training pairs: %i"%(num_train))
print("Number of testing pairs: %i"%(num_test))

# init data data arrays
vec_size = q1_tf.shape[1]
# X_train = np.zeros([num_train, 2, vec_size])
# X_val = np.zeros([num_val, 2, vec_size])
# X_test  = np.zeros([num_test, 2, vec_size])
Y_train = np.zeros([num_train]) 
Y_val = np.zeros([num_val]) 
Y_test = np.zeros([num_test])
from scipy.sparse import csr_matrix
X_train_1 = csr_matrix((num_train, vec_size))
X_train_2 = csr_matrix((num_train, vec_size))
X_val_1 = csr_matrix((num_val, vec_size))
X_val_2 = csr_matrix((num_val, vec_size))
X_test_1 = csr_matrix((num_test, vec_size))
X_test_2 = csr_matrix((num_test, vec_size))

q1_feats = q1_tf
q2_feats = q2_tf
del q1_counts
del q2_counts
del q1_tf
del q2_tf

# fill data arrays with features
X_train_1 = q1_feats[:num_train]
X_train_2 = q2_feats[:num_train]
Y_train = df[:num_train]['is_duplicate'].values

X_val_1 = q1_feats[num_train:num_train+num_val]
X_val_2 = q2_feats[num_train:num_train+num_val]
Y_val = df[num_train:num_train+num_val]['is_duplicate'].values


X_test_1 = q1_feats[num_train+num_val:]
X_test_2 = q2_feats[num_train+num_val:]
Y_test = df[num_train+num_val:]['is_duplicate'].values

# remove useless variables

del q1_feats
del q2_feats


print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K


def euclidean_distance(vects):
	x, y = vects
	# a = raw_input("some:")
	print(x)
	# a = raw_input("some:")
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

def cosine_distance(vests):
	x, y = vests
	x = K.l2_normalize(x, axis=-1)
	y = K.l2_normalize(y, axis=-1)
	return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0],1)

def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(input_dim):
	'''
	Base network for feature extraction.
	'''
	input = Input(shape=(input_dim, ))
	print("#####################################")
	print(input.shape)	
	print("#####################################")
	dense1 = Dense(128)(input)
	bn1 = BatchNormalization()(dense1)
	relu1 = Activation('relu')(bn1)

	dense2 = Dense(128)(relu1)
	bn2 = BatchNormalization()(dense2)
	res2 = merge([relu1, bn2], mode='sum')
	relu2 = Activation('relu')(res2)	

	dense3 = Dense(128)(relu2)
	bn3 = BatchNormalization()(dense3)
	res3 = Merge(mode='sum')([relu2, bn3])
	relu3 = Activation('relu')(res3)   
	
	feats = merge([relu3, relu2, relu1], mode='concat')
	bn4 = BatchNormalization()(feats)

	model = Model(input=input, output=bn4)
	print("#####################################")
	print(bn4.shape)	
	print("#####################################")

	return model


def compute_accuracy(predictions, labels):
	'''
	Compute classification accuracy with a fixed threshold on distances.
	'''
	return np.mean(np.equal(predictions.ravel() < 0.5, labels))

def create_network(input_dim):
	# network definition
	base_network = create_base_network(input_dim)
	
	input_a = Input(shape=(input_dim,))
	input_b = Input(shape=(input_dim,))
	
	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	# a = raw_input("print something:")
	print(processed_a)
	
	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	
	model = Model(input=[input_a, input_b], output=distance)
	return model



print("FFFFFFFFFFFFFF")
# from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
net = create_network(vec_size)

# train
#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

print("GGGGGGGGGGGGGGGGGGG")

for epoch in range(50):
	net.fit([X_train_1, X_train_2], Y_train,
		  validation_data=([X_val_1, X_val_2], Y_val),
		  batch_size=128, nb_epoch=1, shuffle=True, )
	
	# compute final accuracy on training and test sets
	pred = net.predict([X_test_1, X_test_2], batch_size=128)
	te_acc = compute_accuracy(pred, Y_test)
#	print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

model_json = net.to_json()
with open("model_val_tf.json", "w") as json_file:
	json_file.write(model_json)
net.save_weights("model_val_tf.h5")
