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



import spacy
nlp = spacy.load('en')

# vecs1 = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=50)]
# vecs1 =  np.array(vecs1)
# df['q1_feats'] = list(vecs1)

# vecs2 = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=50)]
# vecs2 =  np.array(vecs2)
# df['q2_feats'] = list(vecs2)

# # save features
# pd.to_pickle(df, 'data/1_df.pkl')






print("done with loading word vectors ...")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(df['question1']) + list(df['question2'])

tfidf = TfidfVectorizer(lowercase=False, )
tfidf.fit_transform(questions)

# dict key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))



print("done with sklearn imports ...")

vecs1 = []
for qu in tqdm(list(df['question1'])):
	doc = nlp(qu) 
	mean_vec = np.zeros([len(doc), 384])
	for word in doc:
		# word2vec
		vec = word.vector
		# fetch df score
		try:
			idf = word2tfidf[str(word)]
		except:
			#print word
			idf = 0
		# compute final vec
		# print(mean_vec.shape, vec.shape)
		mean_vec += vec * idf
	mean_vec = mean_vec.mean(axis=0)
	vecs1.append(mean_vec)
# print(df.values())
df['q1_feats'] = list(vecs1)


print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")

vecs2 = []
for qu in tqdm(list(df['question2'])):
	doc = nlp(qu) 
	mean_vec = np.zeros([len(doc), 384])
	for word in doc:
		# word2vec
		vec = word.vector
		# fetch df score
		try:
			idf = word2tfidf[str(word)]
		except:
			#print word
			idf = 0
		# compute final vec
		mean_vec += vec * idf
	mean_vec = mean_vec.mean(axis=0)
	vecs2.append(mean_vec)
df['q2_feats'] = list(vecs2)


print("DDDDDDDDDDDDDDDDDDDDDDDDDDDD")

df = df.reindex(np.random.permutation(df.index))

# set number of train and test instances
num_train = int(df.shape[0] * 0.88)
num_test = df.shape[0] - num_train				 
print("Number of training pairs: %i"%(num_train))
print("Number of testing pairs: %i"%(num_test))

# init data data arrays
X_train = np.zeros([num_train, 2, 384])
X_test  = np.zeros([num_test, 2, 384])
Y_train = np.zeros([num_train]) 
Y_test = np.zeros([num_test])

# format data 
b = [a[None,:] for a in list(df['q1_feats'].values)]
q1_feats = np.concatenate(b, axis=0)

b = [a[None,:] for a in list(df['q2_feats'].values)]
q2_feats = np.concatenate(b, axis=0)

# fill data arrays with features
X_train[:,0,:] = q1_feats[:num_train]
X_train[:,1,:] = q2_feats[:num_train]
Y_train = df[:num_train]['is_duplicate'].values
			
X_test[:,0,:] = q1_feats[num_train:]
X_test[:,1,:] = q2_feats[num_train:]
Y_test = df[num_train:]['is_duplicate'].values

# remove useless variables
del b
del q1_feats
del q2_feats


print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K


def euclidean_distance(vects):
	x, y = vects
	a = raw_input("some:")
	print(x)
	a = raw_input("some:")
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
	a = raw_input("print something:")
	print(processed_a)
	
	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	
	model = Model(input=[input_a, input_b], output=distance)
	return model



print("FFFFFFFFFFFFFF")
# from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
net = create_network(384)

# train
#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

print("GGGGGGGGGGGGGGGGGGG")

for epoch in range(50):
	net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
		  validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
		  batch_size=128, nb_epoch=1, shuffle=True, )
	
	# compute final accuracy on training and test sets
	pred = net.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
	te_acc = compute_accuracy(pred, Y_test)
#	print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
model_json = net.to_json()
with open("model_org.json", "w") as json_file:
	json_file.write(model_json)
net.save_weights("model_org.h5")
