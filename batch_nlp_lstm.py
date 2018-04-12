from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm



# exit()	

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge, Flatten
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model

from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional


def euclidean_distance(vects):
	x, y = vects

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

# def create_bidir_lstm(input_dim):
# 	inputi = Input(shape=(max_queslen,input_dim))
# 	output1 = Bidirectional(LSTM(2, return_sequences=True), merge_mode='ave', input_shape=(max_queslen, input_dim))(inputi)
# 	output2 = (Dense(5, activation='relu'))(output1)
# 	output3 = Flatten()(output2)
# 	model = Model(input=inputi, output=output2)

# 	return model
	
# def average(mats):
# 	k = 0
# 	mat = mats
	# a = raw_input("asome:")
	# print(mat)
	# a = raw_input("asome:")
	# K.reshape(mat.shape[0])
	# for i in mat:
	# 	a = np.zeros([384])
	# 	for j in i:
	# 		a = a + j
	# 	mat[k] = a

	# return mat

def create_base_network(input_dim):
	'''
	Base network for feature extraction.
	'''
	# create_bidirLSTM()
	input = Input(shape=(max_queslen,input_dim))
	output1 = Bidirectional(LSTM(12, return_sequences=True), merge_mode='ave', input_shape=(max_queslen, input_dim))(input)
	print("#####################################")
	print(output1)	
	print("#####################################")
	output2 = TimeDistributed(Dense(30, activation='relu'))(output1)
	print("#####################################")
	print(output2)	
	print("#####################################")
	output3 = Flatten()(output2)
	print("#####################################")
	print(output3)	
	print("#####################################")
	# a = raw_input("some:")
	# newout = output2.Session()
	# print(type(newout))
	# for i in output2:
	#	 newout = newout + i
	# sess = tf.Session()
	# with sess.as_default():
	#	 output2_np = output2.eval()
	# print("#####################################")
	# print(tf.Session().run(output2))	
	# print("#####################################")

	# newout = np.array(newout)
	# print("#####################################")
	# print(newout.shape)	
	# print("#####################################")	

	# network = create_bidir_lstm(input_dim)
	
	# input_a = Input(shape=(max_queslen,input_dim))
	
	# processed_a = network(input_a)
	# distance = Lambda(average, output_shape=eucl_dist_output_shape)([processed_a])

	# print("#####################################")
	# print(processed_a)	
	# print("#####################################")
	# a = raw_input("some:")
	dense1 = Dense(128)(output3)
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
	return np.mean(np.equal(predictions.ravel() < 0.2, labels))

def create_network(input_dim):
	# network definition
	base_network = create_base_network(input_dim)
	
	input_a = Input(shape=(max_queslen,input_dim))
	input_b = Input(shape=(max_queslen,input_dim))
	
	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	
	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	
	model = Model(input=[input_a, input_b], output=distance)
	return model



print("FFFFFFFFFFFFFF")
# from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
import spacy
nlp = spacy.load('en')
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
net = model_from_json(loaded_model_json)
net.load_weights("model.h5")

optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)
print("GGGGGGGGGGGGGGGGGGG")


# for num_file in range(2):
# 	print("#################################")
# 	print(num_file)
# 	print("#################################")
	
# 	filename = "train/quora_train_" + str(num_file) + ".tsv"
# 	df = pd.read_csv(filename,delimiter='\t')

# 	# encode questions to unicode
# 	df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
# 	df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

# 	print("done with loading training data ...")




# 	print("done with loading word vectors ...")

# 	questions = list(df['question1']) + list(df['question2'])
# 	vecs1 = []
# 	max_queslen = 100
# 	# max = 0
# 	# for qu in tqdm(list(df['question1'])):
# 	# 	a = qu.split(' ')
# 	# 	print(len(a))
# 	# 	print(len(nlp(qu)))

# 	corpus1 = []
# 	for qu in tqdm(list(df['question1'])):
# 		doc = nlp(qu)
# 		document = []
# 		# i = 0
# 		# print(1)
# 		for word in doc:
# 			# word2vec
# 			vec = word.vector
# 			document.append(vec)
# 			# i = i + 1
# 			# print(vec)
# 			# fetch df score
# 		document = np.array(document)
# 		corpus1.append(document)
# 		# print(len(document))
# 		# print(len(corpus1))
		
# 		# vecs1.append(doc_3d)
# 	df['q1_feats'] = list(corpus1)

# 	corpus2 = []
# 	for qu in tqdm(list(df['question2'])):
# 		doc = nlp(qu)
# 		document = []
# 		# i = 0
# 		for word in doc:
# 			# word2vec
# 			vec = word.vector
# 			document.append(vec)
# 			# i = i + 1
# 			# print(vec)
# 			# fetch df score
# 		document = np.array(document)
# 		corpus2.append(document)
# 	# exit()
# 		# vecs1.append(doc_3d)
# 	df['q2_feats'] = list(corpus2)

# 	df = df.reindex(np.random.permutation(df.index))

# 	# print(vecs2[1].shape)
# 	# print(df['q1_feats'][1][0])

# 	# set number of train and test instances
# 	num_train = int(df.shape[0] * 0.88)
# 	num_test = df.shape[0] - num_train				 
# 	print("Number of training pairs: %i"%(num_train))
# 	print("Number of testing pairs: %i"%(num_test))

# 	X_train = np.zeros([num_train, 2, 384])
# 	X_test  = np.zeros([num_test, 2, 384])
# 	Y_train = np.zeros([num_train]) 
# 	Y_test = np.zeros([num_test])



# 	Y_train = df[:num_train]['is_duplicate'].values
# 	Y_test = df[num_train:]['is_duplicate'].values

# 	#-----------------------	PADDING		------------------------------------------------------------
# 	corpus1 = np.array(corpus1)
# 	corpus2 = np.array(corpus2)


# 	corpus1_new = np.zeros([corpus1.shape[0], max_queslen, corpus1[0].shape[1]])
# 	corpus2_new = np.zeros([corpus2.shape[0], max_queslen, corpus2[0].shape[1]])

# 	j = 0
# 	for i in corpus1:
# 		if (len(i) == 0):
# 			j = j + 1
# 			continue
# 		if (i.shape[0]<max_queslen):
# 			b = np.zeros([max_queslen - i.shape[0], i.shape[1]])
# 			i = np.concatenate((i,b), axis = 0)
# 		else :
# 			i = i[0:max_queslen,]
# 		corpus1_new[j] = i
# 		j = j + 1

# 	j = 0
# 	for i in corpus2:
# 		if (len(i) == 0):
# 			j = j + 1
# 			continue
# 		if (i.shape[0]<max_queslen):
# 			b = np.zeros([max_queslen - i.shape[0], i.shape[1]])
# 			i = np.concatenate((i,b), axis = 0)
# 		else :
# 			i = i[0:max_queslen,]
# 		corpus2_new[j] = i
# 		j = j + 1

# 	# print(corpus1_new)
# 	# print(corpus2_new)
# 	# print(corpus1_new.shape)
# 	# print(corpus2_new.shape)

# 	X_train1 = corpus1_new[0:num_train,]
# 	X_train2 = corpus2_new[0:num_train,]
# 	X_test1 = corpus1_new[num_train:num_train+num_test,]
# 	X_test2 = corpus2_new[num_train:num_train+num_test,]


# 	# def create_bidirLSTM():
		
# 	#---------------------- 	Input Prepared	---------------------------------------------

	
# 	# if (num_file == 0):
# 	# 	net = create_network(vec_size)

# 	# 	# train
# 	# 	#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
		
# 	# else:


# 	for epoch in range(5):
# 		net.fit([X_train1, X_train2], Y_train,
# 			validation_data=([X_test1, X_test2], Y_test),
# 			batch_size=64, nb_epoch=1, shuffle=True, )
		
# 		# compute final accuracy on training and test sets
# 		# pred = net.predict([X_test1, X_test2], batch_size=128)
# 		# print(pred.shape)
# 		# te_acc = compute_accuracy(pred, Y_test)
# 		# print('* Accuracy on training set: %0.2f%%' % (100 * te_acc))
# 	# print(pred.shape)
# 	# print(Y_test.shape)
# 	# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
	

# model_json = net.to_json()
# with open("model.json", "w") as json_file:
# 	json_file.write(model_json)
# net.save_weights("model.h5")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
net = model_from_json(loaded_model_json)
net.load_weights("model.h5")
cumm_accuracy = 0
#------------------	TESTING STARTS --------------------------------------------------
for num_file in range(80):
	print("#################################")
	print(num_file)
	print("#################################")
	
	filename = "test/quora_test_" + str(num_file) + ".tsv"
	df = pd.read_csv(filename,delimiter='\t')

	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
	df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

	print("done with loading training data ...")




	print("done with loading word vectors ...")

	questions = list(df['question1']) + list(df['question2'])
	vecs1 = []
	max_queslen = 100
	# max = 0
	# for qu in tqdm(list(df['question1'])):
	# 	a = qu.split(' ')
	# 	print(len(a))
	# 	print(len(nlp(qu)))

	corpus1 = []
	for qu in tqdm(list(df['question1'])):
		doc = nlp(qu)
		document = []
		# i = 0
		# print(1)
		for word in doc:
			# word2vec
			vec = word.vector
			document.append(vec)
			# i = i + 1
			# print(vec)
			# fetch df score
		document = np.array(document)
		corpus1.append(document)
		# print(len(document))
		# print(len(corpus1))
		
		# vecs1.append(doc_3d)
	df['q1_feats'] = list(corpus1)

	corpus2 = []
	for qu in tqdm(list(df['question2'])):
		doc = nlp(qu)
		document = []
		# i = 0
		for word in doc:
			# word2vec
			vec = word.vector
			document.append(vec)
			# i = i + 1
			# print(vec)
			# fetch df score
		document = np.array(document)
		corpus2.append(document)
	# exit()
		# vecs1.append(doc_3d)
	df['q2_feats'] = list(corpus2)

	df = df.reindex(np.random.permutation(df.index))

	# print(vecs2[1].shape)
	# print(df['q1_feats'][1][0])

	# set number of train and test instances
	num_test = int(df.shape[0])
	# num_test = df.shape[0] - num_train				 
	print("Number of testing pairs: %i"%(num_test))
	# print("Number of testing pairs: %i"%(num_test))

	X_test = np.zeros([num_test, 2, 384])
	# X_test  = np.zeros([num_test, 2, 384])
	Y_test = np.zeros([num_test]) 
	# Y_test = np.zeros([num_test])



	Y_test = df[:num_test]['is_duplicate'].values
	# Y_test = df[num_train:]['is_duplicate'].values

	#-----------------------	PADDING		------------------------------------------------------------
	corpus1 = np.array(corpus1)
	corpus2 = np.array(corpus2)


	corpus1_new = np.zeros([corpus1.shape[0], max_queslen, corpus1[0].shape[1]])
	corpus2_new = np.zeros([corpus2.shape[0], max_queslen, corpus2[0].shape[1]])

	j = 0
	for i in corpus1:
		if (len(i) == 0):
			j = j + 1
			continue
		if (i.shape[0]<max_queslen):
			b = np.zeros([max_queslen - i.shape[0], i.shape[1]])
			i = np.concatenate((i,b), axis = 0)
		else :
			i = i[0:max_queslen,]
		corpus1_new[j] = i
		j = j + 1

	j = 0
	for i in corpus2:
		if (len(i) == 0):
			j = j + 1
			continue
		if (i.shape[0]<max_queslen):
			b = np.zeros([max_queslen - i.shape[0], i.shape[1]])
			i = np.concatenate((i,b), axis = 0)
		else :
			i = i[0:max_queslen,]
		corpus2_new[j] = i
		j = j + 1

	# print(corpus1_new)
	# print(corpus2_new)
	# print(corpus1_new.shape)
	# print(corpus2_new.shape)

	X_test1 = corpus1_new[0:num_test,]
	X_test2 = corpus2_new[0:num_test,]
	# X_test1 = corpus1_new[num_train:num_train+num_test,]
	# X_test2 = corpus2_new[num_train:num_train+num_test,]


	# def create_bidirLSTM():
		
	#---------------------- 	Input Prepared	---------------------------------------------

	
	# if (num_file == 0):
	# 	net = create_network(384)

	# 	# train
	# 	#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
		
	# else:
	# json_file = open('model.json', 'r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# net = model_from_json(loaded_model_json)

	# optimizer = Adam(lr=0.001)
	# net.compile(loss=contrastive_loss, optimizer=optimizer)
	# print("GGGGGGGGGGGGGGGGGGG")


	for epoch in range(50):
		# net.fit([X_train1, X_train2], Y_train,
			# validation_data=([X_test1, X_test2], Y_test),
			# batch_size=64, nb_epoch=1, shuffle=True, )
		
		# compute final accuracy on training and test sets
		pred = net.predict([X_test1, X_test2], batch_size=64)
		# print(pred.shape)
		te_acc = compute_accuracy(pred, Y_test)
		# print('* Accuracy on training set: %0.2f%%' % (100 * te_acc))
	print(pred.shape)
	print(Y_test.shape)
	print('* Accuracy on test set for this data: %0.2f%%' % (100 * te_acc))
	cumm_accuracy = (cumm_accuracy * num_file + 100 * te_acc)/(num_file + 1)
	print('*Cummulative Accuracy on test set for this data: %0.2f%%' % (cumm_accuracy))	
	# model_json = net.to_json()
	# with open("model.json", "w") as json_file:
	# 	json_file.write(model_json)