from __future__ import absolute_import
from __future__ import print_function

import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import load_model
df = pd.read_csv("quora.tsv",delimiter='\t')

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

print("done with loading training data ...")

stoplist = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam


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


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

for num_file in range(2):
	print("#################################")
	print(num_file)
	print("#################################")
	
	filename = "quora_train_" + str(num_file) + ".tsv"
	df = pd.read_csv(filename,delimiter='\t')

	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
	df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

	print("done with loading training data ...")

	print("done with loading word vectors ...")

	questions = list(df['question1']) + list(df['question2'])
	# vecs1 = []
	# max_queslen = 100
	# max = 0
	# for qu in tqdm(list(df['question1'])):
	# 	a = qu.split(' ')
	# 	print(len(a))
	# 	print(len(nlp(qu)))

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
	# print(vecs2[1].shape)
	# print(df['q1_feats'][1][0])
	# set number of train and test instances
	num_train = int(df.shape[0] * 0.88)
	num_test = df.shape[0] - num_train				 
	print("Number of training pairs: %i"%(num_train))
	print("Number of testing pairs: %i"%(num_test))

	vec_size = q1_tf.shape[1]
	
	X_train = np.zeros([num_train, 2, vec_size])
	X_test  = np.zeros([num_test, 2, vec_size])
	Y_train = np.zeros([num_train]) 
	Y_test = np.zeros([num_test])


	q1_feats = (q1_tf.toarray())
	q2_feats = (q2_tf.toarray())

	X_train[:,0,:] = q1_feats[:num_train]
	X_train[:,1,:] = q2_feats[:num_train]
	Y_train = df[:num_train]['is_duplicate'].values

	X_test[:,0,:] = q1_feats[num_train:]
	X_test[:,1,:] = q2_feats[num_train:]
	Y_test = df[num_train:]['is_duplicate'].values

	del q1_feats
	del q2_feats
	# def create_bidirLSTM():
		
	#---------------------- 	Input Prepared	---------------------------------------------

	
	if (num_file == 0):
		net = create_network(vec_size)

		# train
		#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
		
	else:
		json_file = open('model_batch_tf.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		net = model_from_json(loaded_model_json)
		net.load_weights("model_batch_tf.h5")

	optimizer = Adam(lr=0.001)
	net.compile(loss=contrastive_loss, optimizer=optimizer)
	print("GGGGGGGGGGGGGGGGGGG")


	for epoch in range(1):
		net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
		  validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
		  batch_size=128, nb_epoch=1, shuffle=True, )
		
		# compute final accuracy on training and test sets
		# pred = net.predict([X_test1, X_test2], batch_size=128)
		# print(pred.shape)
		# te_acc = compute_accuracy(pred, Y_test)
		# print('* Accuracy on training set: %0.2f%%' % (100 * te_acc))
	# print(pred.shape)
	# print(Y_test.shape)
	# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
	model_json = net.to_json()
	with open("model_batch_tf.json", "w") as json_file:
		json_file.write(model_json)
	net.save_weights("model_batch_tf.h5")



json_file = open('model_batch_tf.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
net = model_from_json(loaded_model_json)
net.load_weights("model_batch_tf.h5")
cumm_accuracy = 0
#------------------	TESTING STARTS --------------------------------------------------
for num_file in range(2):
	print("#################################")
	print(num_file)
	print("#################################")
	
	filename = "quora_test_" + str(num_file) + ".tsv"
	df = pd.read_csv(filename,delimiter='\t')

	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
	df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

	print("done with loading training data ...")

	print("done with loading word vectors ...")

	questions = list(df['question1']) + list(df['question2'])
	# vecs1 = []
	# max_queslen = 100
	# max = 0
	# for qu in tqdm(list(df['question1'])):
	# 	a = qu.split(' ')
	# 	print(len(a))
	# 	print(len(nlp(qu)))

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
	# print(vecs2[1].shape)
	# print(df['q1_feats'][1][0])
	# set number of train and test instances
	num_test = int(df.shape[0])				 
	# print("Number of training pairs: %i"%(num_train))
	print("Number of testing pairs: %i"%(num_test))

	vec_size = q1_tf.shape[1]
	X_test  = np.zeros([num_test, 2, vec_size])
	Y_test = np.zeros([num_test])


	q1_feats = (q1_tf.toarray())
	q2_feats = (q2_tf.toarray())

	X_test[:,0,:] = q1_feats[num_train+num_val:]
	X_test[:,1,:] = q2_feats[num_train+num_val:]
	Y_test = df[num_train+num_val:]['is_duplicate'].values

	del q1_feats
	del q2_feats

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


	for epoch in range(1):
		# net.fit([X_train1, X_train2], Y_train,
			# validation_data=([X_test1, X_test2], Y_test),
			# batch_size=64, nb_epoch=1, shuffle=True, )
		
		# compute final accuracy on training and test sets
		pred = net.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
		te_acc = compute_accuracy(pred, Y_test)		# print('* Accuracy on training set: %0.2f%%' % (100 * te_acc))
	print(pred.shape)
	print(Y_test.shape)
	print('* Accuracy on test set for this data: %0.2f%%' % (100 * te_acc))
	cumm_accuracy = (cumm_accuracy * num_file + 100 * te_acc)/(num_file + 1)
	print('*Cummulative Accuracy on test set for this data: %0.2f%%' % (cumm_accuracy))	
	# model_json = net.to_json()
	# with open("model.json", "w") as json_file:
	# 	json_file.write(model_json)