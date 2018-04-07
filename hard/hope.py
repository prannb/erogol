import spacy
nlp = spacy.load('en')
import numpy as np
import re
with open("zaidi.tsv") as f:
    my_list = f.readlines()
yh = np.load('y_pred_val.npy')
y = np.load('y_test_val.npy')
yhat = (yh.ravel() < 0.5)*1.0
yambig = (yh.ravel() < 0.6)*1.0
yambig = yambig * ((yh.ravel() >0.4)*1.0)
l = len(my_list)
err = yhat[0:l]-y[0:l]
wrong = []
num = 0
for i in range(0,l):
	if yhat[i]!=y[i] and yh[i]>0.4 and yh[i]<0.6:
		num = num + 1
		wrong = wrong + [my_list[i]]
		questions = my_list[i].split('\t')
		q1 = questions[3]
		q2 = questions[4]
		q1 = unicode(q1, "utf-8")
		q2 = unicode(q2, "utf-8")
		nlp_q1 = nlp(q1)
		nlp_q2 = nlp(q2)
		proper_1 = []
		proper_2 = []
		for token in nlp_q1:
			if (token.pos_ == 'PROPN'):
				proper_1.append(token)
		for token in nlp_q2:
			if (token.pos_ == 'PROPN'):
				proper_2.append(token)
		highest = 0
		pair = ['a', 'a']
		for i in range(len(proper_1)):
			for j in range(len(proper_2)):
				# print '------------------------------------'
				# print word1
				# print word2
				# print '------------------------------------'
				# a1 = unicode(word1, "utf-8")
				# a2 = unicode(word2, "utf-8")
				word1 = proper_1[i]
				word2 = proper_2[j]
				sim = word1.similarity(word2)
				if (sim > highest and sim != 1.0):
					highest = sim
					pair[0] = word1
					pair[1] = word2
		if (pair[0] != pair[1]):
			print '####################################################'
			print pair[0]
			print q1
			print pair[1]
			print q2
			print '####################################################'
		# print q1
		# print q2
print num
wfile = open('wrong.txt', 'w')
for item in wrong:
  wfile.write("%s\n" % item)