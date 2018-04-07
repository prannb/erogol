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
num1 = 0
# ques = ['Did', 'How', 'What', 'Where', 'Can', 'Are', 'Is', 'Why', 'Should', 'Would', 'When', 'Were', 'Is', 'Was', 'Whom', 'Do', 'Does', 'Whom']
ques1 = {'Could':0, 'Would':1, 'Should':2, 'Can':3, 'Will':4, 'Shall':5, 'Did':6, 'How':7, 'What':8, 'Where':9, 'Are':10, 'Is':11, 'Why':12, 'When':13, 'Were':14, 'Is':15, 'Was':16, 'Whom':17, 'Do':18, 'Does':19, 'Whom':20}
for i in range(0,l):
	num1 = num1 + 1
	if yhat[i]!=y[i] and yhat[i]==1 and yh[i]>0.4 and yh[i]<0.6:
		wrong = wrong + [str(yh[i])+my_list[i]]
		# wrong = wrong + [my_list[i]]
		questions = my_list[i].split('\t')
		q1 = questions[3].split(' ')
		q2 = questions[4].split(' ')
		if q1[0] in ques1 and q2[0] in ques1:
			if q1[0] != q2[0]:
				num = num + 1
				# print '##########################'
				# print (q1,str(yh[i]))
				# print (q2,str(yh[i]))
				# print '##########################'

		# 	if q1[0] == ques_word.split('\') :
print num
print num1
# wfile = open('wrong.txt', 'w')
# for item in wrong:
#   wfile.write("%s\n" % item)