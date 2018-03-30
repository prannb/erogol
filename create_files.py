def main():
	file = open('quora_w.tsv', 'r')
	text = file.read()
	text = text.split('\n')
	# print (text[0])
	# del text[0]
	j = 0
	k = 0
	# print (text[0])	
	# exit()
	corpus = "id	qid1	qid2	question1	question2	is_duplicate\n"
	for i in text:
		if (k%1000 == 0 and k!=0):
			filename = "quora_" + str(j) + ".tsv"
			j = j + 1
			res = open(filename, "w");
			res.write(corpus)
			# print corpus
			# a = raw_input("dsf:")
			res.close()
			corpus = "id	qid1	qid2	question1	question2	is_duplicate\n"
			# print i
		corpus = corpus + i + "\n"
		k = k + 1
		# print k

main()