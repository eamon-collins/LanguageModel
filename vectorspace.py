import os
import nltk
#import matplotlib.pyplot as plt
import json
import warnings
import numpy as np
import scipy.sparse as sparse

warnings.filterwarnings("ignore", message="numpy.dtype size changed")


#things to come back to:
#zipfs law graphs are weird with the line being way below the graph visually. intercept at like 11
#atmospher unigram still getting through even though it's in stopwords

def main():
	#UNCOMMENT BELOW TO RUN THE PREPROCESSING

	# train_revs = parse_files('yelp/train')
	# test_revs = parse_files('yelp/test')

	# #Performs tokenization, normalization, stemming, and stopword removal on 'Content' and adds the result to a new key called 'clean' in the same data structure
	# preprocess(train_revs)
	# preprocess(test_revs)


	# #stores the processed reviews for convenience so I don't have to do it every time I run the program
	# write_files('trainfile', train_revs, 'testfile', test_revs)

	train_revs = parse_files('trainfile')
	test_revs = parse_files('testfile')

	#zipfs_law(train_revs, test_revs)

	#zipfs_law(train_revs, test_revs, df=True)

	#bigrams is all ngrams found tied to document frequency, ttf is the same tied to ttf.
	bigrams = construct_ngrams(train_revs)

	sortedBigrams = sorted(bigrams, key=bigrams.get, reverse=True)
	#print(sortedBigrams[:100])
	#sortedBigrams = sortedBigrams[101:]
	vocab = []
	for word in sortedBigrams:
		if bigrams[word] >= 50:
			vocab.append(word)

	#dont uncomment this, just left it in to show that I did add the top 100 bigrams to the stopwords. each run after this incorporates them.
	# with open('stopwords', 'a') as file:
	# 	for word in sortedBigrams[:100]:
	# 		file.write(word+"\n")
	#for k, v in bigrams.items():
		
	print("num ngrams: "+str(len(vocab)))

	print(vocab[:50])
	print("##########################")
	print(vocab[-50:])

	#gets the idfs from the train set, we will use this for all
	idfs = calculate_idfs(vocab, bigrams, len(train_revs))

	# #gets term frequency for the test set, discards df cause we want to use train sets stats
	construct_ngrams(test_revs)

	# #returns sparse matrix of tf-idf scores (unordered colmn indices, use set_order() to output in )
	# tf_idf_matrix = calculate_matrix(vocab, idfs, train_revs)
	tf_idf_matrix = calculate_matrix(vocab, idfs, test_revs)
	#saves the matrix in a sparse format for storage size and ease and convenience (calculate_matrix is quite slow)
	sparse.save_npz('test_matrix', tf_idf_matrix)

	#tf_idf_matrix = sparse.load_npz('test_matrix.npz')

	print("shape of tf-idf matrix: "+ str(tf_idf_matrix.shape))



#calculates the tf-idf matrix for all the reviews.
def calculate_matrix(vocab, idfs, reviews):
	matrix = np.zeros((len(reviews),len(vocab)))
	#matrix = sparse.csr_matrix((len(reviews),len(vocab)))
	for j in range(len(reviews)):
		rev = reviews[j]
		for i in range(len(vocab)):
			term = vocab[i]
			idf = idfs[i]
			if term in rev['tf']:
				tf = rev['tf'][term]
			else:
				tf = 0
			matrix[j,i] = tf * idf

	return sparse.csr_matrix(matrix, (len(reviews),len(vocab)))





#takes dict of bigrams and DF, list of bigrams for idfing, and number of total docs
def calculate_idfs(vocab, freqs, numdocs):
	idfs = []
	for term in vocab:
		if term in freqs:
			idfs.append(1+np.log(numdocs/freqs[term]))
		else:
			idfs.append(0)
	return idfs

#returns a dictionary with the bigrams and unigrams in the reviews as keys and their document frequency as value
def construct_ngrams(reviews, n=2):
	freqs = {}
	for rev in reviews:
		minidict = {}
		for token in rev['clean']:
			if token in minidict:
				minidict[token] += 1
			else:
				minidict[token] = 1
		#bigrams
		for i in range(len(rev['clean'])+1-n):
			token = rev['clean'][i] + "-" + rev['clean'][i+1]
			if token in minidict:
				minidict[token] += 1
			else:
				minidict[token] = 1

		rev['tf'] = minidict

		#collate them into the main dictionary
		for key in minidict:
			if key in freqs:
				freqs[key] += 1
			else:
				freqs[key] = 1
	return freqs




def zipfs_law(train, test, df=False):
	freqs = {}

	if not df:
		for rev in train:
			for token in rev['clean']:
				if token in freqs:
					freqs[token] += 1
				else:
					freqs[token] = 1
		for rev in test:
			for token in rev['clean']:
				if token in freqs:
					freqs[token] += 1
				else:
					freqs[token] = 1
	else:
		for rev in train:
			minidict = {}
			for token in rev['clean']:
				if token in minidict:
					minidict[token] += 1
				else:
					minidict[token] = 1
			for key in minidict:
				if key in freqs:
					freqs[key] += 1
				else:
					freqs[key] = 1
		for rev in test:
			minidict = {}
			for token in rev['clean']:
				if token in minidict:
					minidict[token] += 1
				else:
					minidict[token] = 1
			for key in minidict:
				if key in freqs:
					freqs[key] += 1
				else:
					freqs[key] = 1

	numtokens = len(freqs.keys())

	x = np.linspace(1,numtokens, numtokens)
	y = [k[1] for k in freqs.items()]
	y.sort(reverse=True)


	plt.scatter(x, y, s=3)
	plt.yscale('log',basey=10)
	plt.xscale('log',basex=10)

	loga = np.log(x)
	logb = np.log(y)
	#plt.plot(loga,logb, '--r')

	coefficients = np.polyfit(logb, loga, 1)
	polynomial = np.poly1d(coefficients)
	print(coefficients)

	tau = coefficients[0]
	k = coefficients[1]
	
	plt.plot(x, k*x**tau, '--r')

	plt.ylim(1,100000)
	plt.show()



def preprocess(reviews):
	punctuation = '. , < > / ? ; : \' " ] [ } { - _ = + ) ( \\ | ! @ # $ % ^ & * ` ~'.split()
	stopwords = get_stopwords()

	tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
	stemmer = nltk.stem.snowball.EnglishStemmer()

	for rev in reviews:
		tokens = tokenizer.tokenize(rev['Content'])
		clean = []

		for word in tokens:
			#elims puncuation
			if word in punctuation:
				continue
			#changes to lower case
			word = word.lower()

			#if it can be converted to float it is either float or integer
			try:
				test = float(word)
			except ValueError:
				pass
			else:
				clean.append('NUM')
				continue

			#stems
			word = stemmer.stem(word)
			#removes stopwords
			if word in stopwords:
				continue
			#if it's made it this far, we want this version of the word in our cleaned tokens list
			clean.append(word)

		rev['clean'] = clean




def parse_files(filepath):
	reviews = []
	if os.path.isdir(filepath):
		for file in os.listdir(filepath):
			with open(os.path.join(filepath,file), 'r') as file:
				data = json.loads(file.read())
			for review in data['Reviews']:
				reviews.append(review)
	else:
		with open(filepath, 'r') as file:
			data = json.loads(file.read())
		for review in data:
			reviews.append(review)
	return reviews

def get_stopwords():
	stemmer = nltk.stem.snowball.EnglishStemmer()
	stopwords = []
	with open('stopwords', 'r') as stop:
		for line in stop:
			if line == "\n":
				continue
			else:
				stopwords.append(stemmer.stem(line.strip()))
	return stopwords

def write_files(trainpath, train, testpath, test):
	with open(trainpath, 'w') as trainfile:
		trainfile.write(json.dumps(train))
	with open(testpath, 'w') as testfile:
		testfile.write(json.dumps(test))

if __name__ == '__main__':
	main()