import numpy as np
from collections import Counter

class WordClassificationNeuralNet:
	def __init__(self, reviewsPath, labelsPath, hiddenNodes, learningRate):
		self.reviews = None
		self.labels = None
		self.pos_neg_ratios = None
		self.layer_0 = None
		self.word2index = None
		self.vocab = None
		importFiles(reviewsPath, labelsPath)
		createCounts()


	def importFiles(reviewsPath, labelsPath):
		#Get movie reviews:
		#reviewsPath:  'reviews.txt'
		g = open(reviewsPath,'r')

		self.reviews = list(map(lambda x:x[:-1],g.readlines()))
		g.close()

		#labels path:  'labels.txt'
		g = open(labelsPath,'r') # What we WANT to know!
		self.labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
		g.close()

	def createCounts():
		positive_counts = Counter()
		negative_counts = Counter()
		total_counts = Counter()

		# Create Counter object to store positive/negative ratios
		self.pos_neg_ratios = Counter()

		#Populate the positive/negative counters
		for i in range(0, len(self.reviews)):
			for word in self.reviews[i].split(' '):
				if labels[i] == 'POSITIVE':
					positive_counts[word] += 1
				if labels[i] == 'NEGATIVE':
					negative_counts[word] += 1

		#Calculate the ratios of positive and negative uses of the most common words
		#       Consider words to be "common" if they've been used at least 100 times
		for word, count in positive_counts.items():
			if negative_counts[word] > 0:
				self.pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word] + 1)

    	#Convert ratios to logs
		for word, value in self.pos_neg_ratios.items():
			self.pos_neg_ratios[word] = np.log(value)

    	# words most frequently seen in a review with a "NEGATIVE" label
		#print(list(reversed(pos_neg_ratios.most_common()))[0:30])

	def importData2NeuralNet():
		self.vocab = set()
		for line in self.reviews:
			for word in line.split(' '):
				if word not in self.vocab:
					self.vocab.add(word)

		self.layer_0 = np.zeros((1, len(self.vocab)), np.int8, 'C')

		# Create a dictionary of words in the vocabulary mapped to index positions
		# (to be used in layer_0)
		self.word2index = {}
		for i,word in enumerate(self.vocab):
			self.word2index[word] = i
    
		# display the map of words to indices
		#print(word2index)

	def update_input_layer(review):
		""" Modify the global layer_0 to represent the vector form of review.
		The element at a given index of layer_0 should represent
		how many times the given word occurs in the review.
		Args:
			review(string) - the string of the review
		Returns:
			None
    	"""
		# clear out previous state by resetting the layer to be all 0s
		self.layer_0 *= 0
    
		#Count how many times each word is used in the given review and store the results in layer_0 
		for word in review.split(' '):
		layer_0[0, self.word2index[word]] += 1

	def get_target_for_label(label):
		"""Convert a label to `0` or `1`.
		Args:
			label(string) - Either "POSITIVE" or "NEGATIVE".
		Returns:
			`0` or `1`.
		"""
		# TODO: Your code here
		if label == 'POSITIVE':
			return 1
		else:
			return 0


if __name__ == "__main__":
	nn = WordClassificationNeuralNet('reviews.txt', 'labels.txt', 10, 0.6)