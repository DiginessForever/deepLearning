import time
import sys
import numpy as np
from collections import Counter

def importFiles(reviewsPath, labelsPath):
    #Get movie reviews:
    #reviewsPath:  'reviews.txt'
    g = open(reviewsPath,'r')

    reviews = list(map(lambda x:x[:-1],g.readlines()))
    g.close()

    #labels path:  'labels.txt'
    g = open(labelsPath,'r') # What we WANT to know!
    labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
    g.close()

    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()

    #Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
    for i in range(0, len(reviews)):
        for word in reviews[i].split(' '):
            if labels[i] == 'POSITIVE':
                positive_counts[word] += 1
            if labels[i] == 'NEGATIVE':
                negative_counts[word] += 1

    # Examine the counts of the most common words in positive reviews
    #print(positive_counts.most_common())
    #print(negative_counts.most_common())

    # Create Counter object to store positive/negative ratios
    pos_neg_ratios = Counter()

    # TODO: Calculate the ratios of positive and negative uses of the most common words
    #       Consider words to be "common" if they've been used at least 100 times
    for word, count in positive_counts.items():
        if negative_counts[word] > 0:
            pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word] + 1)

    #print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
    #print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
    #print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

    # TODO: Convert ratios to logs
    for word, value in pos_neg_ratios.items():
        pos_neg_ratios[word] = np.log(value)

    #print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
    #print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
    #print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

    # words most frequently seen in a review with a "POSITIVE" label
    #pos_neg_ratios.most_common()

    # words most frequently seen in a review with a "NEGATIVE" label
    #list(reversed(pos_neg_ratios.most_common()))[0:30]

    # TODO: Create set named "vocab" containing all of the words from all of the reviews
    vocab = set()
    for line in reviews:
        for word in line.split(' '):
            if word not in vocab:
                vocab.add(word)

    vocab_size = len(vocab)
    #print(vocab_size)

    # Create a dictionary of words in the vocabulary mapped to index positions
    # (to be used in layer_0)
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i
    #print(word2index)

    return reviews, labels  #Do I need something else returned?  It's not getting past 50%.

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        #populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        for line in reviews:
            for word in line.split(' '):
                if word not in review_vocab:
                    review_vocab.add(word)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        #populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for i in range(0, len(labels)):
            label_vocab.add(labels[i])
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
            
        # Create a dictionary of words in the vocabulary mapped to index positions
        # populate self.word2index with indices for all the words in self.review_vocab
        self.word2index = {}
        for i,word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of labels mapped to index positions
        # populate self.label2index within indices for all the words in self.label_vocab
        self.label2index = {}
        for i,word in enumerate(self.label_vocab):
            self.label2index[word] = i

        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.layer_0 = np.zeros((1, len(input_nodes)), np.int8, 'C')

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # Initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes), float)
        
        # Initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        
        # Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        #self.layer_0 = np.zeros((1,self.input_nodes))
    
        
    def update_input_layer(self,review):
        # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
        
        #Count how many times each word is used in the given review and store the results in layer_0 
        for word in review.split(' '):
            if word in self.word2index.keys():
                self.layer_0[0][self.word2index[word]] = 1
                
    def get_target_for_label(self,label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            self.update_input_layer(review)
            self.hidden_layer = self.layer_0.dot(self.weights_0_1)
            self.output_layer = self.sigmoid(self.hidden_layer.dot(self.weights_1_2))
            
            # Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.
            self.output_error = self.output_layer - self.get_target_for_label(label)
            self.output_delta = self.output_error * self.sigmoid_output_2_derivative(self.output_layer)
            
            self.hidden_error = self.output_delta.dot(self.weights_1_2.T)
            self.hidden_delta = self.hidden_error
            
            self.weights_1_2 -= self.hidden_layer.T.dot(self.output_delta) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(self.hidden_delta) * self.learning_rate
            
            # Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if np.abs(self.output_error) < 0.5:
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like you did in the
        #   "train" function. That means use the given review to 
        #   update the input layer, then calculate values for the hidden layer,
        #   and finally calculate the output layer.
        #
        #   Note: The review passed into this function for prediction 
        #           might come from anywhere, so you should convert it 
        #           to lower case prior to using it.
        self.update_input_layer(review.lower())
        self.hidden_layer = self.layer_0.dot(self.weights_0_1)
        self.output_layer = self.hidden_layer.dot(self.weights_1_2)
        
        # TODO: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        if self.output_layer[0] > 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"

if __name__ == "__main__":
    reviews, labels = importFiles('reviews.txt','labels.txt')
    mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
    mlp.test(reviews[-1000:],labels[-1000:])
    #mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)
    #mlp.test(reviews[-1000:],labels[-1000:])
    #mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
    #mlp.test(reviews[-1000:],labels[-1000:])