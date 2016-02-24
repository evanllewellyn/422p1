"""
In dumbClassifiers.py, we implement the world's simplest classifiers:
  1) Always predict +1
  2) Always predict the most frequent label from the training data
  3) Just use the sign of the first feature to decide on label
"""

from binary import *
from numpy  import *

import util

class AlwaysPredictOne(BinaryClassifier):
    """
    This defines the classifier that always predicts +1.
    """

    def __init__(self, opts):
        """
        do nothing
        """

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictOne"

    def predict(self, X):
        return 1       # return our constant prediction

    def train(self, X, Y):
        """
        do nothing
        """


class AlwaysPredictMostFrequent(BinaryClassifier):
    """
    This defines the classifier that always predicts the
    most frequent label from the training data.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, assume most frequent class is +1
        """
        self.mostFrequentClass = 1

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictMostFrequent(%d)" % self.mostFrequentClass

    def predict(self, X):
        """
        X is an vector and we want to make a single prediction: Just
        return the most frequent class!
        """

        return self.mostFrequentClass

    def train(self, X, Y):
        '''
        just figure out what the most frequent class is and store it in self.mostFrequentClass
        '''

        pos = Y.tolist().count(1)
        neg = Y.tolist().count(-1)

        if pos >= neg:
            self.mostFrequentClass = 1
        else:
            self.mostFrequentClass = -1

class FirstFeatureClassifier(BinaryClassifier):
    """
    This defines the classifier that always predicts on the basis of
    the first feature only.  In particular, we maintain two
    predictors: one for when the first feature is >0, one for when the
    first feature is <= 0.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, always return 1
        """
        self.classForPos = 1    # what class should we return if X[0] >  0
        self.classForNeg = 1    # what class should we return if X[0] <= 0

    def online(self):
        return False

    def __repr__(self):
        return "FirstFeatureClassifier(%d,%d)" % (self.classForPos, self.classForNeg)

    def predict(self, X):
        """
        check the first feature and make a classification decision based on it
        """

        if X[0] > 0:
            return self.classForPos
        else:
            return self.classForNeg

    def train(self, X, Y):
        '''
        just figure out what the most frequent class is for each value of X[:,0] and store it
        '''

        index = 0

        num_xpos_ypos = 0
        num_xpos_yneg = 0
        num_xneg_ypos = 0
        num_xneg_yneg = 0

        # Iterate through each element of X[:,0], compare to Y[:]
        for item in X:
            if X[index,0] > 0:
                if Y[index] > 0:
                    num_xpos_ypos += 1  # X is 1, Y is 1
                else:
                    num_xpos_yneg += 1  # X is 1, Y is -1
            else:
                if Y[index] > 0:
                    num_xneg_ypos += 1  # X is 0, Y is 1
                else:
                    num_xneg_yneg += 1  # X is 0, Y is -1

            index += 1

        # Compare counted results if X[0] > 0
        if num_xpos_ypos >= num_xpos_yneg:
            self.classForPos = 1
        else:
            self.classForPos = -1

        # Compare counted results if X[0] <= 0
        if num_xneg_ypos >= num_xneg_yneg:
            self.classForNeg = 1
        else:
            self.classForNeg = -1
