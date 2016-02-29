"""
In dt.py, you will implement a basic decision tree classifier for
binary classification.  Your implementation should be based on the
minimum classification error heuristic (even though this isn't ideal,
it's easier to code than the information-based metrics).
"""

from numpy import *

from binary import *
import util
import numpy
import copy

class DT(BinaryClassifier):
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)
        
        self.isLeaf = True
        self.label  = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        """
        Traverse the tree to make predictions.  You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch.
        """
        #returning label if leaf
        if self.isLeaf == True:
            return self.label

        #based on current feature and current test data, recursively continue predicting into the tree 
        if X[self.feature] >= .5:
            return self.right.predict(X)
        else:
            return self.left.predict(X)   

    def trainDT(self, X, Y, maxDepth, used):
        """
        recursively build the decision tree
        """

        # get the size of the data set
        N,D = X.shape

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            
            #setting self as leaf
            self.isLeaf = True
            curY = 0
            curN = 0
            #finding out counts of Y or N for remaining training results (narrowed down to apply to features en route to leaf
            for x in range(len(Y)):
                if Y[x] == 1:
                    curY += 1
                else:    
                    curN += 1
            #setting leaf label according to most popular result for leaf's appropriate features.
            if curN > curY:
                self.label = -1.0
            else :
                self.label = 1.0            

        else:
            # we need to find a feature to split on
            bestFeature = -1     # which feature has lowest error
            bestError   = N      # the number of errors for this feature
            for d in range(D):
                # have we used this feature yet
                if d in used:
                    continue

                # suppose we split on this feature; what labels
                # would go left and right?
                #variables to keep track of features data:
                #leftYes = training data has value <.5 for feature and corresponding training result is 1.0
                #leftNo = training data has value <.5 for feature and corresponding training result is -1.0
                #rightYes = training data has value >=.5 for feature and corresponding training result is 1.0
                #rightNo = training data has value >=.5 for feature and corresponding training result is -1.0
                leftYes = 0
                leftNo = 0
                rightYes = 0
                rightNo = 0

                for x in range(N):
                    if X[x,d] >= .5:
                        if Y[x] == 1.0:
                            rightYes += 1;
                        else:
                            rightNo += 1;
                    else:
                        if Y[x] == 1.0:
                            leftYes += 1;
                        else:
                            leftNo += 1;                            

                #based on above info determining our prediction for feature values 
                if leftYes > leftNo:
                    leftY = 1
                else: 
                    leftY = -1    
               
                if rightYes > rightNo:
                    rightY = 1
                else: 
                    rightY = -1

                #calculating amount of errors for current feature based off of running 
                #training data through a loss function.
                # l(y,y') = {1 | y=y'   }
                #           {0 | otherwise }
                # where y = truth of training data, and y' = our current features prediction 
                errorTracker = 0    
                for x in range(N):
                    if X[x,d] >= .5:
                        featPred = rightY
                    else: 
                        featPred = leftY   
                       
                    if featPred != Y[x]:
                        errorTracker += 1

                # we'll classify the left points as their most
                # common class and ditto right points.  our error
                # is the how many are not their mode.
                error = errorTracker    

                # check to see if this is a better error rate
                if error <= bestError:
                    bestFeature = d
                    bestError   = error

            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                
                #now that weve found best available feature, we need to make new 
                #subsets of training data to pass into left/right according to our feature
                #find what indexes need to be deleted for each side and uses numpy.delete to 
                #remove corresponding indexes from current training data, forming new training data for each side.
                newXleftIdx = []
                newXrightIdx = []
                newYleftIdx = []
                newYrightIdx = []
                for x in range(N): 
                    if X[x,bestFeature] >= .5:
                        newXleftIdx.append(x)
                        newYleftIdx.append(x)
                    else:    
                        newXrightIdx.append(x)
                        newYrightIdx.append(x)
                        
                newXleft = numpy.delete(X, newXleftIdx, 0)
                newXright = numpy.delete(X, newXrightIdx, 0)
                newYleft = numpy.delete(Y, newYleftIdx, 0)
                newYright = numpy.delete(Y, newYrightIdx, 0)

                used.append(bestFeature)

                newUsedRight = copy.deepcopy(used)
                newUsedLeft = copy.deepcopy(used)
            
                self.feature = bestFeature
                self.isLeaf = False



                self.left  = DT({'maxDepth': maxDepth-1})
                self.right = DT({'maxDepth': maxDepth-1})
             
                #going deeper into the tree to form model, using new seperate training data for left/right side 
                #according to new feature selected. 
                self.left.trainDT(newXleft, newYleft, maxDepth-1, newUsedLeft)
                self.right.trainDT(newXright, newYright, maxDepth-1, newUsedRight)

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['maxDepth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponting classes,
            say Y(X(:,5)>=0.5) and if you want the correspnding rows of X,
            say X(X(:,5)>=0.5,:)
            
          - i suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - take a look at the 'mode' and 'uniq' functions in util.py
        """

        self.trainDT(X, Y, self.opts['maxDepth'], [])


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        
        return self

