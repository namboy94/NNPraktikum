
import numpy as np

from util.loss_functions import CrossEntropyError, \
    BinaryCrossEntropyError, SumSquaredError, MeanSquaredError, \
    DifferentError, AbsoluteError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys
import random

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'crossentropy':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Build up the network from specific layers
        self.layers = []

        # Input layer: train.input.shape[1] Inputs, 128 Outputs
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128,
                           None, inputActivation, False, dropout=64))

        # Hidden layer(s)
        if layers is not None:
            self.layers += layers

        # Output layer: 128 Inputs, 10 outputs == amount of possible digits
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10,
                                         None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        _in = inp
        for i, layer in enumerate(self.layers):
            _in = layer.forward(_in)
            if i < len(self.layers) - 1:
                _in = np.insert(_in, 0, 1)
        return _in

    def _compute_error(self, target, inp):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateDerivative(target, self._feed_forward(inp))

    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
                layer.updateWeights(learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")
        pass

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        c = list(zip(self.trainingSet.input, self.trainingSet.label))
        random.shuffle(c)
        self.trainingSet.input, self.trainingSet.label = zip(*c)

        for img, label in zip(self.trainingSet.input,
                              self.trainingSet.label):

            # Use LogisticLayer to do the job
            # Feed it with inputs
            error = self._compute_error(np.array([1 if x == label else 0 for x in range(0, 10)]), img)

            next_derivatives = self._get_output_layer().computeDerivative(error, np.identity(self._get_output_layer().nOut))

            i = len(self.layers) - 2

            while i >= 0:
                deleted = np.delete(self._get_layer(i + 1).weights, 0, 0)
                if self._get_layer(i+1).dropout != 0:
                    l = self._get_layer(1+i)
                    sam = [1] * (l.nOut - l.dropout)
                    sam += [0] * l.dropout
                    random.shuffle(sam)
                    l.outp = l.outp*np.array(sam)

                next_derivatives = self._get_layer(i).computeDerivative(next_derivatives, deleted)
                i -= 1

            self._update_weights(self.learningRate)

    def classify(self, test_instance):
        return np.argmax(self._feed_forward(test_instance))
         

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
