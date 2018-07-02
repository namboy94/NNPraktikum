#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    myMultiLayerPerceptronClassifier = MultilayerPerceptron(data.trainingSet,
                                                            data.validationSet,
                                                            data.testSet,
                                                            learningRate=0.15,
                                                            epochs=20)

    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nMulti Layer Perceptron has been training..")
    myMultiLayerPerceptronClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    multiLayerPerceptronPred = myMultiLayerPerceptronClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Multi-Layer Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, lrPred)
    evaluator.printAccuracy(data.testSet, multiLayerPerceptronPred)
    
    # Draw
    plot = PerformancePlot("Multi-Layer Perceptron validation")
    plot.draw_performance_epoch(myMultiLayerPerceptronClassifier.performances,
                                myMultiLayerPerceptronClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
