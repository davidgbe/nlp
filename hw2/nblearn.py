from naive_bayes_classifier import NaiveBayesClassifer
import sys

model = NaiveBayesClassifer()

if sys.argv[1]:
  model.train_all(sys.argv[1])
  print 'Done'
else:
  print 'Must provide a path to input'
