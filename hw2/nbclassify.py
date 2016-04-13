from naive_bayes_classifier import NaiveBayesClassifer
import sys

model = NaiveBayesClassifer(load = True)

if sys.argv[1]:
  model.predict_all(sys.argv[1])
  print 'Done'
else:
  print 'Must provide a path to input'