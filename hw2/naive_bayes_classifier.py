import re
import os
import pickle
from math import log
from sets import Set

class NaiveBayesClassifer:
  def __init__(self, load = False, path = '.'):
    self.klass_prior_probs = {}
    self.token_probs_given_class = {}
    if load:
      model = self.load(path)
      self.klass_prior_probs = model['priors']
      self.token_probs_given_class = model['token_probs']

  def predict_all(self, dir):
    output = open(os.path.join('.', 'nboutput.txt'), 'w')
    self.predict_dir(os.path.join('.', dir), output)
    output.close()

  def predict_dir(self, path, output):
    if os.path.isfile(path) and path.endswith('.txt'):
      decisions = self.predict(path)
      self.write_result(path, decisions, output)
    elif os.path.isdir(path):
      for name in os.listdir(path):
        child_path = os.path.join(path, name)
        self.predict_dir(child_path, output)

  def predict(self, file, dir = '.', klass_choices = [['truthful', 'deceptive'], ['positive', 'negative']]):
    occurrences = {}

    path = os.path.join(dir, file)
    f = open(path, 'r')
    for line in f:
      words = line.split()
      for word in words:
        token = NaiveBayesClassifer.tokenize(word)
        if occurrences.has_key(token):
          occurrences[token] += 1
        else:
          occurrences[token] = 1
    f.close()
    decisions = []
    for pairing in klass_choices:
      max_score = float('-inf');
      max_name = ''
      for klass in pairing:
        score = self.calc_prob(klass, occurrences)
        if score >= max_score:
          max_score = score
          max_name = klass
      decisions.append(max_name)
    return decisions

  def write_result(self, path, klasses, output):
    to_write = '%s %s\n' % (' '.join(klasses), path)
    output.write(to_write)

  def calc_prob(self, klass, occurrences):
    total = self.klass_prior_probs[klass]
    token_prob_dict_for_klass = self.token_probs_given_class[klass]
    for token, count in occurrences.iteritems():
      if token_prob_dict_for_klass.has_key(token):
        total += (count * token_prob_dict_for_klass[token])
    return total


  def train_all(self, dir):
    klasses = ['positive', 'negative', 'truthful', 'deceptive']
    start = os.path.join('.', dir)
    self.train(start, klasses, Set([]))
    self.normalize(klasses)
    self.save()

  def train(self, path, klasses, applied_klasses):
    new_klasses = applied_klasses | NaiveBayesClassifer.klasses_belongs_to(path, klasses)
    if os.path.isfile(path) and path.endswith('.txt'):
      self.parse_file(path, klasses, applied_klasses)
    elif os.path.isdir(path):
      for name in os.listdir(path):
        child_path = os.path.join(path, name)
        self.train(child_path, klasses, new_klasses)

  def parse_file(self, file_name, klasses, applied_klasses):
    self.add_klass_occurrences(applied_klasses)
    open_file = open(file_name, 'r')
    for line in open_file:
      words = line.split()
      for word in words:
        token = NaiveBayesClassifer.tokenize(word)
        self.add_token(token, klasses, applied_klasses)
    open_file.close()

  def add_klass_occurrences(self, applied_klasses):
    for klass in applied_klasses:
      if not self.klass_prior_probs.has_key(klass):
        self.klass_prior_probs[klass] = 1.0
      else:
        self.klass_prior_probs[klass] += 1.0

  def add_token(self, token, klasses, applied_klasses):
    for klass in klasses:
      if not self.token_probs_given_class.has_key(klass):
        self.token_probs_given_class[klass] = {}
      if not self.token_probs_given_class[klass].has_key(token):
        self.token_probs_given_class[klass][token] = 1.0
    for klass in applied_klasses:
      self.token_probs_given_class[klass][token] += 1.0

  def normalize(self, klasses):
    for klass, token_prob_dict_for_klass in self.token_probs_given_class.iteritems():
      total = sum(token_prob_dict_for_klass.values())
      for token, count in token_prob_dict_for_klass.iteritems():
        token_prob_dict_for_klass[token] = log(count / total)
    klass_count = sum(self.klass_prior_probs.values())
    for klass, count in self.klass_prior_probs.iteritems():
      self.klass_prior_probs[klass] = log(count / klass_count)

  def save(self):
    nb_model_file = open('nbmodel.txt', 'wb')
    model = { 'token_probs': self.token_probs_given_class, 'priors': self.klass_prior_probs }
    pickle.dump(model, nb_model_file)
    nb_model_file.close()

  def load(self, path):
    load_file = open(os.path.join(path, 'nbmodel.txt'), 'r')
    load_to = pickle.load(load_file)
    load_file.close()
    return load_to

  @staticmethod
  def tokenize(word):
    return re.sub(r'[^A-Za-z0-9]', '', word.lower())

  @staticmethod
  def klasses_belongs_to(name, klasses):
    applied = Set([])
    for klass in klasses:
      if klass in name:
        applied.add(klass)
    return applied
