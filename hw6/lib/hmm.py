import os
from math import log
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import time

class HiddenMarkovModel(object):
  def __init__(self, data_path=None, load=False):
    if data_path:
      self.train(data_path)
      self.save()
    if load:
      model = self.load('../hmm_model.txt')
      self.emission_probs = model['em_probs']
      self.transition_probs = model['tran_probs']
      self.states = model['states']
      self.state_distribution = model['state_dist']

  def train(self, data_path):
    self.transition_counts = { 'start': {} }
    self.emission_counts = {}
    self.state_occurences = {}
    self.total_state_occurences = 0
    self.states = [ 'start' ]

    for line in open(os.path.join(os.path.dirname(__file__), '..', data_path)):
      split_line = line.split(' ')
      split_line[-1] = split_line[-1].replace('\n', '')
      split_line = map(lambda x: x.lower(), split_line)
      for i in range(len(split_line)):
        split_line[i] = split_line[i].split('/')
        if i == 0:
          self.add_transition(None, split_line[i])
        else:
          self.add_transition(split_line[i - 1], split_line[i])
        self.add_emission(split_line[i])
    self.build_probs()

  def add_transition(self, prev_element, element):
    start_state = prev_element[1] if prev_element else 'start'
    end_state = element[1]

    if not end_state in self.state_occurences:
      self.state_occurences[end_state] = 1
    else:
      self.state_occurences[end_state] += 1
    self.total_state_occurences += 1

    if not start_state in self.transition_counts:
      self.transition_counts[start_state] = {}

    if not end_state in self.transition_counts[start_state]:
      self.transition_counts[start_state][end_state] = 1
    else:
      self.transition_counts[start_state][end_state] += 1

  def add_emission(self, element):
    state = element[1]
    word = element[0]
    if not state in self.emission_counts:
      self.emission_counts[state] = {}
    if not word in self.emission_counts[state]:
      self.emission_counts[state][word] = 1
    else:
      self.emission_counts[state][word] += 1

  def build_probs(self):
    self.build_state_distribution()
    self.build_transition_probs()
    self.build_emission_probs()

  def build_state_distribution(self):
    for state in self.state_occurences:
      self.state_occurences[state] = np.log(float(self.state_occurences[state]) / self.total_state_occurences)
      self.states.append(state)
    self.state_distribution = self.state_occurences

  def build_transition_probs(self):
    total_num_states = len(self.transition_counts)
    for prev_state in self.states:
      if not prev_state in self.transition_counts:
        self.transition_counts[prev_state] = {}
      total_exiting_state = total_num_states
      for state in self.states:
        exiting_state = 1
        if state in self.transition_counts[prev_state]:
          exiting_state += self.transition_counts[prev_state][state]
        total_exiting_state += exiting_state
        self.transition_counts[prev_state][state] = exiting_state
      for state in self.states:
        self.transition_counts[prev_state][state] = np.log(float(self.transition_counts[prev_state][state]) / total_exiting_state)
    self.transition_probs = self.transition_counts

  def build_emission_probs(self):
    for state in self.emission_counts:
      total_num_emissions = 0
      for word in self.emission_counts[state]:
        total_num_emissions += self.emission_counts[state][word]
      for word in self.emission_counts[state]:
        self.emission_counts[state][word] = np.log(float(self.emission_counts[state][word]) / total_num_emissions)
    self.emission_probs = self.emission_counts

  def predict(self, input_path):
    base_path = os.path.dirname(__file__)
    output = open(os.path.join(base_path, '..', 'hmmoutput.txt'), 'w')
    count = 0
    for line in open(os.path.join(os.path.dirname(__file__), '..', input_path)):
      split_line = line.split(' ')
      split_line_lower = map(lambda x: x.lower(), split_line)
      tagged = self.predict_sentence(split_line_lower)
      newline = ''
      for i in range(len(split_line)):
        if (i != len(split_line) - 1):
          newline += (split_line[i] + '/' + tagged[i+1].upper() + ' ')
        else:
          newline += (split_line[i].replace('\n', '') + '/' + tagged[i+1].upper())
      output.write(newline)
      count += 1
      if count % 50 == 0:
        print count
    output.close()

  def predict_sentence(self, sentence):
    emission_cache = {}
    paths = map(lambda s: ['start'], self.states)
    num_states = len(self.states)
    start = np.zeros((num_states, num_states))
    transition_matrix = self.create_transitions_matrix()
    next = start
    sentence[-1] = sentence[-1].replace('\n', '')
    for word in sentence:
      next = next + self.create_next_step_matrix(word, transition_matrix, emission_cache)
      max_indices = np.array(np.argmax(next, axis=0))[0]
      for i in range(len(paths)):
        paths[i].append(self.states[max_indices[i]])
      maxes = np.amax(next, axis=0)
      next = np.tile(maxes, (len(self.states), 1))
    m = np.argmax(next) % num_states
    return paths[m]

  def fetch_emission_prob(self, state, word):
    if state == 'start':
      return -100
    if word in self.emission_probs[state]:
      return self.emission_probs[state][word]
    else:
      return -100

  def create_transitions_matrix(self):
    transitions = []
    num_states = len(self.states)
    for state in self.states:
      row = []
      transitions.append(row)
      for prev_state in self.states:
        row.append(self.transition_probs[prev_state][state])
    sparse_mat = csr_matrix(transitions)
    return sparse_mat

  def create_next_step_matrix(self, next_word, transition_matrix, emission_cache):
    emissions_mat = None
    if next_word in emission_cache:
      emissions_mat = emission_cache[next_word]
    else:
      emissions = map(lambda s: self.fetch_emission_prob(s, next_word), self.states)
      emissions_mat = np.tile(emissions, (len(self.states), 1)).transpose()
      emission_cache[next_word] = emissions_mat
    return emissions_mat + transition_matrix

  def save(self):
    hmm_model_file = open('hmm_model.txt', 'wb')
    model = { 'em_probs': self.emission_probs, 'tran_probs': self.transition_probs, 'states': self.states, 'state_dist': self.state_distribution}
    pickle.dump(model, hmm_model_file)
    hmm_model_file.close()

  def load(self, path):
    load_file = open(os.path.join(os.path.dirname(__file__), path), 'r')
    load_to = pickle.load(load_file)
    load_file.close()
    return load_to
