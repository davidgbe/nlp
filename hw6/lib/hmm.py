import os

class HiddenMarkovModel(object):
  def __init__(self, data_path):
    self.transition_counts = { 'start': {} }
    self.emission_counts = {}

    self.state_occurences = {}
    self.total_state_occurences = 0

    self.train(data_path)

  def train(self, data_path):
    for line in open(os.path.join(os.path.dirname(__file__), '..', data_path)):
      split_line = line.split(' ')
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
      self.emission_counts[state][word] = 0
    else:
      self.emission_counts[state][word] += 1

  def build_probs(self):
    self.build_state_distribution()
    self.build_transition_probs()
    self.build_emission_probs()

  def build_state_distribution(self):
    for state in self.state_occurences:
      self.state_occurences[state] = float(self.state_occurences[state]) / self.total_state_occurences

  def build_transition_probs(self):
    total_num_states = len(self.transition_counts)
    for prev_state in self.transition_counts:
      total_exiting_state = total_num_states
      for state in self.transition_counts:
        exiting_state = 1
        if state in self.transition_counts[prev_state]:
          exiting_state = self.transition_counts[prev_state][state]
          total_exiting_state += exiting_state
      for state in self.transition_counts:
        self.transition_counts[prev_state][state] = float(exiting_state) / total_exiting_state
    self.transition_probs = self.transition_counts

  def build_emission_probs(self):
    for state in self.emission_counts:
      total_num_emissions = 0
      for word in self.emission_counts[state]:
        total_num_emissions += self.emission_counts[state][word]
      for word in self.emission_counts[state]:
        self.emission_counts[state][word] = float(self.emission_counts[state][word]) / total_num_emissions

