###################################
# CS B551 Fall 2022, Assignment #4
#
# Your names and user ids: Jonathan Hansen and Mark Green
#
# (Based on skeleton code by D. Crandall)
#

# JONATHAN'S RESOURCES:
# https://chat.openai.com/
# https://www.mygreatlearning.com/blog/pos-tagging/

import math
from collections import Counter, defaultdict
from itertools import product
import random
from tqdm import trange


class Solver:
    """
    This class predicts parts-of-speech for words in a sentence
    in three ways using different bayesian network architectures:
        1. a Simple model using only emission probabilities
            for parts-of-speech given a word
        2. a Hidden Markov model implemented by Viterbi algorithm
        3. a Complex bayesian model by Monte Carlo simulations
            implemented by Gibbs sampling
    """

    def __init__(self, default_prob=1e-6):
        """
        These dictionaries store the log probabilities
        corresponding to different edges in the bayesian models:
            1. Emissions: POS --> Word
            2. Transitions: POS --> next POS
            3. Emission2s: POS --> next Word
            4. Transition2s: POs --> next next POS
        
        Args:
            default_prob: float :: the default part-of-speech probability
                                   for words not in the training data.
        """
        self.emissions = {}  # simple
        self.transitions = {}  # hmm
        self.emission2s = {}  # complex
        self.transition2s = {}  # complex
        self.default_prob = math.log(default_prob)

    def train(self, data: list, verbose=False):
        """
        This method fills the log probability dictionaries for the
        emissions, transition, emission2s, and transition2s
        bayesian network edges.
        It takes a list of tuples composed of two objects each:
            1. a tuple of words, representing the sentence
            2. a tuple of parts-of-speech, corresponding to
                each word in the sentence
        The occurence of each edge is counted and normalized
        and LaPlace smoothing is applied for non-existent edges.

        Args:
            data: list :: list of tuples (sentence, labels)
            verbose: bool :: display verbose output for training
        """
        if verbose:
            print("training...")
            print("counting emissions and transitions")
        # Initialize counters for emissions and transitions
        emission_counts = defaultdict(int)
        transition_counts = defaultdict(int)
        tag_counts = defaultdict(int)
        start_tag_counts = defaultdict(int)
        emission2_counts = defaultdict(int)
        transition2_counts = defaultdict(int)

        # Count emissions and transitions from the training data
        for sentence, tags in data:
            start_tag_counts[tags[0]] += 1  # Increment the start tag count
            # count emission from current tag
            for i in range(len(tags)):
                tag = tags[i]
                word = sentence[i].lower()
                tag_counts[tag] += 1
                emission_counts[(tag, word)] += 1  # simple
                # count transition and emission from previous tag
                if i > 0:
                    prev_tag = tags[i - 1]
                    transition_counts[(prev_tag, tag)] += 1  # hmm
                    emission2_counts[(prev_tag, word)] += 1  # complex
                # count transition from previous previous tag
                if i > 1:
                    prev2_tag = tags[i - 2]
                    transition2_counts[(prev2_tag, tag)] += 1  # complex

        # Get sets of all words and tags to determine vocab and number of tags
        self.word_list = list(set(word for sentence, _ in data
                                  for word in sentence))
        self.tag_list = list(set(tag for _, tags in data
                                 for tag in tags))

        # Laplace smoothing - add one to all counts
        if verbose:
            print("Laplace smoothing - Add one to all counts")
        for tag in self.tag_list:
            start_tag_counts[tag] += 1
        for tag in self.tag_list:
            for word in self.word_list:
                emission_counts[(tag, word)] += 1
                emission2_counts[(tag, word)] += 1
        for prev_tag in self.tag_list:
            for tag in self.tag_list:
                transition_counts[(prev_tag, tag)] += 1
                transition2_counts[(prev_tag, tag)] += 1

        # Convert counts to probabilities with smoothing
        # Use total_words for emission probabilities
        if verbose:
            print("computing probabilities...")
        self.initial = {
            k: math.log(v / (tag_counts[k] + len(self.tag_list)))
            for k, v in start_tag_counts.items()
        }
        self.emissions = {
            k: math.log(v / (tag_counts[k[0]] + len(self.word_list)))
            for k, v in emission_counts.items()
        }
        self.transitions = {
            k: math.log(v / (tag_counts[k[0]] + len(self.tag_list)))
            for k, v in transition_counts.items()
        }
        self.emission2s = {
            k: math.log(v / (tag_counts[k[0]] + len(self.word_list)))
            for k, v in emission2_counts.items()
        }
        self.transition2s = {
            k: math.log(v / (tag_counts[k[0]] + len(self.tag_list)))
            for k, v in transition2_counts.items()
        }
        if verbose:
            print("training complete!")

    def posterior(self, model: str, sentence: tuple, label: tuple):
        """
        This method computes the posterior probabilities of a given sentence
        as labeled by a corresponding set of parts-of-speech.
        The method is implemented for each different model,
        Simple, HMM, and MCMC.

        Args:
            model: str :: declare the model by which posterior is computed
            sentence: tuple :: words in the sentence
            label: tuple :: parts-of-speech corresponding to each word
        """
        if model == "Simple":
            # calculate the sum of emission log probabilities
            probability_sum = 0
            for word, tag in zip(sentence, label):
                probability_sum += self.emissions.get((tag, word.lower()),
                                                      self.default_prob)
            return probability_sum

        elif model == "HMM":
            # For the HMM model, calculate the total log probability
            total_log_prob = 0
            for i in range(len(sentence)):
                word = sentence[i].lower()
                tag = label[i]
                # Emission probability
                emission_prob = self.emissions.get((tag, word),
                                                   self.default_prob)
                total_log_prob += emission_prob
                # transition probability
                if i == 0:
                    # Transition from start tag
                    start_transition_prob = self.initial.get(tag,
                                                             self.default_prob)
                    total_log_prob += start_transition_prob
                else:
                    # Transition from previous tag
                    prev_tag = label[i-1]
                    transition_prob = self.transitions.get((prev_tag, tag),
                                                           self.default_prob)
                    total_log_prob += transition_prob
            return total_log_prob

        elif model == "Complex":
            # For the complex model, calculate the total log probability
            total_log_prob = 0
            for i in range(len(sentence)):
                word = sentence[i].lower()
                tag = label[i]
                # emission probability to current word
                emission_prob = self.emissions.get((tag, word),
                                                   self.default_prob)
                total_log_prob += emission_prob
                # transition from start tag
                if i == 0:
                    start_transition_prob = self.initial.get(tag,
                                                             self.default_prob)
                    total_log_prob += start_transition_prob
                # transition from previous tag
                else:
                    prev_tag = label[i-1]
                    transition_prob = self.transitions.get((prev_tag, tag),
                                                           self.default_prob)
                    total_log_prob += transition_prob
                # emission to next word
                if i != range(len(sentence))[-1]:
                    next_word = sentence[i+1].lower()
                    emission2_prob = self.emission2s.get((tag, next_word),
                                                         self.default_prob)
                    total_log_prob += emission2_prob
                # transition from previous previous tag
                if i > 1:
                    prev2_tag = label[i-2]
                    transition2_prob = self.transition2s.get((prev2_tag, tag),
                                                             self.default_prob)
                    total_log_prob += transition2_prob
            return total_log_prob
        else:
            print("Unknown algorithm!")

    def simplified(self, sentence: tuple):
        """
        Simplified POS tagging method assigns the most probable tag
        to each word as the argmax of the emission log probability

        Args:
            sentence: tuple :: words in the sentence
        """
        argmax_list = [
            max(self.tag_list,
                key=lambda tag: self.emissions.get((tag, word.lower()), 0))
            for word in sentence
        ]
        print("Simple Model complete!")
        return argmax_list

    def hmm_viterbi(self, sentence: tuple):
        """
        Implementation of the Viterbi algorithm for HMM mode

        Args:
            sentence: tuple :: words in the sentence
        """
        # Number of tags
        n_tags = len(self.tag_list)
        # Initialize the Viterbi matrix with negative infinity
        viterbi = [[-math.inf for _ in range(n_tags)]
                   for _ in range(len(sentence))]
        # Backpointer matrix to track the best path
        backpointer = [[0 for _ in range(n_tags)]
                       for _ in range(len(sentence))]

        # Initialization step using log probabilities
        first_word = sentence[0].lower()
        for tag_idx, tag in enumerate(self.tag_list):
            # Check for the presence of (word, tag) in emission probabilities
            if (tag, first_word) in self.emissions:
                viterbi[0][tag_idx] = self.initial.get(tag,
                                                       self.default_prob) + \
                                        self.emissions.get((tag, first_word),
                                                           self.default_prob)
            else:
                # Handle unseen (word, tag) pairs
                viterbi[0][tag_idx] = self.initial.get(tag,
                                                       self.default_prob) + \
                                        self.default_prob

        # Iterate through the rest of the sentence
        for t in range(1, len(sentence)):
            word = sentence[t].lower()
            for tag_idx, tag in enumerate(self.tag_list):
                # Find the max log probability and corresponding tag index
                # from the previous step
                max_log_prob, best_tag_idx = max([
                    (
                        viterbi[t-1][prevtag_idx] +
                        self.transitions.get((self.tag_list[prevtag_idx], tag),
                                             self.default_prob),
                        prevtag_idx
                    )
                    for prevtag_idx in range(n_tags)
                ], key=lambda x: x[0])

                # Update the Viterbi matrix
                if (tag, word) in self.emissions:
                    viterbi[t][tag_idx] = max_log_prob + \
                                          self.emissions.get((tag, word),
                                                             self.default_prob)
                # Handle unseen pairs
                else:
                    viterbi[t][tag_idx] = max_log_prob + self.default_prob

                # Update the backpointer matrix
                backpointer[t][tag_idx] = best_tag_idx

        # Termination step
        max_prob, best_tag_idx = max(
            [(viterbi[len(sentence)-1][tag_idx], tag_idx)
             for tag_idx in range(n_tags)],
            key=lambda x: x[0]
        )

        # Backtrack to find the best path
        best_path = [best_tag_idx]
        for t in range(len(sentence)-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        print("Hidden Markov Model complete!")
        # Convert numeric indices to tags
        return [self.tag_list[i] for i in best_path]

    def SampleKeys(self, edge_probs: dict, n: int):
        """
        This method gets random keys (bayesian network edges)
        as given by their values (probabilities).
        This is the mechanism that implements the Monte Carlo simulations.

        Args:
            edge_probs: dict :: dictionary of bayesian network edges and
                                their corresponding log probabilities
            n: int :: number of Monte Carlo simulations for each word
        """
        edges = list(edge_probs.keys())
        probabilities = [math.exp(v) for v in edge_probs.values()]
        sampled_edges = random.choices(edges, probabilities, k=n)
        return sampled_edges

    def mcmc_word_probs(self, sentence: tuple, predicted_pos: list, i: int):
        """
        This method gets all possible POS log probabilities
        for the observed word given its context in the sentence.

        Args: 
            sentence: tuple :: words in the sentence
            predicted_pos: list :: parts-of-speech predicted for earlier
                                   words in the sentence
            i: int :: position of target word in the sentence
        """
        word_probs = {}
        word_probs['emission'] = {
            k: v for k, v in self.emissions.items()
            if k[1] == sentence[i]
        }
        if i == 0:
            word_probs['initial'] = {
                k: v for k, v in self.initial.items()
            }
        if i > 0:
            word_probs['transition'] = {
                k: v for k, v in self.transitions.items()
                if k[0] == predicted_pos[-1]
            }
        if i != range(len(sentence))[-1]:
            word_probs['emission2'] = {
                k: v for k, v in self.emission2s.items()
                if k[1] == sentence[i+1]
            }
        if i > 1:
            word_probs['transition2'] = {
                k: v for k, v in self.transition2s.items()
                if k[0] == predicted_pos[-2]
            }
        # if its a new word, fill probabilities with default log probability
        for k, v in word_probs.items():
            if k == 'initial' and len(v) == 0:
                word_probs[k] = {
                    tag: self.default_prob for tag in self.tag_list
                }
            elif k == 'emission' and len(v) == 0:
                word_probs[k] = {
                    (tag, sentence[i]): self.default_prob
                    for tag in self.tag_list
                }
            elif k == 'transition' and len(v) == 0:
                word_probs[k] = {
                    (predicted_pos[-1], tag): self.default_prob
                    for tag in self.tag_list
                }
            elif k == 'emission2' and len(v) == 0:
                word_probs[k] = {
                    (tag, sentence[i+1]): self.default_prob
                    for tag in self.tag_list
                }
            elif k == 'transition2' and len(v) == 0:
                word_probs[k] = {
                    (predicted_pos[-2], tag): self.default_prob
                    for tag in self.tag_list
                }
        return word_probs

    def complex_mcmc(self, sentence: tuple, n=100):
        """
        This method applies Gibbs sampling to perform Markov Chain Monte Carlo
        simulations to estimate the distribution of the Complex model
        and predict the parts of speech.

        Args:
            sentence: tuple :: words in the sentence
            n: int :: number of Monte Carlo simulations for each word
        """
        print(f"Monte Carlo Markov Chain with {n} simulations per word...")
        # iterate through each word in the sentence to get predicted POS
        predicted_pos = []
        for i in trange(len(sentence)):
            # first get the relevant possible probabilities of POS for the word
            word_probs = self.mcmc_word_probs(sentence=sentence,
                                              predicted_pos=predicted_pos,
                                              i=i)
            # then sample from the probabilities to get the most frequent POS
            pos_samples = []
            # special case for 1st character in 1-character sentence
            if i == 0 and i == range(len(sentence))[-1]:
                combinations = product(word_probs['initial'].keys(),
                                       word_probs['emission'].keys())
                initial_probs = {
                    emk: sum([word_probs['initial'][ink],
                              word_probs['emission'][emk]])
                    for ink, emk in combinations
                    if ink == emk[0]
                }
                pos_samples.extend([pos for pos, word
                                    in self.SampleKeys(initial_probs, n=n)])

            # draw the first POS from the initial probability distribution
            elif i == 0:
                combinations = product(word_probs['initial'].keys(),
                                       word_probs['emission'].keys(),
                                       word_probs['emission2'].keys())
                initial_probs = {
                    emk: sum([word_probs['initial'][ink],
                              word_probs['emission'][emk],
                              word_probs['emission2'][em2k]])
                    for ink, emk, em2k in combinations
                    if ink == emk[0] == em2k[0]
                }
                pos_samples.extend([pos for pos, word
                                    in self.SampleKeys(initial_probs, n=n)])

            # special case for 2nd character in 2-character sentence
            elif i == 1 and i == range(len(sentence))[-1]:
                combinations = product(word_probs['transition'].keys(),
                                       word_probs['emission'].keys())
                final_probs = {
                    emk: sum([word_probs['transition'][trk],
                              word_probs['emission'][emk]])
                    for trk, emk in combinations
                    if trk[1] == emk[0]
                }
                pos_samples.extend([pos for pos, word
                                    in self.SampleKeys(final_probs, n=n)])

            # draw the second POS from emission, transition, and emission2
            elif i == 1:
                combinations = product(word_probs['transition'].keys(),
                                       word_probs['emission'].keys(),
                                       word_probs['emission2'].keys())
                next_probs = {
                    emk: sum([word_probs['transition'][trk],
                              word_probs['emission'][emk],
                              word_probs['emission2'][em2k]])
                    for trk, emk, em2k in combinations
                    if trk[1] == emk[0] == em2k[0]
                }
                pos_samples.extend([pos for pos, word
                                    in self.SampleKeys(next_probs, n=n)])

            # draw POS from emission, transition, emission2, and transition2
            elif i != range(len(sentence))[-1]:
                combinations = product(word_probs['transition'].keys(),
                                       word_probs['emission'].keys(),
                                       word_probs['transition2'].keys(),
                                       word_probs['emission2'].keys())
                next_probs = {
                    emk: sum([word_probs['transition'][trk],
                              word_probs['emission'][emk],
                              word_probs['transition2'][tr2k],
                              word_probs['emission2'][em2k]])
                    for trk, emk, tr2k, em2k in combinations
                    if trk[1] == emk[0] == tr2k[1] == em2k[0]
                }
                pos_samples.extend([pos for pos, word
                                    in self.SampleKeys(next_probs, n=n)])

            # draw the final POS from emission, transition, and transition2
            else:
                combinations = product(word_probs['transition'].keys(),
                                       word_probs['emission'].keys(),
                                       word_probs['transition2'].keys())
                final_probs = {
                    emk: sum([word_probs['transition'][trk],
                              word_probs['emission'][emk],
                              word_probs['transition2'][tr2k]])
                    for trk, emk, tr2k in combinations
                    if trk[1] == emk[0] == tr2k[1]
                }
                pos_samples.extend([pos for pos, word
                                    in self.SampleKeys(final_probs, n=n)])

            # assign part-of-speech as most commonly sampled POS occurence
            best_pos = Counter(pos_samples).most_common(1)[0][0]
            predicted_pos.append(best_pos)
        return predicted_pos

    def solve(self, model: str, sentence: tuple):
        """
        This solve() method is called by label.py,
        so you should keep the interface the same,
        but you can change the code itself.
        It should return a list of part-of-speech labelings of the sentence,
        one part of speech per word.

        Args: 
            model: str :: declare the model by which the
                          parts-of-speech are computed
            sentence: tuple :: words in the sentence
        """
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algorithm!")
