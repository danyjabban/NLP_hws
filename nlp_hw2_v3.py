'''
Spell Check
'''
from functools import cache
import numpy as np

@cache
def levenshtein_distance(w1, w2):
    # get minimum edit distance
    # check two base cases that at least one word has zero letters
    if len(w1) == 0:
        return len(w2) 
    elif len(w2) == 0:
        return len(w1) 

    else:
        if w1[-1] == w2[-1]:
            last_match = 0
        else:
            last_match = 1

        return min((levenshtein_distance(w1, w2[:-1]) + 1, 
                    levenshtein_distance(w1[:-1], w2) + 1, 
                    levenshtein_distance(w1[:-1], w2[:-1]) + last_match))


def spell_check(noisy_word):
    '''
    arguments: noisy_word is your noisy word string
    '''
    p = .0001 # probability of corruption
    word_frequency = {}
    word_frequency_sum = 0 
    # create dictionary of word:frequency pairs
    with open('count_1w.txt') as f:
        for line in f:
            (k, v) = line.split()
            word_frequency_sum += int(v)
            word_frequency[k] = int(v) 
    most_likely_word = ' '
    max_probability = 0
    for word in word_frequency:
        count = word_frequency[word]
        # marginal probability from word frequency
        p_w = count/word_frequency_sum 
        min_edit_distance = levenshtein_distance(noisy_word, word)
        # conditional probability from p times edit distance
        p_x_w = p**(min_edit_distance) 
        # joint probability of x and w
        p_w_x = p_x_w * p_w
        # if word w has greater probability than current max, replace word and update max
        if p_w_x > max_probability:
            max_probability = p_w_x
            most_likely_word = word
    return most_likely_word


if __name__ == "__main__":
    corrected_word = spell_check('xook')
    print(corrected_word)
    #a = levenshtein_distance('fillt', 'still')
    #print(a)
