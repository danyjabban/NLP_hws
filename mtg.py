'''
Markov text generator for n-gram model
'''
import nltk
import numpy as np
import random

def finish_sentence(sentence, n, corpus, deterministic=False):
    # while loop until sentence finished
    while (sentence[-1] not in ['.', '!', '?']) and (len(sentence) < 10):
        n_gram_freq_dict = {}
        n_minus_1_gram_freq = 0
        temp_n = n
        # while there are no n_grams that match reduce n by 1 
        while n_minus_1_gram_freq == 0:
            n_minus_1_gram_freq = corpus_iter(sentence, temp_n, corpus, n_gram_freq_dict, n_minus_1_gram_freq)
            temp_n -= 1 # reduce n by 1
        
        if deterministic:
            new_word = deterministic_prediction(n_gram_freq_dict)
        else:
            new_word = probabilistic_prediction(n_gram_freq_dict, n_minus_1_gram_freq)
        sentence.append(new_word)
    return sentence

def corpus_iter(sentence, n, corpus, n_gram_freq_dict, n_minus_1_gram_freq):
    for i in range(n-1, len(corpus)+1):
        n_minus_1_gram = corpus[i-(n-1):i] # n-minus 1 gram
        n_gram = corpus[i-(n-1):i+1] # n-gram
        str_n_gram = " "
        str_n_gram = str_n_gram.join(n_gram) # create string n-gram
        if (n_minus_1_gram == sentence[-(n-1):]) or n == 1:
            n_minus_1_gram_freq += 1
            if str_n_gram in n_gram_freq_dict:  # if n-gram is present in dictionary
                n_gram_freq_dict[str_n_gram] += 1
            else:                               # if n-gram not yet in dictionary
                n_gram_freq_dict[str_n_gram] = 1
    return n_minus_1_gram_freq




def probabilistic_prediction(n_gram_freq_dict, n_minus_1_gram_freq):
    cum_sum = 0
    n = random.uniform(0, 1) # pick random variable from uniform distribution [0,1]
    for k, v in n_gram_freq_dict.items():
        cum_sum = cum_sum + v/n_minus_1_gram_freq # add to probability mass function
        if n < cum_sum:  # if random var n is in probability range return word from that range
            str_dict = k.split(" ")
            new_word = str_dict[-1] # last word of most frequent n-gram
            return new_word 
        n_gram_freq_dict.update({k:cum_sum})
    

def deterministic_prediction(n_gram_freq_dict):
    max_freq = 0
    new_word = ''
    for k, v in n_gram_freq_dict.items():
        if v > max_freq: # check if new n-gram has highest frequency
            str_dict = k.split(" ")
            new_word = str_dict[-1]
            max_freq = v
    return new_word




if __name__ == '__main__':
    sentence = ['she', 'was', 'in']
    n = 3
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw('austen-sense.txt').lower())
    deterministic = True
    completed_sentence = finish_sentence(sentence, n, corpus, deterministic)
    print(completed_sentence)