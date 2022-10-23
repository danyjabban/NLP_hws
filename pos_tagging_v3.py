import numpy as np
import nltk
from viterbi import viterbi

# note script takes 20 seconds to run :(


def replace_unk(arr):
    word_freq = {}
    for word in arr:
        if word in word_freq.keys():
            word_freq[word] = word_freq[word] + 1
        else:
            word_freq[word] = 1
    unk_word = {word for word in word_freq.keys() if word_freq[word] == 1}
    for i in range(len(arr)):
        if arr[i] in unk_word:
            arr[i] = 'UNK'
    return arr


def preprocess():
    sentence_list = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    pos_tag = [word for sentence in sentence_list for word in sentence]
    word_list = np.array(list(zip(*pos_tag))[0])
    pos_list = np.array(list(list(zip(*pos_tag))[1]))

    word_list = replace_unk(word_list) # function to replace all single occurance words in training corpus with 'UNK'
    word_set = np.unique(word_list)
    pos_set = np.unique(pos_list)

    word_index = {word_set[i]:i for i in range(len(word_set))} # mapping of words to index, length n
    pos_index = {pos_set[i]:i for i in range(len(pos_set))} # mapping of part of speech to index, length p

    A = np.ones((len(pos_set),len(pos_set))) # p x p matrix of ones for smothing
    B = np.ones((len(pos_set),len(word_set))) # p x n matrix of ones for smothing
    pi = np.ones(len(pos_set)) # ones array of length p for smoothing
    return A, B, pi, word_list, pos_list, word_index, pos_index



def hmm_component_generator(A, B, pi, word_list, pos_list, word_index, pos_index):
    '''
    create frequency matrices (vectors) for A, B, (pi)
    '''
    for i in range(1, len(word_list)): # iterate through list of words in training corpus
        if i == 1:
            w_i = word_index[word_list[0]] # get word at index i
            p_i = pos_index[pos_list[0]] # get part of speech at index i
            B[p_i][w_i] = B[p_i][w_i] + 1 # update matrix B
            pi[p_i] = pi[p_i] + 1

        w_i = word_index[word_list[i]] # get word at index i
        p_i = pos_index[pos_list[i]] # get part of speech at index i
        p_i_m_1 = pos_index[pos_list[i-1]] # get part of speech at index i-1
        B[p_i][w_i] = B[p_i][w_i] + 1 # update matrix B
        A[p_i_m_1][p_i] = A[p_i_m_1][p_i] + 1 # update matrix A
        pi[p_i] = pi[p_i] + 1
    
    B = calc_matrix_probabilities(B)
    A = calc_matrix_probabilities(A)
    pi = calc_vector_probabilities(pi)
    return A, B, pi 


def calc_matrix_probabilities(M):
    # convert frequency to probability
    freq_array = np.sum(M, axis=1)
    prob_M = M / freq_array[:,None]
    return prob_M

def calc_vector_probabilities(v):
    # convert frequency to probability
    sum = np.sum(v)
    v_prob = v/sum
    return v_prob

def get_word_index(word_index):
    '''
    iterate through TEST tagged words and get index for each word
    '''
    sentence_list = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
    pos_tag = [word for sentence in sentence_list for word in sentence]
    test_word_list = list(zip(*pos_tag))[0] # list of words from test corpus
    test_pos_list = list(zip(*pos_tag))[1] # list of pos from test corpus
    obs_index = [] 
    # store the index of the word from test corpus list
    for word in test_word_list:
        if word in word_index.keys():
            obs_index.append(word_index[word])
        else:
            obs_index.append(word_index['UNK'])
    return obs_index, test_pos_list, test_word_list

def training():
    A, B, pi, word_list, pos_list, word_index, pos_index = preprocess()
    A, B, pi = hmm_component_generator(A, B, pi, word_list, pos_list, word_index, pos_index)
    return A, B, pi, word_list, pos_list, word_index, pos_index

def main():
    A, B, pi, word_list, pos_list, word_index, pos_index = training()
    obs_index, test_pos_list, test_word_list = get_word_index(word_index)
    output = viterbi(obs_index, pi, A, B)
    
    pos_res_index = dict((v,k) for k,v in pos_index.items())
    hmm_pos_list = [pos_res_index[pos] for pos in output[0]]
    #print(test_word_list)

    # print out all misclassified pos 
    # for i in range(len(hmm_pos_list)):
    #     if not test_pos_list[i] == hmm_pos_list[i]:
    #         print('**************************')
    #         print('actual part of speech   :',test_pos_list[i])
    #         print('predicted part of speech:',hmm_pos_list[i])
    #         print('incorrect word          :',test_word_list[i])

            

if __name__ == "__main__":
    main()