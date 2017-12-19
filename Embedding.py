from gensim.models import KeyedVectors as Word2Vec
import numpy as np
import nltk
from copy import deepcopy
from nltk.tokenize import RegexpTokenizer
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class Embedding(object):

    def __init__(self, GLOVE_FILE, binary=False):
        self.word2vec = Word2Vec.load_word2vec_format(GLOVE_FILE, binary=binary)
        self.word_index = None
        self.embedding_dim = self.word2vec.syn0.shape[1]

    def limit_vocab(self,n_words):
        vocab = {}
        for word, vocab_obj in self.word2vec.vocab.items():
            vocab[word] = vocab_obj.count
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        vocab  = vocab[0:n_words]
        vocab = {word:index for index, word in enumerate(dict(vocab).keys())}
        print("faculty in vocab?",vocab['faculty'])
        self.word_index =  vocab

    def get_embedding_matrix(self, word_index=None):
        if(word_index):
            return self.get_embedding_matrix_limit(word_index)
        if(self.word_index):
            return self.get_embedding_matrix_limit(self.word_index)
        return self.get_embedding_matrix_all()

    def get_embedding_matrix_all(self):
        #print('Preparing embedding matrix.')
        # prepare embedding matrix
        nb_words = len(self.word2vec.vocab)
        print('embedding_dim', self.embedding_dim)
        embedding_matrix = np.zeros((nb_words + 1, self.embedding_dim))
        for i,word in enumerate(self.word2vec.vocab.keys()):
            # words not found in embedding index will be all-zeros.
            #print(word, i)
            embedding_vector = self.word2vec[word]
            #print(embedding_vector)
            embedding_matrix[i] = embedding_vector
        return embedding_matrix, nb_words

    def get_embedding_matrix_limit(self, word_index):
        #print('Preparing embedding matrix.')
        # prepare embedding matrix
        nb_words = len(word_index)
        embedding_dim = self.word2vec.syn0.shape[1]
        print('embedding_dim', embedding_dim)
        embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
        for word, i in word_index.items():
            if word in self.word2vec.vocab:
                # words not found in embedding index will be all-zeros.
                embedding_vector = self.word2vec[word]
                embedding_matrix[i] = embedding_vector
        return embedding_matrix, nb_words


    def text_to_idx(self, text, tokenizer, word_index=None):
        #print(tokenizer, tokenizer(text.lower()))
        if(word_index):
            to_idx = lambda x: [word_index[word]+1 for word in tokenizer(x.lower()) if word in word_index]
        elif(self.word_index):
            to_idx = lambda x: [self.word_index[word]+1 for word in tokenizer(x.lower()) if word in self.word_index]
        else:
            to_idx = lambda x: [self.word2vec.index2word.index(word) + 1 for word in tokenizer(x.lower()) if word in self.word2vec.vocab]
        return to_idx(text)

    def get_actions_toidx(self, actions, tokenizer, word_index=None):
        vec_actions = []
        for action in actions:
            vec_suma = self.text_to_idx(action, tokenizer, word_index)
            if not vec_suma:
                #print("List is empty", actions, word_index)
                vec_suma = [0]
            vec_actions.append(vec_suma)
        return vec_actions

    def get_text_embedding(self, text):
        vec_sum = 0
        cnt = 0
        for word in nltk.word_tokenize(text):
            if word in self.word2vec.vocab:
                #print(word)#,self.word2vec[word])
                vec = self.word2vec[word]
                vec_sum = vec_sum + vec
                cnt = cnt + 1.0
        if cnt > 0:
            vec_sum = vec_sum / cnt
            vec_sum =np.array(vec_sum)
        else:
            vec_sum = np.zeros((1,300), dtype=np.int)
        return vec_sum
    
    def get_actions_embeddings(self, actions):
        vec_actions = np.zeros((len(actions),300))
        #print("vec_actions.shape",vec_actions.shape)
        for i, idx in enumerate(actions):
        #for a in actions:
            #print(i)
            vec_suma = self.get_text_embedding(actions[i])
            vec_actions[i] = vec_suma
            #print(i,vec_actions)
        return vec_actions

    '''def score_sentence(self, sentence, window=5):
        """
        Obtain likelihood score for a single sentence in a fitted skip-gram representaion.
        The sentence is a list of Vocab objects (or None, when the corresponding
        word is not in the vocabulary). Called internally from `Word2Vec.score()`.
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
        """
        log_prob_sentence = 0.0
        word_vocabs = [self.word2vec.vocab[w] for w in sentence if w in self.word2vec.vocab]
        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip
            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - window)
            for pos2, word2 in enumerate(word_vocabs[start: pos + window + 1], start):
                # don't train on OOV words and on the `word` itself
                if word2 is not None and pos2 != pos:
                    l1 = self.word2vec.syn0[word2.index]
                    l2a = deepcopy(self.word2vec.syn1neg[word.point])  # 2d matrix, codelen x layer1_size
                    sgn = (-1.0) ** word.code  # ch function, 0-> 1, 1 -> -1
                    lprob = - np.logaddexp(0, -sgn * np.dot(l1, l2a.T))
                    log_prob_sentence += sum(lprob)

        return log_prob_sentence'''

