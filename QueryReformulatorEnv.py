import random
from gym.utils import seeding
from rl.core import Env
from nltk.tokenize import RegexpTokenizer
import nltk
import corpus_hdf5
import dataset_hdf5
import utils
from time import time
from search import Search
import lucene_search
import os
import numpy as np
import parameters as prm
from tensorforce import TensorForceError
from tensorforce.environments import Environment
# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

class QueryReofrmulatorEnv(Env):
    def __init__(self, DATA_DIR, dset, is_train, verbose, reward = 'RECALL'):
        # this method returns simulator, state/action vocabularies, and the maximum number of actions
        n_words = 100  # 374000 # words for the vocabulary
        vocab_path = os.path.join(DATA_DIR,'data/D_cbow_pdw.pkl')   # Path to the python dictionary containing the vocabulary.
        wordemb_path = os.path.join(DATA_DIR,'data/D_cbow_pdw.pkl')  # Path to the python dictionary containing the word embeddings.
        dataset_path = os.path.join(DATA_DIR, 'data/msa_dataset.hdf5')  # path to load the hdf5 dataset containing queries and ground-truth documents.
        docs_path = os.path.join(DATA_DIR, 'data/msa_corpus.hdf5')  # Path to load the articles and links.
        docs_path_term = os.path.join(DATA_DIR, 'data/msa_corpus.hdf5')  # Path to load the articles and links.
        ############################
        # Search Engine Parameters #
        ############################
        n_threads = 1  # 20 # number of parallel process that will execute the queries on the search engine.
        index_name = 'index'  # index name for the search engine. Used when engine is 'lucene'.
        index_name_term = 'index_terms'  # index name for the search engine. Used when engine is 'lucene'.
        use_cache = False  # If True, cache (query-retrieved docs) pairs. Watch for memory usage.
        max_terms_per_doc = 15  # Maximum number of candidate terms from each feedback doc. Must be always less than max_words_input .
        self.vocab = utils.load_vocab(vocab_path, n_words)
        vocabinv = {}
        for k, v in self.vocab.items():
            vocabinv[v] = k
        self.reward = reward
        self.is_train = is_train
        self.search = Search(engine=lucene_search.LuceneSearch(DATA_DIR, self.vocab, n_threads, max_terms_per_doc, index_name, index_name_term, docs_path, docs_path_term, use_cache))

        t0 = time()
        dh5 = dataset_hdf5.DatasetHDF5(dataset_path)
        self.qi = dh5.get_queries(dset)
        self.dt = dh5.get_doc_ids(dset)
        print("Loading queries and docs {}".format(time() - t0))
        self.reset()


        '''for _, train_index in kf:
            qi, qi_i, qi_lst, D_gt_id, D_gt_url = self.get_samples(qi, dt, vocab, train_index, self.search.engine, max_words_input=self.search.max_words_input)
            # share the current queries with the search engine.
            current_queries = qi_lst
            i=3
            print 'Input Query:       ', qi[i].replace('\n', '\\n')
            print 'Target Docs: ', str(D_gt_url[i])
            print 'Input Query Vocab: ', utils.idx2text(qi_i[i], vocabinv)
            n_iterations = 2 # number of query reformulation iterations.
            for n_iter in range(n_iterations):
                print("current_queries", len(current_queries), current_queries)
                if n_iter < self.search.q_0_fixed_until:
                    ones = np.ones((len(current_queries), self.search.max_words_input))
                    if n_iter > 0:
                        # select everything from the original query in the first iteration.
                        reformulated_query = np.concatenate([ones, ones], axis=1)
                    else:
                        reformulated_query = ones
                print 'reformulated_query', reformulated_query.shape
                #reformulated_query is our action!!!
                metrics, D_i_, D_id_, D_gt_m_ = self.search.perform(reformulated_query, D_gt_id, self.is_train, current_queries)
                print "D_id_", D_id_
                print 'Iteration', n_iter
                print '  '.join(self.search.metrics_map.keys())
                print metrics.mean(0)
                print
                print 'Retrieved Docs:    ', str([self.search.engine.id_title_map[d_id] for d_id in D_id_[i]])
                print
                print 'Reformulated Query:', self.search.reformulated_queries[n_iter][i]
                print
                print 'Query ANS:         ',
                for kk, word in enumerate(current_queries[i][:reformulated_query.shape[1]]):
                    if word not in vocab and word != '':
                        word += '<unk>'
                    if reformulated_query[0, kk] == 1:
                        word = word.upper()
                    print str(word),
                print
                print'''

    def get_samples(self, input_queries, target_docs, vocab, index, engine, max_words_input=200):
        qi = [utils.clean(input_queries[t].lower()) for t in index]
        D_gt_title = [target_docs[t] for t in index]

        D_gt_id_lst = []
        for j, t in enumerate(index):
            #print("j",j)
            D_gt_id_lst.append([])
            for title in D_gt_title[j]:
                #print("title", title)
                if title in engine.title_id_map:
                    D_gt_id_lst[-1].append(engine.title_id_map[title])
                #else:
                #    print 'ground-truth doc not in index:', title

        D_gt_id = utils.lst2matrix(D_gt_id_lst)

        qi_i, qi_lst_ = utils.text2idx2(qi, vocab, max_words_input)
        #print("qi_i", qi_i)
        #print("qi_lst_", qi_lst_)

        qi_lst = []
        for qii_lst in qi_lst_:
            # append empty strings, so the list size becomes <dim>.
            qi_lst.append(qii_lst + max(0, max_words_input - len(qii_lst)) * [''])
        return qi, qi_i, qi_lst, D_gt_id, D_gt_title

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        reformulated_query, current_queries, D_gt_id = action
        metrics, D_i_, D_id_, D_gt_m_ = self.search.perform(reformulated_query, D_gt_id, self.is_train, current_queries)
        print "D_id_", D_id_
        i = 3
        print "ALALALA ", [self.search.engine.id_title_map[d_id] for d_id in D_id_[i]]
        text =  [[self.search.engine.id_title_map[d_id] for d_id in D_id_[i]] for i in range(D_id_.shape[0])]
        actions =current_queries
        metric_idx = self.search.metrics_map[self.reward.upper()]
        reward = metrics[metric_idx]
        if (len(actions) == 0):  # or self.counsteps > 10):
            done = True
        return [text, actions], reward, done, {}

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        t0 = time()
        #for now lets get one sample with all.
        kf = utils.get_minibatches_idx(len(self.qi), len(self.qi), shuffle=True)
        _, train_index = kf[0] #iterate if len(kf)>1 --> for _, train_index in kf:
        print "kf", kf, len(self.qi)
        print("Got minibatch index {}".format(time() - t0))

        qi, qi_i, qi_lst, D_gt_id, D_gt_url = self.get_samples(self.qi, self.dt, self.vocab, train_index, self.search.engine,
                                                               max_words_input=self.search.max_words_input)


        current_queries = qi_lst
        n_iterations = 1  # number of query reformulation iterations.
        if n_iterations < self.search.q_0_fixed_until:
            ones = np.ones((len(current_queries), self.search.max_words_input))
            reformulated_query = ones
            if n_iterations > 0:
                # select everything from the original query in the first iteration.
                reformulated_query = np.concatenate([ones, ones], axis=1)

        print 'reformulated_query', reformulated_query.shape
        # reformulated_query is our action!!!

        actions = reformulated_query, current_queries, D_gt_id
        [text, actions], reward, done, found =  self.step(action)
        print "text", text
        print "actions", actions
        print "rew", reward
        return [text, actions]


    def __del__(self):
        pass

    def get_tokenizers(self):
        state_tokenizer = nltk.word_tokenize
        action_tokenizer = nltk.word_tokenize
        return state_tokenizer, action_tokenizer


