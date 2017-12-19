import random
from gym.utils import seeding
from rl.core import Env
from nltk.tokenize import RegexpTokenizer
import nltk
from keras.preprocessing import sequence
import numpy as np

class FocusedcrawlingEnv(Env):
    def __init__(self, datafile, goalsfile, anchortxt, backaction, verbose, maxlen, embedding):
        # this method returns simulator, state/action vocabularies, and the maximum number of actions
        import pickle
        import webpage
        self.query_dest_id = pickle.load(open(goalsfile, 'r'))#'data/uiucgoal.pickle'
        self.num_goals = len(self.query_dest_id ) * 2
        self.wk = webpage.Webpage(datafile)#'data/uiucdataset.hdf5'
        self.wk.set_cleaned(True)
        self.max_actions, self.min_actions, self.avg_actions = self.wk.get_max_min_avg_nactions()
        self.nstates = len(self.wk.get_titles_pos())
        print('action statistics ', self.max_actions, self.min_actions, self.avg_actions)
        print(' number of states',self.nstates)
        self.anchortxt = anchortxt
        self.backaction = backaction
        self.verbose = verbose
        self.maxlen = maxlen
        self.embedding = embedding
        self.dict_wordId = self.dict_actionId = None
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def execute(self, actions):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        """if reward is 1 then list of actions is empty as we reached a terminal!"""
        """returns (text, list of actions , reward)"""
        action = actions
        if(action >= len(self.currentactions)):
            action = random.randint(0,len(self.currentactions) - 1)
        self.current_state = self.currentactions[action]
        reward = self.AssignReward(self.current_state)
        if (reward == 1 and len(self.goal) <= 1):  # only one goal (left) to find
            self.currentactions = []
            actions_t = []
        else:
            self.currentactions = self.wk.get_article_links(self.current_state)
            if (reward == 1 and len(self.goal) > 0):
                self.goal.remove(self.current_state)
            if (self.backaction):
                self.currentactions.insert(0, self.root)
            if (self.anchortxt):
                actions_t = [self.wk.get_article_title(id) for id in self.currentactions]
            else:
                actions_t = [self.wk.get_article_text(id) for id in self.currentactions]
        text = self.wk.get_article_text(self.current_state)
        done = False
        if(len(actions_t) == 0):# or self.counsteps > 10):
            done = True
        if(self.verbose):
            print("Traversed to ", self.current_state, self.wk.get_article_title(self.current_state), actions_t)
        state_tokenizer, action_tokenizer = self.get_tokenizers()
        vec_sum =  self.embedding.text_to_idx(text, state_tokenizer) #embedding.get_text_embedding(text)
        #vec_actions = self.embedding.get_actions_toidx(actions, action_tokenizer) #embedding.get_actions_embeddings(actions)
        vec_sum = sequence.pad_sequences([vec_sum], maxlen=self.maxlen, padding='post', dtype='int32')
        #vec_actions = sequence.pad_sequences(vec_actions, maxlen=None, padding='post', dtype='int32')
        return vec_sum , done, reward #[vec_sum, vec_actions], reward, done,  self.found

    def AssignReward(self, id):
        """inside here could be a classifier already trained to classify webpages are relevant or not"""
        """OR we could have ids of webpages and already labeled as relevant(1) or not (0)"""
        """return a score for this webpage text or id"""
        if (id in self.goal):  # end page of goal
            print("***Yes!", id)
            self.found[id] = self.wk.get_article_text(id)
            return 1
        return -0.1

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        """load initial embeddings for both actions and webpages (optional)"""
        """make a random StoryNode as the starting point"""
        # self.current_state =  random.choice(self.states)
        self.counsteps = 0
        self.goal = list(zip(*self.query_dest_id.values())[1])  # random.choice(self.query_dest_id.items())
        self.observation = self.query_dest_id.items()[0][1][0]  # self.goal[1][0] #start page of goal
        self.root = self.observation
        self.found = {}
        text = self.wk.get_article_text(self.observation)
        self.currentactions = self.wk.get_article_links(self.observation)
        if (self.anchortxt):
            actions_t = [self.wk.get_article_title(id) for id in self.currentactions]
        else:
            actions_t = [self.wk.get_article_text(id) for id in self.currentactions]
        if (self.verbose):
            print("reset ", self.root, self.wk.get_article_title(self.root), text, actions_t)
        state_tokenizer, action_tokenizer = self.get_tokenizers()
        vec_sum =  self.embedding.text_to_idx(text, state_tokenizer) #embedding.get_text_embedding(text)
        vec_sum = sequence.pad_sequences([vec_sum], maxlen=self.maxlen, padding='post', dtype='int32')
        #vec_actions = self.embedding.get_actions_toidx(actions, action_tokenizer) #embedding.get_actions_embeddings(actions)
        #vec_actions = sequence.pad_sequences(vec_actions, maxlen=None, padding='post', dtype='int32')
        return  vec_sum#[vec_sum, vec_actions]


    def __del__(self):
        pass

    def get_tokenizers(self):
        state_tokenizer = nltk.word_tokenize
        if (self.anchortxt):
            action_tokenizer = RegexpTokenizer(r'\w+').tokenize
            # TODO: links have always wgetdata in front and html back remove it? [1:-1]
        else:
            action_tokenizer = nltk.word_tokenize
        return state_tokenizer, action_tokenizer
