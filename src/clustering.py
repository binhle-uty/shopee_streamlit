from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

from multiprocessing import Pool
from copy import deepcopy
from scipy.sparse.csgraph import connected_components
from collections import OrderedDict

from src.nlp_preprocess import *
from sklearn.feature_extraction.text import CountVectorizer
import nltk

stopwords = set(open('stopwords.txt').read().split('\n')[:-1])
puct_set = set([c for c in '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~'])
THRESHOLD = 0.1

from collections import defaultdict
 
class TextClustering:
    def __init__(self,documents):
        self.documents = documents
        

    @staticmethod
    def generateBigram(paper):
        words = paper.split()
        if len(words) == 1:
            return ''
        bigrams = [words[i] + '_' + words[i+1] for i in range(0,len(words) - 1)]
        return ' '.join(bigrams)

    @staticmethod
    def removeRedundant(text,redundantSet):
        words = text.split()
        for i in range(0,len(words)):
            if words[i].count('_') == 0 and (words[i] in redundantSet or words[i].isdigit()):
                words[i] = ''
            else:
                sub_words = words[i].split('_')
                if any(w in redundantSet or w.isdigit() for w in sub_words):
                    words[i] = ''
        words = [w for w in words if w != '']
        words = ' '.join(words)
        return words

    def preprocessing(self, text):
        text = ' '.join(word_tokenize(text))
        text = text.lower()
        text = ' '.join(text.split())
        text = text + self.generateBigram(text)
        text = self.removeRedundant(text,puct_set | stopwords)
        return text

    def clean_docs(self):
        pool = Pool(10)
        self.clean_documents = pool.map(self.preprocessing,self.documents)
        pool.terminate()
        return self.clean_documents


    def svd(self):
        vectorizer = TfidfVectorizer(token_pattern = "\S+", min_df = 2)
        vectors = vectorizer.fit_transform(self.clean_documents)

        svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
        self.svd_vectors = svd.fit_transform(vectors)

        return self.svd_vectors


    @staticmethod
    def distance(vecs):
        vec1 = vecs[0]
        vecAll = vecs[1]
        Dis_matrix = pairwise_distances(vec1,vecAll,metric = 'cosine',n_jobs=1)
        Dis_matrix = Dis_matrix.astype(np.float16)
        return Dis_matrix

    @staticmethod
    def chunks_vec(l, n):
        for i in range(0, l.shape[0], n):
            yield l[i:i + n]


    def similar_matrix(self):
        vector_chunks = list(self.chunks_vec(self.svd_vectors,1000))
        vector_chunks = [(i,self.svd_vectors) for i in vector_chunks]

        pool = Pool(2)
        self.Dis_matrix = pool.map(self.distance,vector_chunks)
        self.Dis_matrix = np.vstack(self.Dis_matrix)
        pool.terminate()
        return self.Dis_matrix


    def create_graph(self):
        self.graph = deepcopy(self.Dis_matrix)
        self.graph[self.graph <= THRESHOLD] = 2
        self.graph[self.graph != 2] = 0
        self.graph[self.graph == 2] = 1
        self.graph = self.graph.astype(np.int8)
        return self.graph

    def find_connected_components(self):
        self.res = connected_components(self.graph,directed=False)
        return self.res

    def extract_cluster(self):
        cluster_labels = self.res[1]
        num_cluster = self.res[0]
        self.res_cluster = OrderedDict()

        for i in range(0,len(cluster_labels)):
            if cluster_labels[i] in self.res_cluster: self.res_cluster[cluster_labels[i]].append(i)
            else: self.res_cluster[cluster_labels[i]] = [i]

        self.res_cluster = [self.res_cluster[i] for i in range(0,num_cluster)]
        self.res_cluster = [sorted(r) for r in self.res_cluster if len(r) > 1]
        self.res_cluster.sort(key=len,reverse=True)
        return self.res_cluster
    
    def get_longest_common_subseq(self, data):
        data = [x.lower() for x in data]
        if len(data)<=1:
            return data
        else:
            substr = []
            if len(data) > 1 and len(data[0]) > 0:
                for i in range(len(data[0])):
                    for j in range(len(data[0])-i+1):
                        if j > len(substr) and self.is_subseq_of_any(data[0][i:i+j], data):
                            substr = data[0][i:i+j]
            return substr

    def is_subseq_of_any(self, find, data):
        if len(data) < 1 and len(find) < 1:
            return False
        for i in range(len(data)):
            if not self.is_subseq(find, data[i]):
                return False
        return True

    @staticmethod  
    def is_subseq(possible_subseq, seq):
        if len(possible_subseq) > len(seq):
            return False
        def get_length_n_slices(n):
            for i in range(len(seq) + 1 - n):
                yield seq[i:i+n]
        for slyce in get_length_n_slices(len(possible_subseq)):
            if slyce == possible_subseq:
                return True
        return False

    def map_cluster_to_list(self):
        self.mapping = {}

        for i in range(0,len(self.res_cluster)):
            list_docs = [self.documents[idx].split('\n')[0] for idx in self.res_cluster[i]]
            print(list_docs)
            cluster_name = self.get_longest_common_subseq(list_docs)
            for idx in self.res_cluster[i]:
                print(self.documents[idx].lower().strip())
                print(cluster_name)
                self.mapping[self.documents[idx].lower().strip()] = cluster_name
        return self.mapping


    def main_flow(self):
        #strep 1: clean documents
        self.clean_documents = self.clean_docs()
        #step 2: svd
        self.svd_vectors = self.svd()
        #step 3: similarity
        self.Dis_matrix = self.similar_matrix()
        #step 4: graph
        self.graph = self.create_graph()
        #step 5: connected components
        self.res = self.find_connected_components()
        #step 6: extract cluster
        self.res_cluster =self. extract_cluster()
        #step 7: map cluster to list
        self.mapping = self.map_cluster_to_list()
        return self.mapping

