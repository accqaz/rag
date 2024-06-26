from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from pipeline.utils import tf_idf, get_encoder, strip_string, tf_idf2, tf_idf_sort
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from itertools import chain
import torch
#from torch_scatter import segment_csr
from Levenshtein import distance as levenshtein_distance
from pipeline.utils import cal_local_llm_llama
import json


class No_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data):
        return []
    
# class KNN_retrieval(object):
#     def __init__(self, text_encoder, k, k_emb):
#         self.encoder = get_encoder(text_encoder)
#         self.k_emb = k_emb
#         self.k = k
        
#     def index(self, data):
#         self.strip_chunks = data['strip_chunks']
#         self.group_idx = np.array(data['group_idx'])
#         self.chunks = data['chunks']
#         self.np_chunks = np.array(self.chunks)
#         self.chunk_embeds = data['chunk_embeds']
    
#     def query_encode(self, query):
#         return self.encoder.encode([strip_string(query)])
    
#     def retrieve(self, data):
#         self.index(data)
        
#         query_embed = self.query_encode(data['question'])

#         #Search by embedding similarity
#         scores = cosine_similarity(query_embed, self.chunk_embeds).flatten()
#         group_scores = segment_csr(torch.tensor(scores), torch.tensor(self.group_idx), reduce = 'max').numpy()
#         topk = np.argsort(group_scores)[-self.k_emb:].tolist()

#         #Search by text fuzzy matching
#         dist = [levenshtein_distance(data['question'], chunk) for chunk in self.strip_chunks]
#         idxs = np.argsort(dist)

#         for idx in idxs:
#             if idx not in topk:
#                 topk.append(idx)
#                 if len(topk) == self.k:
#                     break

#         return [data['title_chunks'][_][1] for _ in topk]
    
class TF_IDF_retriever(object):
    def __init__(self, k):
        self.vectorizer = TfidfVectorizer()
        self.k = k
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        query_emb = self.vectorizer.transform([data['question']])

        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()

        return [corpus[idx] for idx in cosine_sim.argsort()[-self.k:][::-1]]


class BM25_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        self.bm25 = BM25Okapi([c.split(" ") for c in corpus])

        scores = self.bm25.get_scores(data['question'].split(" "))

        return [corpus[idx] for idx in scores.argsort()[-self.k:][::-1]]
    

class KG_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, query, data, Gs):
        log = []
        log.append("**************************************")
        log.append("**************************************")
        log.append(query)
        log.append("Initial retrieval based on query==============")
        
        # retrieve_graph_idxs = []
        # graph_corpus = [doc['document_type'] + ' ' + doc['title_chunks'][0][1] for doc in data]
        # log.append(f"graph_corpus: {graph_corpus}")
        
        # graphs_idx = tf_idf(query, list(range(len(graph_corpus))), graph_corpus, k=5, visited=retrieve_graph_idxs)
        # log.append(f"Initial retrieved graph indices,graphs_idx: {graphs_idx}")
        graphs_idx = len(Gs)
        
        seed = query
        all_corpus = []
        corpus = []
        total_k = 60 # 多图一共可以检索到的文档数
        
        for graph_idx in range(graphs_idx):
            log.append("================================")
            log.append(f"Processing graph_idx: {graph_idx}")
            
            G = Gs[graph_idx]
            corpus = [text for _, text in data[graph_idx]['title_chunks']]
            candidates_idx = list(range(len(corpus)))
            
            log.append(f"Corpus length: {len(corpus)}")
            log.append(f"Corpus content: {corpus[0]}")
            
            retrieve_idxs = []
            prev_length = 0
            count = 0
            retrieve_num = [5, 5]
            
            while count < len(retrieve_num):
                k_value = retrieve_num[count] if count < len(retrieve_num) else retrieve_num[-1]
                idxs = tf_idf(seed, candidates_idx, corpus, k=k_value, visited=retrieve_idxs)
                
                log.append(f"Step {count + 1} - Retrieved indices based on seed '{seed}': {idxs}")
                retrieve_idxs.extend(idxs[:max(0, total_k - len(retrieve_idxs))])
                log.append(f"Current retrieved indices: {retrieve_idxs}")
                
                candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
                candidates_idx = list(candidates_idx.difference(retrieve_idxs))
                log.append(f"Candidate indices after neighbors and difference operation: {candidates_idx}")

                if len(retrieve_idxs) == prev_length:
                    break
                else:
                    prev_length = len(retrieve_idxs)
                
                count += 1
            
            all_corpus.extend([corpus[idx] for idx in retrieve_idxs])
            log.append(f"One retrieved idxs: {retrieve_idxs}")
            corpus = [] # 一个图遍历完之后，清空
        
        final_retrieved_docs = tf_idf_sort(query, all_corpus, self.k)
        log.append(f"Final retrieved documents: {final_retrieved_docs}")
        # tmp = query[:2]
        # # 将调试信息写入 JSON 文件
        # with open("retrieval_{}.json".format(tmp), "w", encoding="utf-8") as log_file:
        #     json.dump(log, log_file, ensure_ascii=False, indent=4)
        
        # 去重并选择相似度最高的前 10 个文档
        return final_retrieved_docs

        

        # print("KG_retriever==============")
        # #print(data)
        # corpus = [c for _, _, c in data['title_chunks']]
        # candidates_idx = list(range(len(corpus)))
        # print("candidates_idx", candidates_idx)

        # seed = query
        # retrieve_idxs = []

        # prev_length = 0
        # count = 0
        # retrieve_num = [10, 5, 5, 5, 3, 2, 2, 2, 2, 2, 2]
        # while len(retrieve_idxs) < self.k:
        #     for graph_idx in graphs_idx:
        #         G = Gs[graph_idx]
        #         corpus = [text for _, text in data[graph_idx]['title_chunks']]  #每个子图对应的语料库
        #         print("G", G)
        #         idxs = tf_idf(seed, candidates_idx, corpus, k = retrieve_num[count], visited = retrieve_idxs)
        #         print("根据种子初步得到的idxs", idxs)
        #         retrieve_idxs.extend(idxs[:max(0, self.k - len(retrieve_idxs))])
                
        #         candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
        #         candidates_idx = list(candidates_idx.difference(retrieve_idxs))

        #         if len(retrieve_idxs) == prev_length:
        #             break
        #         else:
        #             prev_length = len(retrieve_idxs)
                
        #         count += 1
        #     idxs = tf_idf(seed, candidates_idx, corpus, k = retrieve_num[count], visited = retrieve_idxs)
        #     print("根据种子初步得到的idxs", idxs)
        #     retrieve_idxs.extend(idxs[:max(0, self.k - len(retrieve_idxs))])
            
        #     candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
        #     candidates_idx = list(candidates_idx.difference(retrieve_idxs))

        #     if len(retrieve_idxs) == prev_length:
        #         break
        #     else:
        #         prev_length = len(retrieve_idxs)
            
        #     count += 1

        # return [corpus[idx] for idx in retrieve_idxs]
    
class llm_retriever_LLaMA(object):
    def __init__(self, k, k_nei, port):
        self.k = k
        self.k_nei = k_nei
        self.port = port
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        contexts = []
        
        idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

        for idx in idxs:
            context = seed + '\n' + corpus[idx]

            next_reason = cal_local_llm_llama(context, self.port)

            next_contexts = tf_idf(next_reason, candidates_idx, corpus, k = self.k_nei, visited = [])

            if next_contexts != []:
                contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            else:
                contexts.append(corpus[idx])

        return contexts
    

class llm_retriever_KG_LLaMA(object):
    def __init__(self, k, k_nei, port):
        self.k = k
        self.k_nei = k_nei
        self.port = port
    
    def retrieve(self, data, G):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        contexts = []
        #候选项为所有的corpus
        idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

        for idx in idxs:
            context = seed + '\n' + corpus[idx]

            next_reason = cal_local_llm_llama(context, self.port)

            nei_candidates_idx = list(G.neighbors(idx))
            #候选项为邻居节点
            next_contexts = tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei, visited = [])

            if next_contexts != []:
                contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            else:
                contexts.append(corpus[idx])

        return contexts