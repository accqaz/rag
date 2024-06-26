# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from rank_bm25 import BM25Okapi
# from pipeline.utils import tf_idf, get_encoder, strip_string, tf_idf2, tf_idf_sort
# from langchain.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain
# from itertools import chain
# import torch
# #from torch_scatter import segment_csr
# from Levenshtein import distance as levenshtein_distance
# from pipeline.utils import cal_local_llm_llama, cal_llm
# import json
# import jieba
# from rank_bm25 import BM25Okapi
# from nltk.corpus import stopwords
# import spacy

# #nlp = spacy.load('zh_core_web_sm')
# stops = set(stopwords.words('chinese'))
# # spacy_stopwords = nlp.Defaults.stop_words
# # stops.update(spacy_stopwords)
# chinese_punctuation = set('，。、；：！？（）“”‘’《》【】『』〖〗—…～·')
# whitespace_characters = {'\n', '\r', '\t', ' '}
# stops.update(chinese_punctuation)
# stops.update(whitespace_characters)
# # print("stops", stops)
# # print(len(stops))
# # # 加载NLTK中文停用词
# # stopwords = list(nltk_stopwords.words('chinese'))

# def preprocess(text):
#     words = jieba.lcut(text)
#     return ' '.join([word for word in words if word not in stops and word.strip()])


# class No_retriever(object):
#     def __init__(self, k):
#         self.k = k
    
#     def retrieve(self, data):
#         return []
    
# # class KNN_retrieval(object):
# #     def __init__(self, text_encoder, k, k_emb):
# #         self.encoder = get_encoder(text_encoder)
# #         self.k_emb = k_emb
# #         self.k = k
        
# #     def index(self, data):
# #         self.strip_chunks = data['strip_chunks']
# #         self.group_idx = np.array(data['group_idx'])
# #         self.chunks = data['chunks']
# #         self.np_chunks = np.array(self.chunks)
# #         self.chunk_embeds = data['chunk_embeds']
    
# #     def query_encode(self, query):
# #         return self.encoder.encode([strip_string(query)])
    
# #     def retrieve(self, data):
# #         self.index(data)
        
# #         query_embed = self.query_encode(data['question'])

# #         #Search by embedding similarity
# #         scores = cosine_similarity(query_embed, self.chunk_embeds).flatten()
# #         group_scores = segment_csr(torch.tensor(scores), torch.tensor(self.group_idx), reduce = 'max').numpy()
# #         topk = np.argsort(group_scores)[-self.k_emb:].tolist()

# #         #Search by text fuzzy matching
# #         dist = [levenshtein_distance(data['question'], chunk) for chunk in self.strip_chunks]
# #         idxs = np.argsort(dist)

# #         for idx in idxs:
# #             if idx not in topk:
# #                 topk.append(idx)
# #                 if len(topk) == self.k:
# #                     break

# #         return [data['title_chunks'][_][1] for _ in topk]
    
# class TF_IDF_retriever(object):
#     def __init__(self, k):
#         self.vectorizer = TfidfVectorizer()
#         self.k = k
    
#     def retrieve(self, data):
#         corpus = [c for _, c in data['title_chunks']]
#         tfidf_matrix = self.vectorizer.fit_transform(corpus)
#         query_emb = self.vectorizer.transform([data['question']])

#         cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()

#         return [corpus[idx] for idx in cosine_sim.argsort()[-self.k:][::-1]]


# class BM25_retriever(object):
#     def __init__(self, k):
#         self.k = k
    
#     def retrieve(self, data):
#         corpus = [c for _, c in data['title_chunks']]
#         self.bm25 = BM25Okapi([c.split(" ") for c in corpus])

#         scores = self.bm25.get_scores(data['question'].split(" "))

#         return [corpus[idx] for idx in scores.argsort()[-self.k:][::-1]]
    

# class KG_retriever(object):
#     def __init__(self, k):
#         self.k = k
#         self._bm25 = None
#         self._corpus = []
        
#     def _initialize_bm25(self, corpus):
#         #tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
#         tokenized_corpus = [preprocess(doc) for doc in corpus]
#         self._bm25 = BM25Okapi(tokenized_corpus)
#         self._corpus = tokenized_corpus
#         #print("BM25 initialized.")
#         # print(self._corpus)

#     def bm25_retrieve(self, query_str, data, Gs):
#         tokenized_query = preprocess(query_str)        
#         graphs_idx = len(Gs)
#         log = []
#         log.append(f"tokenized_query: {tokenized_query}")
#         all_results = []

#         for graph_idx in range(graphs_idx):
#             corpus = [text for _, text in data[graph_idx]['title_chunks']]
#             log.append(f"Corpus length: {len(corpus)}")
#             self._initialize_bm25(corpus)
#             scores = self._bm25.get_scores(tokenized_query)
#             idxs = scores.argsort()[::-1][:min(50, len(corpus))]  # 取前50个或更少
#             all_results.extend([(graph_idx, scores[idx]) for idx in idxs])
        
#         all_results.sort(key=lambda x: x[1], reverse=True)
#         # 取all_results的前100
#         top_results = all_results[:100]

#         graph_scores = {}
#         graph_counts = {}
#         for graph_idx, score in top_results:
#             if graph_idx in graph_scores:
#                 graph_scores[graph_idx] += score
#                 graph_counts[graph_idx] += 1
#             else:
#                 graph_scores[graph_idx] = score
#                 graph_counts[graph_idx] = 1

#         # 平均分数计算
#         average_graph_scores = {graph_idx: graph_scores[graph_idx] / graph_counts[graph_idx] for graph_idx in graph_scores}
        
#         sorted_graphs = sorted(average_graph_scores.items(), key=lambda item: item[1], reverse=True)
        
#         log.append(f"Graph scores: {graph_scores}")
#         log.append(f"Graph counts: {graph_counts}")
#         log.append(f"Average graph scores: {average_graph_scores}")
#         log.append(f"Sorted graphs: {sorted_graphs}")
#         log.append(f"graphs index: {[graph_idx for graph_idx, _ in sorted_graphs]}")
#         tmp = query_str[:2]
#         with open(f"./query-test/retrieval_{tmp}.json", "w", encoding="utf-8") as log_file:
#             json.dump(log, log_file, ensure_ascii=False, indent=4)
        
#         return [graph_idx for graph_idx, _ in sorted_graphs]
    
#     def bm25_graph_retrieve(self, query_str, top_k):
#         if not self._bm25:
#             return []
#         tokenized_query = preprocess(query_str)
#         return self._bm25.get_top_n(tokenized_query, self._corpus, n=top_k)
    
#     def retrieve(self, query, data, Gs, llm):
#         log = []
#         graphs_idx = self.bm25_retrieve(query, data, Gs)  # 先初步获取要遍历的图的索引
#         seed = query
#         all_contexts = {}
#         contexts = []
#         log.append({"query": query, "initial_retrieval": graphs_idx})

#         for graph_idx in graphs_idx[:5]:
#             retrieve_idxs = []
#             log.append({"separator": "================================"})
#             log.append({"processing_graph_idx": graph_idx})
#             print(f"Processing graph_idx: {graph_idx}")
#             corpus = []
#             G = Gs[graph_idx]
#             corpus = [text for _, text in data[graph_idx]['title_chunks']]
#             candidates_idx = list(range(len(corpus)))
#             print(f"candidates_idx: {len(candidates_idx)}")
#             log.append({"candidates_idx": len(candidates_idx)})
#             initial_idxs = tf_idf(seed, candidates_idx, corpus, 5, visited=[])
#             log.append({"graph_idx": graph_idx, "initial_idxs": initial_idxs})
#             print(f"initial_idxs:{initial_idxs}")

#             for idx in initial_idxs:

#                 #next_reason = cal_llm(seed, corpus[idx], llm)
#                 #print(f"next_reason:{next_reason}, {type(next_reason)}")
#                 neighbor_idx = list(G.neighbors(idx))
#                 next_contexts = tf_idf(seed, neighbor_idx, corpus, 10, visited=[])
#                 #log.append({"idx": idx, "next_reason": next_reason, "neighbor_idx": neighbor_idx, "next_contexts": next_contexts})
#                 #print(f"next_contexts:{next_contexts}")

#                 if next_contexts:
#                     contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
#                 else:
#                     contexts.append(corpus[idx])
#                 #is_continue = cal_llm(seed, contexts, llm)


#             all_contexts[graph_idx] = contexts
#             log.append({"graph_idx": graph_idx, "contexts_length": len(contexts)})
#             #print(f"length of graph_idx:{graph_idx} contexts: {len(contexts)}")
#             contexts = []

#         context_idx = tf_idf_sort(seed, all_contexts)
#         log.append({"selected_context_idx": context_idx})

#         with open("log_{}.txt".format(query[:2]), "w", encoding="utf-8") as f:
#             for entry in log:
#                 f.write(f"{entry}\n")
#         #return []
#         return all_contexts[context_idx]


      

#         # print("KG_retriever==============")
#         # #print(data)
#         # corpus = [c for _, _, c in data['title_chunks']]
#         # candidates_idx = list(range(len(corpus)))
#         # print("candidates_idx", candidates_idx)

#         # seed = query
#         # retrieve_idxs = []

#         # prev_length = 0
#         # count = 0
#         # retrieve_num = [10, 5, 5, 5, 3, 2, 2, 2, 2, 2, 2]
#         # while len(retrieve_idxs) < self.k:
#         #     for graph_idx in graphs_idx:
#         #         G = Gs[graph_idx]
#         #         corpus = [text for _, text in data[graph_idx]['title_chunks']]  #每个子图对应的语料库
#         #         print("G", G)
#         #         idxs = tf_idf(seed, candidates_idx, corpus, k = retrieve_num[count], visited = retrieve_idxs)
#         #         print("根据种子初步得到的idxs", idxs)
#         #         retrieve_idxs.extend(idxs[:max(0, self.k - len(retrieve_idxs))])
                
#         #         candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
#         #         candidates_idx = list(candidates_idx.difference(retrieve_idxs))

#         #         if len(retrieve_idxs) == prev_length:
#         #             break
#         #         else:
#         #             prev_length = len(retrieve_idxs)
                
#         #         count += 1
#         #     idxs = tf_idf(seed, candidates_idx, corpus, k = retrieve_num[count], visited = retrieve_idxs)
#         #     print("根据种子初步得到的idxs", idxs)
#         #     retrieve_idxs.extend(idxs[:max(0, self.k - len(retrieve_idxs))])
            
#         #     candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
#         #     candidates_idx = list(candidates_idx.difference(retrieve_idxs))

#         #     if len(retrieve_idxs) == prev_length:
#         #         break
#         #     else:
#         #         prev_length = len(retrieve_idxs)
            
#         #     count += 1

#         # return [corpus[idx] for idx in retrieve_idxs]
    
# class llm_retriever_LLaMA(object):
#     def __init__(self, k, k_nei, port):
#         self.k = k
#         self.k_nei = k_nei
#         self.port = port
    
#     def retrieve(self, data):
#         corpus = [c for _, c in data['title_chunks']]
#         candidates_idx = list(range(len(corpus)))

#         seed = data['question']
#         contexts = []
        
#         idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

#         for idx in idxs:
#             context = seed + '\n' + corpus[idx]

#             next_reason = cal_local_llm_llama(context, self.port)

#             next_contexts = tf_idf(next_reason, candidates_idx, corpus, k = self.k_nei, visited = [])

#             if next_contexts != []:
#                 contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
#             else:
#                 contexts.append(corpus[idx])

#         return contexts
    

# class llm_retriever_KG_LLaMA(object):
#     def __init__(self, k, k_nei, port):
#         self.k = k
#         self.k_nei = k_nei
#         self.port = port
    
#     def retrieve(self, data, G):
#         corpus = [c for _, c in data['title_chunks']]
#         candidates_idx = list(range(len(corpus)))

#         seed = data['question']
#         contexts = []
#         #候选项为所有的corpus
#         idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

#         for idx in idxs:
#             context = seed + '\n' + corpus[idx]

#             next_reason = cal_local_llm_llama(context, self.port)

#             nei_candidates_idx = list(G.neighbors(idx))
#             #候选项为邻居节点
#             next_contexts = tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei, visited = [])

#             if next_contexts != []:
#                 contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
#             else:
#                 contexts.append(corpus[idx])

#         return contexts


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
from pipeline.utils import cal_local_llm_llama, cal_llm
import json
import jieba
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import spacy

nlp = spacy.load('zh_core_web_sm')
stops = set(stopwords.words('chinese'))
spacy_stopwords = nlp.Defaults.stop_words
stops.update(spacy_stopwords)
chinese_punctuation = set('，。、；：！？（）“”‘’《》【】『』〖〗—…～·')
whitespace_characters = {'\n', '\r', '\t', ' '}
stops.update(chinese_punctuation)
stops.update(whitespace_characters)
# print("stops", stops)
# print(len(stops))
# # 加载NLTK中文停用词
# stopwords = list(nltk_stopwords.words('chinese'))

def preprocess(text):
    words = jieba.lcut(text)
    return ' '.join([word for word in words if word not in stops and word.strip()])


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
        self._bm25 = None
        self._corpus = []
        
    def _initialize_bm25(self, corpus):
        #tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
        tokenized_corpus = [preprocess(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._corpus = tokenized_corpus
        #print("BM25 initialized.")
        # print(self._corpus)

    def bm25_retrieve(self, query_str, data, Gs):
        tokenized_query = preprocess(query_str)        
        graphs_idx = len(Gs)
        log = []
        log.append(f"tokenized_query: {tokenized_query}")
        all_results = []

        for graph_idx in range(graphs_idx):
            corpus = [text for text in data[graph_idx]['title_chunks']]
            log.append(f"Corpus length: {len(corpus)}")
            self._initialize_bm25(corpus)
            scores = self._bm25.get_scores(tokenized_query)
            idxs = scores.argsort()[::-1][:min(50, len(corpus))]  # 取前50个或更少
            all_results.extend([(graph_idx, scores[idx]) for idx in idxs])
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        # 取all_results的前100
        top_results = all_results[:100]

        graph_scores = {}
        graph_counts = {}
        for graph_idx, score in top_results:
            if graph_idx in graph_scores:
                graph_scores[graph_idx] += score
                graph_counts[graph_idx] += 1
            else:
                graph_scores[graph_idx] = score
                graph_counts[graph_idx] = 1

        # 平均分数计算
        average_graph_scores = {graph_idx: graph_scores[graph_idx] / graph_counts[graph_idx] for graph_idx in graph_scores}
        
        sorted_graphs = sorted(average_graph_scores.items(), key=lambda item: item[1], reverse=True)
        
        log.append(f"Graph scores: {graph_scores}")
        log.append(f"Graph counts: {graph_counts}")
        log.append(f"Average graph scores: {average_graph_scores}")
        log.append(f"Sorted graphs: {sorted_graphs}")
        log.append(f"graphs index: {[graph_idx for graph_idx, _ in sorted_graphs]}")
        tmp = query_str[:2]
        # with open(f"./query-test/retrieval_{tmp}.json", "w", encoding="utf-8") as log_file:
        #     json.dump(log, log_file, ensure_ascii=False, indent=4)
        
        return [graph_idx for graph_idx, _ in sorted_graphs]
    
    def bm25_graph_retrieve(self, query_str, candidates_idx, corpus, top_k, graph_idx):
        log = {}
        self._initialize_bm25(corpus)
        tokenized_query = preprocess(query_str)
        scores = self._bm25.get_scores(tokenized_query)
        #log['scores'] = scores.tolist()
        idxs = scores.argsort()[::-1][:top_k]
        # log['idxs'] = idxs.tolist()
        # log['top_scores'] = scores[idxs].tolist()
        # log['top_contents'] = [corpus[idx] for idx in idxs]
        # log['tokenized_query'] = tokenized_query
        # log['corpus'] = self._corpus
        # # Save log to a JSON file
        # log_file_path = f"./logs/query/log_{query_str[:2]}_{graph_idx}.json"
        # with open(log_file_path, 'w', encoding='utf-8') as log_file:
        #     json.dump(log, log_file, ensure_ascii=False, indent=4)
        
        return idxs.tolist()
    
    def retrieve(self, query, data, Gs, llm):
        log = []
        graphs_idx = self.bm25_retrieve(query, data, Gs)  # 先初步获取要遍历的图的索引
        seed = query
        all_contexts = {}
        yes_context = []
        contexts = []
        yes_counts = {}  # 用于记录每个graph_idx的yes次数
        log.append({"query": query, "initial_retrieval": graphs_idx})

        for graph_idx in graphs_idx[:5]:
            retrieve_idxs = []
            yes_count = 0  # 初始化yes计数
            log.append({" strart_separator": "================================"})
            G = Gs[graph_idx]
            corpus = [text for text in data[graph_idx]['title_chunks']]
            candidates_idx = list(range(len(corpus)))
            log.append({"candidates_idx": len(candidates_idx)})
            #initial_idxs = tf_idf(seed, candidates_idx, corpus, 10, visited=[])
            initial_idxs = self.bm25_graph_retrieve(seed, candidates_idx, corpus, min(40, len(corpus)), graph_idx)
            log.append({"graph_idx": graph_idx, "initial_idxs": initial_idxs})
            context = ""
            for idx in initial_idxs:
                if corpus[idx] not in contexts:
                    answer = cal_llm(seed, corpus[idx], llm)
                    #log.append({"answer": answer, "next_reason": next_reason})
                    if answer == "yes":
                        yes_count += 1  # 统计yes次数
                        yes_context.append(corpus[idx])
                        neighbor_idx = list(G.neighbors(idx))
                        context = seed + '\n' + corpus[idx]
                        next_contexts = tf_idf(context, neighbor_idx, corpus, 5, visited=[])

                        if next_contexts:
                            contexts.append(corpus[idx])
                            for next_idx in next_contexts:
                                log.append({"corpus[": next_idx, "]": corpus[next_idx]})
                                if corpus[next_idx] not in contexts:
                                    log.append({"next_idx": next_idx})
                                    contexts.append(corpus[next_idx])
                        else:
                            contexts.append(corpus[idx])

                        log.append({"start_idx": "++++++++++++++++++++++++++++++++++++++++++"})
                        log.append({"idx": idx, "length of contexts": len(contexts), "context": context, "next_contexts": next_contexts, "contexts": contexts})
                        log.append({"end_idx": "++++++++++++++++++++++++++++++++++++++++++"})
            
            all_contexts[graph_idx] = contexts
            yes_counts[graph_idx] = yes_count  # 记录当前graph_idx的yes次数
            log.append({"graph_idx": graph_idx, "contexts_length": len(contexts), "yes_count": yes_count})
            contexts = []
            log.append({" end_separator": "================================"})

        # 找出yes次数最多的graph_idx
        max_yes_graph = max(yes_counts, key=yes_counts.get)
        max_yes_contexts = all_contexts[max_yes_graph]
        # 合并max_yes_contexts和yes_context并去重
        final_contexts = list(set(max_yes_contexts + yes_context))
        # print(f"length of final_contexts:{len(final_contexts)}")
        log.append({"final_contexts length": len(final_contexts)})
        log.append({"final_contexts": final_contexts})
        with open("./logs/query/log_{}.txt".format(query[:3]), "w", encoding="utf-8") as f:
            for entry in log:
                f.write(f"{entry}\n")
        print(f"final_contexts length: {len(final_contexts)}")
        return final_contexts

      

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