from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import multiprocessing as mp
import concurrent.futures
import requests
from sentence_transformers import SentenceTransformer
import spacy
import torch
from transformers import (AutoConfig, AutoTokenizer)
from nltk.corpus import stopwords
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from nltk.corpus import stopwords as nltk_stopwords
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from networkx.readwrite import json_graph
# # 加载NLTK中文停用词
# stopwords = list(nltk_stopwords.words('chinese'))

# encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# def preprocess(text):
#     words = jieba.cut(text)
#     return ' '.join([word for word in words if word not in stopwords and word.strip()])

embedding = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
    )


def tfidf_kw_extract_chunk(d, n_kw, ngram_l, ngram_h):
    kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
    chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

    chunks = []
    titles = set()
    for title, chunk in d['title_chunks']:
        chunks.append(preprocess(chunk))
        titles.add(title)


    tfidf_vectorizer = TfidfVectorizer(stop_words = stopwords, ngram_range = (ngram_l, ngram_h))
    X = tfidf_vectorizer.fit_transform(chunks)
    term = tfidf_vectorizer.get_feature_names_out()
    score = X.todense()
    #提取每个文本块中排名前n_kw的关键词，并将这些关键词与标题合并，去重后转换为关键词列表
    kws = list(set(list(term[(-score).argsort()[:, :n_kw]][0]) + list(titles)))
    print("******************************************************")
    print(kws)
    print("******************************************************")
    vec = CountVectorizer(vocabulary = kws, binary=True, ngram_range = (ngram_l, ngram_h))
    bow = vec.fit_transform(chunks).toarray()

    num_chunks = bow.shape[0]
    for i in range(num_chunks):
        for j in range(i + 1, num_chunks):
            common_kw = np.logical_and(bow[i], bow[j])
            if np.any(common_kw):
                common_kw_indices = np.where(common_kw)[0]
                for kw_index in common_kw_indices:
                    kw = kws[kw_index]
                    kw2chunk[kw].add(d['title_chunks'][i][1])
                    kw2chunk[kw].add(d['title_chunks'][j][1])

                    chunk2kw[d['title_chunks'][i][1]].add(kw)
                    chunk2kw[d['title_chunks'][j][1]].add(kw)
    # bow_tile = np.tile(bow, (bow.shape[0], 1))
    # bow_repeat = np.repeat(bow, bow.shape[0], axis = 0)
    # common_kw = (bow_tile * bow_repeat).reshape(bow.shape[0], bow.shape[0], -1)
    # node1, node2, kw_id = common_kw.nonzero()

    # for n1, n2, kw in zip(node1, node2, kw_id):
    #     if n1 != n2:
    #         kw2chunk[kws[kw]].add(d['title_chunks'][n1][1])
    #         kw2chunk[kws[kw]].add(d['title_chunks'][n2][1])

    #         chunk2kw[d['title_chunks'][n1][1]].add(kws[kw])
    #         chunk2kw[d['title_chunks'][n2][1]].add(kws[kw])

    for key in kw2chunk:
        kw2chunk[key] = list(kw2chunk[key])

    for key in chunk2kw:
        chunk2kw[key] = list(chunk2kw[key])

    d['kw2chunk'] = kw2chunk
    d['chunk2kw'] = chunk2kw

    print("tfidf_kw_extract_chunk")
    #print(d)
    return d


def tfidf_kw_extract(data, n_kw, ngram_l, ngram_h, num_processes):
    func = partial(tfidf_kw_extract_chunk, n_kw = n_kw, ngram_l = ngram_l, ngram_h = ngram_h)

    with Pool(num_processes) as p:
        data = list(tqdm(p.imap(func, data), total=len(data)))

    return data


def kw_graph_construct(i_d):
    idx, d = i_d

    G = nx.MultiGraph()

    chunk2id = {}
    for i, chunk in enumerate(d['title_chunks']):
        _, chunk = chunk

        G.add_node(i, chunk = chunk)
        chunk2id[chunk] = i
    
    for kw, chunks in d['kw2chunk'].items():
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                G.add_edge(chunk2id[chunks[i]], chunk2id[chunks[j]], kw = kw)
    
    return idx, G


def kw_process_graph(docs):
    pool = mp.Pool(mp.cpu_count())
    graphs = [None] * len(docs)

    for idx, G in tqdm(pool.imap_unordered(kw_graph_construct, enumerate(docs)), total=len(docs)):
        graphs[idx] = G

    pool.close()
    pool.join()
    print("kw_process_graph finished")
    print(type(graphs))
    return graphs

def knn_embs(i_d):
    idx, d = i_d
    # print("idx:-----------------------------------------")
    # print(idx)

    chunks = []
    #preprocessed_chunks = []
    
    for type, chunk in d['title_chunks']:
        # 预处理 chunk
        #preprocessed_chunk = preprocess(chunk)
        chunks.append(chunk)

    # # 将所有 chunk 内容保存到一个 JSON 文件中
    # with open("./{}/chunks_{}.json".format("chunks", idx), "w", encoding='utf-8') as f:
    #     json.dump(chunks, f, ensure_ascii=False, indent=4)

    # 检查可用的CUDA设备
    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices > 0:
        device = 'cuda:{}'.format(idx % num_cuda_devices)
    else:
        device = 'cpu'

    emb = embedding._embed(chunks)

    return idx, emb

def knn_embs_construct(dataset, docs, num_processes):
    pool = mp.Pool(num_processes)
    embs = [None] * len(docs)
    print("++++++++++++++++++++++++")
    print(len(docs))
    for idx, emb in tqdm(pool.imap_unordered(knn_embs, enumerate(docs)), total=len(docs)):
        embs[idx] = emb

    pool.close()
    pool.join()
    # 创建目标目录（如果不存在）
    output_dir = './{}'.format("dataset")
    os.makedirs(output_dir, exist_ok=True)

    pkl.dump(embs, open('./{}/{}_embs.pkl'.format("dataset", dataset), 'wb'))

    # # 将 embs 转换为可序列化的格式
    # def serialize_emb(emb):
    #     if isinstance(emb, torch.Tensor):
    #         return emb.tolist()
    #     elif isinstance(emb, np.ndarray):
    #         return emb.tolist()
    #     else:
    #         return emb

    # # 保存到 json 文件
    # json_embs = [{'idx': idx, 'emb': serialize_emb(emb)} for idx, emb in enumerate(embs)]
    # with open(f'./dataset/{dataset}_embs.json', 'w', encoding='utf-8') as f:
    #     json.dump(json_embs, f, ensure_ascii=False, indent=4)


def knn_graph(i_d, k_knn, embs, strategy='cos'):
    idx, d = i_d
    emb = embs[idx]

    # Build a knn Graph
    if strategy == 'cos':
        sim = cosine_similarity(emb)
    elif strategy == 'dp':
        sim = np.matmul(emb, emb.transpose(1, 0))

    # Top k
    top_idx = np.argsort(-sim, axis=1)[:, 1:k_knn + 1]

    tail_nodes = np.arange(top_idx.shape[0]).repeat(k_knn)
    head_nodes = top_idx.reshape(-1)
    sim_scores = sim[np.arange(sim.shape[0])[:, None], top_idx].flatten()
    edges = [(node1, node2, score) for node1, node2, score in zip(tail_nodes, head_nodes, sim_scores)]

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges, weight='similarity')

    return idx, G

def save_graph_to_json(G, file_path):
    data = json_graph.node_link_data(G)
    # Convert numpy data types to native Python data types
    for node in data['nodes']:
        for k, v in node.items():
            if isinstance(v, (np.integer, np.floating)):
                node[k] = v.item()
    for link in data['links']:
        for k, v in link.items():
            if isinstance(v, (np.integer, np.floating)):
                link[k] = v.item()
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)





# def knn_graph(i_d, k_knn, embs, strategy = 'cos'):
#     idx, d = i_d

#     emb = embs[idx]

#     #build a knn Graph
#     if strategy == 'cos':
#         sim = cosine_similarity(emb, emb)

#     elif strategy == 'dp':
#         sim = np.matmul(emb, emb.transpose(1, 0))

#     #topk
#     top_idx = np.argsort(-sim, axis = 1)[:, 1:k_knn+1]

#     tail_nodes = np.arange(top_idx.shape[0]).repeat(k_knn)
#     head_nodes = top_idx.reshape(-1)
#     edges = [(node1, node2) for node1, node2 in zip(tail_nodes, head_nodes)]

#     G = nx.DiGraph()
#     G.add_edges_from(edges)

#     return idx, G

def knn_graph_construct(dataset, docs, k_knn, num_processes, strategy = 'cos'):
    pool = mp.Pool(num_processes)
    graphs = [None] * len(docs)
    #这个里面的knn_embs.pkl是所有（idx，chunk）对
    func = partial(knn_graph, k_knn = k_knn, embs = pkl.load(open('./{}/{}_embs.pkl'.format("dataset", dataset), 'rb')), strategy = strategy)

    for idx, G in tqdm(pool.imap_unordered(func, enumerate(docs)), total=len(docs)):
        graphs[idx] = G

    pool.close()
    pool.join()

    return graphs

def visualize_graph(G, filename):
    """可视化图并保存为图片文件"""
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(filename)
    plt.close()

def visualize_graphs(graphs, output_dir):
    """可视化图列表并保存为图片文件"""
    os.makedirs(output_dir, exist_ok=True)
    for idx, G in enumerate(graphs):
        filename = os.path.join(output_dir, f'graph_{idx}.png')
        visualize_graph(G, filename)