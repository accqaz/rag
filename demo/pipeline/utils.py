from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import requests
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from dotenv import dotenv_values
from llama_index.legacy.llms import OpenAILike as OpenAI
# 初始化 LLM 模型
config = dotenv_values(".env")
llm = OpenAI(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )

embedding = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
    )
def get_embeddings(texts):
    return embedding._embed(texts)

def tf_idf(seed, candidates_idx, corpus, k, visited):
    
    try:
        seed_emb = get_embeddings([seed])[0]
        candidates_texts = [corpus[_] for _ in candidates_idx]
        candidates_embs = get_embeddings(candidates_texts)

        # Compute cosine similarity between the seed embedding and the candidate embeddings
        cosine_sim = cosine_similarity([seed_emb], candidates_embs).flatten()
        # print(seed + " fenshu...")
        # print(cosine_sim)

        # Get the indices of the top-k most similar candidates
        idxs = cosine_sim.argsort()[::-1]

        tmp_idxs = []
        for idx in idxs:
            if candidates_idx[idx] not in visited:
                tmp_idxs.append(candidates_idx[idx])
            
            k -= 1

            if k == 0:
                break
        print("tmp_idxs")
        print(tmp_idxs)
        return tmp_idxs

    except Exception as e:
        return []


def tf_idf_sort(question, corpus, k):
    sub_corpus = [c for c in corpus]
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sub_corpus)
        # print("tfidf_matrix")
        # print(tfidf_matrix)
        query_emb = vectorizer.transform([question])
        print("query_emb")
        print(query_emb)
        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()
        idxs = cosine_sim.argsort()[::-1][:k]
        print(question +" fenshu...")
        print(cosine_sim)
        return [sub_corpus[_] for _ in idxs], [cosine_sim[_] for _ in idxs]
        #return [(sub_corpus[_], cosine_sim[_]) for _ in idxs]
    
    except Exception as e:
        return []

def tf_idf2(question, corpus, corpus_idx, k):
    sub_corpus = [corpus[_] for _ in corpus_idx]
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sub_corpus)

        query_emb = vectorizer.transform([question])
        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()
        idxs = cosine_sim.argsort()[::-1][:k]

        return [sub_corpus[_] for _ in idxs]
    
    except Exception as e:
        return []


def get_encoder(encoder_type):
    return SentenceTransformer(encoder_type)


def strip_string(string, only_stopwords = False):
    if only_stopwords:
        return ' '.join([str(t) for t in nlp(string) if not t.is_stop])
    else:
        return ' '.join([str(t) for t in nlp(string) if t.pos_ in ['NOUN', 'PROPN']])
    

def window_encodings(sentence, window_size, overlap):
    """Compute encodings for a string by splitting it into windows of size window_size with overlap"""
    tokens = sentence.split()

    if len(tokens) <= window_size:
        return [sentence]
    
    return [' '.join(tokens[i:i + window_size]) for i in range(0, len(tokens) - window_size, overlap)]


def cal_local_llm_llama(input, port):
    # Define the url of the API
    url = "http://localhost:{}/api/ask".format(port)

    # Define the headers for the request
    headers = {
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the POST request
    # Replace with actual instruction and input data
    data = {
        'instruction': 'What evidence do we need to answer the question given the current evidence?',
        'input': input
        }

    # print(data)
    # Convert the data to JSON format
    data_json = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data_json)

    # Get the json response
    response_json = response.json()

    return response_json['answer']


def cal_llm(seed, context, llm):
    QA_TEMPLATE = """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    请你基于上下文信息而不是自己的知识，回答以下问题，可以分点作答，如果上下文信息没有相关知识，可以回答不确定，不要复述上下文信息：
    {query_str}

    回答：\
    """
    final_query = QA_TEMPLATE.format(context_str=context, query_str=seed)
    # 获取模型的回答
    final_answer = llm.complete(final_query)
    print("call_llm")
    print(final_answer)
    return final_answer