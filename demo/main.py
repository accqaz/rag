import asyncio
import torch.multiprocessing as mp
from kg.graph_construct import knn_graph_construct, knn_embs_construct, visualize_graphs, save_graph_to_json
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import models
from tqdm.asyncio import tqdm
from pipeline.ingestion import read_data, build_chunk
# from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
import pickle as pkl
from pipeline.retriever import *
import json
from json import dumps, loads
import os
from nltk.corpus import stopwords
import spacy
from networkx.readwrite import json_graph

QA_TEMPLATE = """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    请你基于上下文信息而不是自己的知识，首先判断能否根据上下文信息回答以下问题，如果不能或者不确定请直接回答‘no’，否则回答‘yes’。如果你的回答为‘yes’，请根据上下文信息回答以下问题，可以分点作答，不要复述上下文信息：
    {query_str}
    回答：\
    """

async def reciprocal_rank_fusion(results: list[list[dict]], k=60) -> list[dict]:
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents 
       and an optional parameter k used in the RRF formula"""
    
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    sorted_docs_with_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_results = [
        {"doc": loads(doc), "score": score}
        for doc, score in sorted_docs_with_scores
    ][:10]
    return reranked_results


async def main():
    
    config = dotenv_values(".env")

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = Ollama(
        model="qwen", base_url="http://localhost:11434", temperature=0, request_timeout=150
    )
    embedding = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
    )
    Settings.embed_model = embedding

    retriever = KG_retriever(k = 20)
    queries = read_jsonl("question1.jsonl")
    queries = queries[:1]

    data = read_data("data")
    # pipeline = await build_chunk(embedding)
    pipeline = await build_chunk()
    processed_data = await pipeline.arun(documents=data, show_progress=True, num_workers=1)
    
    # # 将 processed_data 保存到文件中
    # with open("processed_data.json", "w", encoding='utf-8') as f:
    #     json.dump([doc.to_dict() for doc in processed_data], f, ensure_ascii=False, indent=4)
    
    # 创建一个字典来根据 graph_type 和 document_type 合并文档
    graph_type_dict = {}
    for doc in processed_data:
        graph_type = doc.metadata['graph_type']
        doc_type = doc.metadata['document_type']
        if graph_type not in graph_type_dict:
            graph_type_dict[graph_type] = {}
        if doc_type not in graph_type_dict[graph_type]:
            graph_type_dict[graph_type][doc_type] = []
        graph_type_dict[graph_type][doc_type].append((doc.metadata['document_title'], doc.text))
    
    # 将字典转换为所需格式的列表
    formatted_docs_dict = {}
    for graph_type, doc_dict in graph_type_dict.items():
        formatted_docs_dict[graph_type] = [{'idx': idx, 'document_type': doc_type, 'title_chunks': title_chunks} for idx, (doc_type, title_chunks) in enumerate(doc_dict.items())]

    # # 将 formatted_docs_dict 保存到文件中
    # with open("formatted_docs.json", "w", encoding='utf-8') as f:
    #     json.dump(formatted_docs_dict, f, ensure_ascii=False, indent=4)

    dataset = "data"
    # subfolders = [f.name for f in os.scandir(dataset) if f.is_dir()]
    subfolders = ['director']
    Gs = {}
    for folder in subfolders:
        if folder in formatted_docs_dict:
            #knn_embs_construct(folder, formatted_docs_dict[folder], 2)
            Gs[folder] = knn_graph_construct(folder, formatted_docs_dict[folder], 20, 2)
            #graph_folder = 'graph'
            # os.makedirs(os.path.join(graph_folder, folder), exist_ok=True)  # 使用 os.path.join 来确保路径正确
            # for idx, G in zip(range(len(Gs[folder])), Gs[folder]):
            #     file_path = os.path.join(graph_folder, folder, f"graph_{folder}_{idx}.json")  # 同样使用 os.path.join
            #     save_graph_to_json(G, file_path)
            # 可视化图并保存
            # output_dir = f'./{folder}_graphs'
            # visualize_graphs(Gs[folder], output_dir)

    print("KGs加载成功。")
    print({k: type(v) for k, v in Gs.items()})

    results = []

    # 使用检索器进行检索并生成答案
    for query in tqdm(queries, desc="Processing queries"):
        doc_type = query["document"]
        if doc_type in Gs:
            print(f"Processing query: {query['query']}, doc_type: {doc_type}")
            print(f"First data in Gs[doc_type]: {Gs[doc_type][0] if Gs[doc_type] else 'Empty Gs'}")
            retrieved_docs = retriever.retrieve(query["query"], formatted_docs_dict[doc_type], Gs[doc_type], llm)
            print(f"type of retrieved_docs:{type(retrieved_docs)}")
            qa_prompt_template = PromptTemplate(input_variables=["context_str", "query_str"], template=QA_TEMPLATE)
            for doc in retrieved_docs:
                print(f"retrieved_docs[{doc}]:{retrieved_docs[doc]}")
                final_query = qa_prompt_template.format(context_str='\n'.join(retrieved_docs[doc]), query_str=query["query"])
                
                final_answer = await llm.acomplete(final_query)
                print(f"final_answer:{final_answer}")
            results.append(final_answer)
        
    #save_answers(queries, results, "kg_retrieval_result1.jsonl")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    asyncio.run(main())
