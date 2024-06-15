import asyncio
from typing import Iterable
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import models
from tqdm.asyncio import tqdm
import jsonlines
from json import dumps, loads
from typing import Iterable, List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers, save_inter_answers, save_inter_re_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from langchain.prompts import ChatPromptTemplate
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)

async def reciprocal_rank_fusion(results: List[List[Dict]], k=60) -> List[Dict]:
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
    # for doc_with_score in reranked_results:
    #     print(f"Doc: {doc_with_score['doc']}, Score: {doc_with_score['score']}")
    # reranked_results = [
    #     loads(doc)
    #     for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    # ][:10]
    # print(f"reranked_results: {reranked_results}")
    



QA_TEMPLATE = """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    请你基于上下文信息而不是自己的知识，回答以下问题，可以分点作答，如果上下文信息没有相关知识，可以回答不确定，不要复述上下文信息：
    {query_str}

    回答：\
    """

# Multi Query: Different Perspectives
template = """你是一个可靠的AI语言模型助手。你的任务是生成三个不同版本的给定用户问题，以从向量数据库中检索相关文档。通过对用户问题生成多种视角，你的目标是帮助用户克服基于距离的相似性搜索的一些限制。原始问题：{question} \n
输出（3个查询）："""
prompt_perspectives = ChatPromptTemplate.from_template(template)

async def generate_multi_queries(question, llm):
    prompt = prompt_perspectives.format(question=question)
    response = await llm.acomplete(prompt)
    multi_queries = response.text.strip().split('\n')
    return multi_queries



async def main():
    config = dotenv_values(".env")

    # Initialize LLM embedding model and Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=150
    )
    embedding = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
    )
    Settings.embed_model = embedding

    # Initialize data ingestion pipeline and vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        data = read_data("data")
        corpus = [doc.text for doc in data]
        print(len(data))
        pipeline = build_pipeline(llm, embedding, vector_store=vector_store)
        # Temporarily stop real-time indexing
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # Resume real-time indexing
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print("embedding done!")

    retriever = QdrantRetriever(vector_store, embedding, similarity_top_k=3, corpus=corpus)

    queries = read_jsonl("question.jsonl")
    queries = queries[:1]
  
    # Generate answers
    print("Start generating answers...")

    results = []
    inter_docs = []
    multi_queries_list = []
    for query in tqdm(queries, total=len(queries)):
        # Generate multiple queries
        multi_queries = await generate_multi_queries(query["query"], llm)
        multi_queries_list.append(multi_queries)
        
        # Retrieve information for each query
        all_search_res = []
        for mq in multi_queries:
            multi_search_res = await retriever.multi_retrieve(QueryBundle(query_str=mq))
            all_search_res.extend(multi_search_res)
       
        reranked_docs = await reciprocal_rank_fusion([all_search_res])
        
        print(f"Original query: {query['query']}")

        # Generate final answer using Ollama with QA_TEMPLATE
        qa_prompt_template = PromptTemplate(QA_TEMPLATE)
        final_query = qa_prompt_template.format(context_str=reranked_docs, query_str=query["query"])
        final_answer = await llm.acomplete(final_query)
        results.append(final_answer)
        inter_docs.append(reranked_docs)

    # Process results
    save_answers(queries, results, "multi_retrieval_result.jsonl")
    save_inter_re_answers(queries, results, inter_docs, multi_queries_list, "inter_data_multi_retrieval.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
