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

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers, save_inter_answers, save_inter_fusion_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from langchain.prompts import ChatPromptTemplate
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)

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
template = """你是一个能够基于单一输入查询生成多个搜索查询的助手。\n 生成与以下内容相关的多个搜索查询：{question} \n
输出（5个查询）："""
prompt_perspectives = ChatPromptTemplate.from_template(template)

async def generate_multi_queries(question, llm):
    prompt = prompt_perspectives.format(question=question)
    response = await llm.acomplete(prompt)
    multi_queries = response.text.strip().split('\n')
    return multi_queries


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """ 
    fused_scores = {}
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            doc_str = dumps({"text": doc.node.text, "metadata": doc.node.metadata})
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    reranked_nodes_with_scores = [(doc[0], doc[1]) for doc in reranked_results]
    
    return reranked_nodes_with_scores


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

    retriever = QdrantRetriever(vector_store, embedding, similarity_top_k=3)

    queries = read_jsonl("question.jsonl")
    #queries = queries[:5]

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
        all_nodes_with_scores = []
        for mq in multi_queries:
            nodes_with_scores = await retriever.aretrieve(QueryBundle(query_str=mq))
            all_nodes_with_scores.append(nodes_with_scores)
        
        # Apply reciprocal rank fusion to the retrieved results
        fused_nodes_with_scores = reciprocal_rank_fusion(all_nodes_with_scores)
        
        # Print out the retrieved content for each unique document
        print(f"Original query: {query['query']}")
        # for node, score in fused_nodes_with_scores:
        #     print(f"Retrieved text: {node['text']}, Score: {score}")

        # Format context for QA
        context_str = "\n\n".join(
            [f"{node['metadata']['document_title']}: {node['text']}" for node, score in fused_nodes_with_scores]
        )
        
        # Generate final answer using Ollama with QA_TEMPLATE
        qa_prompt_template = PromptTemplate(QA_TEMPLATE)
        final_query = qa_prompt_template.format(context_str=context_str, query_str=query["query"])
        final_answer = await llm.acomplete(final_query)
        results.append(final_answer)
        inter_docs.append(fused_nodes_with_scores)

    # Process results
    save_answers(queries, results, "rag_fusion_result.jsonl")
    save_inter_fusion_answers(queries, results, inter_docs, multi_queries_list, "inter_data_fusion.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
