import asyncio
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import models
from tqdm.asyncio import tqdm
from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers, save_de_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from langchain.prompts import ChatPromptTemplate
from llama_index.core import QueryBundle, PromptTemplate, StorageContext, VectorStoreIndex

synthesis_prompt_template = """\
    以下是一些问题和答案：
    ----------
    {context}
    ----------
    使用这些信息来综合回答这个问题：
    {question}

    回答：\
    """

template = """你是一个AI语言模型助手。你的任务是生成三个与给定用户问题相关的子问题或子查询。你的目标是将输入问题分解为一组可以独立回答的子问题。生成多个子查询以检索相关文档。原始问题：{question} \n
输出（3个查询）："""
prompt_perspectives = ChatPromptTemplate.from_template(template)

async def generate_decomposition_queries(question, llm):
    prompt = prompt_perspectives.format(question=question)
    response = await llm.acomplete(prompt)
    multi_queries = response.text.strip().split('\n')
    return multi_queries

async def retrieve_and_rag(question, retriever, llm, sub_question_generator_chain):
    sub_questions = await sub_question_generator_chain(question, llm)
    
    sub_answers = []
    for sub_question in sub_questions:
        answer = await generation_with_knowledge_retrieval(sub_question, retriever, llm)
        sub_answers.append((sub_question, answer.text.strip()))
    
    return sub_answers

def format_qa_pairs(questions, answers):
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"问题 {i}: {question}\n回答 {i}: {answer}\n\n"
    return formatted_string.strip()

async def print_query_embeddings(query, embedding_model):
    # 获取查询的token及其对应的embedding
    tokens = embedding_model._model.tokenizer.tokenize(query)  # 使用模型的tokenizer获取token
    print(f"Token: {tokens}")


async def main():
    config = dotenv_values(".env")

    # Initialize LLM embedding model and Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
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
    #queries = queries[:1]

    # Generate answers
    print("Start generating answers...")

    results = []
    inter_docs = []
    multi_queries_list = []
    for query in tqdm(queries, total=len(queries)):
        print(f"Original query: {query['query']}")
        await print_query_embeddings(query['query'], embedding)  # 打印每个查询的token嵌入
        rag_results = await retrieve_and_rag(query["query"], retriever, llm, generate_decomposition_queries)
        
        
        sub_questions, sub_answers = zip(*rag_results)
        multi_queries_list.append(sub_questions)
        context = format_qa_pairs(sub_questions, sub_answers)
        
        for mq in sub_questions:
            embedding_vector = await embedding._aget_query_embedding(mq)
            print(f"Query: {mq}")
            print(f"Embedding: {embedding_vector[0]}")


        final_synthesis_query = synthesis_prompt_template.format(context=context, question=query["query"])
        final_answer = await llm.acomplete(final_synthesis_query)

        results.append(final_answer)
        inter_docs.append(sub_answers)


    save_answers(queries, results, "decomposition_query_result.jsonl")
    save_de_answers(queries, results, inter_docs, multi_queries_list, "de_inter_data.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
