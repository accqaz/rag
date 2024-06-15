from typing import Iterable
import jsonlines


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def save_answers(
    queries: Iterable, results: Iterable, path: str = "data/answers.jsonl"
):
    answers = []
    for query, result in zip(queries, results):
        answers.append(
            {"id": query["id"], "query": query["query"], "answer": result.text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/answers.jsonl
    write_jsonl(path, answers)

    
def save_inter_answers(
    queries: Iterable, results: Iterable, docs: Iterable, multi_queries_list: Iterable, path: str = "data/inter_answers.jsonl"
):
    answers = []
    for query, doc_list, result, multi_queries in zip(queries, docs, results, multi_queries_list):
        doc_texts = [doc.node.text for doc in doc_list]
        answers.append(
            {"id": query["id"], "query": query["query"], "multi_queries": multi_queries, "docs": doc_texts, "answer": result.text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/inter_answers.jsonl
    write_jsonl(path, answers)


def save_de_answers(
    queries: Iterable, results: Iterable, docs: Iterable, multi_queries_list: Iterable, path: str = "data/inter_answers.jsonl"
):
    answers = []
    for query, doc_list, result, multi_queries in zip(queries, docs, results, multi_queries_list):
        # Extract text from CompletionResponse if necessary
        if hasattr(result, 'text'):
            result_text = result.text
        else:
            result_text = str(result)
            
        answers.append(
            {
                "id": query["id"],
                "query": query["query"],
                "multi_queries": multi_queries,
                "docs": doc_list,
                "answer": result_text
            }
        )

    # Save answers to the specified path using jsonlines
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(answers)



def save_inter_fusion_answers(
    queries: Iterable, results: Iterable, docs_with_scores: Iterable, multi_queries_list: Iterable, path: str = "data/inter_answers.jsonl"
):
    answers = []
    for query, doc_list_with_scores, result, multi_queries in zip(queries, docs_with_scores, results, multi_queries_list):
        doc_texts_with_scores = [{"text": doc["text"], "score": score} for doc, score in doc_list_with_scores]
        answers.append(
            {"id": query["id"], "query": query["query"], "multi_queries": multi_queries, "docs_with_scores": doc_texts_with_scores, "answer": result.text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/inter_answers.jsonl
    write_jsonl(path, answers)



def save_inter_re_answers(
    queries: Iterable, results: Iterable, docs: Iterable, multi_queries_list: Iterable, path: str = "data/inter_answers.jsonl"
):
    answers = []
    for query, doc, result, multi_queries in zip(queries, docs, results, multi_queries_list):
        answers.append(
            {"id": query["id"], "query": query["query"], "multi_queries": multi_queries, "docs": doc, "answer": result.text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/inter_answers.jsonl
    write_jsonl(path, answers)