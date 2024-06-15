from typing import List, Optional
import qdrant_client

from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse

from custom.template import QA_TEMPLATE
import jieba
from rank_bm25 import BM25Okapi


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
        corpus: Optional[List[str]] = None,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        self._corpus = corpus
        self._bm25 = None
        if corpus:
            self._initialize_bm25(corpus)
        super().__init__()
        
    def _initialize_bm25(self, corpus: List[str]):
        tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def bm25_retrieve(self, query_str: str, top_k: int = 5) -> List[str]:
        if not self._bm25:
            return []
        tokenized_query = jieba.lcut(query_str)
        return self._bm25.get_top_n(tokenized_query, self._corpus, n=top_k)
        
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    async def multi_retrieve(self, query_bundle: QueryBundle, methods: Optional[List[str]] = None) -> List[str]:
        if methods is None:
            methods = ['bm25', 'emb']
        
        search_res = list()
        for method in methods:
            if method == 'bm25':
                bm25_res = self.bm25_retrieve(query_bundle.query_str, top_k=self._similarity_top_k)
                for item in bm25_res:
                    if item not in search_res:
                        search_res.append(item)
                #print(f"bm25: search_res: {search_res}")
            elif method == 'emb':
                query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
                vector_store_query = VectorStoreQuery(
                    query_embedding, similarity_top_k=self._similarity_top_k
                )
                query_result = await self._vector_store.aquery(vector_store_query)
                for node in query_result.nodes:
                    if node.text not in search_res:
                        search_res.append(node.text)
                #print(f"emb: search_res: {search_res}")
            
        return search_res



async def generation_with_knowledge_retrieval(
    query_str: str,
    retriever: BaseRetriever,
    llm: LLM,
    qa_template: str = QA_TEMPLATE,
    reranker: BaseNodePostprocessor | None = None,
    debug: bool = False,
    progress=None,
) -> CompletionResponse:
    query_bundle = QueryBundle(query_str=query_str)
    node_with_scores = await retriever.aretrieve(query_bundle)
    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret

