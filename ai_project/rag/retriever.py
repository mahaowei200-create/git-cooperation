import chromadb
from chromadb.utils import embedding_functions
import jieba
from rank_bm25 import BM25Okapi
import torch
torch.set_num_threads(1)
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, embed_model, rerank_model):
        print("⚙️ 正在初始化双路召回 + 精排引擎...")
        
        print("📡 雷达 1: 准备启动 Chroma...")
        self.db_client = chromadb.EphemeralClient()
        
        print("📡 雷达 2: 准备加载右眼 (向量模型)...")
        self.bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        
        print("📡 雷达 3: 准备创建知识库集合...")
        self.collection = self.db_client.create_collection(name="hybrid_kb", embedding_function=self.bge_ef)
        
        self.bm25 = None 
        self.chunks = [] 
        
        print("📡 雷达 4: 准备加载主考官 (Reranker 模型，约 1GB，加载较慢且最容易闪退)...")
        self.reranker = CrossEncoder(rerank_model, max_length=512, device='cpu')
        
        print("✅ 后厨三大件全部装配成功！")

    def ingest_docs(self,chunks):
        self.chunks = chunks

        chunk_ids = [f"chun_{i}" for i in range(len(chunks))]
        self.collection.add(documents= chunks,ids = chunk_ids)

        tokenized_chunks = [jieba.lcut(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        print("知识库构建完毕")

    def search(self, query, top_k=5, final_k=3):
        self.query = query

        vector_docs = self.collection.query(query_texts=query,n_results=5)["documents"][0]
        bm25_docs = self.bm25.get_top_n(jieba.lcut(query),self.chunks,n = top_k)

        candidate_docs = list(set(vector_docs+bm25_docs))

        cross_inputs = [[query,doc]for doc in candidate_docs]
        scores = self.reranker.predict(cross_inputs)

        scored_docs  =sorted(zip(candidate_docs,scores),key=lambda x:x[1],reverse = True)
        return scored_docs[:final_k]
        



