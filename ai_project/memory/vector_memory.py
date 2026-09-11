from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# 加载.env环境变量
load_dotenv()

class VectorMemory:
    # DeepSeek 向量固定输出1024维，dim统一设为1024
    def __init__(self, dim=1024):
        self.embeddings = OpenAIEmbeddings(
            model="deepseek-embedding",
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(dim)
        # 保存原始对话文本，和向量一一对应
        self.texts = []

    def add(self, text: str):
        """新增记忆向量，增加异常捕获，接口失败不会直接崩整个服务"""
        try:
            # 文本转向量
            vec = self.embeddings.embed_query(text)
            # 转为FAISS要求的二维float32数组
            vec = np.array([vec]).astype("float32")
            # 存入向量库
            self.index.add(vec)
            # 同步保存原文
            self.texts.append(text)
        except Exception as e:
            # 打印错误，跳过本条记忆，不阻断主业务流程
            print(f"⚠️ 向量记忆写入失败，跳过本条内容：{text}，错误详情：{str(e)}")

    def search(self, query: str, top_k=3):
        """根据问题语义检索相似历史记忆，兜底空库判断"""
        # 无记忆直接返回空列表
        if len(self.texts) == 0:
            return []
        try:
            # 查询文本转向量
            qvec = self.embeddings.embed_query(query)
            qvec = np.array([qvec]).astype("float32")
            # FAISS检索，返回距离D、下标I
            D, I = self.index.search(qvec, top_k)
            # 根据下标还原原始文本，过滤越界下标
            return [self.texts[i] for i in I[0] if i < len(self.texts)]
        except Exception as e:
            print(f"⚠️ 向量记忆检索失败，返回空记忆，错误详情：{str(e)}")
            return []