import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 重新启用国内极速镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

PDF_PATH = "data/test1.pdf"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

CHUNK_MAX_LENGTH = 150
CHUNK_OVERLAP = 1