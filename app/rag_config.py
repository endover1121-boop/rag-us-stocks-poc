EMBEDDING_MODEL_NAME = "sbintuitions/sarashina-embedding-v2-1b"
LLM_MODEL_NAME = "sbintuitions/sarashina2.2-1b-instruct-v0.1"

QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "us_stocks_rag"

# sarashina-embedding-v2-1b は1792次元ベクトル
EMBEDDING_DIM = 1792
TOP_K = 5