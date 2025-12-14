"""
FastAPI で動く超シンプルな RAG API のサンプル。
- Embedding: sarashina-embedding-v2-1b（SentenceTransformer）
- LLM      : sarashina2.2-3b-instruct-v0.1（transformers）
- VectorDB : Qdrant（docker-compose で立ち上がっている前提）
"""

from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_DIM,
    TOP_K,
)

# ============================================================
# FastAPI アプリ生成
# ============================================================

app = FastAPI(title="Sarashina RAG (US Stocks PoC)")

# ============================================================
# モデル & Qdrant クライアントの初期化
# ============================================================


print("Loading embedding model:", EMBEDDING_MODEL_NAME)
# SentenceTransformer を使って sarashina 埋め込みモデルをロード
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Loading LLM model:", LLM_MODEL_NAME)
# sarashina LLM 用の tokenizer と model をロード
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

# GPU が使えれば GPU、なければ CPU を使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
llm_model.to(device)
llm_model.eval()

print(f"Connecting Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
# Qdrant に接続（ホスト名とポートは rag_config から）
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# コレクションがなければ作り直す
existing_collections = [c.name for c in qdrant.get_collections().collections]
if QDRANT_COLLECTION not in existing_collections:
    print(f"Creating Qdrant collection: {QDRANT_COLLECTION}")
    qdrant.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            # ベクトル次元数も rag_config から
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

# ============================================================
# Pydantic モデル（API のリクエスト / レスポンス定義）
# ============================================================

class IndexItem(BaseModel):
    """
    1つのチャンク（分割テキスト）を表すデータ構造。
    後で「米株の決算チャンク」をここに詰めて /index に投げるイメージ。
    """
    id: int               # 一意なID（"AAPL_10K_2024_001" など）
    text: str             # 実際のテキスト
    ticker: str           # 銘柄コード（例: "AAPL"）
    source: Optional[str] = None  # "10-K" / "10-Q" / "press_release" など
    date: Optional[str] = None    # "YYYY-MM-DD" 形式の文字列

class IndexRequest(BaseModel):
    """
    複数の IndexItem をまとめてインデックスするためのリクエスト。
    """
    items: List[IndexItem]

class QueryRequest(BaseModel):
    """
    ユーザからの RAG クエリ。
    question: 実際の質問（日本語でも英語でもOK）
    ticker  : 特定の銘柄に絞りたい場合に指定（任意）
    """
    question: str
    ticker: Optional[str] = None

class QueryResponse(BaseModel):
    """
    RAG の応答。
    answer : LLM が生成した回答
    used_ids : 参照に使ったチャンクのIDリスト（どの情報を見たかのトレース用）
    """
    answer: str
    used_ids: List[str]

# ============================================================
# ユーティリティ関数
# ============================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    複数のテキストを埋め込みベクトルに変換する。
    - embedding_model は SentenceTransformer なので
      .encode() でリストをまとめて処理できる。
    """
    vecs = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.tolist()

def generate_answer(context: str, question: str) -> str:
    """
    取得したコンテキスト（関連チャンクの結合）とユーザ質問から、
    sarashina LLM に回答を生成させる。
    """
    # プロンプト（systemメッセージっぽい部分も自前で書く）
    prompt = (
        "あなたは日本語で投資リサーチを行うプロの投資家・トレーダーです。\n"
        "以下のコンテキストに基づいて、ユーザーの質問に答えてください。\n"
        "コンテキストに書かれていないことは推測で断定せず、分からないと答えてください。\n"
        "将来の株価を断定的に予測することは禁止です。\n\n"
        "【コンテキスト】\n"
        f"{context}\n\n"
        "【質問】\n"
        f"{question}\n\n"
        "【回答】"
    )

    # tokenizer でトークン化し、モデルに渡す
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )

    # 生成結果を文字列に戻す
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # "【回答】" より前はプロンプトなので、それ以降だけを返す
    if "【回答】" in text:
        return text.split("【回答】", 1)[-1].strip()
    else:
        # 念のため fallback
        return text.strip()

# ============================================================
# エンドポイント定義
# ============================================================

@app.get("/health")
def health():
    """
    ヘルスチェック用。
    ブラウザで http://localhost:8000/health を開いて {"status": "ok"} が出ればOK。
    """
    return {"status": "ok"}

@app.post("/index")
def index_documents(req: IndexRequest):
    """
    ドキュメント（チャンク）を Qdrant にインデックスするエンドポイント。
    - 本番運用だと、ここは「一度だけ呼ぶバッチ」の役割になる。
    - PoC のうちは、curl や HTTP クライアントから直接叩いてもOK。
    """
    texts = [item.text for item in req.items]

    # テキストをまとめてベクトル化
    vectors = embed_texts(texts)


    # Qdrant に upsert するための PointStruct を組み立てる
    points = []
    for item, vec in zip(req.items, vectors):
        points.append(
            PointStruct(
                id=item.id,
                vector=vec,
                payload={
                    "text": item.text,
                    "ticker": item.ticker,
                    "source": item.source,
                    "date": item.date,
                },
            )
        )

    # Qdrant に upsert（既に同じ id があれば上書き）
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    return {"status": "ok", "count": len(points)}

@app.post("/rag/query", response_model=QueryResponse)
def rag_query(req: QueryRequest):
    """
    実際のRAGクエリ用エンドポイント。
    - 質問を埋め込みベクトル化
    - Qdrant から類似チャンクを TOP_K 件取得
    - それらをコンテキストとして LLM に回答を生成させる
    """
    # 質問文を1件だけ埋め込み（戻り値は1要素のリスト）
    query_vec = embed_texts([req.question])[0]

    # ticker が指定されていればフィルタをかける
    search_filter = None
    if req.ticker:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="ticker",
                    match=MatchValue(value=req.ticker),
                )
            ]
        )

    # Qdrant に類似検索
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=TOP_K,           # ← ここでも rag_config の TOP_K を使用
        query_filter=search_filter,
    )

    if not hits:
        return QueryResponse(
            answer="関連する情報が見つかりませんでした。",
            used_ids=[],
        )

    # 取得したチャンクをコンテキスト文字列にまとめる
    context_chunks = []
    used_ids: List[str] = []

    for hit in hits:
        payload = hit.payload or {}
        context_chunks.append(payload.get("text", ""))
        used_ids.append(str(hit.id))


    context = "\n\n".join(context_chunks)

    # LLM で回答生成
    answer = generate_answer(context, req.question)

    return QueryResponse(answer=answer, used_ids=used_ids)

# uvicorn から直接このファイルを実行する場合用（docker では CMD で起動するので必須ではない）
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

