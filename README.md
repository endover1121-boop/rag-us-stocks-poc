# RAG US Stocks PoC

米国株の 10-K / 10-Q を対象にした、**投資リサーチ支援用 RAG（Retrieval-Augmented Generation）PoC** です。

- EDGAR から取得した 10-K / 10-Q（PDF）
- ベクトルDB（Qdrant）
- 埋め込みモデル（Sentence-Transformers）
- LLM（sarashina 系モデルを想定）

を組み合わせて、銘柄別に決算書の内容を検索・要約できるようにしています。

> このリポジトリは「コード＋構成＋再現手順」を提供し、  
> モデル本体やキャッシュは Git 管理の対象外にしています。


---

## 機能概要

- 10-K/10-Q PDF をテキスト化し、ページ単位でチャンク化
- チャンクを埋め込みベクトルに変換し、Qdrant にインデックス
- FastAPI ベースの REST API 提供
  - `POST /index` : チャンクの一括インデックス
  - `POST /rag/query` : 質問＋（任意で ticker）から RAG 回答を生成
  - `GET /health` : ヘルスチェック
- Docker Compose によるローカル・スタック構築
  - `rag-app` : FastAPI + 埋め込みモデル + LLM
  - `qdrant` : ベクトルDBコンテナ

---

## ディレクトリ構成

ベースとなる構成は以下のとおりです。

```bash
rag-us-stocks-poc/
├── app/
│   ├── main.py          # FastAPI アプリ本体（APIエンドポイント定義）
│   ├── rag_config.py    # モデル名や Qdrant 設定などの構成
│   ├── requirements.txt # app コンテナ用 Python 依存パッケージ
│   └── Dockerfile       # rag-app コンテナのビルド定義
│
├── docker-compose.yml   # rag-app + Qdrant のコンテナオーケストレーション
│
├── ingest_10k.py        # 10-K / 10-Q PDF → /index へ投入するバッチスクリプト
├── company10kq_pdfs/    # EDGAR 等から取得した PDF を置くフォルダ
│   └── (例) AAPL_10K_2024.pdf, MSFT_10K_2023.pdf, ...
│
├── data/
│   ├── qdrant/          # Qdrant の永続ボリューム
│   └── models/          # モデルキャッシュなど（Git では無視）
│
└── .gitignore           # .venv や dataを除外

```
---

## 前提環境

- OS: Windows 11（開発・動作確認想定）
- 必須
  - Docker Desktop
  - docker-compose（Docker Desktop 同梱）
  - Python 3.10 以降（`ingest_10k.py` 実行用）
- 推奨
  - GPU（今回はNVIDIA GeForce RTX 3060 Ti）
  - docker-compose（Docker Desktop 同梱）

    → rag-app で LLM を GPU 実行する場合

---
## セットアップ
1. リポジトリのクローン

```bash
https://github.com/endover1121-boop/rag-us-stocks-poc.git
cd rag-us-stocks-poc
```
2. Python 仮想環境
```bash
python -m venv .venv
.\.venv\Scripts\activate  # PowerShell の場合
pip install -r app/requirements.txt
pip install pdfplumber    # ingest_10k.py で使用
```
　※ `.venv/` は `.gitignore` で除外しています。

---
## Docker でバックエンドを起動
`<docker-compose.yml` があるフォルダで実行します。

```bash
docker compose up --build
```
- `rag-app` コンテナ: FastAPI + 埋め込みモデル + LLM 
- `qdrant` コンテナ: ベクトルDB

起動後、ブラウザで以下にアクセスして動作確認できます。

- ヘルスチェック:
  
  http://localhost:8000/health
- OpenAPI ドキュメント（Swagger UI）:

  http://localhost:8000/docs

---
## 10-K / 10-Q のインデックス投入フロー
1. PDFを配置

   `company10kq_pdfs/` フォルダに対象の PDF を配置します。

    ファイル名は、簡易パースのために以下の形式を前提にしています：

```text
<TICKER>_10Q[K]_<YEAR><DATE>.pdf
例:
  aapl-20250628.pdf
  msft-20250630.pdf

```

`parse_pdf_filename()` で `[TICKER, YEAR, DATE]` の3要素を想定しています。

2. インデックス投入スクリプトを実行

   Docker で rag-app / qdrant が起動している状態で、別ターミナルから：

```bash
.\.venv\Scripts\activate   # 仮想環境を使う場合
python ingest_10k.py
```
`ingest_10k.py` の主な処理：
- `company10kq_pdfs/` 配下の PDF を列挙
- `pdfplumber` でページ単位にテキスト抽出
- 各ページを 1 チャンクとして扱い、必要に応じて先頭 N 文字に切り詰め
- `id` / `text` / `ticker` / `source="10-K"`(後ほど削除予定) / `date` を構築
- バッチ（小さめの batch_size）単位で `POST /index` に送信
- `/index` 側で埋め込み → Qdrant へ upsert

メモリ制約を考慮して、以下のような工夫をしています。
- PDF を 1ファイルずつ処理
- ページ単位でチャンク化（1ページ=1チャンク）
- 1バッチあたりの送信件数を制限（例: 5 件程度）
- LLM は lazy load（インデックス時はロードしない）にする想定

---
## API の使い方
1. `/index` エンドポイント 

　　手動でテストしたい場合は、Swagger UI から POST /index を選択し、例えば：

```json
{
  "items": [
    {
      "id": 1,
      "text": "Microsoftはクラウド事業Azureの売上が前年比20%成長したと発表した。",
      "ticker": "MSFT",
      "source": "news",
      "date": "2024-11-01"
    }
  ]
}

```
　　のような JSON を送ると、Qdrant に1件登録されます。 

　　通常運用では、人手で `/index` を叩くのではなく、`ingest_10k.py`がまとめて投げる想定です。

2. `/rag/query`エンドポイント

    Swagger UI から POST /rag/query を選択し、例えば：

```json
{
  "question": "AAPLの直近10-Kの内容から、成長ドライバーと主要なリスクを教えてください。",
  "ticker": "AAPL"
}
```
  のようなリクエストを送ります。
  - `ticker`を指定した場合：
    - その銘柄に紐づくチャンクのみをQdrantから検索し。RAGで回答
  - `ticker`を`null`または省略した場合：
    - 全銘柄横断で類似チャンクを検索

レスポンス例（イメージ）：

```json
{
  "answer": "AAPLの直近10-Kでは、主な成長ドライバーとして...（省略）",
  "used_ids": [1000001, 1000002, 1000003]
}
```
used_ids は Qdrant 上で参照したチャンクIDの一覧です（DOC_ID_STRIDE に基づき銘柄ごとにユニークな範囲を持たせています）。

---

## モデル・データの扱いについて

### Git 管理の対象外にしているもの

- `data/models/`
  - LLM / 埋め込みモデルのキャッシュ・実体
  - 容量が大きく、ライセンス上の再配布リスクもあるため Git 管理対象外
- `.venv/`
  - 各環境ごとの仮想環境（OS依存のため Git 管理不要）
- `data/qdrant/`
  - Qdrant のストレージ（生成物）

これらは `.gitignore` に含めている想定です。

### モデルの再現方法

実際のモデル名や設定値は app/rag_config.py で定義しています（例）：
- `LLM_MODEL_NAME` : sarashina の 1B / 3B モデル名
- `EMBEDDING_MODEL_NAME` : Sentence-Transformers のモデル名
- Qdrant のホスト / ポート / コレクション名

FastAPI / Docker 起動時に、`transformers` / `sentence-transformers` 経由で自動的にモデルがダウンロード・キャッシュされる前提です。

---
## このPoCの技術的なポイント

- RAG をゼロから設計・構築
  - 10-K/10-Q という金融ドメイン特有の長文・大容量PDFを対象にした構成
- ベクトルDB（Qdrant）と LLM の組み合わせ
  - Python + FastAPI で `/index` / `/rag/query` を実装
- メモリ制約への対応 
  - -Windows 環境でのメモリ不足（OOM, exit code 137, MemoryError）を踏まえ、
  - モデルサイズ調整（例: 3B → 1B） 
  - ページ単位チャンク、文字数制限
  - インデックスのバッチサイズ最適化
  - LLM の lazy load 
  
  など、実行環境に合わせたチューニングを実施

- Docker Compose によるローカルスタック再現性
  - `docker compose up` で Qdrant + `rag-app` を一括起動できるよう整理
  
---
## ライセンス／注意事項

- このリポジトリ内のコードは、個人の学習・PoC目的で作成したものです。
- 利用するモデル（sarashina 系、Sentence-Transformers 等）のライセンスは、それぞれの配布元の条件に従ってください。
- EDGAR等から取得した10-K/10-QのPDFやテキストは、各配布元の利用規約に従って扱ってください。

---
## 今後の拡張アイデア

- チャンク戦略の高度化（セクション単位 / 見出し単位）
- 10-Q / 8-K、決算説明会トランスクリプトへの対応
- フロントエンド（銘柄選択 + よく使う質問テンプレ）の実装
- スコアに基づく「投資アイデア候補リスト」生成
