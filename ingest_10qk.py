
import os
import glob
from typing import List, Dict, Tuple
import requests
import math

# pdfplumber
import pdfplumber

API_URL = "http://localhost:8000/index"
PDF_DIR = "./company10kq_pdfs"

# 1チャンクの最大文字数とオーバーラップ
MAX_CHARS = 1000
OVERLAP_CHARS = 200

# QdrantのID用に、ドキュメントごとに大きめのオフセットを取る
# doc_index * DOC_ID_STRIDE + chunk_index という形で整数IDを作る
DOC_ID_STRIDE = 1_000_000

# 進捗の重みづけ（全体のうち何%を占めるか）
PDF_PHASE_WEIGHT = 0.5     # PDF処理フェーズ = 全体の50%
INDEX_PHASE_WEIGHT = 0.5   # インデックスフェーズ = 全体の50%

def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """
    テキストを max_chars 文字程度のチャンクに分割する。
    overlap 文字ぶんだけ前チャンクと重なるようにする（文脈保持用）。
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # 次のチャンクの開始位置を、オーバーラップを考慮して進める
        start = end - overlap
        if start < 0:
            start = 0

        if start >= length:
            break

    return chunks


def parse_pdf_filename(filename: str) -> Tuple[str, str]:
    """
    ファイル名から ticker と year をざっくり取る。
    例: AAPL_10K_2024.pdf → ("AAPL", "2024")
    """
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]  # 拡張子除去

    parts = name.split("-")
    # 想定: [TICKER,"YYYY","MMDD"]
    if len(parts) >= 2:
        ticker = parts[0].upper()
        year = parts[1][0:4]
        date = parts[1][4:8]
    else:
        # 適当にfallback
        ticker = parts[0].upper()
        year = "0000"
        date = "0101"
    return ticker, year, date


def count_pdf_pages(pdf_path: str) -> int:
    """PDF のページ数だけを数える"""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def build_items_for_pdf_with_page_progress(
    pdf_path: str,
    doc_index: int,
    pages_done: int,
    total_pages: int,
) -> Tuple[List[Dict], int]:
    """
    1つのPDFについて、ページ単位でテキスト抽出しながら
    全体進捗（%）をページ単位で更新していく。
    各ページは「そのまま1チャンク」として扱う。
    """
    ticker, year, date = parse_pdf_filename(pdf_path)
    date_str = f"{year}-{date}"  # 超ざっくり。あとでちゃんと決算日を入れてもOK。

    print(f"[INFO] Processing PDF #{doc_index}: {pdf_path} (ticker={ticker}, year={year}), date={date}")

    items: List[Dict] = []
    base_id = doc_index * DOC_ID_STRIDE
    chunk_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

        for page_idx, page in enumerate(pdf.pages):
            # ページのテキストを抽出（None になることもあるので空文字扱い）
            page_text = page.extract_text() or ""
            page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")

            # 完全に空のページはスキップ
            if not page_text.strip():
                pages_done += 1
                ratio_pages = pages_done / max(total_pages, 1)
                overall_progress = ratio_pages * PDF_PHASE_WEIGHT * 100.0
                print(
                    f"[PROGRESS] overall {overall_progress:.1f}% "
                    f"(PDF {doc_index}, page {page_idx + 1}/{page_count}, total pages {pages_done}/{total_pages}, empty page)"
                )
                continue
            chunk_counter += 1
            point_id = base_id + chunk_counter  # 整数ID（Qdrant用）
            items.append(
                {
                    "id": point_id,
                    "text": page_text,
                    "ticker": ticker,
                    "source": "10-K",
                    "date": date_str,
                }
            )

            # このページの処理が終わったので、ページ進捗を更新
            pages_done += 1
            ratio_pages = pages_done / max(total_pages, 1)
            overall_progress = ratio_pages * PDF_PHASE_WEIGHT * 100.0
            print(
                f"[PROGRESS] overall {overall_progress:.1f}% "
                f"(PDF {doc_index}, page {page_idx + 1}/{page_count}, total pages {pages_done}/{total_pages})"
            )

    print(f"[INFO] Created {len(items)} chunks for {ticker} ({pdf_path})")
    return items, pages_done


def send_items_in_batches(
    items: List[Dict],
    batch_size: int = 50,
    start_progress: float = 50.0,    # 全体のうち何%からスタートするか（このPDFのインデックス開始点）
    total_weight: float = 10.0       # このPDFのインデックスが全体の何%分を受け持つか
):
    """
    items を /index にまとめて送る。
    batch_size ごとに分割。
    進捗を全体%で表示する。
    """
    total = len(items)
    if total == 0:
        print("[WARN] No items to send for this PDF.")
        return

    num_batches = math.ceil(total / batch_size)
    processed = 0

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, total)
        batch = items[start:end]

        payload = {"items": batch}
        print(f"[INFO] Sending batch {b+1}/{num_batches} ({len(batch)} items)")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
        except Exception as e:
            print(f"[ERROR] HTTP request failed for batch {b+1}: {e}")
            continue

        if resp.status_code != 200:
            print(f"[ERROR] Failed to index batch {b+1}: {resp.status_code} {resp.text}")
        else:
            print(f"[OK] Batch {b+1} indexed: {resp.json()}")

        # 進捗計算（インデックスフェーズ内の進み具合）
        processed += len(batch)
        ratio_in_phase = processed / total
        overall_progress = start_progress + ratio_in_phase * total_weight
        if overall_progress > 100.0:
            overall_progress = 100.0
        print(
            f"[PROGRESS] overall {overall_progress:.1f}% "
            f"(indexed {processed}/{total} chunks for this PDF)"
        )


def main():
    # PDFフォルダの中の .pdf を全部列挙
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {PDF_DIR}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDF files in {PDF_DIR}")

    # まず全PDFの総ページ数を数える（ページ単位進捗のため）
    per_pdf_pages = []
    total_pages = 0
    for pdf_path in pdf_files:
        n_pages = count_pdf_pages(pdf_path)
        per_pdf_pages.append((pdf_path, n_pages))
        total_pages += n_pages

    print(f"[INFO] Total pages across all PDFs: {total_pages}")

    pages_done = 0
    total_docs = len(per_pdf_pages)

    # 各PDFごとに：
    # - ページ単位でチャンク化（進捗 0〜50%）
    # - そのPDF分だけ /index 送信（インデックスフェーズの一部）
    index_phase_start = PDF_PHASE_WEIGHT * 100.0     # 例: 50.0
    total_index_weight = INDEX_PHASE_WEIGHT * 100.0  # 例: 50.0
    per_doc_index_weight = total_index_weight / max(total_docs, 1)

    for doc_index, (pdf_path, page_count) in enumerate(per_pdf_pages, start=1):
        # ===== PDF → チャンク化（ページ単位で進捗） =====
        items, pages_done = build_items_for_pdf_with_page_progress(
            pdf_path=pdf_path,
            doc_index=doc_index,
            pages_done=pages_done,
            total_pages=total_pages,
        )

        # ===== このPDF分だけ /index 送信 =====
        this_start = index_phase_start + per_doc_index_weight * (doc_index - 1)
        this_weight = per_doc_index_weight

        print(
            f"[INFO] Indexing chunks for PDF #{doc_index} "
            f"(overall progress range: {this_start:.1f}% → {this_start + this_weight:.1f}%)"
        )

        send_items_in_batches(
            items,
            batch_size=3,
            start_progress=this_start,
            total_weight=this_weight,
        )

        # このPDFのitemsはもう使わないので、参照を消してGCに任せる
        del items

    print("[INFO] All done.")


if __name__ == "__main__":
    main()