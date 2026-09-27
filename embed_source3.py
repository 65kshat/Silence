"""
embed_source3.py

Executes Step 2: Dense (BAAI/bge-m3) + Sparse (Word & Char TF-IDF) Representations
for Source 3 (Train & Test splits) on NVIDIA GPU (RTX 5050 Laptop).

Usage:
  # Embed Train Source 3:
  python embed_source3.py --split train

  # Embed Test Source 3:
  python embed_source3.py --split test

  # Custom batch size or paths:
  python embed_source3.py --split train --batch-size 512
"""
import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# ── STRICT GPU CHECK ──
if not torch.cuda.is_available():
    print("=" * 70)
    print("ERROR: CUDA GPU is required! Please run in your GPU environment (.venv).")
    print("=" * 70)
    sys.exit(1)

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on Ampere+
torch.cuda.empty_cache()

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def get_vram_info() -> str:
    alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    return f"VRAM Allocated: {alloc:.1f} MB | Reserved: {reserved:.1f} MB"


def build_combined_text(name: str, address: str) -> str:
    name = (name or "").strip()
    address = (address or "").strip()
    if not name and not address:
        return ""
    if not address:
        return name
    if not name:
        return address
    return f"{name} | {address}"


def fit_or_load_tfidf(cleaned_dir: Path, shared_dir: Path, sample_size: int = 300_000):
    word_vec_path = shared_dir / "tfidf_word.joblib"
    char_vec_path = shared_dir / "tfidf_char.joblib"

    if word_vec_path.exists() and char_vec_path.exists():
        print(f"Loading existing shared TF-IDF models from: {shared_dir}")
        word_vec = joblib.load(word_vec_path)
        char_vec = joblib.load(char_vec_path)
        print(f"  Word Vocab: {len(word_vec.vocabulary_):,} | Char Vocab: {len(char_vec.vocabulary_):,}")
        return word_vec, char_vec

    print("Fitting new shared TF-IDF vectorizers (sampling across cleaned parquet parts)...")
    parquet_files = sorted(cleaned_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet parts found in {cleaned_dir}")

    sampled_texts = []
    per_file_sample = max(10_000, sample_size // len(parquet_files))
    
    for f in tqdm(parquet_files, desc="Sampling text for TF-IDF"):
        df = pd.read_parquet(f, columns=["name_full_clean", "address_clean"])
        combined = [
            build_combined_text(n, a) for n, a in zip(df["name_full_clean"], df["address_clean"])
        ]
        if len(combined) > per_file_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combined), size=per_file_sample, replace=False)
            sampled_texts.extend([combined[i] for i in idx])
        else:
            sampled_texts.extend(combined)

    print(f"Fitting TF-IDF on {len(sampled_texts):,} sampled records...")
    
    # Word TF-IDF
    t0 = time.time()
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=200_000
    )
    word_vec.fit(sampled_texts)
    joblib.dump(word_vec, word_vec_path)
    print(f"  Word TF-IDF fit in {time.time() - t0:.1f}s (Vocab: {len(word_vec.vocabulary_):,})")

    # Char TF-IDF
    t1 = time.time()
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=200_000
    )
    char_vec.fit(sampled_texts)
    joblib.dump(char_vec, char_vec_path)
    print(f"  Char TF-IDF fit in {time.time() - t1:.1f}s (Vocab: {len(char_vec.vocabulary_):,})")

    return word_vec, char_vec


def process_part(
    file_path: Path,
    output_dir: Path,
    tokenizer,
    model,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    batch_size: int = 512,
    max_length: int = 160
):
    stem = file_path.stem  # e.g. "part_00000"
    meta_out = output_dir / f"{stem}_meta.parquet"
    emb_npy_out = output_dir / f"{stem}_emb.npy"
    word_out = output_dir / f"{stem}_tfidf_word.npz"
    char_out = output_dir / f"{stem}_tfidf_char.npz"
    done_file = output_dir / f"{stem}.done"

    if done_file.exists() and meta_out.exists() and emb_npy_out.exists():
        print(f"[SKIP] {stem} already completed.")
        return

    t0 = time.time()
    df = pd.read_parquet(file_path, columns=["entity_id", "name_full_clean", "address_clean", "country"])
    df["combined_text"] = [
        build_combined_text(n, a) for n, a in zip(df["name_full_clean"], df["address_clean"])
    ]
    texts = df["combined_text"].tolist()
    n_rows = len(texts)

    # 1. Sparse TF-IDF representations
    word_mat = word_vec.transform(texts)
    sp.save_npz(word_out, word_mat.astype(np.float32))

    if char_vec is not None:
        char_mat = char_vec.transform(texts)
        sp.save_npz(char_out, char_mat.astype(np.float32))

    # 2. Dense BGE-M3 Embeddings on GPU
    # Directly tokenize + run model, taking CLS token as dense vector (same as FlagEmbedding internally)
    all_embeddings = []
    for i in range(0, n_rows, batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()  # CLS token
        all_embeddings.append(batch_emb)
    embeddings = l2_normalize(np.vstack(all_embeddings), axis=1)

    # 3. Save raw float32 matrix + metadata parquet
    np.save(emb_npy_out, embeddings)

    df_meta = df[["entity_id", "combined_text", "country"]].copy()
    df_meta.to_parquet(meta_out, index=False)

    done_file.write_text(f"DONE {n_rows}\n", encoding="utf-8")
    dt = time.time() - t0
    rate = n_rows / dt if dt > 0 else 0
    print(f"[{stem}] Encoded {n_rows:,} records in {dt:.1f}s ({rate:,.1f} items/sec) | {get_vram_info()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dense BGE-M3 + Sparse TF-IDF for Source 3")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split (train or test)")
    parser.add_argument("--batch-size", type=int, default=512, help="GPU batch size (default: 512)")
    parser.add_argument("--max-length", type=int, default=160, help="Max token length (default: 160)")
    parser.add_argument("--input-dir", type=Path, default=None, help="Custom input cleaned parquet directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(r"c:\Users\Akshat\Desktop\Over Here\Projects\Amazon ML (2026)")
    input_dir = args.input_dir if args.input_dir else (base_dir / "dataset" / "cleaned" / args.split)
    output_dir = args.output_dir if args.output_dir else (base_dir / "dataset" / "cleaned" / "embeddings" / f"source3_{args.split}")
    shared_dir = base_dir / "dataset" / "cleaned" / "embeddings" / "shared"

    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet parts found in {input_dir}. Please run clean_source3.py first.")

    print("=" * 70)
    print(f"STEP 2: BGE-M3 DENSE + DUAL TF-IDF REPRESENTATION — SOURCE 3 ({args.split.upper()})")
    print(f"GPU:          {GPU_NAME} ({TOTAL_VRAM_GB:.2f} GB VRAM)")
    print(f"Input Dir:    {input_dir} ({len(parquet_files)} parts)")
    print(f"Output Dir:   {output_dir}")
    print(f"Shared Dir:   {shared_dir}")
    print(f"Batch Size:   {args.batch_size} | Max Length: {args.max_length}")
    print("=" * 70)

    # 1. Fit or Load Shared TF-IDF
    word_vec, char_vec = fit_or_load_tfidf(input_dir, shared_dir)

    # 2. Load BGE-M3 onto GPU directly via transformers (bypasses FlagEmbedding compatibility issues)
    print(f"\nLoading BAAI/bge-m3 on {GPU_NAME} with FP16...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.float16).to("cuda")
    model.eval()
    print(f"BGE-M3 loaded in {time.time() - t_load:.1f}s | [{get_vram_info()}]")

    # 3. Process every parquet part
    print(f"\nStarting GPU encoding across {len(parquet_files)} parts...")
    start_time = time.time()

    for f in tqdm(parquet_files, desc=f"Source 3 ({args.split})"):
        process_part(
            file_path=f,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            word_vec=word_vec,
            char_vec=char_vec,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

    (output_dir / "_SUCCESS").write_text(f"Completed at {time.ctime()}\n", encoding="utf-8")
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"STEP 2 EMBEDDING COMPLETE FOR SOURCE 3 ({args.split.upper()})!")
    print(f"Total Time:    {total_time / 60:.1f} minutes")
    print(f"Output Path:   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
