"""
clean_source2.py

Production Multilingual Cleaning Pipeline for Source 2 (Train & Test).
Streams 5+ million records in chunks with parquet output and full resumability.

Usage:
  # Clean Train Source 2 (Default):
  python clean_source2.py

  # Clean Test Source 2:
  python clean_source2.py --split test

  # Custom paths:
  python clean_source2.py --input path/to/source2.tsv --output path/to/output_dir
"""
import sys
import os
import re
import time
import argparse
import unicodedata
from pathlib import Path
import pandas as pd
from unidecode import unidecode
from tqdm.auto import tqdm

# ============================================================
# MULTILINGUAL REGIONAL & LEGAL MAPS
# ============================================================

INDIC_STATE_MAP = {
    # Gujarati
    "ગુજરાત": "gujarat",
    # Tamil
    "தமிழ்நாடு": "tamil nadu", "தமிழ் நாடு": "tamil nadu",
    # Telugu
    "తెలంగాణ": "telangana", "తెలంగాణ రాష్ట్రం": "telangana",
    "ఆంధ్ర ప్రదేశ్": "andhra pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh",
    # Malayalam
    "കേരളം": "kerala", "കേരള": "kerala",
    # Kannada
    "ಕರ್ನಾಟಕ": "karnataka",
    # Devanagari (Hindi / Marathi)
    "महाराष्ट्र": "maharashtra", "उत्तर प्रदेश": "uttar pradesh", "मध्य प्रदेश": "madhya pradesh",
    "राजस्थान": "rajasthan", "हरियाणा": "haryana", "दिल्ली": "delhi", "नई दिल्ली": "new delhi",
    "बिहार": "bihar", "झारखंड": "jharkhand", "उत्तराखंड": "uttarakhand", "छत्तीसगढ़": "chhattisgarh",
    # Bengali
    "পশ্চিমবঙ্গ": "west bengal", "পশ্চিম বঙ্গ": "west bengal",
    # Gurmukhi (Punjabi)
    "ਪੰਜਾਬ": "punjab",
    # Oriya
    "ଓଡ଼ିଶା": "odisha", "ଓଡିଶା": "odisha"
}

INDIC_LEGAL_SUFFIXES = [
    # Multi-word patterns
    (r"\b(प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|प्रा\s+लि|प्रा\.\s*लिमिटेड)\b", " private limited "),
    (r"\b(પ્રાઇવેટ\s+લિમિટેડ|પ્રા\.\s*લિ\.|પ્રા\s+લિ)\b", " private limited "),
    (r"\b(பிரைவேட்\s+லிமிடெட்)\b", " private limited "),
    (r"\b(ಪ್ರೈವೇಟ್\s+ಲಿಮಿಟೆಡ್)\b", " private limited "),
    (r"\b(ప్రైవేట్\s+లిమిటెడ్)\b", " private limited "),
    (r"\b(പ്രൈവറ്റ്\s+ലിമിറ്റഡ്|പ്രൈവറ്റ്\s+ലി\.)\b", " private limited "),
    (r"\b(প্রাইভেট\s+লিমিটেড)\b", " private limited "),
    (r"\b(ਪ੍ਰਾਈਵੇਟ\s+ਲਿਮਿਟੇਡ|ਪ੍ਰਾ\.\s*ਲਿ\.)\b", " private limited "),
    (r"\b(ପ୍ରାଇଭେଟ\s+ଲିମିଟେଡ)\b", " private limited "),
    # Single words
    (r"\b(लिमिटेड|લિમિટેડ|லிமிடெட்|ಲಿಮಿಟೆಡ್|లిమిటెడ్|ലിമിറ്റഡ്|লিমিটেড|ਲਿਮਿਟੇਡ|ଲିମିଟେଡ)\b", " limited "),
    (r"\b(प्राइवेट|પ્રાઇવેટ|பிரைவேட்|ಪ್ರೈವೇಟ್|ప్రైవేట్|പ്രൈവറ്റ്|প্রাইভেট|ਪ੍ਰਾਈਵੇਟ|ପ୍ରାଇଭେଟ)\b", " private "),
    (r"\b(एलएलपी|એલએલપી|எல்எல்பி|ಎಲ್ಎಲ್ಪಿ|ఎల్ఎల్పీ|എൽഎൽപി|এলএলপি|ਐਲਐਲਪੀ)\b", " llp "),
]

NAME_ABBREVIATIONS = {
    "pvt": "private", "ltd": "limited", "inc": "incorporated",
    "corp": "corporation", "co": "company", "llc": "llc", "llp": "llp",
    "plc": "plc", "st": "saint", "dept": "department", "mfg": "manufacturing",
    "intl": "international", "tech": "technologies", "serv": "services"
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "apt": "apartment",
    "ste": "suite", "fl": "floor", "bldg": "building", "pk": "park",
    "dist": "district", "tq": "taluk", "sec": "sector", "blk": "block",
    "opp": "opposite", "nr": "near", "c/o": "care of"
}

HARD_LEGAL_SUFFIXES = {
    "company", "co", "corporation", "corp", "inc", "incorporated",
    "limited", "ltd", "private", "pvt", "llc", "llp", "plc",
    "sarl", "sas", "sasu", "eurl", "sa", "gmbh", "ag", "spa", "srl",
    "bv", "nv", "limitdda", "praivetta", "pvt ltd", "private limited"
}

SOFT_LEGAL_SUFFIXES = {
    "enterprises", "solutions", "services", "technologies", "holdings",
    "international", "global", "industries", "ventures", "group", "associates"
}

LEADING_STOPWORDS = {"the", "a", "an", "m/s", "ms", "dr", "mr", "mrs", "shri", "sri"}
LANDMARK_PATTERN = re.compile(r"\b(near|opposite|behind|beside|next to|close to|adjacent to|in front of|opp|nr|c/o)\b", re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\b\d+[a-z]?(?:[/-]\d+[a-z]?)*\b", re.IGNORECASE)
PIN_INDIA_RE = re.compile(r"\b[1-9]\d{5}\b")
ZIP_US_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
WORD_RE = re.compile(r"\b[a-z0-9]{2,}\b")


def normalize_indic_text(text: str) -> str:
    if not text:
        return ""
    for native_state, eng_state in INDIC_STATE_MAP.items():
        if native_state in text:
            text = text.replace(native_state, f" {eng_state} ")
    for pattern, repl in INDIC_LEGAL_SUFFIXES:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def clean_latin_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_indic_text(text)
    text = unidecode(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = re.sub(r"['\"`]", "", text)
    text = re.sub(r"[^\w\s/-]", " ", text)
    return MULTI_SPACE_RE.sub(" ", text).strip()


def expand_tokens(text: str, mapping: dict) -> str:
    if not text:
        return ""
    tokens = text.split()
    return " ".join(mapping.get(tok, tok) for tok in tokens)


def clean_name(raw_name: str) -> str:
    cleaned = clean_latin_text(raw_name)
    return expand_tokens(cleaned, NAME_ABBREVIATIONS)


def clean_address(raw_addr: str) -> str:
    cleaned = clean_latin_text(raw_addr)
    return expand_tokens(cleaned, ADDRESS_ABBREVIATIONS)


def make_name_core(cleaned_name: str) -> str:
    if not cleaned_name:
        return ""
    tokens = cleaned_name.split()
    while tokens and tokens[0] in LEADING_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in HARD_LEGAL_SUFFIXES:
        tokens.pop()
    if len(tokens) >= 3 and tokens[-1] in SOFT_LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def split_landmark(address_clean: str):
    if not address_clean:
        return "", ""
    match = LANDMARK_PATTERN.search(address_clean)
    if not match:
        return address_clean, ""
    return address_clean[:match.start()].strip(), address_clean[match.start():].strip()


def extract_postal_code(address_clean: str, country: str) -> str:
    if not address_clean:
        return ""
    c_upper = (country or "").strip().upper()
    if c_upper in {"INDIA", "IN"}:
        matches = PIN_INDIA_RE.findall(address_clean)
        return matches[-1] if matches else ""
    elif c_upper in {"US", "USA", "UNITED STATES"}:
        matches = ZIP_US_RE.findall(address_clean)
        if matches:
            return matches[-1].split("-")[0]
    return ""


def extract_numbers(text: str) -> str:
    if not text:
        return ""
    matches = NUMBER_RE.findall(text)
    seen = set()
    res = []
    for m in matches:
        if m not in seen and len(m) <= 10:
            seen.add(m)
            res.append(m)
    return "|".join(res)


def tokenize(text: str) -> str:
    if not text:
        return ""
    tokens = [w for w in WORD_RE.findall(text) if len(w) >= 2]
    return "|".join(tokens)


def process_chunk(chunk: pd.DataFrame, chunk_id: int, output_dir: Path) -> Path:
    final_file = output_dir / f"part_{chunk_id:05d}.parquet"
    done_file = output_dir / f"part_{chunk_id:05d}.done"

    if final_file.exists() and done_file.exists():
        print(f"[SKIP] Chunk {chunk_id:05d} already processed.")
        return final_file

    chunk["business_name_original"] = chunk["business_name"].fillna("").astype(str)
    chunk["business_address_original"] = chunk["business_address"].fillna("").astype(str)
    chunk["country"] = chunk["country"].fillna("").astype(str).str.strip().str.upper()

    # Step 1: Clean names
    chunk["name_full_clean"] = chunk["business_name_original"].map(clean_name)
    chunk["name_core"] = chunk["name_full_clean"].map(make_name_core)
    chunk["name_latin"] = chunk["name_full_clean"]

    # Step 2: Clean addresses
    chunk["address_clean"] = chunk["business_address_original"].map(clean_address)
    landmark_split = chunk["address_clean"].map(split_landmark)
    chunk["address_core"] = landmark_split.map(lambda pair: pair[0])
    chunk["landmark_text"] = landmark_split.map(lambda pair: pair[1])
    chunk["address_numbers"] = chunk["address_clean"].map(extract_numbers)
    chunk["postal_code"] = [
        extract_postal_code(addr, cntry)
        for addr, cntry in zip(chunk["address_clean"], chunk["country"])
    ]

    # Step 3: Tokens
    chunk["name_tokens"] = chunk["name_full_clean"].map(tokenize)
    chunk["address_tokens"] = chunk["address_clean"].map(tokenize)

    output_columns = [
        "entity_id", "business_name_original", "business_address_original",
        "name_full_clean", "name_core", "name_latin",
        "address_clean", "address_core", "landmark_text",
        "name_tokens", "address_tokens", "address_numbers",
        "postal_code", "country"
    ]
    chunk_clean = chunk[output_columns]

    # Step 4: Write atomic parquet
    tmp_file = output_dir / f"part_{chunk_id:05d}.parquet.tmp"
    chunk_clean.to_parquet(tmp_file, engine="pyarrow", index=False, compression="zstd")
    if tmp_file.exists():
        tmp_file.replace(final_file)
    done_file.write_text("DONE\n", encoding="utf-8")

    return final_file


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Data Cleaning for Source 2")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split (train or test)")
    parser.add_argument("--input", type=Path, default=None, help="Custom input TSV path")
    parser.add_argument("--output", type=Path, default=None, help="Custom output directory")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Chunk size (rows per parquet part)")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(r"c:\Users\Akshat\Desktop\Over Here\Projects\Amazon ML (2026)")
    raw_dir = base_dir / "dataset" / args.split
    
    input_file = args.input if args.input else (raw_dir / f"{args.split}_source3.tsv")
    output_dir = args.output if args.output else (base_dir / "dataset" / "cleaned" / args.split)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print("=" * 70)
    print(f"MULTILINGUAL DATA CLEANING — SOURCE 2 ({args.split.upper()} SPLIT)")
    print(f"Input File:   {input_file}")
    print(f"Output Dir:   {output_dir}")
    print(f"Chunk Size:   {args.chunk_size:,} rows per part")
    print("=" * 70)

    start_time = time.time()
    total_processed = 0

    reader = pd.read_csv(
        input_file,
        sep="\t",
        chunksize=args.chunk_size,
        dtype=str,
        keep_default_na=False
    )

    for chunk_id, chunk in enumerate(reader):
        t0 = time.time()
        chunk_len = len(chunk)
        process_chunk(chunk, chunk_id, output_dir)
        total_processed += chunk_len
        dt = time.time() - t0
        rate = chunk_len / dt if dt > 0 else 0
        print(f"[Chunk {chunk_id:04d}] Processed {chunk_len:,} rows in {dt:.1f}s ({rate:,.0f} rows/sec) | Total: {total_processed:,}")

    total_time = time.time() - start_time
    (output_dir / "_SUCCESS").write_text(f"Completed {total_processed} rows in {total_time:.1f}s\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"CLEANING COMPLETE FOR SOURCE 2 ({args.split.upper()})")
    print(f"Total Records: {total_processed:,}")
    print(f"Total Time:    {total_time:.1f}s ({total_processed / total_time:,.0f} rows/sec)")
    print(f"Output Path:   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
