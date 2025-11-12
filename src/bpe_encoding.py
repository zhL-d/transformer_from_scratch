from src.tokenizer import Tokenizer
import os
import sys
import io
import numpy as np
import logging
from pathlib import Path
from google.cloud import storage

# ---- logging setup ----
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def is_gcs(path: str | os.PathLike) -> bool:
    s = os.fspath(path)
    return s.startswith("gs://")


def read_text(path: str | os.PathLike) -> str:
    if is_gcs(path):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data_bytes = blob.download_as_bytes()
        return data_bytes.decode("utf-8", "surrogatepass")
    else:
        return Path(path).read_text(encoding="utf-8", errors="surrogatepass")


def write_artifact(path: str | os.PathLike, arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    if is_gcs(path):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(buf.getvalue())
    else:
        Path(path).write_bytes(buf.getvalue())


def normalize_vocab_merge_path(vocab_path: str, merge_path: str) -> tuple[str, str]:
    if is_gcs(vocab_path) and is_gcs(merge_path):
        vocab_json_text = read_text(vocab_path)
        merge_json_text = read_text(merge_path)

        tmp_path = Path("/tmp/tokenizer")
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_vocab_path = tmp_path / "vocab.json"
        tmp_merge_path = tmp_path / "merge.json"

        tmp_vocab_path.write_text(vocab_json_text, encoding="utf-8", errors="surrogatepass")
        tmp_merge_path.write_text(merge_json_text, encoding="utf-8", errors="surrogatepass")

        return str(tmp_vocab_path), str(tmp_merge_path)
    else:
        return vocab_path, merge_path


def build_artifact_path(base_path: str, corpus_path: str) -> str | Path:
    corpus_basename = Path(corpus_path).stem
    filename = f"token_ids_uint16_{corpus_basename}.npy"
    if is_gcs(base_path):
        return base_path.rstrip("/") + "/" + filename
    return Path(base_path) / filename


def main():
    corpus_path = os.getenv("CORPUS_PATH")
    vocab_path = os.getenv("VOCAB_PATH")
    merge_path = os.getenv("MERGE_PATH")
    artifact_path = os.getenv("ARTIFACT_PATH")

    if corpus_path is None:
        print("Error: the env variable CORPUS_PATH is not set")
        sys.exit(1)

    if vocab_path is None:
        print("Error: the env variable VOCAB_PATH is not set")
        sys.exit(1)

    if merge_path is None:
        print("Error: the env variable MERGE_PATH is not set")
        sys.exit(1)

    if artifact_path is None:
        print("Error: the env variable ARTIFACT_PATH is not set")
        sys.exit(1)

    artifact_final_path = build_artifact_path(artifact_path, corpus_path)

    logging.info(
        "Job start | vocab_path=%s | merge_path=%s | corpus_path=%s | artifact_path=%s",
        vocab_path,
        merge_path,
        corpus_path,
        artifact_path,
    )

    vocab_path, merge_path = normalize_vocab_merge_path(vocab_path, merge_path)

    tokenizer = Tokenizer.from_files(vocab_path, merge_path, ["<|endoftext|>"])

    corpus = read_text(corpus_path)

    token_ids = tokenizer.encode(corpus)

    token_ids_np = np.array(token_ids, dtype=np.uint16)

    # np.save(artifact_final_path, token_ids_np)

    write_artifact(artifact_final_path, token_ids_np)

    # print(f"Save {len(token_ids_np)} in {artifact_final_path}")
    logger.info("Encode done | token_count=%d | artifact_path=%s", len(token_ids), artifact_final_path)


if __name__ == "__main__":
    main()
