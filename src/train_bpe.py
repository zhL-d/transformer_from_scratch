from src.bpetokenizer_trainer import BPETokenizerTrainer
import time
import tracemalloc
import os
import psutil
import contextlib
import yaml

@contextlib.contextmanager
def perf_monitor(enabled: bool = True):
    if not enabled:
        yield {}
        return
    
   # Stat time and memory
    tracemalloc.start()
    start_time = time.perf_counter()
    
    try:
        yield {}
    finally:
        # Stat time and memory
        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory stat(rss)
        process = psutil.Process(os.getpid())
        rss_mem = process.memory_info().rss / (1024 * 1024)

        # Build report
        report = f"""
        Performence report
        -------------------------------
        Total time                      :{(end_time - start_time):.2f} seconds
        Peak memory managed by python   :{peak / 1024 / 1024:.2f} MB
        Total physical memory used(RSS) :{rss_mem:.2f} MB
        """

        print(report)

def main():
    with perf_monitor(enabled=False):

        with open("cs336_basics/config_azure_trial.yaml", "r") as f:
            config = yaml.safe_load(f)

            # Apply env-var overrides (env > YAML)
            config["traindata_path"] = os.getenv("TRAINDATA_PATH", config["traindata_path"])
            config["vocab_size"] = int(os.getenv("VOCAB_SIZE", config["vocab_size"]))
            config["outputs_path"] = os.getenv("OUTPUTS_PATH", config["outputs_path"])
        # Init tokenizer
        tokenizer = BPETokenizerTrainer(
            config["special_tokens"],
            config["outputs_path"],
            # os.getenv("OUTPUTS_PATH", config["outputs_path"]),
            enable_log=config["enable_log"], 
            # log_path=config["log_path"],
            serialization=config["serialization"],
            # serialization_vocab_path= config["serialization_vocab_path"],
            # serialization_merge_path= config["serialization_merge_path"]
        )
    
        # Training
        vocab, merges = tokenizer.train_bpe(
            config["traindata_path"], 
            vocab_size=config["vocab_size"], 
            gpt2_regex=config["gpt2_regex"], 
            enable_parallel=config["parallel"]
        )

    # Build report
    report = f"""
        BPE Tokenizer Training report
        -------------------------------
        Vocabuary size                  :{len(vocab)}
        Number of merges                :{len(merges)}
        First 5 merges                  :{merges[:5]}
    """

    print(report)

    op = config["outputs_path"]
    print(f"outputs_path: {op}")

if __name__ == "__main__":
    main()
