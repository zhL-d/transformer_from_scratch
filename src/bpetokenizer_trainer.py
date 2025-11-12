import regex as re
import json
import logging
from collections import defaultdict
from typing import BinaryIO
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
from datetime import datetime
from pathlib import Path

PAT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT_PATTERN)

# Wrapper for heap comparasion, lexical greater
class _Desc:
    # __slots__ = ['x']

    def __init__(self, x):
        self.x = x
    
    def __lt__(self, other):
        """
        Overwrite reversice, lexical greater
        """
        return self.x > other.x
    

class BPETokenizerTrainer:
    def __init__(
            self,
            special_tokens: list[str] | None = None,
            output_path: str | None = None,
            enable_log: bool = False, 
            # log_path: str = "",
            serialization: bool = False,
            # serialization_vocab_path: str | None = None,
            # serialization_merge_path: str | None = None,
        ):
        self.special_tokens = special_tokens or []
        self.next_id = 0
        self.vocab: dict[int, bytes] = {}
        self.merge: list[tuple[bytes, bytes]] = []
        self.enable_log = enable_log
        self._heap = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.serialization = serialization
        # self.serialization_vocab_path = serialization_vocab_path
        # self.serialization_merge_path = serialization_merge_path

        # Create output path dir
        # default path
        if not output_path:
            self.output_path = Path(__file__).parent / "outputs"
        else:
            p = Path(output_path)
            self.output_path = p if p.is_absolute() else Path(__file__).parent.parent / output_path
        
        self.output_path.mkdir(parents=True, exist_ok=True)

        if enable_log:
            # if not log_path:
            #     raise ValueError("Logging is enable but no log path was provided")
            
            log_name = f"training_{self.timestamp}.log"
            log_path = self.output_path / log_name
            
            self._set_log_conifg(log_path)

    @staticmethod
    def _set_log_conifg(log_path: str):
        logging.basicConfig(
            filename=log_path, 
            filemode="w", 
            level=logging.INFO, 
            format="%(message)s"
        )
    
    def dump_pair_count(self, pair_count: dict[tuple[bytes], int], merged_token: tuple[tuple[bytes], int], index: int):
        if self.enable_log:
            serial = { str(k): v for k, v in pair_count.items() }
            serial_merged_token = {str(merged_token[0]): merged_token[1]}
            logging.info(json.dumps({"step": index, "pair": serial, "merged": serial_merged_token}, ensure_ascii=False, sort_keys=True))
    
    def init_vocab(self):
        self.vocab = {x: bytes([x]) for x in range (256)}
        token_id_start = 256

        for i, special_token in enumerate(self.special_tokens):
            s_bytes = special_token.encode("utf-8")
            special_token_id = token_id_start + i
            self.vocab[special_token_id] = s_bytes

        # Start assigning after base bytes + special tokens
        self.next_id = 256 + len(self.special_tokens)
    
    def remove_special_tokens(self, text: str) -> list[str]:
        stokens_escaped = [re.escape(stoken) for stoken in self.special_tokens]
        return re.split("|".join(stokens_escaped), text)
    
    def remove_special_tokens_static(text: str, special_tokens: list[str]) -> list[str]:
        stokens_escaped = [re.escape(stoken) for stoken in special_tokens]
        return re.split("|".join(stokens_escaped), text)
    
    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )
    
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
    
        chunk_size = file_size // desired_num_chunks
    
        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
    
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
    
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
    
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
    
        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    # @staticmethod
    # def pretokenize_and_count(docs: list[str], gpt2_regex: bool = False) -> dict[tuple[bytes], int]:
    #     token_count : dict[tuple[bytes], int] = {}
    
    #     for doc in docs:
    #         pre_tokens = None
    #         # Use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
    #         if gpt2_regex:
    #             PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #             pre_tokens = re.finditer(PAT, doc)
    #             pre_tokens = [match.group(0) for match in pre_tokens]
    #         else:
    #             pre_tokens = doc.split()
    
    #         for token in pre_tokens:
    #             bytes_token = token.encode("utf-8")
                
    #             tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #             token_count[tuple_bytes_token] = token_count.get(tuple_bytes_token, 0) + 1
            
    #     return token_count

    # @staticmethod
    # def pretokenize_and_count(docs: list[str], gpt2_regex: bool = False) -> dict[tuple[bytes], int]:
    #     token_count : dict[tuple[bytes], int] = {}
    #     # token_count_get = token_count.get
    
    #     for doc in docs:
    #         pre_tokens = None
    #         # Use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
    #         if gpt2_regex:
    #             # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #             pre_tokens = PAT.finditer(doc)
    #             pre_tokens = [match.group(0) for match in pre_tokens]
    #         else:
    #             pre_tokens = doc.split()
    
    #         for token in pre_tokens:
    #             bytes_token = token.encode("utf-8")
                
    #             tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #             token_count[tuple_bytes_token] = token_count.get(tuple_bytes_token, 0) + 1
            
    #     return token_count

    # @staticmethod
    # def pretokenize_and_count(docs: list[str], gpt2_regex: bool = False) -> dict[tuple[bytes], int]:
    #     token_count : dict[tuple[bytes], int] = {}
    #     token_count_get = token_count.get
    
    #     for doc in docs:
    #         pre_tokens = None
    #         # Use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
    #         if gpt2_regex:
    #             # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #             pre_tokens = PAT.finditer(doc)
    #             pre_tokens = [match.group(0) for match in pre_tokens]
    #         else:
    #             pre_tokens = doc.split()
    
    #         for token in pre_tokens:
    #             bytes_token = token.encode("utf-8")
                
    #             tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #             token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1
            
    #     return token_count
    
    # #OPED
    # @staticmethod
    # def pretokenize_and_count(docs: list[str], gpt2_regex: bool = False) -> dict[tuple[bytes], int]:
    #     token_count : dict[tuple[bytes], int] = {}
    #     token_count_get = token_count.get
    
    #     for doc in docs:
    #         # pre_tokens = None
    #         # Use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
    #         if gpt2_regex:
    #             for token in PAT.finditer(doc):
    #                 token_str = token.group(0)
    #                 bytes_token = token_str.encode("utf-8")

    #                 tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #                 token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1
    #         else:
    #             for token in doc.split():
    #                 bytes_token = token.encode("utf-8")

    #                 tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #                 token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1

    #         # if gpt2_regex:
    #         #     # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #         #     pre_tokens = PAT.finditer(doc)
    #         #     pre_tokens = [match.group(0) for match in pre_tokens]
    #         # else:
    #         #     pre_tokens = doc.split()
    
    #         # for token in pre_tokens:
    #         #     bytes_token = token.encode("utf-8")
                
    #         #     tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #         #     token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1
            
    #     return token_count
    
    # @staticmethod
    # def pretokenize_and_count(docs: list[str], gpt2_regex: bool = False) -> dict[tuple[bytes], int]:
    #     token_count : dict[tuple[bytes], int] = {}
    #     token_count_get = token_count.get
    
    #     for doc in docs:
    #         # pre_tokens = None
    #         # Use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
    #         if gpt2_regex:
    #             for token in PAT.finditer(doc):
    #                 token_str = token.group(0)
    #                 bytes_token = token_str.encode("utf-8")

    #                 length_bytes_token = len(bytes_token)

    #                 tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (length_bytes_token))
    #                 token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1
    #         else:
    #             for token in doc.split():
    #                 bytes_token = token.encode("utf-8")

    #                 tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #                 token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1

    #         # if gpt2_regex:
    #         #     # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #         #     pre_tokens = PAT.finditer(doc)
    #         #     pre_tokens = [match.group(0) for match in pre_tokens]
    #         # else:
    #         #     pre_tokens = doc.split()
    
    #         # for token in pre_tokens:
    #         #     bytes_token = token.encode("utf-8")
                
    #         #     tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
    #         #     token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1
            
    #     return token_count
    
    @staticmethod
    def pretokenize_and_count(docs: list[str], gpt2_regex: bool = False) -> dict[tuple[bytes], int]:
        token_count : dict[tuple[bytes], int] = {}
        token_count_get = token_count.get
        
        # Build cache mapping pretoken bytes format to tuple byte format 
        cache :dict[bytes, tuple[bytes]] = {}
        cache_get = cache.get
    
        for doc in docs:
            # pre_tokens = None
            # Use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
            if gpt2_regex:
                for token in PAT.finditer(doc):
                    token_str = token.group(0)
                    bytes_token = token_str.encode("utf-8")

                    pretoken_tuplebytes = cache_get(bytes_token)

                    if pretoken_tuplebytes is None:
                        # Build cache
                        length_bytes_token = len(bytes_token)
                        pretoken_tuplebytes = tuple(bytes_token[i : i+1] for i in range (length_bytes_token))
                        cache[bytes_token] = pretoken_tuplebytes
                                        
                    token_count[pretoken_tuplebytes] = token_count_get(pretoken_tuplebytes, 0) + 1
            else:
                for token in doc.split():
                    bytes_token = token.encode("utf-8")

                    tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
                    token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1

            # if gpt2_regex:
            #     # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            #     pre_tokens = PAT.finditer(doc)
            #     pre_tokens = [match.group(0) for match in pre_tokens]
            # else:
            #     pre_tokens = doc.split()
    
            # for token in pre_tokens:
            #     bytes_token = token.encode("utf-8")
                
            #     tuple_bytes_token = tuple(bytes_token[i : i+1] for i in range (len(bytes_token)))
            #     token_count[tuple_bytes_token] = token_count_get(tuple_bytes_token, 0) + 1
            
        return token_count
    
    @staticmethod
    def pretokenize_and_count_task(path: str, start: int, end :int, special_token: list[str], gpt2_regex: bool) -> dict[tuple[bytes], int]:
        # Get chunk
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Remove special token
        docs = BPETokenizerTrainer.remove_special_tokens_static(chunk, special_token)

        # Build pretoken counts dict
        pretoken_counts = BPETokenizerTrainer.pretokenize_and_count(docs, gpt2_regex)

        return pretoken_counts
    
    def pretokenize(self, input_path: str, gpt2_regex: bool) -> dict[tuple[bytes], int]:
        # Read training data
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    
        # Removing special tokens
        docs = self.remove_special_tokens(text)

        # Pre-tokenization
        pretokens = self.pretokenize_and_count(docs, gpt2_regex)

        return pretokens
    
    def pretokenize_parallel(self, path: str, gpt2_regex: bool) -> dict[tuple[bytes], int]:
        # Get logical core number (allow override via env to control memory spikes)
        core_num = int(os.getenv("PRETOKEN_PROCS", os.cpu_count() or 1))

        # Get boundaries of chunks
        # Use provided special tokens if available, fallback to endoftext
        split_tok = (self.special_tokens[0] if self.special_tokens else "<|endoftext|>")
        with open(path, "rb") as f:
            boundaries = BPETokenizerTrainer.find_chunk_boundaries(
                f, core_num, split_tok.encode("utf-8"))
        
        # Parallel pretoken
        with ProcessPoolExecutor(max_workers=core_num) as executor:
            futures = [executor.submit(BPETokenizerTrainer.pretokenize_and_count_task, path, start, end, self.special_tokens, gpt2_regex) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        pretoken_counts = {}
        pretoken_counts_get = pretoken_counts.get

        for future in as_completed(futures):
            for pretoken, count in future.result().items():
                pretoken_counts[pretoken] = pretoken_counts_get(pretoken, 0) + count
        
        return pretoken_counts
  
    # @staticmethod
    # def build_paircount_and_cache(
    #     pretokens : dict[tuple[bytes, ...], int]
    # ) -> tuple[
    #     dict[tuple[bytes], int], 
    #     dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]
    #     ]:
    
    #     pair_count: dict[tuple[bytes], int] = {}
    #     cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]] = defaultdict(set)
    
    #     for k, v in pretokens.items():
    #         for i in range(len(k)-1):
    #             pair_count[k[i : i+2]] = pair_count.get(k[i : i+2], 0) + v
    
    #             cache[k[i : i+2]].add((k, v))
    
    #     return pair_count, cache


    # @staticmethod
    # def build_paircount_and_cache(
    #     pretokens : dict[tuple[bytes, ...], int]
    # ) -> tuple[
    #     dict[tuple[bytes], int], 
    #     dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]
    #     ]:
    
    #     pair_count: dict[tuple[bytes], int] = {}
    #     cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]] = defaultdict(set)
    
    #     for k, v in pretokens.items():
    #         for i in range(len(k)-1):
    #             pair = k[i : i+2]

    #             pair_count[pair] = pair_count.get(pair, 0) + v
    
    #             cache[pair].add((k, v))
    
    #     return pair_count, cache
    
    # @staticmethod
    # def build_paircount_and_cache(
    #     pretokens : dict[tuple[bytes, ...], int]
    # ) -> tuple[
    #     dict[tuple[bytes], int], 
    #     dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]
    #     ]:
    
    #     pair_count: dict[tuple[bytes], int] = {}
    #     cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]] = defaultdict(set)

    #     pc_get = pair_count.get
    
    #     for k, v in pretokens.items():
    #         for i in range(len(k)-1):
    #             pair = k[i : i+2]

    #             # pair_count[pair] = pair_count.get(pair, 0) + v
    #             pair_count[pair] = pc_get(pair, 0) + v

    #             cache[pair].add((k, v))
    
    #     return pair_count, cache
    
    @staticmethod
    def build_paircount_and_cache(
        pretokens : dict[tuple[bytes, ...], int]
    ) -> tuple[
        dict[tuple[bytes], int], 
        dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]
        ]:
    
        pair_count: dict[tuple[bytes], int] = {}
        cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]] = defaultdict(set)

        pc_get = pair_count.get
    
        for k, v in pretokens.items():
            length_k = len(k)-1
            for i in range(length_k):
                pair = k[i : i+2]

                # pair_count[pair] = pair_count.get(pair, 0) + v
                pair_count[pair] = pc_get(pair, 0) + v

                cache[pair].add((k, v))
    
        return pair_count, cache
    
    def build_heap(self, pair_count: dict[tuple[bytes], int]):
        # Prepare for heapify
        self._heap = [(-count, _Desc(pair)) for pair, count in pair_count.items()]

        heapq.heapify(self._heap)
    
    def update_heap(self, changed_paircount: dict[tuple[bytes], int]):
        for pair, count in changed_paircount.items():
            heapq.heappush(self._heap, (-count, _Desc(pair)))
    
    # @staticmethod
    # def _pick_best_mergetoken(pair_count: dict[tuple[bytes], int]) -> tuple[tuple[bytes], int]:
    #     try:
    #         return max(
    #             pair_count.items(),
    #             key = lambda kv: (kv[1], kv[0])
    #         )
    #     except Exception as e:

    #     # Log or print the freqs that caused the failure
    #         print("Error picking best token, pair_count was:", pair_count)
    #         raise
    
    def _pick_best_mergetoken(self, pair_count: dict[tuple[bytes], int]) -> tuple[tuple[bytes], int]:
        while len(self._heap):
            best_heap = self._heap[0]
            pair = best_heap[1].x
            count = -best_heap[0]

            if pair_count.get(pair, 0) == count:
                return (pair, count)
            else:
                heapq.heappop(self._heap)
    
    # @staticmethod
    # def _build_new_pretoken(
    #     old_pretoken: tuple[tuple[bytes, ...], int], 
    #     best_paircount: tuple[bytes, ...]
    #     ) ->  tuple[tuple[bytes, ...], int]:
    
    #     new_pretoken_pair = ()
    #     old_pretoken_pair = old_pretoken[0]
    #     best_pair = best_paircount
    #     i = 0
    
    #     while i < len(old_pretoken_pair)-1:
    #         if old_pretoken_pair[i : i+2] == best_pair:
    #             new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i] + old_pretoken_pair[i+1],)
    
    #             if i == len(old_pretoken_pair)-3:
    #                 new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i+2],)
    
    #             i = i+2
    #         else:
    #             new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i],)
    
    #             if i == len(old_pretoken_pair)-2:
    #                 new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i+1],)
    
    #             i = i+1
        
    #     new_pretoken = (new_pretoken_pair, old_pretoken[1])
    
    #     return new_pretoken
    
    @staticmethod
    def _build_new_pretoken(
        old_pretoken: tuple[tuple[bytes, ...], int], 
        best_paircount: tuple[bytes, ...]
        ) ->  tuple[tuple[bytes, ...], int]:
    
        # new_pretoken_pair = ()
        new_pretoken_pair: list[bytes] = []

        old_pretoken_pair = old_pretoken[0]
        best_pair = best_paircount

        i = 0
        L = len(old_pretoken_pair) - 1
    
        while i < L:
            if old_pretoken_pair[i : i+2] == best_pair:
                # new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i] + old_pretoken_pair[i+1],)
                new_pretoken_pair.append(old_pretoken_pair[i] + old_pretoken_pair[i+1])
    
                # if i == len(old_pretoken_pair)-3:
                #     new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i+2],)
    
                i = i+2
            else:
                # new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i],)
                new_pretoken_pair.append(old_pretoken_pair[i])
    
                # if i == len(old_pretoken_pair)-2:
                #     new_pretoken_pair = new_pretoken_pair + (old_pretoken_pair[i+1],)
    
                i = i+1
            
            if i == L:
                new_pretoken_pair.append(old_pretoken_pair[i])
        
        new_pretoken = (tuple(new_pretoken_pair), old_pretoken[1])
    
        return new_pretoken
    
    # @staticmethod
    # def _delete_old_contribution(
    #     pretoken: tuple[tuple[bytes, ...], int], 
    #     pair_count: dict[tuple[bytes], int], 
    #     reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]
    #     ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]]:

    #     pretoken_pair = pretoken[0]
    #     pretoken_count = pretoken[1]
    
    #     for i in range (len(pretoken_pair)-1):
    #         pair = pretoken_pair[i : i+2]
    
    #         pair_count[pair] = pair_count[pair] - pretoken_count
    #         if pair_count[pair] == 0:
    #             del pair_count[pair]
    
    #         reversed_cache[pair].discard(pretoken)
    #         if not reversed_cache[pair]:
    #             del reversed_cache[pair]
        
    #     return pair_count, reversed_cache
    
    # @staticmethod
    # def _delete_old_contribution(
    #     pretoken: tuple[tuple[bytes, ...], int], 
    #     pair_count: dict[tuple[bytes], int], 
    #     reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]],
    #     changed_paircount: dict[tuple[bytes], int]
    #     ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]], dict[tuple[bytes], int]]:

    #     pretoken_pair = pretoken[0]
    #     pretoken_count = pretoken[1]

    #     changed_paircount_get = changed_paircount.get
    
    #     for i in range (len(pretoken_pair)-1):
    #         pair = pretoken_pair[i : i+2]
    
    #         pair_count[pair] = pair_count[pair] - pretoken_count
            
    #         # Record negative change 
    #         changed_paircount[pair] = changed_paircount_get(pair, 0) - pretoken_count

    #         if pair_count[pair] == 0:
    #             del pair_count[pair]
    
    #         reversed_cache[pair].discard(pretoken)
    #         if not reversed_cache[pair]:
    #             del reversed_cache[pair]
        
    #     return pair_count, reversed_cache, changed_paircount
    
    @staticmethod
    def _delete_old_contribution(
        pretoken: tuple[tuple[bytes, ...], int], 
        pair_count: dict[tuple[bytes], int], 
        reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]],
        changed_paircount: dict[tuple[bytes], int]
        ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]], dict[tuple[bytes], int]]:

        pretoken_pair = pretoken[0]
        pretoken_count = pretoken[1]

        changed_paircount_get = changed_paircount.get
        length_pretoken_pair = len(pretoken_pair)-1
    
        for i in range (length_pretoken_pair):
            pair = pretoken_pair[i : i+2]
    
            pair_count[pair] = pair_count[pair] - pretoken_count
            
            # Record negative change 
            changed_paircount[pair] = changed_paircount_get(pair, 0) - pretoken_count

            if pair_count[pair] == 0:
                del pair_count[pair]
    
            reversed_cache[pair].discard(pretoken)
            if not reversed_cache[pair]:
                del reversed_cache[pair]
        
        return pair_count, reversed_cache, changed_paircount
    
    # @staticmethod
    # def _add_new_contribution(
    #     pretoken: tuple[tuple[bytes, ...], int], 
    #     pair_count: dict[tuple[bytes], int], 
    #     reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]
    #     ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]]:

    #     reversed_cache = defaultdict(set, reversed_cache)
    #     pretoken_pair = pretoken[0]
    #     pretoken_count = pretoken[1]
    
    #     for i in range (len(pretoken_pair)-1):
    #         pair = pretoken_pair[i : i+2]
    
    #         pair_count[pair] = pair_count.get(pair, 0) + pretoken_count
    
    #         reversed_cache[pair].add(pretoken)
        
    #     return pair_count, reversed_cache
    
    @staticmethod
    def _add_new_contribution(
        pretoken: tuple[tuple[bytes, ...], int], 
        pair_count: dict[tuple[bytes], int], 
        reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]],
        changed_paircount: dict[tuple[bytes], int]
        ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]], dict[tuple[bytes], int]]:

        reversed_cache = defaultdict(set, reversed_cache)
        pretoken_pair = pretoken[0]
        pretoken_count = pretoken[1]

        pair_count_get = pair_count.get
        changed_paircount_get = changed_paircount.get

        length_pretoken_pair = len(pretoken_pair)-1
    
        for i in range (length_pretoken_pair):
            pair = pretoken_pair[i : i+2]
    
            pair_count[pair] = pair_count_get(pair, 0) + pretoken_count

            # Record positive change
            changed_paircount[pair] = changed_paircount_get(pair, 0) + pretoken_count
    
            reversed_cache[pair].add(pretoken)
        
        return pair_count, reversed_cache, changed_paircount

    # @staticmethod
    # def merge_new(
    #     pair_counts: dict[tuple[bytes], int], 
    #     reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]],
    #     best_pair: tuple[bytes, ...]
    # ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]]]:

    #     affected_pretokens = reversed_cache[best_pair].copy()
    
    #     for old_pretoken in affected_pretokens:
    #         new_pretoken = Tokenizer._build_new_pretoken(old_pretoken, best_pair)
    
    #         # Update, delete old pretoken contribution
    #         pair_counts, reversed_cache = Tokenizer._delete_old_contribution(old_pretoken, pair_counts, reversed_cache)
    #         # update, add new pretoken contrbution
    #         pair_counts, reversed_cache = Tokenizer._add_new_contribution(new_pretoken, pair_counts, reversed_cache)

    #     return pair_counts, reversed_cache
    
    @staticmethod
    def merge_new(
        pair_counts: dict[tuple[bytes], int], 
        reversed_cache: dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]],
        best_pair: tuple[bytes, ...]
    ) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes, ...], set[tuple[tuple[bytes, ...], int]]], dict[tuple[bytes], int]]:

        affected_pretokens = reversed_cache[best_pair].copy()

        delta_changed_paircount: dict[tuple[bytes], int] = {}
    
        for old_pretoken in affected_pretokens:
            new_pretoken = BPETokenizerTrainer._build_new_pretoken(old_pretoken, best_pair)
    
            # Update, delete old pretoken contribution
            pair_counts, reversed_cache, delta_changed_paircount = BPETokenizerTrainer._delete_old_contribution(old_pretoken, pair_counts, reversed_cache, delta_changed_paircount)
            # Update, add new pretoken contrbution
            pair_counts, reversed_cache, delta_changed_paircount = BPETokenizerTrainer._add_new_contribution(new_pretoken, pair_counts, reversed_cache, delta_changed_paircount)
        
        # Build changed pair count dict
        changed_paircount = {}

        for changed_pair, changed_count in delta_changed_paircount.items():
            # If the changed count is zero, which means no changed, should not include here
            if changed_count and changed_pair in pair_counts:
            # if changed_count and pair_counts.get(changed_pair, 0) != 0:
                changed_paircount[changed_pair] = pair_counts[changed_pair]

        return pair_counts, reversed_cache, changed_paircount
    
    
    def update_vocab(self, best_pair: tuple[tuple[bytes], int]):
        # sorted_vocab = sorted(self.vocab.items(), reverse=True)
        # new_index =  sorted_vocab[0][0] + 1
        
        k = best_pair[0]
        k = k[0] + k[1]
    
        self.vocab[self.next_id] = k

        self.next_id += 1
    
    def _save_vocabulary_merges(self):
        """
        Serialize the resulting vocabulary and merges to disk for further inspection

        Args:

        Returns:
        """
        vocab_serialized = {
            str(token_id): vocab_bytes.decode("utf-8", "surrogateescape")
            for token_id, vocab_bytes in self.vocab.items()
        }

        merge_serialized = [
            [first_bytes.decode("utf-8", "surrogateescape"), second_bytes.decode("utf-8", "surrogateescape")]
            for first_bytes, second_bytes in self.merge
        ]

        # Create path
        self.serialization_vocab_path = self.output_path / f"{self.traindata_path.stem}_serialization_vocab_{self.timestamp}.json"
        self.serialization_merge_path = self.output_path / f"{self.traindata_path.stem}_serialization_merge_{self.timestamp}.json"


        with open(self.serialization_vocab_path, 'w', encoding="utf-8", errors="surrogatepass") as f:
            json.dump(vocab_serialized, f, indent=2, ensure_ascii=False)
        
        with open(self.serialization_merge_path, 'w', encoding="utf-8", errors="surrogatepass") as f:
            json.dump(merge_serialized, f, indent=2, ensure_ascii=False)
    
    # def train_bpe(self, input_path: str, vocab_size: int, gpt2_regex: bool, enable_parallel: bool) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #     # Init vocab
    #     self.init_vocab()

    #     if enable_parallel:
    #         pretokens = self.pretokenize_parallel(input_path, gpt2_regex)
    #     else:
    #         pretokens = self.pretokenize(input_path, gpt2_regex)
      
    #     # Build the first pair count and cache(pair to corresponding pretokens)
    #     pair_counts, reversed_cache = self.build_paircount_and_cache(pretokens)
    
    #     for i in range(vocab_size - 256 - len(self.special_tokens)):
    #         # Pick best adjcent tokens to merge
    #         best_pair = self._pick_best_mergetoken(pair_counts)
    
    #         # Log pair counts, best pair and step
    #         self.dump_pair_count(pair_counts, best_pair, i)
    
    #         # Update pair counts and cache
    #         pair_counts,  reversed_cache = self.merge_new(pair_counts, reversed_cache, best_pair[0])
    
    #         # TODO: optimize point, insert vocab and merges two times
    #         # Update vocabs
    #         self.update_vocab(best_pair)
    #         # Update merges
    #         self.merge.append((best_pair[0][0], best_pair[0][1]))
        
    #     return self.vocab, self.merge

    def train_bpe(self, input_path: str, vocab_size: int, gpt2_regex: bool, enable_parallel: bool) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.traindata_path = Path(input_path)

        # Init vocab
        self.init_vocab()

        if enable_parallel:
            pretokens = self.pretokenize_parallel(input_path, gpt2_regex)
        else:
            pretokens = self.pretokenize(input_path, gpt2_regex)
      
        # Build the first pair count and cache(pair to corresponding pretokens)
        pair_counts, reversed_cache = self.build_paircount_and_cache(pretokens)

        self.build_heap(pair_counts)

        merge_size = vocab_size - 256 - len(self.special_tokens)
    
        for i in range(merge_size):
            # Pick best adjcent tokens to merge
            best_pair = self._pick_best_mergetoken(pair_counts)
    
            # Log pair counts, best pair and step
            self.dump_pair_count(pair_counts, best_pair, i)
    
            # Update pair counts and cache
            pair_counts, reversed_cache, changed_paircount = self.merge_new(pair_counts, reversed_cache, best_pair[0])

            self.update_heap(changed_paircount)
    
            # TODO: optimize point, insert vocab and merges two times
            # Update vocabs
            self.update_vocab(best_pair)
            # Update merges
            self.merge.append((best_pair[0][0], best_pair[0][1]))
        
        if self.serialization:
            self._save_vocabulary_merges()
        
        return self.vocab, self.merge
