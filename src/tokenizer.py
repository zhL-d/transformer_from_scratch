import regex as re
import logging
import json
from collections.abc import Iterable
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# )

logger = logging.getLogger(__name__)


PAT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT_PATTERN)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: dict[int, bytes]:

            merges: list[tuple[bytes, bytes]]

            special_tokens: list[str] | None = None
        """

        logger.debug("Starting construct tokenizer")

        self.vocab = vocab
        self.reverse_vocab = {token: id for id, token in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

        logger.debug(f"Constructed tokenizer with vocab, merges, special tokens: vocab={self.vocab}, merges={self.merges}, special tokens={self.special_tokens}")
        logger.info("Construct tokenizer complete")
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):

        logger.debug("Starting construct tokenizer from files")

        with open(vocab_filepath, encoding="utf-8", errors="surrogatepass") as vf:
            vocab_raw = json.load(vf)
            vocab_serialized = {int(token_id): token.encode("utf-8", errors="surrogateescape") for token_id, token in vocab_raw.items()}
        with open(merges_filepath, encoding="utf-8", errors="surrogatepass") as mf:
            merges_raw = json.load(mf)
            merges_serialized = [(merge[0].encode("utf-8", errors="surrogateescape"), merge[1].encode("utf-8", errors="surrogateescape")) for merge in merges_raw]
        
        logger.debug(f"Constructed from files vocab, merge: vocab={vocab_serialized}, merges={merges_serialized}")
        logger.info("Construct tokenizer from files complete")

        return cls(vocab_serialized, merges_serialized, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            test: Corpus used for tokenization

        Returns:
            List of tokenid corresponding to the tokenized corpus
        """
        logger.debug(f"Starting encode: text_length={len(text)}, corpus_preview={text[:50]}")

        token_ids: list[int] = []

        if self.special_tokens:
            
            logger.debug(f"Special token has set: {self.special_tokens}")

            corpus_parts = self._spillt_by_specialtoken(text)

            logger.debug(f"Corpus parts: {corpus_parts}")

            for part_idx, part in enumerate(corpus_parts):        
                # TODO: optimize performance
                if part in self.special_tokens:

                    logger.debug(f"Part[{part_idx}]: '{part}' is special token")

                    token_id = self._encode_specialtoken(part)

                    logger.debug(f"Mapping token id: Token={part} -> Token_id={token_id}")

                    token_ids.append(token_id)
                else:
                    logger.debug(f"Part[{part_idx}]: '{part}' is normal part")

                    part_token_ids = self._encode_nomal(part)

                    token_ids.extend(part_token_ids)
        else:
            logger.debug("Special token has NOT set")

            token_ids = self._encode_nomal(text)

        logger.info("Encoding complete")

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:

        logger.debug("Starting iterable encode:")

        for chunk_idx, chunk in enumerate(iterable):
            logger.debug(f"Process chunk[{chunk_idx}]: text_length={len(chunk)}, corpus_preview={chunk[:50]}")

            if self.special_tokens:
                
                logger.debug(f"Special token has set: {self.special_tokens}")

                corpus_parts = self._spillt_by_specialtoken(chunk)

                logger.debug(f"Corpus parts: {corpus_parts}")

                for part_idx, part in enumerate(corpus_parts):        
                    # TODO: optimize performance
                    if part in self.special_tokens:

                        logger.debug(f"Part[{part_idx}]: '{part}' is special token")

                        token_id = self._encode_specialtoken(part)

                        logger.debug(f"Mapping token id: Token={part} -> Token_id={token_id}")

                        yield token_id
                    else:
                        logger.debug(f"Part[{part_idx}]: '{part}' is normal part")

                        pretokens = Tokenizer.pretokenize(part)

                        for idx, pretoken in enumerate(pretokens):
                            # May include multi tokens
                            tokenized_pretoken = self.tokenize(pretoken)

                            logger.debug(f"Tokenizing: Pretoken[{idx}]={pretoken} -> Tokenized Pretoken={tokenized_pretoken}")

                            for token_idx, token in enumerate(tokenized_pretoken):
                                try:
                                    token_id = self.reverse_vocab[token]
                                except KeyError:
                                    raise ValueError(
                                        f"Token {token} not found in vocabulary table."
                                    )

                                logger.debug(f"Mapping token id: Token[{token_idx}]={token} -> Token_id={token_id}")

                                yield token_id                      
            else:
                logger.debug("Special token has NOT set")

                pretokens = Tokenizer.pretokenize(chunk)

                for idx, pretoken in enumerate(pretokens):
                    # May include multi tokens
                    tokenized_pretoken = self.tokenize(pretoken)

                    logger.debug(f"Tokenizing: Pretoken[{idx}]={pretoken} -> Tokenized Pretoken={tokenized_pretoken}")

                    for token_idx, token in enumerate(tokenized_pretoken):
                        try:
                            token_id = self.reverse_vocab[token]
                        except KeyError:
                            raise ValueError(
                                f"Token {token} not found in vocabulary table."
                            )

                        logger.debug(f"Mapping token id: Token[{token_idx}]={token} -> Token_id={token_id}")

                        yield token_id

            logger.debug(f"Encoding chunk[{chunk_idx}] complete")
        logger.info("Iterable encoding complete")
    
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text

        Args:
            ids: a sequence of token IDs

        Returns:
            text 
        """
        logger.debug(f"Starting decode: length={len(ids)}, preview token ids={ids[:20]}")

        text_bytes: bytes = b""

        for token_id in ids:
            try:
                token_bytes = self.vocab[token_id]
            except KeyError:
                raise ValueError(
                    f"Token id {token_id} is not found in vocabulary table"
                )
            
            logger.debug(f"Maping token id to token bytes: token id={token_id} -> token bytes={token_bytes}")
            
            # TODO: optimize performence
            text_bytes = text_bytes + token_bytes
        
        logger.debug(f"Converting token bytes to text: length={len(text_bytes)}, preview token bytes={text_bytes[:20]}")

        decoded_text = text_bytes.decode("utf-8", errors='replace')

        logger.debug(f"Decoded text: preview token ids={ids[:20]} -> preview decoded text={decoded_text[:20]}")
        logger.info("Decode complete")

        return decoded_text
        

    @staticmethod
    def pretokenize(text: str, is_gpt: bool = True) -> list[list[bytes]]:
        """Pre-tokenize the corpus and represent each pre-token as a list of UTF-8 bytes

        Args:
            text: Corpus used for tokenization
            is_gpt: Whether use gpt pattern to pretokenize corpus

        Returns:
            List of pretokens
        """
        logger.debug(f"Starting pretokenize: text_length={len(text)}, gpt_pattern={is_gpt}")

        pretokens: list[list[bytes]] = []

        if is_gpt:
            for pretoken in PAT.finditer(text):
                pretoken_str = pretoken.group(0)
                pretoken_bytes = pretoken_str.encode("utf-8")

                pretoken_byteslist: list[bytes] = []
                
                pretoken_byteslist = [bytes([pretoken_byte]) for pretoken_byte in pretoken_bytes]

                pretokens.append(pretoken_byteslist)
        else:
            pass

        logger.debug(f"Pretokenization result: {pretokens}")
        logger.info("Pretokenization complete")

        return pretokens

    def tokenize(self, pretoken: list[bytes]) -> list[bytes]:
        """Tokenize every pretoken

        Args:
            pretoken: Single pretoken

        Returns:
            Tokenized pretoken
        """

        # logger.debug(f"Start tokenize: pretoken={pretoken}")

        for merge_idx, merge in enumerate(self.merges):

            logger.debug(f"Start merge[{merge_idx}]: merge item={merge}, pretoken={pretoken}")

            i = 0
            while i < len(pretoken)-1:
                if pretoken[i:i+2] == list(merge):    
                    token = pretoken[i] + pretoken[i+1]
                    temp_pretoken = pretoken[:i] + [token]
                    if i+1 == len(pretoken)-1:
                        pretoken = temp_pretoken
                    else:
                        pretoken = temp_pretoken + pretoken[i+2:]
                    i += 1
                else:
                    i += 1

            # logger.debug(f"Complete merge[{merge_idx}]: merge item={merge}, pretoken{pretoken}")
        
        # logger.debug(f"Finish tokenize: tokenized pretoken={pretoken}")
        return pretoken
    
    def _spillt_by_specialtoken(self, text: str) -> list[str]:
        """Spillt corpus into parts by special tokens and keep the special tokens in resulting parts list

            Args:
                corpus: Corpus used to tokenize

            Returns:
                Corpus in parts list format while keeping special tokens in the parts list
        """
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_specialtokens = [re.escape(stoken) for stoken in sorted_special_tokens]
        pattern = "(" + "|".join(escaped_specialtokens) + ")"
        return re.split(pattern, text)
    
    def _encode_specialtoken(self, part: str) -> int:
        specialtoken_bytes = part.encode("utf-8")
        try:
            token_id = self.reverse_vocab[specialtoken_bytes]
        except KeyError:
            raise ValueError(
                f"Special token {part} is not found in vocabulary table"
            )
        return token_id
    
    def _encode_nomal(self, part: str) -> list[int]:
        token_ids: list[int] = []

        pretokens = Tokenizer.pretokenize(part)

        for idx, pretoken in enumerate(pretokens):
            # May include multi tokens
            tokenized_pretoken = self.tokenize(pretoken)

            logger.debug(f"Tokenizing: Pretoken[{idx}]={pretoken} -> Tokenized Pretoken={tokenized_pretoken}")

            for token_idx, token in enumerate(tokenized_pretoken):
                try:
                    token_id = self.reverse_vocab[token]
                except KeyError:
                    raise ValueError(
                        f"Token {token} not found in vocabulary table."
                    )

                logger.debug(f"Mapping token id: Token[{token_idx}]={token} -> Token_id={token_id}")

                token_ids.append(token_id)
        return token_ids