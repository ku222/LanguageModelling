
from tqdm import tqdm
from typing import List, Tuple
from dataclasses import dataclass
import random


BADUNK1 = "< unk >"
BADUNK2 = "<unk>"
ATSIGN = "@"
UNKTOKEN = "[UNK]"
SENTENCE_SEPARATOR = " . "
MIN_LINE_CHARS = 80


@dataclass
class SentencePair:
    first: str
    second: str
    classlabel: int

    def to_tuple(self) -> Tuple[str, str]:
        return (self.first, self.second)


class DataMaker:
    def __init__(self, max_pairs=100000, traincorpus_fpath="data/wiki.train.tokens", validcorpus_fpath="data/wiki.valid.tokens") -> None:
        self.traindata = self.create_data(traincorpus_fpath, max_pairs=max_pairs)
        self.validdata = self.create_data(validcorpus_fpath, max_pairs=100)

    def create_data(self, fpath: str, max_pairs: int) -> List[SentencePair]:
        with open(fpath, encoding="utf8") as file:
            lines = file.readlines()
        # Create positive pair groups
        pos_groups = self.create_positive_groups(lines, max_pairs=max_pairs//2)
        # Now create negative pairs of sentences using positive pairs
        neg_groups = self.create_negative_groups(pos_groups, max_pairs=max_pairs//2)
        # Convenience function for flattening List[List[Pairs]] -> List[Pairs]
        flatten = lambda lstlstpairs: [pair for lstpairs in lstlstpairs for pair in lstpairs]
        # Concatenate the two groups
        combined = flatten(pos_groups) + flatten(neg_groups)
        return combined[ :max_pairs]

    def create_positive_groups(self, lines: List[str], max_pairs: int) -> List[List[SentencePair]]:
        pair_groups: List[List[SentencePair]] = []
        num_pairs = 0
        for line in tqdm(lines):
            if num_pairs < max_pairs:
                (valid, sentences) = self.is_valid(line)
                if valid:
                    sents = self.clean_sentences(sentences)
                    pairs = [SentencePair(sents[i], sents[i+1], classlabel=1) for i in range(len(sents)-1)]
                    pair_groups.append(pairs)
                    num_pairs += len(pairs)
        return pair_groups
    
    def clean_sentences(self, sentences: List[str]) -> List[str]:
        sents = [s.replace(BADUNK1, UNKTOKEN) for s in sentences]
        sents = [s.replace(BADUNK2, UNKTOKEN) for s in sentences]
        sents = [s.replace(ATSIGN, " ") for s in sentences]
        return sents
    
    def is_valid(self, line: str) -> Tuple[bool, List[str]]:
        if not line.startswith(" = "):
            line = line.strip()
            if line:
                splitted = line.split(SENTENCE_SEPARATOR)
                if len(splitted) > 1 and len(line) > MIN_LINE_CHARS:
                    return (True, splitted)
        return (False, None)

    def create_negative_groups(self, pos_groups: List[List[SentencePair]], max_pairs: int) -> List[List[SentencePair]]:
        minoffset, maxoffset = 50, 500
        neg_groups = []
        num_pairs = 0
        for i in tqdm(range(len(pos_groups))):
            if num_pairs < max_pairs:
                randidx = random.randint(i+minoffset, i+maxoffset)
                l, r = pos_groups[i], pos_groups[randidx % len(pos_groups)]
                l_firsts, l_seconds = (pair.first for pair in l), (pair.second for pair in l)
                r_firsts, r_seconds = (pair.first for pair in r), (pair.second for pair in r)
                pairs = (
                    [SentencePair(f, s, 0) for (f, s) in zip(l_firsts, r_firsts)] +
                    [SentencePair(f, s, 0) for (f, s) in zip(l_seconds, r_seconds)])
                neg_groups.append(pairs)
                num_pairs += len(pairs)
        return neg_groups
