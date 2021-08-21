
import os
from dataclasses import dataclass
from typing import List, Tuple
import requests
import zipfile
import io


from tqdm import tqdm


WIKI103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
WIKI103_FOLDER = "wikitext-103"
TRAIN_DATA = "wiki.train.tokens"
VALID_DATA = "wiki.valid.tokens"
BADUNK1 = "< unk >"
BADUNK2 = "<unk>"
ATSIGN = "@"
UNKTOKEN = "[UNK]"
SENTENCE_SEPARATOR = " . "
MIN_LINE_CHARS = 80



@dataclass
class SentencePair:
    """
    Convenient dataclass to hold our sentence pairs for Next Sentence Prediction & Masked Language
    Modelling objectives. Holds a pair of related or unrelated (sentence1, sentence2)
    sentences. If sentences are related, `classlabel==1`, else `classlabel==0` if they are unrelated
    (i.e. randomly paired sentences)
    """
    first: str
    second: str
    classlabel: int

    def to_tuple(self) -> Tuple[str, str]:
        return (self.first, self.second)



class DataMaker:
    """
    Helper class to handle downloading and preprocessing our Language Modelling
    dataset into a list of SentencePair of objects, which makes conversion into
    Tensors much simpler for downstream operations.
    """
    def __init__(self, datafolder="data/", max_train_pairs=10000, max_valid_pairs=1000) -> None:
        self.datafolder = datafolder
        self.maxtrainpairs = max_train_pairs
        self.maxvalidpairs = max_valid_pairs
        self.traindata = list()
        self.validdata = list()

    def build(self) -> Tuple[List[SentencePair], List[SentencePair]]:
        """
        Main function to initiate making of the data.
        Peforms the following steps:
            - download data if not exists
            - build train / validation lists of SentencePairs
            - Store and return the tuple of (TrainList, ValidationList)
        """
        folder = self.datafolder
        if not (TRAIN_DATA and VALID_DATA) in os.listdir(f"{folder}/{WIKI103_FOLDER}"):
            self._download_data(folder)
        self.traindata = self._create_data(f"{folder}/{WIKI103_FOLDER}/{TRAIN_DATA}", max_pairs=self.maxtrainpairs)
        self.validdata = self._create_data(f"{folder}/{WIKI103_FOLDER}/{VALID_DATA}", max_pairs=self.maxvalidpairs)
        return (self.traindata, self.validdata)

    def _download_data(self, datafolder: str) -> None:
        """
        Download Wiki103 dataset from an online-hosted zip file, extracts contents to given `datafolder` directory.
        See page at https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
        """
        print(f"No training data found - downloading {WIKI103_URL} from source (~180MB size)")
        resp = requests.get(url=WIKI103_URL, stream=True)
        byte = io.BytesIO(initial_bytes=resp.content)
        zipp = zipfile.ZipFile(file=byte)
        zipp.extractall(path=datafolder)

    def _create_data(self, fpath: str, max_pairs: int) -> List[SentencePair]:
        """
        Main function to creates a list of SentencePair objects from a given Corpus,
        which will be fed downstream for preprocessing into DataLoaders.
        """
        with open(fpath, encoding="utf8") as file:
            lines = file.readlines()
        # Create positive pair groups
        pos_groups = self._create_positive_groups(lines, max_pairs=max_pairs//2)
        # Now create negative pairs of sentences using positive pairs
        neg_groups = self._create_negative_groups(pos_groups, max_pairs=max_pairs//2)
        # Convenience function for flattening List[List[Pairs]] -> List[Pairs]
        flatten = lambda lstlstpairs: [pair for lstpairs in lstlstpairs for pair in lstpairs]
        # Concatenate the two groups
        combined = flatten(pos_groups) + flatten(neg_groups)
        return combined[ :max_pairs]

    def _create_positive_groups(self, lines: List[str], max_pairs: int) -> List[List[SentencePair]]:
        """
        Creates groups of Positive Pairs from the corpus by pairing together
        sentences [i] with sentences [i+1] within the same paragraph of text.
        Will create pairs up to the given `max_pairs` limit.

        For instance:
            - Original paragraph: "First he warmed up. Then he ran around the park. Then down the road. Then over the bridge."
            - Output: [
                [
                    ("First he warmed up", "Then he ran around the park"),
                    ("Then he ran around the park", "Then down the road"),
                    ("Then down the road", "Then over the bridge"),
                ]
            ]
        """
        pair_groups: List[List[SentencePair]] = []
        num_pairs = 0
        for line in tqdm(lines):
            if num_pairs < max_pairs:
                (valid, sentences) = self._is_valid(line)
                if valid:
                    sents = self._clean_sentences(sentences)
                    pairs = [SentencePair(sents[i], sents[i+1], classlabel=1) for i in range(len(sents)-1)]
                    pair_groups.append(pairs)
                    num_pairs += len(pairs)
        return pair_groups

    def _clean_sentences(self, sentences: List[str]) -> List[str]:
        """
        Replaces data-specific unknown tokens with recognized unknown tokens,
        and replaces @ signs with spaces.
        """
        sents = [s.replace(BADUNK1, UNKTOKEN) for s in sentences]
        sents = [s.replace(BADUNK2, UNKTOKEN) for s in sentences]
        sents = [s.replace(ATSIGN, " ") for s in sentences]
        return sents

    def _is_valid(self, line: str) -> Tuple[bool, List[str]]:
        """
        Checks if a given line/paragraph should be processed further,
        or ignored entirely. If should be processed, will return a
        tuple containing the paragraph splitted on sentences boundaries.
        """
        if not line.startswith(" = "):
            line = line.strip()
            if line:
                splitted = line.split(SENTENCE_SEPARATOR)
                if len(splitted) > 1 and len(line) > MIN_LINE_CHARS:
                    return (True, splitted)
        return (False, list())

    def _create_negative_groups(self, pos_groups: List[List[SentencePair]], max_pairs: int, offset=50) -> List[List[SentencePair]]:
        """
        Given a list of positive sentence pair groupings, synthetically creates negative
        pairs from those, by pairing up the halves of those positive pairs with random
        halves from other pairs elsewhere in the corpus. How far we 'jump' forward to find
        those random pairs is determined by the `minoffset` and `maxoffset` params.
        """
        neg_groups = []
        num_pairs = 0
        for i in tqdm(range(len(pos_groups))):
            if num_pairs < max_pairs:
                # Pull out the two chosen pair groups - lets call these 'Left' and 'Right'
                rightidx = i + offset
                l, r = pos_groups[i], pos_groups[rightidx % len(pos_groups)]
                # Create iterators for the various first/second items of the selected pairs
                l_firsts, l_seconds = (pair.first for pair in l), (pair.second for pair in l)
                r_firsts, r_seconds = (pair.first for pair in r), (pair.second for pair in r)
                pairs = (
                    # Pair the first halves of the 'Left' group with second halves of the 'Right' group
                    [SentencePair(f, s, 0) for (f, s) in zip(l_firsts, r_seconds)] +
                    # Pair the first halves of the 'Right' group with second halves of the 'Left' group
                    [SentencePair(f, s, 0) for (f, s) in zip(r_firsts, l_seconds)])
                neg_groups.append(pairs)
                num_pairs += len(pairs)
        return neg_groups
