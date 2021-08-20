
#%%

from typing import List, Tuple, Union

import torch
import numpy as np
from torch import LongTensor
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from data import SentencePair


class MLMPreprocessor:
    def __init__(self, tokenizer: BertTokenizer, maxseqlen: int, mask_percentage=0.10) -> None:
        self.tokenizer = tokenizer
        self.maskpct = mask_percentage
        self.maxseqlen = maxseqlen

    @property
    def paddingidx(self) -> int:
        return self.tokenizer.pad_token_id

    def decode(self, tokens: Union[int, List[int], np.ndarray, LongTensor]) -> str:
        return self.tokenizer.decode(tokens)

    def create_dataloaders(self, trainpairs: List[SentencePair], validpairs: List[SentencePair], batch_size: int) -> Tuple[DataLoader, DataLoader]:
        return (
            self._preprocess_pairs(trainpairs, batch_size),
            self._preprocess_pairs(validpairs, batch_size))

    def _preprocess_pairs(self, sentence_pairs: List[SentencePair], batch_size: int) -> DataLoader:
        sentence_tuples = [pair.to_tuple() for pair in sentence_pairs]
        enc = self.tokenizer.batch_encode_plus(sentence_tuples, add_special_tokens=True, max_length=self.maxseqlen, padding=True, truncation=True)
        stack = lambda lstlst: torch.stack([LongTensor(lst) for lst in lstlst])
        input_ids, attention_mask, token_type_ids = [stack(enc[key]) for key in ("input_ids", "attention_mask", "token_type_ids")]
        masked_input_ids = self._mask_input_ids(input_ids)
        class_labels = torch.LongTensor([pair.classlabel for pair in sentence_pairs])
        return DataLoader(
            shuffle=True,
            batch_size=batch_size,
            dataset=TensorDataset(
                masked_input_ids,
                attention_mask,
                token_type_ids,
                input_ids,
                class_labels))

    def _mask_input_ids(self, input_ids: LongTensor) -> LongTensor:
        """
        For each sentence in the input_id batches, randomly masks it with a % chance of `self.maskpct`.
        Returns a new LongTensor containing the masked inputs
        """
        output = torch.zeros(input_ids.shape, dtype=torch.long)
        for i in range(len(input_ids)):
            # Clone tensor, convert to numpy array
            arr = input_ids[i].clone().numpy()
            # Get index of [SEP] tokens - don't want to replace those
            sep1, sep2 = np.where(arr == self.tokenizer.sep_token_id)[0]
            # Select random indices between [CLS] ---- [SEP] ---- [SEP]
            valid_inds = np.concatenate([np.arange(1, sep1), np.arange(sep1+1, sep2)])
            select_inds = np.random.choice(valid_inds, size=int(self.maskpct * sep2))
            # Overwrite selected indices with [MASK]
            arr[select_inds] = self.tokenizer.mask_token_id
            # Add to output tensor
            output[i] = LongTensor(arr)
        return output
