
import torch
from torch import LongTensor, nn
from transformers import BertTokenizerFast

from data import DataMaker
from modules import BertConfig, BertPreTraining
from modelling import MLMPreprocessor, LanguageModelTrainer


NLAYERS = 6
MAXSEQLEN = 200
MASK_PCT = 0.15
EPOCHS = 10
BATCH_SIZE = 4
ACCUMULATION_K = 3
PROGRESS_K = 100
MAX_TRAIN_PAIRS = 5000


bert = BertPreTraining(BertConfig(nlayers=NLAYERS, maxseqlen=MAXSEQLEN))
datamaker = DataMaker(max_pairs=MAX_TRAIN_PAIRS)

preprocessor = MLMPreprocessor(
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased"), 
    maxseqlen=MAXSEQLEN,
    mask_percentage=MASK_PCT)

trainloader, validloader = preprocessor.create_dataloaders(
    trainpairs=datamaker.traindata,
    validpairs=datamaker.validdata,
    batch_size=BATCH_SIZE)

trainer = LanguageModelTrainer(
    model=bert,
    preprocessor=preprocessor,
    device=torch.device("cuda"))

trainer.commence_training(trainloader, validloader, epochs=EPOCHS, accumulation_k=ACCUMULATION_K, progress_k=PROGRESS_K)
