
import torch
from transformers import BertTokenizerFast
import argparse

from data import DataMaker
from modules import BertConfig, BertPreTraining
from modelling import MLMPreprocessor, LanguageModelTrainer



def main(args: argparse.Namespace) -> None:
    # Initialise Bert Pretraining Model
    bert = BertPreTraining(BertConfig(nlayers=args.nlayers, maxseqlen=args.maxseqlen))
    
    # Create DataMaker to preprocess/prepare our WikiText training data into sentence pairs
    datamaker = DataMaker(max_pairs=args.max_train_pairs)

    # Create Data Preprocessor to perform [MASK]ing and create DataLoaders
    preprocessor = MLMPreprocessor(
        tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased"), 
        maxseqlen=args.maxseqlen,
        mask_percentage=args.maskpct)

    # Create dataloaders from preprocessor
    trainloader, validloader = preprocessor.create_dataloaders(
        trainpairs=datamaker.traindata,
        validpairs=datamaker.validdata,
        batch_size=args.batchsize)

    # Instantiate Trainer class to wrap around model
    trainer = LanguageModelTrainer(
        model=bert,
        preprocessor=preprocessor,
        device=torch.device("cuda"))

    # Commence training loop
    trainer.commence_training(
        trainloader,
        validloader,
        epochs=args.epochs,
        accumulation_k=args.accumulation_k,
        progress_k=args.progress_k)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayers", default=6, help="Number of Bert self-attention layers")
    parser.add_argument("--maxseqlen", default=80, help="Maximum number of tokens in training data fed to the model")
    parser.add_argument("--maskpct", default=0.15, help="Percentage of training tokens replaced with [MASK]")
    parser.add_argument("--epochs", default=20, help="Number of epochs to train for")
    parser.add_argument("--batchsize", default=8, help="Minibatch size for training")
    parser.add_argument("--accumulation_k", default=4, help="Number of minibatches to accumulate gradients on before taking a step() with optimizer")
    parser.add_argument("--progress_k", default=100, help="Number of minibatches to pass through before logging current average epoch loss values")
    parser.add_argument("--max_train_pairs", default=10000, help="Number of sentence pairs (training examples) to include in training set out of the whole text corpus")
    (known_args, _) = parser.parse_known_args()
    main(args=known_args)
