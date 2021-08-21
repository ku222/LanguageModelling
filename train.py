
import argparse
import functools
import os


import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import BertTokenizerFast

from data import DataMaker
from modelling import LanguageModelTrainer, MLMPreprocessor
from modules import BertConfig, BertPreTraining


class Namespace:
    def __init__(self, **kwargs) -> None:
        for (varname, value) in kwargs.items():
            setattr(self, varname, value)



def setup(master_addr: str, master_port: str, backend: str, rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method="env://")


def cleanup():
    dist.destroy_process_group()


def train_distributed(rank: int, **kwargs) -> None:
    # Place kwargs into a Namespace object
    args = Namespace(**kwargs)
    
    # Set Google Application credentials
    if args.gcredentialspath:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gcredentialspath

    # Setup distributed training env
    setup(args.master_addr, args.master_port, args.backend, rank, args.world_size)

    # Initialise Bert Pretraining Model
    bert = BertPreTraining(BertConfig(nlayers=args.nlayers, maxseqlen=args.maxseqlen))

    # Create DataMaker to preprocess/prepare our WikiText training data into sentence pairs
    datamaker = DataMaker(datafolder=args.datafolder, max_train_pairs=args.max_train_pairs, max_valid_pairs=args.max_valid_pairs)
    (traindata, validdata) = datamaker.build()

    # Create Data Preprocessor to perform [MASK]ing and create DataLoaders
    preprocessor = MLMPreprocessor(
        tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased"),
        maxseqlen=args.maxseqlen,
        mask_percentage=args.maskpct)

    # Create dataloaders from preprocessor
    (trainloader, validloader) = preprocessor.create_dataloaders(
        trainpairs=traindata,
        validpairs=validdata,
        batch_size=args.batchsize)

    # Instantiate Trainer class to wrap around model
    trainer = LanguageModelTrainer(
        model=bert,
        preprocessor=preprocessor,
        rank=rank)

    # Set checkpointing strategy
    trainer.set_checkpoint_strategy(
        localpath=args.saved_weights_path,
        gcpbucketname=args.gcpbucketname,
        blobname=args.gcpblobname)
    
    # Set logging strategy
    trainer.set_logging_strategy(use_cloud_logging=True)

    # Load model weights if weights exist & user has desire to continue
    if (args.continue_from_saved and trainer.has_existing_weights()):
        print("\n Continuing from existing weight checkpoint \n")
        trainer.load_weights(args.saved_weights_path)

    # Commence training loop
    print("\n Commencing Training Loop \n")
    trainer.commence_training(
        trainloader=trainloader,
        validloader=validloader,
        epochs=args.epochs,
        accumulation_k=args.accumulation_k,
        progress_k=args.progress_k,
        checkpoint_k=args.checkpoint_k)

    # Destroy process group upon finishing
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcredentialspath",       type=str,   default="",                         help="Path to Google Application Credentials")
    parser.add_argument("--master_addr",            type=str,   default="localhost",                help="Address of the Rank 0 Node")
    parser.add_argument("--master_port",            type=str,   default="8880",                     help="Free Port on machine with rank 0")
    parser.add_argument("--world_size",             type=int,   default=2,                          help="Number of distributed GPUs/processes to use")
    parser.add_argument("--backend",                type=str,   default="gloo",                     help="Backend library to use for distributed processing")
    parser.add_argument("--continue_from_saved",    type=bool,  default=False,                      help="Whether to continue from existing model checkpoint or not")
    parser.add_argument("--saved_weights_path",     type=str,   default="saved_weights/bert.pt",    help="Local location to save checkpoint weights of the trained model to")
    parser.add_argument("--gcpbucketname",          type=str,   default="ptorchdistributed",        help="Name of the GCP bucket to store model weights in")
    parser.add_argument("--gcpblobname",            type=str,   default="bert.pt",                  help="Name of the blob file to store model weights in")
    parser.add_argument("--use_cloud_logging",      type=bool,  default=True,                       help="Whether to use Cloud Logging for message logging")
    parser.add_argument("--datafolder",             type=str,   default="data/",                    help="Relative folder reference to find or store the Wiki103 language modelling corpus used for training the model")
    parser.add_argument("--nlayers",                type=int,   default=6,                          help="Number of Bert self-attention layers")
    parser.add_argument("--maxseqlen",              type=int,   default=100,                        help="Maximum number of tokens in training data fed to the model")
    parser.add_argument("--maskpct",                type=float, default=0.15,                       help="Percentage of training tokens replaced with [MASK]")
    parser.add_argument("--epochs",                 type=int,   default=20,                         help="Number of epochs to train for")
    parser.add_argument("--batchsize",              type=int,   default=8,                          help="Minibatch size for training")
    parser.add_argument("--accumulation_k",         type=int,   default=2,                          help="Number of minibatches to accumulate gradients on before taking a step() with optimizer")
    parser.add_argument("--progress_k",             type=int,   default=512,                        help="Number of minibatches to pass through before logging current average epoch loss values")
    parser.add_argument("--checkpoint_k",           type=int,   default=10,                         help="How many epochs to wait before checkpointing model weights")
    parser.add_argument("--max_train_pairs",        type=int,   default=1000,                       help="Number of sentence pairs (training examples) to include in training set out of the whole text corpus")
    parser.add_argument("--max_valid_pairs",        type=int,   default=2048,                       help="Number of sentence pairs to include in the validation set out of the whole text corpus")
    (args, _) = parser.parse_known_args()
    assert args.backend.lower() in ("nccl", "gloo", "mpi")
    # pre-fill main_distributed with keyword args
    main = functools.partial(train_distributed, **args.__dict__)
    # Start multiprocessing
    mp.spawn(main, nprocs=args.world_size)
