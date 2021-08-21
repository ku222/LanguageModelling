


import logging
import os
import random
from typing import Tuple

import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging
import google.cloud.storage as storage
import torch
import torch.distributed as dist
from modules import BertPreTraining
from torch import FloatTensor, LongTensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling import MLMPreprocessor


class Batch:
    """
    Helper class to wrap around a batch of tensors from a given Dataloader,
    so that those tensors can be accessed via named attributes
    """
    def __init__(self, batch: Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor], device: torch.device) -> None:
        self.device = device
        batch = batch if len(batch[0].shape) == 2 else [tnsr.unsqueeze(0) for tnsr in batch]
        self.masked_input_ids = batch[0].to(device)     # (batch, seqlen) e.g. [101, 246, 26245...]
        self.attention_mask = batch[1].to(device)       # (batch, seqlen) e.g. [1, 1, 0...]
        self.token_type_ids = batch[2].to(device)       # (batch, seqlen) e.g. [0, 0, 0]
        self.input_ids = batch[3].to(device)            # (batch, seqlen) e.g. [101, 246, 26245...]
        self.class_labels = batch[4].to(device)         # (batch,) e.g [1] or [0]



class DistributedTrainer:
    def __init__(self, model: nn.Module, rank: int) -> None:
        self._model = model.to(rank)
        self.model = DDP(self._model, device_ids=[rank])
        self.device = self.rank = rank


class LanguageModelTrainer(DistributedTrainer):
    """
    Helper class to handle the training logic for the Bert model
    """
    def __init__(self, model: BertPreTraining, preprocessor: MLMPreprocessor, rank: int) -> None:
        super().__init__(model, rank)
        self.preprocessor = preprocessor
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=self.preprocessor.paddingidx)
        self.nsp_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)
        self.trainloss_mlm = 0.0
        self.trainloss_nsp = 0.0
        self.validloss_mlm = 0.0
        self.validloss_nsp = 0.0
        self.checkpoint_is_setup = False
        self.logging_is_setup = False

    def set_checkpoint_strategy(self, localpath: str, gcpbucketname="", blobname="") -> None:
        """
        Creates a method `self.checkpoint()` to be called at the end of each epoch, which
        will save the weights of the model to localpath & GCP bucket if passed.

        Calling self.checkpoint() will either:
            - Save locally ONLY if `gcpbucketname` is not given
            - Save locally AND THEN upload to the given GCP bucket otherwise
        """
        def save_locally() -> None:
            torch.save(self.model.state_dict(), localpath)

        def save_to_gcp() -> None:
            client = storage.Client()
            bucket = client.get_bucket(gcpbucketname)
            blob = bucket.blob(blobname)
            blob.upload_from_filename(localpath)

        if gcpbucketname:
            self.checkpoint = lambda: (save_locally(), save_to_gcp())
        else:
            self.checkpoint = lambda: save_locally()

        self.checkpoint_is_setup = True

    def set_logging_strategy(self, use_cloud_logging=True) -> None:
        logging.getLogger().setLevel(logging.INFO)
        if use_cloud_logging:
            client = google.cloud.logging.Client()
            handler = CloudLoggingHandler(client)
            setup_logging(handler)
        self.logging_is_setup = True

    def commence_training(self, trainloader: DataLoader, validloader: DataLoader, epochs: int, accumulation_k: int, progress_k=100, checkpoint_k=10) -> None:
        """
        Core training loop:
            - Trains for a total of `epochs` rounds, and checkpoints model weights after each one
            - Accumulates gradients, steps optimizer only every `accumulation_k` batches
            - Prints progress every `progress_k` batches
        """
        # Ensure model is on GPU, set to training mode
        self.model.train()
        # Shortcut for a pretty-printing function
        pprint = lambda strng: logging.info(f"GPU {self.rank}: {30*'='} {strng} {30*'='}")
        # Commence core training loop
        for e in tqdm(range(1, epochs+1)):

            # Loop over all train mini-batches
            pprint(f"Training Steps Epoch {e}/{epochs}")
            for (i, batch) in enumerate(trainloader, 1):
                (mlm_loss, nsp_loss) = self.forward(Batch(batch, device=self.device))
                self.trainloss_mlm += mlm_loss.item()
                self.trainloss_nsp += nsp_loss.item()
                self.progress_log(i, len(trainloader), progress_k)
                loss = (mlm_loss + nsp_loss) / accumulation_k
                loss.backward()
                # Every accumulation_k batches, take step() on accumulated gradients
                if i % accumulation_k == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Loop over all validation mini-batches
            pprint(f"Validation Steps Epoch {e}/{epochs}")
            for (i, batch) in enumerate(validloader, 1):
                with torch.no_grad():
                    (mlm_loss, nsp_loss) = self.forward(Batch(batch, device=self.device))
                    self.validloss_mlm += mlm_loss.item()
                    self.validloss_nsp += nsp_loss.item()
                    self.progress_log(i, len(validloader), progress_k, is_train=False)

            # After end of each epoch, evaluate, reset, checkpoint
            self.evaluate_step(validloader)
            self.reset_step()

            # Checkpoint every k epochs
            if e % checkpoint_k == 0:
                self.checkpoint()

    def progress_log(self, i: int, maxi: int, progress_k: int, is_train=True) -> None:
        """
        Every time `progress_k` is a multiple of `i`, print average losses so far
        """
        if i % progress_k == 0:
            rounded = lambda val: round(val, 5)
            (mlmloss, nsploss) = (self.trainloss_mlm, self.trainloss_nsp) if is_train else (self.validloss_mlm, self.validloss_nsp)
            logging.info(f"\t GPU {self.rank}: batch {i}/{maxi} | MLM_epoch_loss={rounded(mlmloss/i)} | NSP_epoch_loss={rounded(nsploss/i)}")

    def load_weights(self, localpath: str) -> None:
        """
        Loads checkpoint model weights from disk
        """
        self.model.load_state_dict(torch.load(localpath))

    def has_existing_weights(self, localpath: str) -> bool:
        """
        Checks to see if weights already exist from previous training run
        """
        splitslashes = localpath.split("/")
        (folder, fname) = "".join(splitslashes[:-1]), "".join(splitslashes[-1])
        return fname in os.listdir(folder)

    def forward(self, b: Batch) -> Tuple[FloatTensor, FloatTensor]:
        """
        Forward propagate a batch through the Bert model, and retrieve MLM & NSP losses
        """
        (mlm_logits, nsp_logits) = self.model(b.masked_input_ids, b.attention_mask, b.token_type_ids)   # (batch, seqlen, vocabsize); (batch, 2)
        # Get MLM component of loss
        (batch, seqlen, vocabsize) = mlm_logits.shape
        mlm_loss = self.mlm_criterion(mlm_logits.view(batch * seqlen, vocabsize), b.input_ids.flatten())
        # Get NSP component of loss
        nsp_loss = self.nsp_criterion(nsp_logits, b.class_labels)
        return (mlm_loss, nsp_loss)

    def evaluate_step(self, dataloader: DataLoader) -> None:
        """
        Perform evaluation of the model on MLM and NSP tasks
        """
        b = self.random_batch(dataloader)
        (mlm_logits, nsp_logits) = self.model(b.masked_input_ids, b.attention_mask, b.token_type_ids)   # (batch, seqlen, vocabsize); (batch, 2)
        mlm_logits = mlm_logits.argmax(dim=2)                                                           # (batch, seqlen)
        # Evaluate MLM performance
        maskedi = self.preprocessor.decode(b.masked_input_ids[0])
        predict = self.preprocessor.decode(mlm_logits[0])
        actuals = self.preprocessor.decode(b.input_ids[0])
        # Evaluate NSP Performance
        nsp_actuals = b.class_labels[0]
        nsp_predictions = nsp_logits[0].argmax(-1)
        logging.info({
            "GPU ID": self.rank,
            "[MLM] Masked": maskedi,
            "[MLM] Actual": actuals,
            "[MLM] Predicted": predict,
            "[NSP] Actual": nsp_actuals.tolist(),
            "[NSP] Predicted": nsp_predictions.tolist()
        })

    def reset_step(self) -> None:
        """
        Resets all loss accumulators after each epoch, and zeroes out gradients
        """
        self.trainloss_mlm = self.trainloss_nsp = self.validloss_mlm = self.validloss_nsp = 0.0
        self.optimizer.zero_grad()

    def random_batch(self, dataloader: DataLoader) -> Batch:
        """
        Select a single random batch from the given dataloader
        """
        dataset = dataloader.dataset
        randidx = int(random.random() * len(dataset))
        return Batch(dataset[randidx], device=self.device)
