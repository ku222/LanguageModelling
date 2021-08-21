

# Language Modelling Distributed Training Repo

From-scratch adaptation of BERT Transformer architecture for Masked Language Modelling & Next Sentence Prediction


## Training

Run train.py, which will spawn multiple independent PyTorch GPU nodes, each of which will:
- Download the Wiki103 dataset from https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
- Extract dataset contents into given `datafolder`/wiki-103/
- Create MLM/NSP training & validation datasets
- Commence training
- Periodically log losses & subjective performance metrics
- Periodically checkpoint the model weights to local/cloud storage


## TODO

- Add code to create & submit distributed PyTorch training job to GCP AI Platform
