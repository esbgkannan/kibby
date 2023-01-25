# Introduction

This respository contains the code and dataset for:

[**Alignment-free estimation of sequence conservation for identifying functional sites using protein sequence embeddings.**](https://academic.oup.com/bib/article/24/1/bbac599/6984799)\
Wayland Yeung, Zhongliang Zhou, Sheng Li, Natarajan Kannan.\
*Briefings in Bioinformatics.* (2023)

# Quick start guide

Don't have time for the full guide? Thats okay!

Here are the BASH commands:
```
# Download this repository
git clone https://github.com/waylandy/kibby
cd kibby

# Install Python dependencies (specific versions are in environment.yml)
python3 -m pip install numpy pandas matplotlib torch
python3 -m pip install fair-esm transformers

# Example 1: estimate the residue conservation of each sequence in a fasta
python3 conservation_from_fasta.py sample_data/sequences.fasta sample_data/conservation.csv

# Optional program for example 2
sudo apt install dssp

# Example 2: estimate conservation for uniprot sequences, map values to alphafold models, and make annotated figures
python3 conservation_from_uniprot.py sample_data/uniprot.txt sample_data/annotated_outputs
```

# Installing dependencies

### Downloading this repository

This repository contains the code and files needed to generate sequence embeddings using a pre-trained protein language model, then estimate sequence conservation using linear regression.
```
# Download this repository
git clone https://github.com/waylandy/kibby

cd kibby
```

If you are an advanced user with pre-generated sequence embeddings, our regression parameters for various pre-trained protein language model can be found under the "linear_models" directory. Parameters are stored and compressed numpy arrays and consist of the linear coefficients and intercept. To make predictions, simply take the dot product of the sequence embedding and the linear coefficients then add the intercept.

### Installing dependencies

Our code is written in Python 3 and requires several Python packages to run.

We provide instructions for installing these dependencies using either `pip` or `conda`.

#### Installing dependencies with `pip`

```
python3 -m pip install numpy pandas matplotlib torch
python3 -m pip install fair-esm transformers
```

#### Installing dependencies with `conda`

If you do not have `conda` installed, you can download from the [Anaconda website](https://www.anaconda.com/).
```
# Create a conda environment using the yml file
conda env create -f environment.yml

# Activate the environment
conda activate kibby
```
If you want to exit the environment:
```
conda deactivate
```
If you want to start over, you can delete the conda environment:
```
conda env remove -n kibby
```

#### Notes on installing PyTorch

The most computationally expensive part of this pipeline is generating the sequence embedding using a protein language model. This uses PyTorch which can benefit from GPU acceleration if installed with the appropriate CUDA version. To check CUDA version, try running `nvidia-smi`. The PyTorch packages installed in the previous section should work on CPU, but you may need to re-install a PyTorch package that is compatible with your specific GPU and CUDA version to benefit from GPU acceleration.

For more information on installing PyTorch, see the installation guide on the [PyTorch website](https://pytorch.org/). 

#### Other dependencies

One optional step in our annotation pipeline uses `dssp` to assign secondary structure labels to residues using AlphaFold models.

```
sudo apt install dssp
```

# Running the script

We provide two scripts for embedding-based conservation analysis.

If this is the first time you're using a particular protein language model, please note that the script will automatically download the model onto your computer for future runs. The initial download may take some time, but this only needs to be one once per computer.

### Conservation analysis of sequences in a FASTA file

A minimal script which sequences from a fasta file and estimates conservation scores for each sequence.

```
# run using example data
python3 conservation_from_fasta.py sample_data/sequences.fasta sample_data/conservation.csv
```

To see the help screen, use `python3 conservation_from_fasta.py -h`

```
Estimate the conservation of each sequence in a fasta file using sequence embeddings.
Sequence embeddings can be generated from any ESM2 protein language model.

For performance reasons, the default model is "esm2_t33_650M_UR50D", while the
original manuscript uses results from "esm2_t36_3B_UR50D" which is also available.

Usage: python3 conservation_from_fasta.py input_fasta output_csv [options]

optional arguments:
  -h, --help        show this help message and exit
  -model MODEL      Name of the protein language model to use.
                      esm2_t6_8M_UR50D
                      esm2_t12_35M_UR50D
                      esm2_t30_150M_UR50D
                      esm2_t33_650M_UR50D (default)
                      esm2_t36_3B_UR50D
                      esm2_t48_15B_UR50D
  -device DEVICE    Used for generating embeddings in PyTorch; use either cpu or cuda. (default: cpu)
  -threads THREADS  Used for generating embeddings in PyTorch; number of cpu threads to use. (default: 3)
```

### Conservation analysis of sequences from UniProt accessions

A script which takes a list of UniProt IDs, downloads each sequence and estimates conservation scores. The script will also download the AlphaFold model and map the conservation scores onto the pdb file using the b-factor column. These scores can be viewed on using the PyMOL command `spectrum b, white_red, minimum=0.35, maximum=0.8`. Domain annotations will be downloaded from UniProt in gff format and secondary structure annotations will be added by running `dssp` on the AlphaFold model.

```
# run using example data
python3 conservation_from_uniprot.py sample_data/uniprot.txt sample_data/annotated_outputs
```

To see the help screen, use `python3 conservation_from_uniprot.py -h`

```
Downloads sequences based on a list of UniProt accessions.
Estimate the conservation of each sequence using sequence embeddings.
Sequence embeddings can be generated from any ESM2 protein language model.

Downloads AlphaFold models and UniProt domain annotations.
Maps conservation scores to AlphaFold pdb files on the bfactor column.
Draws a bar chart showing conservation alongside domain & structure annotations.

For performance reasons, the default model is "esm2_t33_650M_UR50D", while the
original manuscript uses results from "esm2_t36_3B_UR50D" which is also available.

Usage: python3 conservation_from_uniprot.py accession_txt output_dir [options]

optional arguments:
  -h, --help        show this help message and exit
  -model MODEL      Name of the protein language model to use.
                      esm2_t6_8M_UR50D
                      esm2_t12_35M_UR50D
                      esm2_t30_150M_UR50D
                      esm2_t33_650M_UR50D (default)
                      esm2_t36_3B_UR50D
                      esm2_t48_15B_UR50D
  -device DEVICE    Used for generating embeddings in PyTorch; use either cpu or cuda. (default: cpu)
  -threads THREADS  Used for generating embeddings in PyTorch; number of cpu threads to use. (default: 3)
  -dssp DSSP        The dssp binary for generating secondary structure annotations (default: dssp)
  -sleep SLEEP      How many seconds to wait between each entry (default: 0.5)
```
# How to do this manually

This section describes the bare minimum code that is required for generating embedding-based conservation analysis.

First, we import the necesary modules.

```
import numpy as np
import torch
import esm
```

Next, we load a protein language model. Different protein language models may be loaded in different ways, but for this example, we will be using the ESM2 model with 650M parameters.

```
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # the model
batch_converter = alphabet.get_batch_converter() # the tokenizer
model = model.eval() # disable dropout
```

Then, we take an example protein sequence and tokenize it using the batch converter.
```
sequence = 'GGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDS'
batch_tokens = batch_converter([[None,sequence]])[2]
```

After that, we can send the tokenized protein sequence into the language model to generate a sequence embedding vector. Special tokens are also removed from the embedding vector.
```
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
sequence_embedding = results["representations"][33][0][1:-1].numpy()    
```

Load the parameters of the appropriate trained linear regression model, all of which are provided in this repository.
```
linear = np.load('linear_models/esm2_t33_650M_UR50D.npz')
coef, intercept = linear['coef'], linear['intercept']
```

Matrix multiply the embedding vector with the linear coefficients, add the intercept, then clip the range from zero to one. The result should be the conservation values.
```
conservation = np.clip(np.matmul(sequence_embedding, coef) + intercept, 0, 1)
```




