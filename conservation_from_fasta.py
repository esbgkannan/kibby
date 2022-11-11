import os
import sys
import argparse

import torch

sys.dont_write_bytecode = True
from my_library import *

description = """
Estimate the conservation of each sequence in a fasta file using sequence embeddings.
Sequence embeddings can be generated from any ESM2 protein language model.

For performance reasons, the default model is "esm2_t33_650M_UR50D", while the
original manuscript uses results from "esm2_t36_3B_UR50D" which is also available.

Usage: python3 conservation_from_fasta.py input_fasta output_csv [options]

"""

parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, formatter_class=argparse.RawTextHelpFormatter, description=description)

parser.add_argument('-model', type=str, default='esm2_t33_650M_UR50D',
                    help="Name of the protein language model to use.\n  esm2_t6_8M_UR50D\n  esm2_t12_35M_UR50D\n  esm2_t30_150M_UR50D\n  esm2_t33_650M_UR50D (default)\n  esm2_t36_3B_UR50D\n  esm2_t48_15B_UR50D")
parser.add_argument('-device', type=str, default='cpu',
                    help="Used for generating embeddings in PyTorch; use either cpu or cuda. (default: cpu)")
parser.add_argument('-threads', type=int, default=3,
                    help="Used for generating embeddings in PyTorch; number of cpu threads to use. (default: 3)")

args, unk   = parser.parse_known_args()

INPUT_FASTA, OUTPUT_CSV = unk

MODEL_NAME  = args.model
DEVICE      = args.device
N_THREADS   = args.threads

try:
    torch.Tensor().to(DEVICE)
except:
    sys.stderr.write(f'Cannot move data to device "{DEVICE}", defaulting to "cpu"\n\n')
    DEVICE = 'cpu' 

sys.stderr.write(f'Loading protein language model: {MODEL_NAME}\n')
esm = ESM_Model()
esm.load(MODEL_NAME)

sys.stderr.write(f'Loading regression model\n')
npz_file = f'linear_models/{MODEL_NAME}.npz'
regression_model = RegressionModel.from_npz(npz_file)
estimate_chunk = lambda x: regression_model.predict(esm.encode(x, device=DEVICE, threads=N_THREADS)[1:-1])
estimate_full = lambda x: estimate_full_length(x, estimate_chunk, chunk_size=1022, min_overlap=300)

with open(OUTPUT_CSV, 'w') as w:
    w.write('header,sequence,conservation\n')
    for n, (header, sequence) in enumerate(read_fasta(INPUT_FASTA)):
        sys.stderr.write(f'Processing sequence {1+n}\r')
        conservation = estimate_full(sequence)
        conservation = ' '.join(f'{i:.4f}' for i in conservation)
        w.write(f'{header},{sequence},{conservation}\n')
    w.write('\n')

sys.stderr.write(f'\ndone\n')

