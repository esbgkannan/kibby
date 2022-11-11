import os
import sys
import time
import argparse

import torch

sys.dont_write_bytecode = True

from my_library import *

description = """
Downloads sequences based on a list of UniProt accessions.
Estimate the conservation of each sequence using sequence embeddings.
Sequence embeddings can be generated from any ESM2 protein language model.

Downloads AlphaFold models and UniProt domain annotations.
Maps conservation scores to AlphaFold pdb files on the bfactor column.
Draws a bar chart showing conservation alongside domain & structure annotations.

For performance reasons, the default model is "esm2_t33_650M_UR50D", while the
original manuscript uses results from "esm2_t36_3B_UR50D" which is also available.

Usage: python3 conservation_from_uniprot.py accession_txt output_dir [options]

"""

parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, formatter_class=argparse.RawTextHelpFormatter, description=description)

parser.add_argument('-model', type=str, default='esm2_t33_650M_UR50D',
                    help="Name of the protein language model to use.\n  esm2_t6_8M_UR50D\n  esm2_t12_35M_UR50D\n  esm2_t30_150M_UR50D\n  esm2_t33_650M_UR50D (default)\n  esm2_t36_3B_UR50D\n  esm2_t48_15B_UR50D")
parser.add_argument('-device', type=str, default='cpu',
                    help="Used for generating embeddings in PyTorch; use either cpu or cuda. (default: cpu)")
parser.add_argument('-threads', type=int, default=3,
                    help="Used for generating embeddings in PyTorch; number of cpu threads to use. (default: 3)")
parser.add_argument('-dssp', type=str, default='dssp',
                    help="The dssp binary for generating secondary structure annotations (default: dssp)")
parser.add_argument('-sleep', type=float, default=0.5,
                    help="How many seconds to wait between each entry (default: 0.5)")


args, unk   = parser.parse_known_args()

ACCESIONS_TXT, OUTPUT_DIR = unk

MODEL_NAME  = args.model
DEVICE      = args.device
N_THREADS   = args.threads
DSSP_BIN    = args.dssp
SLEEP       = args.sleep

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

if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR += '/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

accessions = [i.strip() for i in open(ACCESIONS_TXT) if not i.isspace()]

def _pipeline(uniprot, basedir=OUTPUT_DIR, estimator=estimate_full, dssp_bin=DSSP_BIN, sleep=SLEEP):
    
    sys.stderr.write(f'  {uniprot} : Downloading fasta sequence from UniProt ')
    fasta_buffer = download_fasta(uniprot)
    
    if fasta_buffer == None:
        sys.stderr.write(f'[FAILED : accession not found]\n')
        return
    
    sequence_name = fasta_buffer.split()[0].split('|')[2]
    outdir = f'{basedir}{sequence_name}/'
        
    fasta_file = f'{outdir}/{uniprot}.fasta'
    gff_file   = f'{outdir}/{uniprot}.gff'
    pdb_file   = f'{outdir}/{uniprot}.pdb'
    pdf_output = f'{outdir}/{uniprot}.pdf'
    csv_output = f'{outdir}/{uniprot}.csv'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    with open(fasta_file, 'w') as w:
        w.write(fasta_buffer)
    
    header, sequence = next(read_fasta(fasta_file))
    sys.stderr.write(f'[DONE]\n')
    
    sys.stderr.write(f'  {uniprot} : Generated embeddings & estimated conservation scores ')
    conservation = estimator(sequence)
    with open(csv_output, 'w') as w:
        w.write(''.join(f'{i},{j:.4f}\n' for i, j in zip(sequence,conservation)))
    sys.stderr.write(f'[DONE]\n')
    
    sys.stderr.write(f'  {uniprot} : Downloading gff annotations from UniProt ')
    gff_buffer = download_gff(uniprot)
    if gff_buffer == None:
        sys.stderr.write(f'[FAILED : gff not found]\n')
    else:
        with open(gff_file, 'w') as w:
            w.write(gff_buffer)
        sys.stderr.write(f'[DONE]\n')
    
    sys.stderr.write(f'  {uniprot} : Downloading AlphaFold model & mapping conservation scores ')
    pdb_buffer = download_alphafold(uniprot)
    if pdb_buffer == None:
        sys.stderr.write(f'[FAILED : pdb not found]\n')
    else:
        try:
            with open(pdb_file, 'w') as w:
                w.write(pdb_buffer)
            map_bfactor_values(pdb_file, conservation, overwrite=True)
            sys.stderr.write(f'[DONE]\n')
        except:
            if os.path.exists(pdb_buffer):
                os.remove(pdb_buffer)
            sys.stderr.write(f'[FAILED : mapping error]\n')
    
    sys.stderr.write(f'  {uniprot} : Adding secondary structure annotations to gff ')
    if (gff_buffer != None) and (pdb_buffer != None):
        try:
            dssp_buffer = generate_dssp_gff(pdb_file, seqname=uniprot, dssp_bin=dssp_bin)
            with open(gff_file, 'a') as w:
                w.write(dssp_buffer)
            sys.stderr.write(f'[DONE]\n')
        except:
            sys.stderr.write(f'[FAILED : dssp error]\n')
    else:
        sys.stderr.write(f'[FAILED : missing gff or pdb]\n')
            
    if len(sequence) > 3000:
        sys.stderr.write(f'  {uniprot} : Skipping plot (sequence is too long)\n')
    else:
        sys.stderr.write(f'  {uniprot} : Drawing plot ')
        try:
            if gff_buffer == None:
                plot_conservation(sequence, conservation, 
                    savefig=pdf_output)
            else:
                gff = GFF(gff_file)
                plot_conservation(sequence, conservation, 
                    annot_struc=gff.get_secondary_structure(),
                    annot_topology=gff.get_topology(),
                    annot_domains=gff.get_domains(),
                    savefig=pdf_output)
            sys.stderr.write(f'[DONE]\n')
        except:
            sys.stderr.write(f'[FAILED : matplotlib error]\n')
    time.sleep(sleep)

skip = set()
for subdir, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        if file.endswith('.csv'):
            skip.add(file[:-4])

for uniprot in accessions:
    if uniprot in skip:
        sys.stderr.write(f'  {uniprot} : skipped\n')
    else:
        _pipeline(uniprot, basedir=OUTPUT_DIR, estimator=estimate_full, dssp_bin=DSSP_BIN, sleep=SLEEP)

