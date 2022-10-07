import os
import sys
import requests
from itertools import groupby
from subprocess import check_output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

class ESM_Model:
    # esm1b_t33_650M_UR50S
    # esm2_t6_8M_UR50D
    # esm2_t12_35M_UR50D
    # esm2_t30_150M_UR50D
    # esm2_t33_650M_UR50D
    # esm2_t36_3B_UR50D
    
    def __init__(self, *args):
        if len(args) == 1:
            self.load(args[0])
    
    def load(self, model_name):
        import esm
        self.model_name = model_name
        self.model, alphabet = eval(f'esm.pretrained.{self.model_name}()')
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.embed_dim = self.model._modules['layers'][0].embed_dim
        self.layers = sum(1 for i in self.model._modules['layers'])
        
    def encode(self, sequence, device='cuda', threads=1):
        try:
            torch.cuda.empty_cache()
            torch.set_num_threads(threads)

            batch_labels, batch_strs, batch_tokens = self.batch_converter([['',sequence]])
            batch_tokens = batch_tokens.to(device)
            self.model = self.model.to(device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.layers], return_contacts=False)
                results = results["representations"][self.layers].to('cpu')[0]
            return results
        except:
            if device != 'cpu':
                return self.encode(sequence, device='cpu', threads=threads)
            else:
                return

class T5_Model:
    # Rostlab/prot_t5_xl_uniref50
    
    def __init__(self, *args):
        if len(args) == 1:
            self.load(args[0])
    
    def load(self, model_name):
        from transformers import T5EncoderModel, T5Tokenizer
        self.model     = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        self.layers    = sum(1 for i in self.model._modules['encoder']._modules['block']._modules)
        self.model.eval()
        
    def encode(self, sequence, device='cuda', threads=1):
        torch.set_num_threads(threads)
        try:
            torch.cuda.empty_cache()
            d     = {'U':'X','Z':'X','O':'X','B':'X'}
            s     = ' '.join(d[j] if j in d else j for j in ''.join(i.strip() for i in sequence))
            ids   = self.tokenizer.batch_encode_plus([s], add_special_tokens=True, padding=True)
            model = self.model.to(device)

            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)
                embedding = embedding.last_hidden_state.cpu()[0]
            return embedding
        except:
            if device != 'cpu':
                return self.encode(sequence, device='cpu', threads=threads)
            else:
                return

class RegressionModel:
    def __init__(self, coef, intercept):
        self._coef = coef
        self._intercept = intercept
    
    @staticmethod
    def from_npz(npz_file):
        npz = np.load(npz_file)
        coef = np.squeeze(npz['coef'])
        intercept = npz['intercept'].item()
        return RegressionModel(coef, intercept)
    
    def predict(self, X):
        return self._intercept + np.dot(X, self._coef)

class GFF:
    def __init__(self, gff_file):
        gff = pd.DataFrame(pd.read_csv(gff_file, sep='\t', comment='#', header=None).values[:,:9])
        gff.columns = ['seqname','source','feature','start','end','score','strand','frame','attribute']
        self.data = gff

    def get_secondary_structure(self):
        sele = self.data[self.data['source'] == 'AlphaFold']
        data = sele[['start','end','feature']].iterrows()
        return [{'start': i['start'], 
                 'end'  : i['end'],
                 'label': i['feature']} for _, i in data]
    
    def get_domains(self):
        sele = self.data[self.data['feature'] == 'Domain']
        data = sele[['start','end','attribute']].iterrows()
        return [{'start': i['start'], 
                 'end'  : i['end'],
                 'label': i['attribute'].split(';')[0].split('=',1)[1]} for _, i in data]
    
    def get_topology(self):
        sele = self.data[(self.data['feature'] == 'Transmembrane') | (self.data['feature'] == 'Intramembrane')]
        data = sele[['start','end','feature']].iterrows()
        return [{'start': i['start'], 
                 'end'  : i['end'],
                 'label': i['feature']} for _, i in data]
            
def read_fasta(file):
    is_header = lambda x: x.startswith('>')
    compress  = lambda x: ''.join(_.strip() for _ in x)
    reader    = iter(groupby(open(file), is_header))
    reader    = iter(groupby(open(file), is_header)) if next(reader)[0] else reader
    for key, group in reader:
        if key:
            for header in group:
                header = header[1:].strip()
        else:
            sequence = compress(group)
            if sequence != '':
                yield header, sequence

def download_fasta(uniprot):
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot}.fasta'
    response = requests.get(url)
    if response.status_code == 200:
        return response.content.decode('UTF-8')
    else:
        return None

def download_gff(uniprot):
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot}.gff'
    response = requests.get(url)
    if response.status_code == 200:
        return response.content.decode('UTF-8')
    else:
        return None

def download_alphafold(uniprot, version='v3'):
    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_{version}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        return response.content.decode('UTF-8')
    else:
        return None

def generate_dssp_gff(pdb_file, seqname='.', dssp_bin='dssp'):
    dssp = check_output([dssp_bin,'-i',pdb_file]).decode()
    dssp_contents = [i for i in dssp.split('\n') if not i.endswith('.')][1:-1]
    ss8 = [(int(i[5:10]), i[16].replace(' ','C')) for i in dssp_contents]
    d = {
        'S': 'Loop',
        'T': 'Loop',
        'C': 'Loop',
        'B': 'Loop',
        'E': 'Beta strand',
        'I': 'Helix',
        'G': 'Helix',
        'H': 'Helix'}
    ss3 = [(i, d[j])for i, j in ss8]
    gff = []
    for k, g in groupby(ss3, lambda x: x[1]):
        g = [i[0] for i in g]
        gff += ['\t'.join((seqname,'AlphaFold',k,str(min(g)),str(max(g)),'.','.','.','.'))]
    return '\n'.join(gff)

def map_bfactor_values(pdb_file, values, overwrite=False):
    # pymol command for viewing b-factor
    # spectrum b, white_red, minimum=0.35, maximum=0.8
    groups = []
    for line in open(pdb_file):
        if line[:6] == 'ATOM  ':
            resname = line[17:20]
            chain   = line[21]
            resnum  = line[22:26]
            bfactor = line[60:66]
            groups += [(chain, resnum, resname)]
    residues = [k for k, g in groupby(groups)]
    assert len(residues) == len(values)
    vdic = dict(zip(residues, values))
    
    buffer = ''
    for line in open(pdb_file):
        if line[:6] == 'ATOM  ':
            resname = line[17:20]
            chain   = line[21]
            resnum  = line[22:26]
            
            value   = vdic[(chain, resnum, resname)]
            value   = f'{value:.2f}'.rjust(6,' ')
            
            line    = list(line)
            line[60:66] = list(value)
            buffer += ''.join(line)
    
    if overwrite:
        with open(pdb_file, 'w') as w:
            w.write(buffer)
    else:
        return buffer

def estimate_full_length(sequence, regressor, chunk_size=100, min_overlap=20):
    total_size = len(sequence)
    if total_size > chunk_size:
        # determine sequence chunks based on the max chunk size
        sequence = np.array(list(sequence))
        n_chunks = 1 + int(np.ceil((total_size - chunk_size + min_overlap) / (chunk_size - min_overlap)))
        stride = (total_size - chunk_size) / (n_chunks - 1)
        
        # set up weights for stitching together chunks
        chunk_weights = np.array([np.arange(1,1+chunk_size),np.arange(1,1+chunk_size)[::-1]]).min(0)
        weights = np.zeros((n_chunks, total_size))
        for n, i in enumerate(range(n_chunks)):
            i = round(i*stride)
            weights[n,i:i+chunk_size] = chunk_weights
        assert (weights.sum(0)!=0).all()
        
        # calculate scores for each chunk and stitch them using weights
        scores = np.zeros((n_chunks, total_size))
        for n, r in enumerate(weights):
            scores[n,r!=0] = regressor(''.join(sequence[r!=0]))
        return np.clip((scores * (weights / weights.sum(0))).sum(0), 0, 1)
    else:
        # if sequence is smaller than chunk size, no stitching is needed
        return np.clip(regressor(sequence), 0, 1)

def plot_conservation(sequence, conservation, cutoff=0.35, annot_struc=[], annot_topology=[], annot_domains=[], savefig=None):
    residue_numbers = 1 + np.arange(len(sequence))
    n_residues = residue_numbers.size

    toggle = 0 if sum(map(len,(annot_struc,annot_domains,annot_topology)))==0  else 1
    
    fig, ax = plt.subplots(2 + toggle, 1, figsize=(n_residues * 0.12, 3 + (toggle * 1.5)),
                          gridspec_kw={'height_ratios':[1,0]+[0.5 for i in range(toggle)], 'hspace':0.3})
    
    ax[0].bar(residue_numbers, conservation, width=1, color='tomato')
    ax[0].set_ylim(cutoff, 1)
    ax[0].set_ylabel('conservation')
    ax[0].set_xlim(residue_numbers.min()-0.5, 0.5+residue_numbers.max())
    ax[0].set_xticks(residue_numbers)
    ax[0].set_xticklabels(sequence, family='monospace', ha='center')
    ax[0].tick_params(length=0)
    
    ax[1].set_xlim(residue_numbers.min()-0.5, 0.5+residue_numbers.max())
    ax[1].set_xticks(residue_numbers[(residue_numbers % 10)==0])
    ax[1].set_yticks([])
    
    if toggle != 0:
        ax[2].set_yticks([])
        ax[2].set_ylim(6, 0)
        ax[2].set_xticks([])
        ax[2].set_xlim(residue_numbers.min()-0.5, 0.5+residue_numbers.max())
        
        args_kwargs = []
        xticks, xticklabels = [], []
        
        height = 1
        args_kwargs += [(((residue_numbers.min()-0.5, 0.5+residue_numbers.max()),(height, height)), {'solid_capstyle':'butt','color':'black', 'lw':2})]
        for annot in annot_struc:
            if annot['label'] == 'Helix':
                args_kwargs += [(((annot['start']-0.5,0.5+annot['end']),(height, height)), {'solid_capstyle':'butt','color':'red', 'lw':10})]
            elif annot['label'] == 'Beta strand':
                args_kwargs += [(((annot['start']-0.5,0.5+annot['end']),(height, height)), {'solid_capstyle':'butt','color':'blue', 'lw':10})]
        
        height = 3
        args_kwargs += [(((residue_numbers.min()-0.5, 0.5+residue_numbers.max()),(height, height)), {'solid_capstyle':'butt','color':'black', 'lw':2})]
        for annot in annot_topology:
            if annot['label'] == 'Transmembrane':
                args_kwargs += [(((annot['start']-0.5,0.5+annot['end']),(height, height)), {'solid_capstyle':'butt','color':'red', 'lw':10})]
            elif annot['label'] == 'Intramembrane':
                args_kwargs += [(((annot['start']-0.5,0.5+annot['end']),(height, height)), {'solid_capstyle':'butt','color':'gold', 'lw':10})]
        
        height = 5
        args_kwargs += [(((residue_numbers.min()-0.5, 0.5+residue_numbers.max()),(height, height)), {'solid_capstyle':'butt','color':'black', 'lw':2})]
        for annot in annot_domains:
            args_kwargs += [(((annot['start']-0.5,0.5+annot['end']),(height, height)), {'solid_capstyle':'butt','color':'gray', 'lw':10})]
            xticks += [(annot['start'] + annot['end']) / 2]
            xticklabels += [annot['label']]
        
        for args, kwargs in args_kwargs:
            plt.plot(*args, **kwargs)
        
        ax[2].set_xticks(xticks)
        ax[2].set_xticklabels(xticklabels)
        ax[2].set_yticks([1,3,5])
        ax[2].set_yticklabels(['secondary structure','membrane topology','protein domain'])
        ax[2].set_frame_on(False)
        ax[2].tick_params(length=0)

    if savefig != None:
        plt.savefig(savefig)
    
    # plt.show()
    plt.close()

