# Feature

*AlphaFold Wiki - Supported by [Bozitao Zhong](mailto:zbztzhz@gmail.com)*

If we divided AlphaFold into 2 main parts: CPU part and GPU part, we can also call the CPU part as feature/preprocessing part. In this part, AlphaFold did MSA (multiple sequence alignment), template searching and build the `feature.pkl` file for further neural network inference. In following pages, we will call this part as **feature** part for this name demonstrate the purpose of this pipeline. 

Why did we want to know detail of feature part? If we want to modify AlphaFold pipeline, or we want to accelerate it, or for some different applications for further usage, the easiest way the achieve these jobs is modify feature part.

Up to now, we already know that the modification of feature part can achieve:

- skip MSA and template searching steps
- parallelize the CPU step to obtain a around 60% accelerate
- use pre-computed MSA for inference
- give AlphaFold given template for modeling



## Output: `feature.pkl` file

Feature file is different in monomer model and multimer model (maybe monomer and monomer_ptm models are same)

These model have common parts:

- `aatype`: Sequence amino acid type information. $N_{res}*21$ (one-hot) in monomer, $N_{res}*1$ (0 to 20, actually 0 to 19) in multimer
- `residue_index`: Index of target sequence. Same as `np.arange(0,N_res,1)` in monomer model, multimer model is concatenate of sequences.
- `seq_length`: Sequence length. $N_{res}*1$, each number is the length in monomer, one length number (Scalar) in multimer
- `msa`: MSA information in matrix format. $N_{MSA}*N_{res}$ in monomer, $N_{MSA} = 512$ in multimer (maybe it has a lower limit or default value), number indicate the amino acid type (0 to 21).
- `num_alignments`: $N_{MSA}$. In monomer, $N_{res}*1$, Scalar in multimer, this is not equal to 512.
- `template_aatype`: Sequence information of templates. $N_{template}*N_{res}*22$ (one-hot) in monomer, $N_{template}*N_{res}$ (0 to 21) in multimer
- `template_all_atom_masks`/`template_all_atom_mask`: Unknown function, maybe masks for template? $N_{template}*N_{res}*37$, binary tensor
- `template_all_atom_positions`: Unknown function, maybe coordinates. $N_{template}*N_{res}*37*3$
- 





Monomer model have these keys specifically:

- `between_segment_residues`: Unknown function. $N_{res}*1$, normally is all 0.
- `domain_name`: Name of this protein. String.
- `sequence`: Sequence. Normal amino acid sequence in string format
- `deletion_matrix_int`: Unknown function, maybe showed the point that have deletion in MSA, number might correspond to length of deletion (gap) in MSA. $N_{MSA}*N_{res}$
- `msa_uniprot_accession_identifiers`: List of string indicate the UniProt Accession ID of each MSA sequence. $N_{MSA}*1$ 
- `msa_species_identifiers`: Similar to former one, don't know the species identifiers come from, it's a 5-character string.
-  `template_domain_names`: List of string that indicate names of templates (PDB names), $N_{MSA}-1$
- `template_sequence`: List of string of template sequence, $N_{template}*N_{res}$
- `template_sum_probs`: Unknown function. $N_{template}*1$, numbers



Multimer model have these keys specifically:

- `asym_id`: Number mark each chain, from 0 to $N_{chain}$, shape: $N_{res}*1$
- `sym_id`: Similar as former one
- `entity_id`: All 1, unknown function, shape $N_{res}*1$
- `deletion_matrix`: $N_{res}*512$
- `deletion_mean`: $N_{res}*1$
- `all_atom_mask`: $N_{res}*37$ 
- `all_atom_positions`:  Initial atom position, all zero. $N_{res}*37*3$ 
- `assembly_num_chains`: Number of chain in complex. Scalar 
- `entity_mask`: Unknown $N_{res}*1$
- `num_templates`: Number of templates, scalar
- `cluster_bias_mask`: $512*1$ 
- `bert_mask`: $512*N_{res}$ 
- `seq_mask`: $N_{res}$ 
- `msa_mask`: $N_{res}$ 



In monomer model, feature file contains these parts:

- 'aatype', 'between_segment_residues', 'domain_name', 'residue_index', 'seq_length', 'sequence', 'deletion_matrix_int', 'msa', 'num_alignments', 'msa_uniprot_accession_identifiers', 'msa_species_identifiers', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_domain_names', 'template_sequence', 'template_sum_probs'

In multimer model, feature file contains these parts:

- 'aatype', 'residue_index', 'seq_length', 'msa', 'num_alignments', 'template_aatype', 'template_all_atom_mask', 'template_all_atom_positions', 