
# Code for the thesis project "Improving Identification of Homodimers from AlphaFold-Multimer Outputs Using Tree Ensembles".

## Relationship to prior work

This project builds upon and extends the study of Narrowe Danielsson & Elofsson (2025), whose code is available at
<https://github.com/SarahND97/alphafold-homodimers>.

The AlphaFold pipeline used codes from the previous work of Claudio Mirabello and Sarah Narrowe Danielsson <https://github.com/clami66/AF_cache>. Minor adjustments were made.

Three scripts named "*\*\_Narrowe.py*" in this repository are based on codes provided by Sarah Narrowe Danielsson.

## Repository layout

The directory numbering follows the pipeline order, with one gap: between stages 2 and 3 the sequences are modelled as homodimers with AlphaFold-Multimer (2.3), using code from <https://github.com/clami66/AF_cache>. Those predictions are the input to feature extraction.

```
homodimer_public/
├── 1_Select_protein_ids_labelling/
│   ├── 1_data_curation_Narrowe.ipynb     # PDB → clustering → temporal split → filters
│   ├── 2_make_stoichiometry_Narrowe.py   # surviving entries → (id, chain, stoichiometry)
│   └── 3_create_labels.py                # stoichiometry → binary label (homodimer = 1)
├── 2_Prepare_sequences/
│   ├── 1_prepare_sequences.py            # chain-aware FASTA download from RCSB
│   └── 2_filter_nonprotein.py            # drop nucleic-acid / non-standard entities
├── 3_Feature_extraction/
│   ├── 1_extract_features_Narrowe.py     # AlphaFold + interface + homology features
│   └── 2_check_homology_missingness.py   # QC on the homology feature block
└── 4_Prediction/
    └── 1_predict_homodimer.py            # model training, evaluation, comparison
```