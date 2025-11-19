# GRNforge
An integrated multi-feature single-cell multiomics gene regulatory network inference method based on transcriptomics, chromatin accessibility and transcription factor cooperativity

## 1. Overview

The pipeline consists of four main components:

1. **Data preprocessing**  
2. **TFBS detection (motif scanning + footprinting)**  
3. **Base eGRN construction**  
4. **Predictive modelling using XGBoost and EnhLink**  
5. **Post-processing and scoring of final interactions**  

A small **test dataset** is included so users can easily try the full workflow without installing heavy external tools like GimmeMotifs or TOBIAS.

## 2. Installation

```bash
conda create -n grnforge_env python=3.11
conda activate grnforge_env
```
Python dependencies are documented at the top of each script and include:

- numpy, pandas, scipy

- scanpy

- snapatac2

- pybedtools

- xgboost


External tools used (optional for test dataset):

- GimmeMotifs (motif scanning): You can go to this reference and follow the instruction - [https://github.com/vanheeringen-lab/gimmemotifs](https://github.com/vanheeringen-lab/gimmemotifs) 

- TOBIAS (ATAC footprinting): You can go to this reference and follow the instruction - [https://github.com/loosolab/TOBIAS](https://github.com/loosolab/TOBIAS) 

- EnhLink (enhancer–target regression)


## 3. Repository structure
```
GRNforge/
├── data/
│   ├── universal_files/          # independent reusable reference files
│   └── test/                     # sample dataset (10 TFs, 100 TGs)
│       ├── data/                 # preprocessed matrices and peak/motif files
│       ├── scripts/
│       └── results/
│
├── models/                       # stored predictive models
├── results/                      # output from full pipeline runs
│
├── scripts/
│   ├── tfbs_detection/           # motif scanning + footprint extraction
│   ├── grn_prediction/           # base eGRN + XGBoost prediction
│   └── grn_postprocessing/       # scoring, integration, and ranking
```
## 4. Data preprocessing
### 1. RNA-seq
- RNA-seq & ATAC-seq preprocessing
    - Output of each file is a h5da file, which contain cell and gene information for each activity (gene activity from ATAC and gene expresison from RNA)
- FBS detection: to detect TF binding sites, we use
    - GimmeMotifs for motif scanning
    - TOBIAS for ATAC footprint correction and footprint scoring
    - Generates TF–RE connections using motif matches, footprints, and RE–TG distance
    ```
    python scripts/GRNforge/tfbs_detection/filter_peak_and_find_motifs.py
    ```
For easy to run, we provide subset dataset includes gene activity extract from file  and RNA preprocessing file precomputed files of GimmeMotifs and TOBIAS , so you may skip the preprocesisng step and step using  GimmeMotifs and tobias. You only need to run the python file to gain the TFBS_peak file for input of model.

<img width="647" height="409" alt="image" src="https://github.com/user-attachments/assets/b76ad54f-829d-4537-8e99-61e82c53192c" />


## 5. Model

The network inference consists of (1) base-network construction and (2) predictive modelling.
    - Base eGRN construction `baseGRN.py`: This step builds RE–TG links (distance-based), TF–RE links (motif + footprint), TF–TG candidate interactions
    - Predictive modelling using XGBoost `modelling_v5.py`: Each target gene (TG) gets its own model, predicting TG expression/activity from TF expression/activity.
    
<img width="945" height="661" alt="image" src="https://github.com/user-attachments/assets/d89ed854-6c00-4f5a-9890-051caaf77570" />

**Support model modes**
    | Model        | Description                                |
| ------------ | ------------------------------------------ |
| **exp_exp**  | TF expression → TG expression              |
| **act_act**  | TF accessibility → TG accessibility        |
| **exp_act**  | TF expression → TG accessibility           |
| **act_exp**  | TF accessibility → TG expression           |
| **expCOP_exp** | Expression-based models using TF complexes |
| **actCOP_act** | Accessibility-based complex models         |
| **expCOP_act** | Expression-based models using TF complexes |
| **actCOP_exp** | Accessibility-based complex models         |

After run model, we do model postprocessing. This step:

- normalizes raw importance scores
- integrates results across folds/models
- ranks final TF–TG interactions
- aggregates all outputs into final eGRN files

## 6. Running the pipeline (quick start)

Start in the project root.
You can run with test data or after you run preprocessing to extarct gene activity and gene expression from preprocessing, tf-peak and motif score (GimmeMotifs and TOBIAS).
Before run pipelines, please download gene annotation (**section 7**) and store it in `data/universal_files`

Step 1 — Generate TFBS 
```
python scripts/GRNforge/tfbs_detection/filter_peak_and_find_motifs.py
```

Step 2 — Build the base eGRN
```
python scripts/GRNforge/grn_prediction/baseGRN.py
```

Step 3 — Train predictive models
```
python scripts/GRNforge/grn_prediction/modelling_v5.py exp_exp 30 50 #model_mode start_chunk end_chunk
```

Step 4 — Post-processing
```
python scripts/GRNforge/grn_postprocessing/grn_postprocessing.py
```

## 7. Additional data
Gene annotation (GENCODE v38)
```
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gff3.gz
```
** Please store the file in directory data/universal_files* *

## 8. Test dataset
Test dataset is a subset from full data, which extract from chromoseom 4 base on 10 TF and 100 target gene. PLease download and store the gene annotation please run the code.
The folder data/test contains:
- processed RNA expression matrices
- processed ATAC gene activity matrices
- precomputed peak-motif matches
- precomputed footprint scores

This allows running the entire workflow without needing preprocessing, or using GimmeMotifs or TOBIAS.

## 9. Citation


