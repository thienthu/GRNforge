import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from gimmemotifs.motif import default_motifs
import pickle
import pyBigWig
import sys
# sys.path.append('/scratch/gent/vo/000/gvo00027/projects/CBIGR/software/pybiomart')
# sys.path.append('/scratch/gent/vo/000/gvo00027/projects/CBIGR/software/tmp/JLR_pybedtools/')
from pybiomart import Server
import pybedtools

data_folder = 'data/test/'
data_universal_folder = 'data/universal_files/'
result_folder = 'results/human_PBMC_10x_10k/'
input_peak_file           = 'pbmc_granulocyte_sorted_10k_atac_peaks_chr4only.bed'
input_peak_file_anotation = 'pbmc_granulocyte_sorted_10k_atac_peak_annotation_chr4only.tsv'
motif_scan_file = 'pbmc_granulocyte_sorted_10k_atac_peaks_motif_matches_chr4only.bed'
motif_occupation_matrix_file = 'pbmc_granulocyte_sorted_10k_atac_peaks_motif_matches_chr4only.tsv'
tobias_footprint_file = 'pbmc_granulocyte_sorted_10k_atac_peaks_footprints_chr4only.bw'
peaks_footprint_file = 'PBMC10k_peaks_footprints.csv'
gene_annotation_file = 'gencode.v38.annotation.gff3'
TSS_annotation_file = 'TSS_annotation_HG38.csv'

peaks = pd.read_csv(data_folder + input_peak_file_anotation, sep='\t')

# load scanned motifs and check what was found 

scanned_motifs = pd.read_csv(data_folder + motif_scan_file, sep='\t', skiprows=5, header=None)

scanned_motifs['length'] = abs(scanned_motifs[2] - scanned_motifs[1])

peaks['length'] = abs(peaks['end'] - peaks['start'] )

motif_occupation_matrix = pd.read_csv(data_folder + motif_occupation_matrix_file, sep='\t', skiprows=5, index_col=0)

# 1. Prepare the motif mapper
# prepare the TF_motif_data dataframe
# for CellOracle motifs, only to be done once, then loaded from file:

# we can prepare one file, including an indicator 
# it turns out to be most conveniently saved as dicionary pickle for now
motif_mapper = {}
all_motifs = default_motifs()
for m in all_motifs:
    motif_mapper[m] = m.factors

# export the pickle
motif_mapper_file = 'gimmemotifs_mapping_file.pickle'
with open(result_folder + motif_mapper_file, 'wb') as handle:
    pickle.dump(motif_mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # 2. Prepare the peak file
# start the annotation with TOBIAS footprints
bw = pyBigWig.open(result_folder + tobias_footprint_file)

scores_mean = []
scores_sum  = []
scores_max  = []

# score all peaks
for chrom, start, end in zip(peaks['chrom'], peaks['start'], peaks['end']):
    try:
        values = bw.values(chrom, start, end)
        scores_mean.append(np.mean(values))
        scores_sum.append(np.sum(values))
        scores_max.append(np.max(values))
    except:
        print('skip entry', chrom, start, end)
        scores_mean.append(0)
        scores_sum.append(0)
        scores_max.append(0)
        
    
# add to file
peaks['ftp_scores_mean'] = np.nan_to_num(scores_mean)
peaks['ftp_scores_sum']  = np.nan_to_num(scores_sum)
peaks['ftp_scores_max']  = np.nan_to_num(scores_max)


# save peaks to result
peaks.to_csv(result_folder+peaks_footprint_file)

# load peaks to result
peaks2 = pd.read_csv(result_folder+peaks_footprint_file, index_col=0)

# # 3. TSS peak annotation
# load HG38 annotation file to receive IDs
infile = open(data_universal_folder + gene_annotation_file,'r')
new_lines = []

for line in infile.read().split('\n'):
    if len(line) == 0 or line [0] == '#':
        continue
    
    else:
        line_info = []
        info      = line.split('\t')
        line_info.append(info[0])
        line_info.append(info[3])
        line_info.append(info[4])
        line_info.append(info[2])
        line_info.append(info[6])
        tmp = info[8].split(';')
        id_1      = '.'
        gene_name = '.'
        gene_type = '.'
        hgnc_id   = '.'
        for i in tmp:
            if i.startswith('ID='):
                id_1      = i.split('=')[1]
            if i.startswith('gene_type'):
                gene_type = i.split('=')[1]
            if i.startswith('gene_name'):
                gene_name = i.split('=')[1]     
            if i.startswith('hgnc_id'):
                hgnc_id = i.split('=')[1]     
        line_info.append(id_1)
        line_info.append(gene_name)
        line_info.append(gene_type)
        line_info.append(hgnc_id)    
    new_lines.append(line_info)
    
genome_annotation = pd.DataFrame(new_lines)

server  = Server(host='http://www.ensembl.org')
dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']

results = dataset.query(attributes=[
    'ensembl_gene_id',
    'external_gene_name',
    'transcription_start_site'
])

# dictionary from gene_dict
gene_dict = results.set_index('Gene stable ID')['Transcription start site (TSS)'].to_dict()

genome_annotation[9] = [x.split('.')[0] for x in list(genome_annotation[5])]

new_tss_anntation = []
for i in genome_annotation[9]:
    if i in gene_dict:
        new_tss_anntation.append(gene_dict[i])
    else:
        new_tss_anntation.append('.')

genome_annotation['TSS'] = new_tss_anntation

genome_annotation = genome_annotation.rename(columns={0: 'chr', 1: 'start', 2:'end', 3:'type',  
                                                      4:'strand', 5: 'id_full', 6: 'gene_name', 7: 'gene_type', 8:'hng_symbol', 9: 'id_short'})

genome_annotation.to_csv(result_folder + TSS_annotation_file)

genome_annotation_genes = genome_annotation[genome_annotation['type'] == 'gene']

# now, prepare two bed files, one for the extended TSS files (+ threshold_1, - thredhold_2), and one encoding the peaks

# for the peaks
bed_records = zip(peaks2['chrom'], peaks2['start'], peaks2['end'])

# Write to a BED file
with open(result_folder + '/peaks.bed', 'w') as bedfile:
    for record in bed_records:
        bedfile.write('\t'.join(map(str, record)) + '\n')
        
bed_records = zip(peaks2['chrom'], peaks2['start'], peaks2['end'])


# Convert the 'TSS' column to numeric (if it isn't already)
genome_annotation_genes['TSS'] = pd.to_numeric(genome_annotation_genes['TSS'], errors='coerce')

chr_new       = []
start_new     = []
end_new       = []
tss_new       = []
id_new        = []
gene_name_new = []

threshold_up   = 2500000
threshold_down = 2500000

for chr_, id_, tss_, gn in zip( genome_annotation_genes['chr'], genome_annotation_genes['id_short'], genome_annotation_genes['TSS'], genome_annotation_genes['gene_name']):
    if not np.isnan(tss_):
        chr_new.append(chr_)
        end_new.append(int(tss_+threshold_down))
        tss_new.append(int(tss_))
        id_new.append(id_)
        gene_name_new.append(gn)
        if tss_ <= threshold_up:
            start_new.append(0)
        else:
            start_new.append(int(tss_-threshold_up))
        

# for TSS sites
bed_records = zip(chr_new, start_new, end_new, tss_new, id_new, gene_name_new )

# Write to a BED file
with open(result_folder + '/tss.bed', 'w') as bedfile:
    for record in bed_records:
        bedfile.write('\t'.join(map(str, record)) + '\n')

# now intersect them to get the annotations

# Create BedTool objects for the two BED files
bedfile1 = pybedtools.BedTool(result_folder + '/peaks.bed')  # Gene regions with thresholds
bedfile2 = pybedtools.BedTool(result_folder + '/tss.bed')  # Peak locations

# Intersect peaks with gene regions, including annotations from bedfile1
intersected = bedfile1.intersect(bedfile2, wa=True, wb=True)

# Save the annotated results or process further
intersected.saveas(result_folder + '/TSS_annotated_peaks.bed')

chr_new       = []
start_new     = []
end_new       = []
tss_new       = []
id_new        = []
gene_name_new = []

threshold_up   = 2500000
threshold_down = 2500000

for chr_, id_, tss_, gn in zip( genome_annotation_genes['chr'], genome_annotation_genes['id_short'], genome_annotation_genes['TSS'], genome_annotation_genes['gene_name']):
    if not np.isnan(tss_):
        chr_new.append(chr_)
        end_new.append(tss_+threshold_down)
        tss_new.append(tss_)
        id_new.append(id_)
        gene_name_new.append(gn)
        if tss_ <= threshold_up:
            start_new.append(0)
        else:
            start_new.append(tss_-threshold_up)
        
# # 4. TFBS-space annotation
# Load the BED files as BedTool objects
bedfile_a = pybedtools.BedTool(data_folder + motif_scan_file)
bedfile_b = pybedtools.BedTool(result_folder + '/peaks.bed')

# Intersect the two BED files to get overlapping regions
# This will include all entries from file A and the overlapping entries from file B
intersected = bedfile_a.intersect(bedfile_b, wa=True, wb=True)

# Save the result or process further
intersected.saveas(result_folder + 'TFBS_to_peak.bed')




