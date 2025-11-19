import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import pickle

# 1. base GRN inference
# The idea is to consolidate the potential GRN space, and allow for effective pre-filtering. The structure will be able to load information on four different layers,
# and order them into fast-access dictonary relationship files. An base eGRN object should store all necessary information about the current filter applied, and not discard source information in case a less strict filtering is required.
#   
#   
# The current input files were prepared with the FilterPeaksAndFindMotifs.ipynb script
# - First input:    TF-motif mapping table (pre-computable)
# - Second input:   Full peak space (+ optional motif footprints per peak)
# - Third input:    Full TFBS space annoatead with their respective peak 
# - Fourth input:   A distance - relationship mapping between TGs and peaks (pre-computable)
# 
# The recomended handle of this objects are as follows:
# - initalize the objects
# - load the source files
# - set a filtering and build the corresponding eGRN structure


class EnhancerGRN:
    """
    Core enhancer GRN dictionary structure.
    Lets make it as simple as possible, and reverse each dictionary order
    for maximal lookup speed (given the slowness of python of course).
    """
    def __init__(self):
        # pandas dataframe for the full data
        self.TF_motif_data     = None
        self.Full_peak_data    = None
        self.Full_TFBS_data    = None
        self.Full_TG_peak_data = None
        self.TF_set            = None
        self.test_variable     = None

        # TF - Motif layer and inverse
        self.tf_tfbs_map = {}  # connects all TFs to a TFBS (motif)
        self.tfbs_tf_map = {}  # connects a TFBS (motif) to all its TFs
        # TFBS - RE layer and inverse
        self.tfbs_re_map = {}  # map each tfbs to all its REs
        self.re_tfbs_map = {}  # map each RE to all its TFBS
        # TF - RE layer and inverse
        self.tf_re_map   = {}  # map each tfbs to all its REs
        self.re_tf_map   = {}  # map each RE to all its TFBS
        # TG - RE layer and inverse
        self.tg_re_map   = {}  # map each tfbs to all its REs
        self.re_tg_map   = {}  # map each RE to all its TFBS
        # TF - TG layer and inverse
        self.tf_tg_map   = {}  # map each tfbs to all its REs
        self.tg_tf_map   = {}  # map each RE to all its TFBS

    def load_TF_list(self, path):
        """
        Expected: A csv file that contains motifs classified as
        Alternative input: a list or set of TFs
        """
        if type(path) == str:
            self.TF_set = set(pd.read_csv(Lovering_TF_list, header=None)[0])
        if type(path) == set or type(path) == list:
            self.TF_set = set(path)
        
    def load_tf_motif_data(self, path):
        """
        Expected: A python dictionary that maps for each motif what TFs are binding.
        """
        with open(path, 'rb') as handle:
            self.TF_motif_data = pickle.load(handle)

    def load_peak_data(self, path):
        """
        Expected: A csv file, containing the peak, as well as footprint scores
        """
        self.Full_peak_data = pd.read_csv(path, index_col=0)
        
    def load_TG_peak_data(self, path):
        """
        Expected: A bed file, containing 
        """
        self.Full_TG_peak_data = pd.read_csv(path, sep='\t', header=None)

    def load_TFBS_peak_data(self, path):
        """
        Expected: A csv file, containing the peak, as well as 
        """
        self.Full_TFBS_data = pd.read_csv(path, sep='\t', header=None) 
        
    def create_tf_tfbs_map(self, TF_filter=True, capitalize_TFs = True, features = ['direct', 'indirect\nor predicted']):
        """
        Function to make a tf_tfbs mapping as well as its inverse. All non-mapping motifs are removed.
        If TF_filter is set to True, all motifs are additionally filtered to target only TFs 
        """

        for motif in self.TF_motif_data:
            
            # get current targets
            current_targets = self.TF_motif_data[motif]
            # strip motif
            motif = str(motif).rpartition('_')[0]
            new_target_list = []
            # add all TFs in all defined feature variables
            for feature in features:
                new_target_list.extend(current_targets[feature])
            # capitalize all elements and remove redundance
            if capitalize_TFs == True:
                new_target_list = set([x.upper() for x in  new_target_list])
            else:
                new_target_list = set(new_target_list)
            # intersect with the TF set
            new_target_list = self.TF_set.intersection(new_target_list)
            #print(new_target_list)
            # add features if set is not empty
            if len(new_target_list) > 0:
                self.tfbs_tf_map[str(motif)] = new_target_list
            # add inverse 
            for TF in new_target_list: 
                if TF not in self.tf_tfbs_map:
                    self.tf_tfbs_map[TF] = {str(motif)}
                else:
                    self.tf_tfbs_map[TF].add(str(motif))
                    
    def create_tfbs_re_map(self, motif_score_threshold = 3, motif_name_index = 3, peak_start_index = 6, motif_score_index = 4):
        """
        Create a mapping between motif positions and their respective REs. This is done 
        on the base of the already prepared Full_TFBS_data. This contains at the inital
        positions the information of the motif, and on the last positions the information 
        of the peaks. With motif_name_index and peak_start_index, one can indicate the corresponding
        columbs of the motif name (identifier) as well as the name of the RE (generallt tracked
        asexpressed as chr_start_end in this analysis).
        """
        # filter TFBS for motif score threshold
        filtered_peaks = self.Full_TFBS_data[self.Full_TFBS_data[motif_score_index] >=  motif_score_threshold]

        for index, row in filtered_peaks.iterrows():
            # get the name of the motif, as well as its peak position
            source = row[motif_name_index]
            target = str(row[peak_start_index])+'_'+str(row[peak_start_index+1])+'_'+str(row[peak_start_index+2])
            # add it to the tfbs_re_map
            if source not in self.tfbs_re_map:
                self.tfbs_re_map[str(source)] = {str(target)}
            else:
                self.tfbs_re_map[str(source)].add(str(target))
            # add the inverse to re_tfbs_map
            if target not in self.re_tfbs_map:
                self.re_tfbs_map[target] = {str(str(source))}
            else:
                self.re_tfbs_map[target].add(str(str(source)))
                
    def create_tf_re_map(self):
        """
        Map the TFs to the RE via its motifs. Function needs the previously functions 
        create_tf_tfbs_map() and create_tfbs_re_map() to be run.
        """
        # check status of self.tfbs_re_map and self.tfbs_tf_map:
        if len(self.tfbs_re_map) == 0 or len(self.tfbs_tf_map) == 0:
            print("Run create_tf_tfbs_map() and create_tfbs_re_map() first to enable the mapping function")
            return 'Error'

        for motif1 in self.tfbs_re_map:
            # Check whether the motif has annotated TFs that bind it
            if motif1 in self.tfbs_tf_map:

                # add for each TF the set of REs it binds too
                for TF in self.tfbs_tf_map[motif1]:
                    if TF not in self.tf_re_map:
                        self.tf_re_map[TF] = self.tfbs_re_map[motif1]
                    else:
                        self.tf_re_map[TF].union(self.tfbs_re_map[motif1])

                # add for each RE all TF that bind it
                for RE in self.tfbs_re_map[motif1]:
                    if RE not in self.re_tf_map:
                        self.re_tf_map[RE] = self.tfbs_tf_map[motif1]
                    else:
                        self.re_tf_map[RE].union(self.tfbs_tf_map[motif1])
                        
    def create_re_tg_map(self, TSS_threshold_up=500000, TSS_threshold_down=500000, chromosome_index=0, peak_start_index=1, peak_end_index = 2, tss_index = 6, gene_id_index=8):
        """
        Takes the TSS-peak annotations and transforms them into dictionaries.
        In case the file name has differetn headers, the index parameters can be changed to
        match column headers
        
        TODO: since there was an error in the first implementation, TSS_threshold_up will be used for both directions
        for now, instead of custom TSS
        """
        # check precondition for function
        if len(self.Full_TG_peak_data) == 0:
            print("Provide Full_TG_peak_data")
        
        # first, filter to desired max TSS distance. 
        Filtered_TG_peak_data = self.Full_TG_peak_data.copy()

        # since we need to check rather specific condtions, we do it in a zip loop and not in the df
        pass_filter = []
        for i,j,k in zip(Filtered_TG_peak_data[peak_start_index], Filtered_TG_peak_data[peak_end_index], Filtered_TG_peak_data[tss_index]):
            center_of_peak = j-(int(j-i)/2)
            # check position for upstream condition
            if center_of_peak <= k:
                if abs(k-center_of_peak) <= TSS_threshold_up:
                    pass_filter.append(True)
                else:
                    pass_filter.append(False)
            if center_of_peak > k:
                if abs(k-center_of_peak) <= TSS_threshold_down:
                    pass_filter.append(True)
                else:
                    pass_filter.append(False)
        Filtered_TG_peak_data['pass_filter'] = pass_filter
        Filtered_TG_peak_data = Filtered_TG_peak_data[Filtered_TG_peak_data['pass_filter'] == True]
        
#        Filtered_TG_peak_data = Filtered_TG_peak_data[ abs( abs(Filtered_TG_peak_data[peak_start_index]-Filtered_TG_peak_data[peak_end_index]) - Filtered_TG_peak_data[tss_index]) <= TSS_threshold_up   ]
        self.test_variable = Filtered_TG_peak_data.copy()
        # Last, create re_tg_map and tg_re_map based on the filter set-up
        
        for index, row in Filtered_TG_peak_data.iterrows():
            TG   = str(row[gene_id_index])
            peak = str(row[chromosome_index])+'_'+str(row[peak_start_index])+'_'+str(row[peak_end_index])
            # add TG_RE layer
            if TG not in self.tg_re_map:
                self.tg_re_map[TG] = set()
                self.tg_re_map[TG].add(peak)
            else:
                self.tg_re_map[TG].add(peak)
            # add RE_TG layer
            if peak not in self.re_tg_map:
                self.re_tg_map[peak] = set()
                self.re_tg_map[peak].add(TG)
            else:
                self.re_tg_map[peak].add(TG)
                
    def create_tf_tg_map(self):
        """
        Map all TF to their potential target genes and vice versa.
        Carefull: saturation effects occur if too many combinations are considered, leading
        to an effect that every TF connects to every TG
        """

        # Create TF-TG relation based on TF-RE and RE-TG relation
        for TF in self.tf_re_map.keys():
            for RE in self.tf_re_map[TF]:
                if RE in self.re_tg_map:
                    for TG in self.re_tg_map[RE]:
                        # fill tf_tg_map
                        if TF not in self.tf_tg_map:
                            self.tf_tg_map[TF] = set()
                            self.tf_tg_map[TF].add(TG)
                        else:
                            self.tf_tg_map[TF].add(TG)
                        # fill tg_tf_map
                        if TG not in self.tg_tf_map:
                            self.tg_tf_map[TG] = set()
                            self.tg_tf_map[TG].add(TF)
                        else:
                            self.tg_tf_map[TG].add(TF)



if __name__ == "__main__" : 

    # different pathes 
    tf_motif_mapping_file = 'results/human_PBMC_10x_10k/gimmemotifs_mapping_file.pickle'
    peak_footprint_file   = 'results/human_PBMC_10x_10k/PBMC10k_peaks_footprints.csv'
    TSS_annotated_peaks   = 'results/human_PBMC_10x_10k/TSS_annotated_peaks.bed'
    TFBS_peak_anntoation  = 'results/human_PBMC_10x_10k/TFBS_to_peak.bed'
    Lovering_TF_list      = 'data/universal_files/Lovering_TF_16_12_22.txt'

    # initalize object
    print("***************** baseGRN *********************")
    print("***********************************************")
    print("Start running the model...")
    base_eGRN = EnhancerGRN()
    print("Integrating data...")
    # Provide the file path to the TF motif mapping file
    base_eGRN.load_tf_motif_data(tf_motif_mapping_file)
    # provide the file path to the peak_tf_file
    base_eGRN.load_peak_data(peak_footprint_file)
    # load TSS annotation for the peaks
    base_eGRN.load_TG_peak_data(TSS_annotated_peaks)
    # load TSS annotation for the peaks
    base_eGRN.load_TFBS_peak_data(TFBS_peak_anntoation)
    # Load lovering TF list
    base_eGRN.load_TF_list(Lovering_TF_list)
    print("Start mapping process...")
    # map TF to TFBS
    base_eGRN.create_tf_tfbs_map()
    # map TFBS to REs
    base_eGRN.create_tfbs_re_map()
    # map TF to REs
    base_eGRN.create_tf_re_map()
    # map REs to TG
    base_eGRN.create_re_tg_map()
    # map TF to TG
    base_eGRN.create_tf_tg_map()

    # lets pickle the baseGRN
    # first delete the variable that causes problems with reloading since it needs CellOracle:
    base_eGRN.TF_motif_data = None
    # export the pickle
    with open('models/baseGRN_500k.pickle', 'wb') as handle:
        pickle.dump(base_eGRN, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model saved!")