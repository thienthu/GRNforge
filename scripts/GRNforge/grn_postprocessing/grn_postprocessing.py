import sys
import os
import glob, re
import pandas as pd
import numpy as np
import pickle 
import gzip
import anndata as ad
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import sklearn.metrics
import sklearn as sk
import scanpy as sc
from pyscenic.aucell import create_rankings, enrichment, aucell, derive_auc_threshold, create_rankings
from ctxcore.genesig import GeneSignature
from pySCENIC_binarization import binarize
from pySCENIC_binarization import derive_threshold
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.append(os.path.join(parent_dir, "grn_prediction"))
# print(os.path.join(parent_dir, "grn_prediction"))
# sys.path.append("../grn_prediction")

from baseGRN import EnhancerGRN
from sklearn.preprocessing import MinMaxScaler

# Function to generate singature files (gmt)
def generateSignature(singnature_name, singature_genes, description='Description'):
    '''
    Generates one line of a signature file for gtm files
    Arguments:
        singnature_name      : type string
        singature_genes      : type list of strings
        description          : (Optional) singature description if available
    '''
    return singnature_name + '\t' + description + '\t'+ '\t'.join(singature_genes)

def loadSCENICResult(path, genes=None, min_regulon_size= 1, weights=None):
    """
    Helper function to load the scenig result (zero-one file for all TF-targets per regulon)
    Carefull: colnames can be duplicate gene names, depending on which motifs were found 
    """
    weight_dic = {}
    # load and process weights
    if weights != None:
        df_w = pd.read_csv(weights,sep='\t')
        for TF, TG, W in zip(df_w['TF'], df_w['target'],df_w['importance']):
            weight_dic[TF+'~'+TG] = W
           # print(TF+'~'+TG,  W)

    # Load GRN
    GRN_full_df =  pd.read_csv(path, sep=',', index_col = 0)

    # get all Regulon names
    TF_names = {}
    for i in list(GRN_full_df.columns):
        if i != 'regulonMatrix.gene' and i != 'genes':
            TF_names[i] = i.replace('regulonMatrix.', '').split('_')[0]
    
    # clean up wrong column names:
    if 'gene' in TF_names:
        del TF_names['gene']
    
    #print(TF_names)
    TF_TG_interactions          = {}
    TF_TG_interactions_weighted = {}
    
    if genes != None:
        for i in TF_names:
            interactions  = []
            status        = []
            weight_list   = []
            for index, row in GRN_full_df[i].items():
                # return
                # make a filter step to check whether gene name is present in dataframe
                if index in genes:
                    interactions.append(TF_names[i]+'~'+index)
                    status.append(abs(int(row)))
                    if weights!= None:
                        if TF_names[i]+'~'+index in weight_dic:
                            weight_list.append(weight_dic[TF_names[i]+'~'+index])
                        else:
                             weight_list.append(0)
                
            # add a regulon size filter at this position, setting a minimum amount of edges per regulon
            df = pd.DataFrame(list(zip(interactions, status)), columns =['Interaction', 'Status'])
            df_2 = pd.DataFrame(list(zip(interactions, weight_list)), columns =['Interaction', 'weight_list'])
            if len(df[df['Status'] > 0]) >= min_regulon_size:
                TF_TG_interactions[TF_names[i]] = pd.DataFrame(list(zip(interactions, status)), columns =['Interaction', 'Status'])
                if weights!= None:
                    TF_TG_interactions_weighted[TF_names[i]] = pd.DataFrame(list(zip(interactions, weight_list)), columns =['Interaction', 'Status'])
    else:
        for i in TF_names:
            #print(i)
            interactions  = []
            status        = []
            weight_list   = []
            for index, row in GRN_full_df[i].items():
                interactions.append(TF_names[i]+'~'+index)
                status.append(abs(int(row)))
                if weights!= None:
                    if TF_names[i]+'~'+index in weight_dic:
                        weight_list.append(weight_dic[TF_names[i]+'~'+index])
                    else:
                        weight_list.append(0)
            #print(len(interactions), len(status), len(weight_list))  
            # add a regulon size filter at this position, setting a minimum amount of edges per regulon
            df = pd.DataFrame(list(zip(interactions, status)), columns =['Interaction', 'Status'])
            df_2 = pd.DataFrame(list(zip(interactions, status)), columns =['Interaction', 'weight_list'])
            if len(df[df['Status'] > 0]) >= min_regulon_size:
                TF_TG_interactions[TF_names[i]] = pd.DataFrame(list(zip(interactions, status)), columns =['Interaction', 'Status'])
                if weights!= None:
                    TF_TG_interactions_weighted[TF_names[i]] = pd.DataFrame(list(zip(interactions, weight_list)), columns =['Interaction', 'Status'])
    #print(TF_TG_interactions_weighted)

    return TF_TG_interactions, TF_TG_interactions_weighted

def intersection(lst1, lst2):
    lst2 = dict.fromkeys(lst2, [])
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def intersect_pf(Regulon_to_compare, ChIP_gold_df):
    """
    The goal of this function is to intersect entries that were consideret inthe inferred networks with interations present in the gold standard
    Also remove duplicates, in case they could appear for some reason in the Gold Standard.
    The function works because both dataframes have ALL possible interactions realised, meaning each TG-TF pair that was measured
    or was in the gold standard has an id (and can be 0 or 1, whether it exists or not).
    Each regulon TF in a sample is therefore compared to the gold standard for ALL possible gene interactions. 
    The filtered out interactions with genes that were not measured in the gold standard, or were not measured in the dataset 
    """
    # Drop dupliate interactions (normally they do not exist)
    Regulon_to_compare.drop_duplicates(subset=['Interaction'], inplace = True)
    ChIP_gold_df.drop_duplicates(subset=['Interaction'], inplace = True)
    # Get interactions of the same id's
    same_ids = intersection(list(Regulon_to_compare['Interaction']),list(ChIP_gold_df['Interaction']))
    #print("check stats:", len(same_ids))
    #print(list(Regulon_to_compare['Interaction']))
    #print(ChIP_gold_df['Interaction'])
    I_TG =  Regulon_to_compare[Regulon_to_compare['Interaction'].isin(same_ids)] 
    I_Gold =  ChIP_gold_df[ChIP_gold_df['Interaction'].isin(same_ids)] 
    
    # sort both datasets, so that the index of an interaction is the same between both
    I_TG   = I_TG.sort_values(by=['Interaction'], ignore_index=True, inplace=False)
    I_Gold = I_Gold.sort_values(by=['Interaction'],  ignore_index=True, inplace=False)
    
    return I_TG,I_Gold

def prepareCollecTRIGoldStandard(path, evidence_score=1, complex_filter = 'complex_removed'):
    """
    Function to load CollecTRI, filter it based on evidence, and return two gold standards: Full and filtered.
    For complex_filter, one can set all (no filtering), complex_only (only complexes), complex_removed (only non 
    complex interactions)
    """
    # load CollecTRI into pandas dataframe
    CTRI              = pd.read_csv(path,sep='\t')
    # annotate whether an interaction is a complex (allows to remove them later)
    complex_indicator = [True if 'COMPLEX' in i else False for i in list(CTRI['source'])]
    CTRI['is_complex'] = complex_indicator

    # Test filter
    if complex_filter == 'complex_only':
        CTRI = CTRI[ CTRI['is_complex'] == True]
    elif complex_filter == 'complex_removed':
        CTRI = CTRI[ CTRI['is_complex'] == False]
    else:
        CTRI = CTRI
    
    # Make gold standard for unfiltered case:
    # Get TG and TF names
    TFs = {}
    TGs = {}
    for i in CTRI['source']:
        TFs[i] = []
    for i in CTRI['target']:
        TGs[i] = []    

    for i in list(zip(CTRI['source'], CTRI['target'])):
        TFs[i[0]].append(i[1])
        
    #print(TFs)
    
    # Ceate TF-TG interaction dataframes for all
    TF_TG_interactions_full = {}
    for TF in TFs:
        interaction_list = []
        status_list      = []
       # print(TF)
        
        for TG in TGs:         
            if TG in TFs[TF]:
                interaction_list.append(TF+'~'+TG)
                status_list.append(1)
            else:
                interaction_list.append(TF+'~'+TG)
                status_list.append(0)
        TF_TG_interactions_full[TF] = pd.DataFrame(list(zip(interaction_list, status_list)), columns =['Interaction', 'Status'])
    #print(TF_TG_interactions_full)

    # Make gold standard for filtered case:
    CTRI = CTRI[CTRI['curation_effort'] >= evidence_score]
    
    # Get TG and TF names
    TFs = {}
    TGs = {}
    for i in CTRI['source']:
        TFs[i] = []
    for i in CTRI['target']:
        TGs[i] = []    

    for i in list(zip(CTRI['source'], CTRI['target'])):
        TFs[i[0]].append(i[1])
    
    # Ceate TF-TG interaction dataframes for all
    TF_TG_interactions_filtered = {}
    for TF in TFs:
        interaction_list = []
        status_list      = []

        for TG in TGs:
            if TG in TFs[TF]:
                interaction_list.append(TF+'~'+TG)
                status_list.append(1)
            else:
                interaction_list.append(TF+'~'+TG)
                status_list.append(0)
        TF_TG_interactions_filtered[TF] = pd.DataFrame(list(zip(interaction_list, status_list)), columns =['Interaction', 'Status'])

    
        
    return(TF_TG_interactions_full, TF_TG_interactions_filtered)

def normalizeImportanceScores(scores, top_k_threshold=None, scaling_factor=100):
    """
    Take all importance scores, and rank them by their relative contribution with regard to 
    the total sum of importance scores.
    """
    new_scores = scores.copy()
    for TG in scores:
        if top_k_threshold == None:
            sum_over_scores = np.sum(list(scores[TG].values()))
        else:
            sum_over_scores = np.sum(sorted(list(scores[TG].values()), reverse=True)[:top_k_threshold]) 
        for TF in scores[TG]:
            new_scores[TG][TF] = (scores[TG][TF]/sum_over_scores)*scaling_factor
    return new_scores

def normalizeForModelPerformance(scores, performance, scaling_factor=1, score_type=1):
    """
    Take all importance scores, and multiply them by a factor that represents
    the [0-1] scaled model performance. Model performance is calcuated relatively to offset the different
    expression ranges by validating it against a mean baseline (mae_prediction/mean_TG)
    The performance dictionary contains tuples of (mean, train_error, test_error)
    """
    new_scores = scores.copy()

    # prepare model scaling factor
    error             = []
    mean              = []
    error_mean_ratio  = {}
    for i in performance:
        #if performance[i][0]:
        mean.append(performance[i][0])
        error.append(performance[i][1])
        if mean[-1] > 0:
            error_mean_ratio[i] = error[-1]/mean[-1]   # if error < mean, the coefficient is < 1
        else:
            error_mean_ratio[i] = 5

    # Divide each error_mean_ratio by the maxium mean_error ration.
    # Then build the inverse, as the lower values indicate better performance
    max_error_mean_ratio = max(error_mean_ratio.values())
    
    for TG in error_mean_ratio:
        error_mean_ratio[TG] =  1-(error_mean_ratio[TG]/max_error_mean_ratio)    
    
    for TG in scores:
        for TF in scores[TG]:
            new_scores[TG][TF] = scores[TG][TF] * error_mean_ratio[TG]
    return new_scores

def makePandasNetwork(regulons, interactions):
    """
    Use on regulons and interactions to create a data structure: 
    Dictionary with TFs as keys and pandas DataFrame of all TGs
    Two columns: Interactions [TF~TG] and Status. Status takes 0 or 1 for binary indication of an edge, or the weight of the edge.
    """
    # Create TF_TG_interactions_binary and TF_TG_interactions_weighted
    #all_target_genes = set(TG for _, TG, _ in selected_interactions)
    all_target_genes = set([x.split('~')[1] for x in  list(interactions.keys())])
    
    TF_TG_interactions_binary = {}
    TF_TG_interactions_weighted = {}

    for TF, tg_dict in regulons.items():
        # Prepare the interaction and status lists
        interaction_list = []
        status_binary_list = []
        status_weighted_list = []

        # Iterate over all target genes to build interaction keys and status
        for target_gene in all_target_genes:
            interaction_key = f"{TF}~{target_gene}"
            interaction_list.append(interaction_key)

            # For binary interaction (1 if interaction exists, 0 if not)
            status_binary = 1 if interaction_key in interactions else 0
            status_binary_list.append(status_binary)

            # For weighted interaction (importance score)
            status_weighted = tg_dict.get(target_gene, 0)
            status_weighted_list.append(status_weighted)

        # Create DataFrames
        TF_TG_interactions_binary[TF] = pd.DataFrame({
            'Interaction': interaction_list,
            'Status': status_binary_list
        })

        TF_TG_interactions_weighted[TF] = pd.DataFrame({
            'Interaction': interaction_list,
            'Status': status_weighted_list
        })    

    return TF_TG_interactions_binary, TF_TG_interactions_weighted

def makeCustomNetwork(scores, minimum_regulon_size=10, number_edges=100000, filter_TGs=None, filter_interactions=None, filter_interactions_keep=True, verbose=True):
    """
    Make a custom network based on the XGBoost scores.
        scores:                scored TF interactions per model
        minimum_regulon_size:  limit size of regulons
        number_edges:          target goal of number of egdes to include
        filter_TFs:            set of TGs to keep. Useful to either include or exclude models based on their quality
        filter_interactions:   set of interactions to keep
    Important note: regulon size is optimized in conjugation with number of edges, meaning if a regulon does not satisfy 
    minimum_regulon_size close to the max. number_edges, it will not be inlcuded and regulons with 'worse' importance score will be added
    if they can deliver the required amount of edges to satisfy number_edges.
    """
    # Reverse dictionary structure: TG -> {TF: score} -> TF -> {TG: score}
    inital_interactions = []
    TF_list             = []
    TG_list             = []
    score_list          = []
    print('before TG filter', len(scores.items()))
    for TG, tf_dict in scores.items():
        # filter out TGs that are not in set if filter_TGs == True
        if filter_TGs !=  None and TG not in filter_TGs:
            continue
        for TF, score in tf_dict.items():
            inital_interactions.append((TF, TG, score))
            TF_list.append(TF)
            TG_list.append(TG)
            score_list.append(score)
    print('After TG filter', len(set(TG_list)))
    
    df = pd.DataFrame({'TF':TF_list, 'TG':TG_list, 'score':score_list})
    df.sort_values(by='score', inplace=True, ascending=False)
    
    inital_interactions   = [tuple(row) for row in df.itertuples(index=False, name=None)]
    inital_interactions_2 = [] 
    # add part to filter out initial interactions that are not allowed

    if filter_interactions !=  None:
        print('initial length of interaction:', len(inital_interactions) )
        for i in inital_interactions:
            if filter_interactions_keep == True and i[0] + "~" + i[1] in filter_interactions:
                inital_interactions_2.append(i)
            if filter_interactions_keep == False and i[0] + "~" + i[1] not in filter_interactions:
                inital_interactions_2.append(i)
        print('length after interaction filter:', len(inital_interactions_2))
    
        inital_interactions =  inital_interactions_2           
           


    if verbose:
        print(f"Total initial interactions: {len(inital_interactions)}")

    # Ensure we do not exceed available edges
    number_edges = min(number_edges, len(inital_interactions))

    # Find the threshold score at position `number_edges`
    thres = inital_interactions[number_edges - 1][2]  # (TF, TG, score) → score
    #print(len(inital_interactions), number_edges, thres, inital_interactions[number_edges - 1][2], inital_interactions[number_edges - 1])
    if np.isnan(thres):
        thres = 0
    #print(thres)
    if verbose:
        print(f"Threshold: {thres}")

    # Collect exactly `number_edges` edges, handling ties deterministically
    selected_interactions = []

    for TF, TG, score in inital_interactions:
        #print(TF, TG, score)

        if len(selected_interactions) >= number_edges:
            break
        if score > thres or (score == thres and len(selected_interactions) < number_edges):
            selected_interactions.append((TF, TG, score))
            
    if verbose:
        print(f"Final edge count: {len(selected_interactions)}")

    # Convert back to TF -> {TG: score} structure
    regulons = {}
    interactions = {}
    
    for TF, TG, score in selected_interactions:
        regulons.setdefault(TF, {})[TG] = score
        interactions[f"{TF}~{TG}"] = score

    # Remove small regulons
    regulons = {TF: tg_dict for TF, tg_dict in regulons.items() if len(tg_dict) >= minimum_regulon_size}
    interactions = {f"{TF}~{TG}": score for TF, tg_dict in regulons.items() for TG, score in tg_dict.items()}


    # Refill the network if needed
    remaining_edges = number_edges - len(interactions)
    if remaining_edges > 0:
        remaining_interactions = [i for i in inital_interactions if f"{i[0]}~{i[1]}" not in interactions]
        added_edges = 0
        candidate_regulons = {}

        # Group remaining interactions by TF
        for TF, TG, score in remaining_interactions:
            candidate_regulons.setdefault(TF, []).append((TG, score))

        # Sort remaining interactions per TF by score
        for TF in candidate_regulons:
            candidate_regulons[TF].sort(key=lambda x: x[1], reverse=True)

        # Add interactions to grow existing regulons first
        for TF, TG_scores in candidate_regulons.items():
            if added_edges >= remaining_edges:
                break
            if TF in regulons:
                for TG, score in TG_scores:
                    if added_edges >= remaining_edges:
                        break
                    if TG not in regulons[TF]:
                        regulons[TF][TG] = score
                        interactions[f"{TF}~{TG}"] = score
                        added_edges += 1

        # Add new regulons only if they meet minimum size
        for TF, TG_scores in candidate_regulons.items():
            if added_edges >= remaining_edges:
                break
            if TF not in regulons and len(TG_scores) >= minimum_regulon_size:
                regulons[TF] = {TG: score for TG, score in TG_scores[:minimum_regulon_size]}
                for TG, score in regulons[TF].items():
                    interactions[f"{TF}~{TG}"] = score
                    added_edges += 1

    
    # create weighted and binary pandas dataframes
    TF_TG_interactions_binary, TF_TG_interactions_weighted = makePandasNetwork(regulons, interactions)
        
    return regulons, interactions, TF_TG_interactions_binary, TF_TG_interactions_weighted

def rank_dict_values(interactions):
    """
    Replace values in a dictionary with their rank.
    The highest value gets the highest numerical rank.
    """
    sorted_items = sorted(interactions.items(), key=lambda x: x[1], reverse=False)
    #print(sorted_items[:1000])
    ranks = {key: rank+1 for rank, (key, _) in enumerate(sorted_items)}
    #print(sorted_items[0], sorted_items[100], sorted_items[1000], sorted_items[-1])
    #print(ranks[0], ranks[100], ranks[1000], ranks[-1])
    #print(ranks)
    #break
    return ranks

def rankIntegration(list_interactions, mode='maxrank', interactions = 30000, minimum_regulon_size = 10, verbose=False):
    """
    Function to integrate two or more networks based on the rank of interactions. Differernt integration modes are:
    maxrank:    Per interaction, take the maximum rank across all networks => This mode preserves best individual high ranks between networks
    meanrank:   This gives an option to downrank/uprank a gene based on consensus mean
    medianrank: This gives an option to downrank/uprank a gene based on consensus median
    minrank:    A meme implementation for test reasons. Expected to give you the worst network in comparison
    boostrank:  Tries to given an advantage ot edges that were found in multiple models. Same as meanrank, but considers the worsed rank for
                edges that do not exist in one model

    Input:  list_interactions
    Output: regulons_integrated, interactions_integrated, TF_TG_interactions_binary, TF_TG_interactions_weighted
    """

    # get all possible interactions 
    all_possible_interactions = set()
    for network in list_interactions:
        all_possible_interactions.update(set(list(network.keys())))

    print(len(all_possible_interactions))
        
    if mode != 'PCA':
        ranked_networks = []
        # calculate ranks (lowest rank = highest value)
        for network in list_interactions:
            ranked_networks.append( rank_dict_values(network) )
        #print(ranked_networks[1])
    
    integrated_interactions = {}
    # start the integration
    #print(all_possible_interactions)
    for interaction in all_possible_interactions:
        if mode == 'maxrank':
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
            integrated_interactions[interaction] = max(ranks)
        if mode == 'minrank':
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
            integrated_interactions[interaction] = min(ranks)
        if mode == 'meanrank':
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
            integrated_interactions[interaction] = np.mean(ranks)
        if mode == 'medianrank':
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
            integrated_interactions[interaction] = np.median(ranks)
        if mode == 'boostrank':
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
                else:
                    ranks.append(1)
            integrated_interactions[interaction] = np.mean(ranks)
        if mode == 'sumrank':
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
            integrated_interactions[interaction] = np.sum(ranks)
        if mode == 'supportrank':
            support = 2
            ranks = []
            for network in ranked_networks:
                if interaction in network:
                    ranks.append(network[interaction])
            if len(ranks) >= support:
                integrated_interactions[interaction] = max(ranks)

    if mode == 'PCA':
        print('make df')
        # this might penanlize missing edges too much
        df = pd.DataFrame(list_interactions)#.fillna(0)
        df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
        # lets mix in the mean approach
       #df = pd.DataFrame(list_interactions).fillna(df.mean(axis=1))       
        print('scale')
        df = pd.DataFrame(MinMaxScaler().fit_transform(df), index=df.index, columns=df.columns)
        print('transpose')
        df = df.T
        
        print('PCA')
        pcn = PCA(n_components=2).fit_transform(df)
        pc1 = pcn[:, 0] # 0 = first
        print('check direction and fix if necessary')
        mean_scores = df.mean(axis=1).values   # average across all models for each interaction
        print(np.corrcoef(pc1.flatten(), mean_scores)[0, 1])
        if np.corrcoef(pc1.flatten(), mean_scores)[0, 1] < 0:
            print('flip axis direction')
            pc1 = -pc1
        
        print('integrated_interactions')
        integrated_interactions = dict(zip(df.index, pc1.flatten()))
  
    #print(integrated_interactions.keys())
    # prune edges to desired number if necessary
    if interactions == 'all':
        interactions = len(integrated_interactions)

    # Collect exactly `number_edges` edges, handling ties deterministically
    print(len(integrated_interactions))

    if verbose:
        print(f"Total initial interactions: {len(inital_interactions)}")

    # Ensure we do not exceed available edges
    number_edges = min(interactions, len(integrated_interactions))

    # Find the threshold score at position `number_edges`
    thres = sorted(list(integrated_interactions.values()), reverse=True)[number_edges - 1]  # (TF, TG, score) → score
    #print(sorted(list(integrated_interactions.values()), reverse=True))
    #print(sorted(list(integrated_interactions.values()), reverse=True))
    #print(thres, number_edges)
    #print(number_edges)
    selected_interactions = []
    for edge in integrated_interactions:
        TF = edge.split('~')[0]
        TG = edge.split('~')[1]
        score = integrated_interactions[edge]
        #print(TF, TG, score)
        if len(selected_interactions) >= number_edges:
            break
        if score > thres:# or (score == thres and len(integrated_interactions) < number_edges):
            selected_interactions.append((TF, TG, score))
        
    #print(selected_interactions)
    if verbose:
        print(f"Final edge count: {len(selected_interactions)}")

    # Convert back to TF -> {TG: score} structure
    regulons = {}
    interactions = {}
    print(len(selected_interactions))
    for TF, TG, score in selected_interactions:
        regulons.setdefault(TF, {})[TG] = score
        interactions[f"{TF}~{TG}"] = score
    print(len(interactions))
    # Remove small regulons
    regulons = {TF: tg_dict for TF, tg_dict in regulons.items() if len(tg_dict) >= minimum_regulon_size}
    interactions = {f"{TF}~{TG}": score for TF, tg_dict in regulons.items() for TG, score in tg_dict.items()}

    #interactions_dict = {f"{tf}~{tg}": sc for tf, tgmap in regulons.items() for tg, sc in tgmap.items()} 
    
   # Refill the network if needed
    remaining_edges = number_edges - len(interactions)
    #print('remaining_edges',remaining_edges)
    #print(integrated_interactions)
    if remaining_edges > 0:
        #remaining_interactions = [i for i in all_possible_interactions if f"{i[0]}~{i[1]}" not in interactions]
        #remaining_interactions = [i for i in integrated_interactions if f"{i[0]}~{i[1]}" not in interactions]

        triplet_interactions = [(k.split('~')[0], k.split('~')[1], v) for k, v in integrated_interactions.items()]
        remaining_interactions = [
            (TF, TG, score)
            for TF, TG, score in triplet_interactions
            if f"{TF}~{TG}" not in interactions
        ]

        
        #print('remaining_interactions', remaining_interactions)
        added_edges = 0
        candidate_regulons = {}

        # Group remaining interactions by TF
        for TF, TG, score in remaining_interactions:
            candidate_regulons.setdefault(TF, []).append((TG, score))

        # Sort remaining interactions per TF by score
        for TF in candidate_regulons:
            candidate_regulons[TF].sort(key=lambda x: x[1], reverse=True)

        # Add interactions to grow existing regulons first
        for TF, TG_scores in candidate_regulons.items():
            if added_edges >= remaining_edges:
                break
            if TF in regulons:
                for TG, score in TG_scores:
                    if added_edges >= remaining_edges:
                        break
                    if TG not in regulons[TF]:
                        regulons[TF][TG] = score
                        interactions[f"{TF}~{TG}"] = score
                        added_edges += 1

        # Add new regulons only if they meet minimum size
        for TF, TG_scores in candidate_regulons.items():
            if added_edges >= remaining_edges:
                break
            if TF not in regulons and len(TG_scores) >= minimum_regulon_size:
                regulons[TF] = {TG: score for TG, score in TG_scores[:minimum_regulon_size]}
                for TG, score in regulons[TF].items():
                    interactions[f"{TF}~{TG}"] = score
                    added_edges += 1

                        
    # create weighted and binary pandas dataframes
    TF_TG_interactions_binary, TF_TG_interactions_weighted = makePandasNetwork(regulons, interactions)
    # get regulons from integrated_interactions:
    return regulons, interactions, TF_TG_interactions_binary , TF_TG_interactions_weighted       

def delineateComplexScores(scores, mode='max'):
    """
    This function asigns individual scores to each TF that is involved in a complex.
    Right now, every TF is assigned it's highest score that it archived across all complexes
    """
    new_scores =  {}
    for TG in scores:
        new_scores[TG] = {}
        for TF in scores[TG]:
            #print(TF, scores[TG][TF])
            if '~' in TF:
                TFs = TF.split('~')
                #print(TFs)
                for i in TFs:
                    if i not in new_scores[TG]:
                        new_scores[TG][i] = scores[TG][TF]
                    else:
                        if new_scores[TG][i] < scores[TG][TF]:
                             new_scores[TG][i] =  scores[TG][TF]
            else:
                if TF not in new_scores[TG]:
                    new_scores[TG][TF] = scores[TG][TF]
                else:
                    if new_scores[TG][TF] < scores[TG][TF]:
                        new_scores[TG][i] =  scores[TG][TF]    
    return new_scores

def makeTFBSenrichedNetwork(df, minimum_regulon_size=10, number_edges=100000, filter_TGs=None, verbose=True):
    """
    Take a pandas DataFrame with information about motif enrichment per TF and make a network out of it.
        df:                  DataFrame with columns ['TF', 'TG', 'log2_odds_ratio']
        minimum_regulon_size: Minimum size of a regulon to keep
        number_edges:         Maximum number of edges to include
        filter_TGs:           Set of TGs to keep. Useful to include/exclude specific TGs
    """
    # Sort by log2 odds ratio (descending)
    df_sorted = df.sort_values(by='log2_odds_ratio', ascending=False)

    # Filter out rows with infinite scores
    df_sorted = df_sorted[np.isfinite(df_sorted['log2_odds_ratio'])]
    
    # Apply target gene filtering if provided
    if filter_TGs is not None:
        df_sorted = df_sorted[df_sorted['TG'].isin(filter_TGs)]
    
    # Limit to the top N edges
    df_sorted = df_sorted.head(number_edges)
    
    if verbose:
        print(f"Total edges after filtering and sorting: {len(df_sorted)}")
    
    # Build regulon structure
    regulons = {}
    interactions = {}
    
    for row in df_sorted.itertuples(index=False):
        TF, TG, score = row.TF, row.TG, row.log2_odds_ratio
        regulons.setdefault(TF, {})[TG] = score
        interactions[f"{TF}~{TG}"] = score
    
    # Remove small regulons
    regulons = {TF: tg_dict for TF, tg_dict in regulons.items() if len(tg_dict) >= minimum_regulon_size}
    
    # Re-check and prune interactions to match the updated regulons
    interactions = {f"{TF}~{TG}": score for TF, tg_dict in regulons.items() for TG, score in tg_dict.items()}
    
    if verbose:
        print(f"Number of regulons after size filtering: {len(regulons)}")
    
    # Create binary and weighted DataFrames
    TF_TG_interactions_binary, TF_TG_interactions_weighted = makePandasNetwork(regulons, interactions)
    
    return regulons, interactions, TF_TG_interactions_binary, TF_TG_interactions_weighted

def summarize_networks(regulons_all):
    """
    Simply delivers some quick topological information with regard to a network
    """
    summary_data = []

    for network_name, regulons in regulons_all.items():
        num_TFs = len(regulons)
        num_TGs = len(set(TG for TF in regulons for TG in regulons[TF]))
        num_interactions = sum(len(tgs) for tgs in regulons.values())
        regulon_sizes = [len(tgs) for tgs in regulons.values()]
        

        summary_data.append({
            'Network': network_name,
            '#TF': num_TFs,
            '#TG': num_TGs,
            '#Interactions': num_interactions,
            'Mean Regulon Size': np.mean(regulon_sizes),
            'Median Regulon Size': np.median(regulon_sizes),
            'Min Regulon Size': np.min(regulon_sizes),
            'Max Regulon Size': np.max(regulon_sizes)
        })

    summary_df = pd.DataFrame(summary_data).set_index('Network')
    return summary_df

def save_dict_to_pickle(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def calculate_hypergeometric_probability(n1, n2, total_elements=804):
    # Create a range of possible overlaps (0 to min(n1, n2))
    k_values = np.arange(0, min(n1, n2) + 1)
    
    # Calculate the hypergeometric probability mass function for each k
    probabilities = hypergeom.pmf(k_values, total_elements, n1, n2)
    
    # Display probabilities
    for k, prob in zip(k_values, probabilities):
        print(f"Probability of {k} overlaps: {prob:.4f}")
    
    return k_values, probabilities

if __name__ == "__main__" :

    print("-----------------Post-processing step-----------------")
    print("------------------------------------------------------")
    # Folder set-up
    base_result_path = 'results/human_PBMC_10x_10k/GRN_test_runs/'
    run_name         = 'run_1'
    result_folder_1       = 'results/'
    path_to_expression    = 'data/test/PBMC10k_gex_normalized_log1p_testing.h5ad'
    path_to_geneactivity = 'data/test/PBMC10k_gene_activity_matrix_testing.h5ad'
    # path_to_regulons      = 'results/GRN_test_runs/run_1/models_pruned/'
    path_to_signature = 'results/human_PBMC_10x_10k/GRN_test_runs/run_1'
    get_annotation_adata  = 'data/test/PBMC10k_gex_normalized_log1p_ann_testing.h5ad'
    models = ['exp_exp',  'act_act', 'exp_act', 'act_exp', 'expCOP_exp', 'actCOP_act', 'expCOP_act', 'actCOP_exp']
    path_to_thresholds = path_to_signature + '/threshold.csv'
    # load blacklist for genes with overlapping gene bodies in the ATAC-activity based networks to avoid artifical gene activity correlations
    print("Loading data...")
    path_bed_overlap = 'data/universal_files/overlapping_genes.bed'
    bed_file_bl      = pd.read_csv(path_bed_overlap, sep='\t', header=None)
    bed_file_bl[9] = bed_file_bl[8]/(bed_file_bl[2] - bed_file_bl[1])
    blacklist = set()
    for i in zip(bed_file_bl[3], bed_file_bl[7]):
        if i[0] != i[1]:
            blacklist.add(i[0]+'~'+i[1])

    # Load activity and gene activity matrices
    adata_annotation   = sc.read_h5ad( get_annotation_adata)
    adata              = sc.read_h5ad( path_to_expression)
    adata_filtered     = adata[adata.obs.index.isin(adata_annotation.obs_names)].copy()
    ex_matrix          = pd.DataFrame(data = adata_filtered.X.toarray(), index=adata_filtered.obs_names, columns=adata_filtered.var_names)
    geneactivity_adata = sc.read_h5ad(path_to_geneactivity)
    ac_matrix          = pd.DataFrame(data = geneactivity_adata.X.toarray(), index=geneactivity_adata.obs_names, columns=geneactivity_adata.var_names)

    # contains only portein-conding genes (in case we want to clean up a bit the TGs)
    transcript_annotation = pd.read_csv("data/universal_files/all_ensembl_transcripts.csv", index_col=0)
    only_proteincoding_genes = set(transcript_annotation['symbol'])

    # save protein coding gene list in a txt format Style
    with open("data/universal_files/protein_coding_genes.txt", "w") as f:
        for item in only_proteincoding_genes:
            f.write(f"{item}\n")

    # cluster_df = pd.DataFrame({'Barcode': geneactivity_adata.obs['leiden'].index, 'Cluster': geneactivity_adata.obs['leiden']})
    # cluster_df.to_csv('../../../../data//human_PBMC_10x_10k/enhlink/clusters.tsv', index=False, sep='\t')
    print("Identifying threshold...")
    # Calculate the dropout rate for each feature (gene) in the activity matrix. For expression, we already have it at this point
    dropout_percentage = np.sum(geneactivity_adata.X == 0, axis=0) / geneactivity_adata.shape[0] * 100

    # Reshape the dropout percentages to be a column vector
    dropout_percentage = dropout_percentage.reshape(-1, 1)

    # Create a DataFrame to store the dropout percentage for each feature
    dropout_df = pd.DataFrame(
        dropout_percentage,
        index=geneactivity_adata.var_names,
        columns=['pct_dropout_by_counts']
    )
    # Sort the DataFrame to see features with the highest dropout percentage
    dropout_df = dropout_df.sort_values(by='pct_dropout_by_counts', ascending=False)
    geneactivity_adata.var['pct_dropout_by_counts'] = dropout_percentage

    # Define a gene list that satisfies a maximum dropout ratio. Based on the quality plots in chapter 3, we go for 99%
    gene_names_act = geneactivity_adata.var[geneactivity_adata.var['pct_dropout_by_counts'] <= 99].index.tolist()
    gene_names_exp = adata_filtered.var[adata_filtered.var['pct_dropout_by_counts']         <= 99].index.tolist()

    # Here we can use our base GRN to make different TSS_thresholds. Since this takes a long time to calculate, I did save the intermediate results as
    # pickle, and load it at the end of this cell. 

    all_interactions_to_keep = {}
    all_network_sizes        = {}

    # load eGRN structure
    # influence of TSS distance on network performance
    with open('models/baseGRN_500k.pickle', 'rb') as handle:
        base_eGRN = pickle.load(handle)

    for TSS_thres in [500000, 250000, 100000, 50000, 25000, 10000, 2500]:
        # Reset only the necessary maps
        base_eGRN.re_tg_map = {}
        base_eGRN.tf_tg_map = {}
        base_eGRN.tg_tf_map = {}

        # Re-run only the parts that depend on the threshold
        base_eGRN.create_re_tg_map(
            TSS_threshold_up=TSS_thres,
            TSS_threshold_down=TSS_thres,
            chromosome_index=0,
            peak_start_index=1,
            peak_end_index=2,
            tss_index=6,
            gene_id_index=8
        )

        base_eGRN.create_tf_tg_map()

        # Collect interactions
        interactions_to_keep = set()
        for TF, TGs in base_eGRN.tf_tg_map.items():
            for TG in TGs:
                interactions_to_keep.add(f"{TF}~{TG}")

        all_interactions_to_keep[TSS_thres] = interactions_to_keep
        all_network_sizes[TSS_thres] = len(interactions_to_keep)

    save_dict_to_pickle(all_interactions_to_keep, result_folder_1+ '/all_interactions_to_keep_per_TSS_thresold.pkl')

    # It is the fastest to just pickle the data and load it instead of recalcauting the base GRN every time
    all_interactions_to_keep = load_dict_from_pickle(result_folder_1+ '/all_interactions_to_keep_per_TSS_thresold.pkl')

    # lets start with loading the XGBoost outputs for the 8 inter- and intra omics models first. The ranges indicate different blocks of parallelization
    # that were done while calculating the XGBoost outputs, and signify a set of genes respectively
    print("Running model...")
    models = ['exp_exp',  'act_act', 'exp_act', 'act_exp', 'expCOP_exp', 'actCOP_act', 'expCOP_act', 'actCOP_exp']

    # Define data structure
    TG_models_scores            = {}
    TG_models_rank_deviation    = {}
    TG_models_evals             = {}
    TG_models_pred_performance  = {}

    # load results: adapt ranges towards how the model was split
    for m in models:
        # load scores
        folder_path = gparent_dir + '/' +base_result_path+run_name + '/' + 'model_' + m
        file_names = [f for f in os.listdir(folder_path) if f.endswith("_scores.pkl")]
        file_names = sorted(file_names)
        
        for file_name in file_names:
            with gzip.open(folder_path + '/' +file_name, 'rb') as handle:
                scores = pickle.load(handle)
                if m not in TG_models_scores:
                    TG_models_scores[m] = scores
                else:
                    TG_models_scores[m].update(scores)
                    
        # load rank deviations
        folder_path = gparent_dir + '/' +base_result_path+run_name + '/' + 'model_' + m
        file_names = [f for f in os.listdir(folder_path) if f.endswith("_scores_rank_deviation.pkl")]
        file_names = sorted(file_names)
        
        for file_name in file_names:
            with gzip.open(folder_path + '/' +file_name, 'rb') as handle:
                scores = pickle.load(handle)
                if m not in TG_models_rank_deviation:
                    TG_models_rank_deviation[m] = scores
                else:
                    TG_models_rank_deviation[m].update(scores)
                    
        # load model evals
        folder_path = gparent_dir + '/' +base_result_path+run_name + '/' + 'model_' + m
        file_names = [f for f in os.listdir(folder_path) if f.endswith("_evals.pkl")]
        file_names = sorted(file_names)
        
        for file_name in file_names:
            with gzip.open(folder_path + '/' +file_name, 'rb') as handle:
                scores = pickle.load(handle)
                if m not in TG_models_evals:
                    TG_models_evals[m] = scores
                else:
                    TG_models_evals[m].update(scores)        
                    
        # load model performacnce
        folder_path = gparent_dir + '/' +base_result_path+run_name + '/' + 'model_' + m
        file_names = [f for f in os.listdir(folder_path) if f.endswith("_predictive_performance.pkl")]
        file_names = sorted(file_names)
        
        for file_name in file_names:
            with gzip.open(folder_path + '/' +file_name, 'rb') as handle:
                scores = pickle.load(handle)
                if m not in TG_models_pred_performance:
                    TG_models_pred_performance[m] = scores
                else:
                    TG_models_pred_performance[m].update(scores)
    

    # get a set with all RE names
    re_set = set(base_eGRN.re_tfbs_map.keys())

    a = base_eGRN.Full_peak_data.copy()
    #a['id'] = a['chrom']+'_'+str(a['start'])+'_'+str(a['end'])
    a['start'] = a['start'].astype(str)
    a['end'] = a['end'].astype(str)
    a['id']=a['chrom']+'_'+a['start']+'_'+a['end']
    sorted(a['id'])

    # Now genereate the actual GRNs for each model
    # Current settings:
    # Only models with <99% dropout
    # Only protein coding genes
    # TSS +- 100000
    # Edges? Less is better, but more means we can effectively benchmark deeper due to the overlap with collecTRI...

    # make networks
    print("Generating networks...")
    k            = 30 # maximum anmount of scoring scale per TF-> meaning we scale for the sum of the top 30 regulators, to avoid a bit spourious tails which would de-preioritize TFs in models with lagere number of potential TF regulators
    n_edges      = 20000 # better to do 50000 in the benchmark, as we can effectilvey only benchmark 6000/20000 interactions
    min_reg_size = 10 # Ten is fine. I would not go lower
    #filter_TGs  = None
    filter_TGs   = only_proteincoding_genes
        
    filter_int_TSS   = all_interactions_to_keep[100000]

    regulons_all                    = {}
    interactions_all                = {}
    TF_TG_interactions_binary_all   = {}
    TF_TG_interactions_weighted_all = {}


    # create the desiderd networks
    for model in models:
        # make the TG_filters for exp and act models
        if model.split('_')[1] == 'exp':
            filter_TGs = set(gene_names_exp).intersection(only_proteincoding_genes)
        else:
            filter_TGs = set(gene_names_act).intersection(only_proteincoding_genes)

        # get scores, normalize them by model, using the sum of the top k scores
        new_scores     = normalizeImportanceScores(TG_models_scores[model], top_k_threshold=k)

        # delineate the complexes if score is in model
        if "COP" in model:
            new_scores     = delineateComplexScores(TG_models_scores[model])

        # multiply models by performance score to limit the influence of worse performing models
        new_scores_all = normalizeForModelPerformance(new_scores, TG_models_pred_performance[model])

        # Filter act models also for blacklisted interactions
        if model == 'act_act' or model == 'actCOP_act':
            filter_int_TSS.difference_update(blacklist)
            regulons_all[model], interactions_all[model], TF_TG_interactions_binary_all[model], TF_TG_interactions_weighted_all[model] = makeCustomNetwork(new_scores_all, minimum_regulon_size=min_reg_size, number_edges=n_edges, filter_TGs=filter_TGs, filter_interactions=filter_int_TSS, filter_interactions_keep=True, verbose=False)
        else:
            regulons_all[model], interactions_all[model], TF_TG_interactions_binary_all[model], TF_TG_interactions_weighted_all[model] = makeCustomNetwork(new_scores_all, minimum_regulon_size=min_reg_size, number_edges=n_edges, filter_TGs=filter_TGs, filter_interactions=filter_int_TSS, filter_interactions_keep=True, verbose=False)

    # load enhancer-based models
    # for model in ['enh_exp', 'enh_act']:
    #     # Filter act models also for blacklisted interactions
    #     regulons_all[model], interactions_all[model], TF_TG_interactions_binary_all[model], TF_TG_interactions_weighted_all[model] = makeCustomNetwork(TG_models_scores[model], minimum_regulon_size=min_reg_size, number_edges=n_edges, filter_TGs=filter_TGs, filter_interactions=filter_int_TSS, filter_interactions_keep=True, verbose=False)

    # save models 
    path_to_save_models = result_folder_1
    for model in models:
        with gzip.open(path_to_save_models+'/model_'+model+'_'+str(n_edges/1000)+'k_regulons.pkl.gz', 'wb') as handle:
            pickle.dump(regulons_all[model], handle)
        with gzip.open(path_to_save_models+'/model_'+model+'_'+str(n_edges/1000)+'k_interactions.pkl.gz', 'wb') as handle:
            pickle.dump(interactions_all[model], handle)
        with gzip.open(path_to_save_models+'/model_'+model+'_'+str(n_edges/1000)+'k_binary.pkl.gz', 'wb') as handle:
            pickle.dump(TF_TG_interactions_binary_all[model], handle)
        with gzip.open(path_to_save_models+'/model_'+model+'_'+str(n_edges/1000)+'k_weighted.pkl.gz', 'wb') as handle:
            pickle.dump(TF_TG_interactions_weighted_all[model], handle)

    # Make integrated networks (with or without PCA-based integration model is possible)
    integrating_interactions = [interactions_all[x] for x in models]
    for integration_model in ['maxrank', 'meanrank', 'supportrank', 'PCA']:  
        regulons_all[integration_model+'_1'], interactions_all[integration_model+'_1'], TF_TG_interactions_binary_all[integration_model+'_1'], TF_TG_interactions_weighted_all[integration_model+'_1'] = rankIntegration(integrating_interactions, mode=integration_model, interactions = 20000, minimum_regulon_size = 10, verbose=False)
    integrating_interactions = [interactions_all[x] for x in models[:-2]]
    for integration_model in ['maxrank', 'meanrank', 'supportrank', 'PCA']:  
        regulons_all[integration_model+'_2'], interactions_all[integration_model+'_2'], TF_TG_interactions_binary_all[integration_model+'_2'], TF_TG_interactions_weighted_all[integration_model+'_2'] = rankIntegration(integrating_interactions, mode=integration_model, interactions = 20000, minimum_regulon_size = 10, verbose=False) 

    new_samples = ['exp_exp',  'act_act', 'exp_act', 'act_exp', 'expCOP_exp', 'actCOP_act', 'expCOP_act', 'actCOP_exp', 'maxrank_1', 'meanrank_1', 'supportrank_1', 'PCA_1', 'maxrank_2', 'meanrank_2', 'supportrank_2', 'PCA_2']

    # Calculate Regulon Activities for networks with AUCell
    print("Calculating regulon activities...")
    all_activity_matrices = {}
    for sample in new_samples:
        net = regulons_all[sample]
            
        outfile = open(path_to_signature+'/'+sample+'_singatures.tsv', 'w')
        for regulator in net:
            targets = list(net[regulator].keys())
            outfile.write(generateSignature(regulator, targets)+'\n')
        outfile.close()

        # import sigantures
        signatures = GeneSignature.from_gmt(path_to_signature+'/'+sample+'_singatures.tsv', field_separator='\t', gene_separator='\t')
        
        # Check some percentiles => or simply use default of 0.05 as cut-off
        percentiles = derive_auc_threshold(ex_matrix)
        percentiles

        # Calculate AUCell
        aucs_mtx = aucell(ex_matrix, signatures, auc_threshold=0.05, seed=250, num_workers=4)
        all_activity_matrices[sample] = aucs_mtx

        # export
        aucs_mtx.T.to_csv(base_result_path+run_name+'/regulon_activity_AUCell'+sample+'regulon_activity.csv')

        # Calculate thresholds 
        thresholds = binarize(aucs_mtx.T.values, seed=43, num_workers=3, method="hdt")
        pd.DataFrame({'regulon': list(aucs_mtx.columns) , 'thresholds':list(thresholds)}).to_csv(path_to_thresholds, index=False)

        # Create binary matrix
        bin_mat = pd.DataFrame(aucs_mtx.T.values > thresholds.values.reshape(-1,1), dtype=int)
        bin_mat.index = aucs_mtx.T.index
        bin_mat.columns = aucs_mtx.T.columns
        bin_mat.to_csv(base_result_path+run_name+'/regulon_activity_AUCell'+sample+'regulon_binary_activity.csv')
    print("Finished!")
