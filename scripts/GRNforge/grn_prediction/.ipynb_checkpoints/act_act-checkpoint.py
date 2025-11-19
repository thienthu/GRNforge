import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import pickle
import anndata as ad
# now, make regression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sklearn.metrics
import scanpy as sc
import importlib
import pickle
import gzip
sys.path.append('/scratch/gent/vo/000/gvo00027/projects/CBIGR/24JLR_GRN/scripts/23JLR_regulation/')
sys.path.append('/scratch/gent/vo/000/gvo00027/projects/CBIGR/24JLR_GRN/scripts/23JLR_regulation/GRNforge')
from GRNforge.eGRN import EnhancerGRN

class ModelGRN:
    """
    This class should be able to run the modelling for each GRN. It should save all model relevenat information
    for further look-up, and be able to perform all necessary functionalities for the GRN model. Ideally, models
    will be named according to their respective targets.
    """
    def __init__(self, X, y, feature_map, params=None):
        # pandas dataframe for the full data
        self.params                = params    # Dictionalry to pass to XGboost
        self.X                     = X
        self.y                     = y
        self.feature_map           = feature_map
        self.X_train               = None
        self.X_test                = None
        self.y_train               = None
        self.y_test                = None
        self.y_pred                = None
        self.y_pred_train          = None
        self.model                 = None
        self.random_seed           = None
        self.num_training_rounds   = 300
        self.early_stopping_rounds = 20
        self.importance_scores     = None
        self.evals_result          = {}
        self.test_mae              = None
        self.train_mae             = None
        self.test_mae_std          = None
        self.train_mae_std         = None
        self.data_mean             = None

    def makeTrainTestSplit(self, test_size=0.2):
        """
        Make a train and test split with sklearn function
        """
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Train model

    def trainModel(self, importance_type="gain"):
        """
        Make a train and test split with sklearn function
        """

        # DMatrix for XGBoost (optional, but efficient)
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest  = xgb.DMatrix(self.X_test, label=self.y_test)
        print(dtrain)
        num_round = self.num_training_rounds
        watchlist = [(dtrain, "train"), (dtest, "eval")]
        #print(params)
        self.model = xgb.train(self.params, dtrain, num_round, evals=watchlist,  early_stopping_rounds=20, evals_result=self.evals_result, verbose_eval=False)
        
        # Predictions
        self.y_pred_test  = self.model.predict(dtest)
        self.y_pred_train = self.model.predict(dtrain)
        mse_train = mean_squared_error(self.y_train, self.y_pred_train)
        mse_test  = mean_squared_error(self.y_test, self.y_pred_test)
        mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
        mae_test  = mean_absolute_error(self.y_test, self.y_pred_test)
        print(f"Mean Squared Error train: {mse_train}")
        print(f"Mean Squared Error test: {mse_test}")
        print(f"Mean: {np.mean(self.y)}")
        print(f"Rooe mean Squared Error: {mse_test**0.5}")

        # get importance scores
        self.importance_scores        = self.model.get_score(importance_type=importance_type)  # Default is 'weight'
        self.mapped_importance_scores = {self.feature_map.get(key, key): value for key, value in self.importance_scores.items() }
        self.test_mae                 = mae_test
        self.train_mae                = mae_train
        self.data_mean                = np.mean(self.y)
    
    def cleanup(self):
        """
        Since for now I want to have access to all models via pickle, it uses up way too much space on the disk.
        Lets see how much remains when we clean up everything but the model itself...
        """
        self.params                = params    # Dictionalry to pass to XGboost
        self.X                     = X
        self.y                     = y
        self.feature_map           = feature_map
        self.X_train               = None
        self.X_test                = None
        self.y_train               = None
        self.y_test                = None
        self.y_pred                = None
        self.y_pred_train          = None
        self.random_seed           = None
        self.num_training_rounds   = 300
        self.early_stopping_rounds = 20
        #self.importance_scores     = None
        self.evals_result          = {}
        
    def trainModelCV(self, importance_type="gain", n_splits=5):
        """
        Train and evaluate the model using k-fold cross-validation, integrating feature importance scores across folds.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
        # Initialize lists to store metrics
        train_mse_list = []
        test_mse_list = []
        train_mae_list = []
        test_mae_list = []
        
        # Initialize a defaultdict to accumulate importance scores
        importance_accumulator = defaultdict(float)
        
        # Loop over the folds
        for train_index, test_index in kf.split(self.X_train):
            X_train_fold, X_test_fold = self.X_train[train_index], self.X_train[test_index]
            y_train_fold, y_test_fold = self.y_train[train_index], self.y_train[test_index]
            
            # Create DMatrix for this fold
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dtest = xgb.DMatrix(X_test_fold, label=y_test_fold)
            
            num_round = self.num_training_rounds
            watchlist = [(dtrain, "train"), (dtest, "eval")]
            
            # Train the model
            model = xgb.train(self.params, dtrain, num_round, evals=watchlist, early_stopping_rounds=20, evals_result=self.evals_result, verbose_eval=False)
            
            # Predictions
            y_pred_train = model.predict(dtrain)
            y_pred_test = model.predict(dtest)
    
            # Compute metrics
            mse_train = mean_squared_error(y_train_fold, y_pred_train)
            mse_test = mean_squared_error(y_test_fold, y_pred_test)
            mae_train = mean_absolute_error(y_train_fold, y_pred_train)
            mae_test = mean_absolute_error(y_test_fold, y_pred_test)
    
            # Append to lists
            train_mse_list.append(mse_train)
            test_mse_list.append(mse_test)
            train_mae_list.append(mae_train)
            test_mae_list.append(mae_test)
            
            # Accumulate importance scores for this fold
            fold_importance_scores = model.get_score(importance_type=importance_type)
            for feature, score in fold_importance_scores.items():
                importance_accumulator[feature] += score
        
        # Average the importance scores across all folds
        #num_folds = len(kf.split(self.X_train))
        num_folds = sum(1 for _ in kf.split(self.X_train))
        averaged_importance_scores = {feature: score / num_folds for feature, score in importance_accumulator.items()}
    
        # Calculate mean and std of metrics across all folds
        mean_train_mse = np.mean(train_mse_list)
        std_train_mse = np.std(train_mse_list)
        mean_test_mse = np.mean(test_mse_list)
        std_test_mse = np.std(test_mse_list)
        mean_train_mae = np.mean(train_mae_list)
        std_train_mae = np.std(train_mae_list)
        mean_test_mae = np.mean(test_mae_list)
        std_test_mae = np.std(test_mae_list)
    
        # Print the results
        print(f"Mean Train MSE: {mean_train_mse} ± {std_train_mse}")
        print(f"Mean Test MSE: {mean_test_mse} ± {std_test_mse}")
        print(f"Mean Train MAE: {mean_train_mae} ± {std_train_mae}")
        print(f"Mean Test MAE: {mean_test_mae} ± {std_test_mae}")
        print(f"Mean: {np.mean(self.y)}")
        
        # After training, map the averaged importance scores
        #self.importance_scores        = self.model.get_score(importance_type=importance_type)  # Default is 'weight'
        self.mapped_importance_scores = {self.feature_map.get(key, key): value for key, value in averaged_importance_scores.items()}
    
        # Store final metrics
        self.test_mae = mean_test_mae
        self.train_mae = mean_train_mae
        self.test_mae_std = mean_test_mae
        self.train_mae_std = mean_train_mae
        
        self.data_mean = np.mean(self.y)

def getTargetAndPredictorExpression(adata, base_eGRN, targets, random_state=42, verbose=True):
    """
    Extract the corresponding features from an adata and returns a matrix.
    Right now, the target should only be one gene, and predictors can be multiple genes
    """
    # prepare target gene
    filtered_adata = adata[:, adata.var.index.isin(targets)]
    y = filtered_adata.X.toarray()

    if verbose == True:
        print('potential regulators:', len(base_eGRN.tg_tf_map[target[0]]))

    potential_regulators = base_eGRN.tg_tf_map[target[0]]
    # remove target if necessary
    if target[0] in potential_regulators:
        potential_regulators.remove(target[0])             

    filtered_adata = adata[:, adata.var.index.isin(potential_regulators)]
    X              = filtered_adata.X.toarray()

    if verbose == True:
        print('expressed regulators:', len(X[0]))
    feature_map = {f"f{i}": name for i, name in enumerate(list(filtered_adata.var.index))}
    return y, X, feature_map



if __name__ == '__main__':
    # import files
    # import the base_eGRN pickle
    with open('/scratch/gent/vo/000/gvo00027/projects/CBIGR/24JLR_GRN/results/human_PBMC_10x_10k/baseGRN_500k.pickle', 'rb') as handle:
        base_eGRN = pickle.load(handle)

    # import the expression data
    result_folder      = '/scratch/gent/vo/000/gvo00027/projects/CBIGR/24JLR_GRN/results/human_PBMC_10x_10k/'
    data_folder        = '/scratch/gent/vo/000/gvo00027/projects/CBIGR/24JLR_GRN/data/human_PBMC_10x_10k/'
    expression_adata   = ad.read_h5ad(data_folder   + 'pbmc_granulocyte_sorted_10k_gex_molecule_info_filtered_mg_mc_gc_mt_tc.h5ad')
    geneactivity_adata = ad.read_h5ad(result_folder + 'PBMC10k_gene_activity_matrix.h5ad')

    overlapping_genes  = set(expression_adata.var_names).intersection(set(geneactivity_adata.var_names))
    expression_adata_i = expression_adata[:, list(overlapping_genes)].copy()
    geneactivity_adata_i = geneactivity_adata[:, list(overlapping_genes)].copy()
    
    params = {
    "booster": "gbtree",          # Model type ('gbtree', 'gblinear', 'dart', 'gpu_hist')
    "objective": "reg:squarederror",  # Regression loss squarederror
    "learning_rate": 0.1,         # Step size shrinkage (alias: eta)
    "max_depth": 2,               # Maximum depth of a tree
    "min_child_weight": 1,        # Minimum sum of instance weight (hessian) in a child
    "gamma": 0,                   # Minimum loss reduction required to make a split
    "subsample": 0.8,             # Subsample ratio of the training instances
    "colsample_bytree": 0.8,      # Subsample ratio of columns when constructing each tree
    "colsample_bylevel": 1,       # Subsample ratio of columns for each split
    "colsample_bynode": 1,        # Subsample ratio of columns for each tree node
    "lambda": 0,                  # L2 regularization term on weights
    "alpha": 0.8,                   # L1 regularization term on weights
    "tree_method": "auto",        # Tree construction algorithm
    "device": "cpu",   
    "scale_pos_weight": 5,        # Balancing of positive and negative weights
    "eval_metric": "mae",        # Evaluation metric ('rmse', 'mae', etc.)   => I use this as I do not want to have uninterpretable resutls when dooing root over [0.1] intervall compared to > 1 
    "nthread": -1,                # Number of parallel threads (-1 = use all cores)
    "seed": 41,                    # Random seed
    }

    base_result_path = '/scratch/gent/vo/000/gvo00027/projects/CBIGR/24JLR_GRN/results/human_PBMC_10x_10k/GRN_test_runs/'
    run_name         = 'run_2/'
    
    # train all models 
    TG_models_exp_exp_scores            = {}
    TG_models_exp_exp_evals             = {}
    TG_models_exp_exp_pred_performance  = {}
    
    TG_models_act_act_scores            = {}
    TG_models_act_act_evals             = {}
    TG_models_act_act_pred_performance  = {}
    
    TG_models_exp_act_scores            = {}
    TG_models_exp_act_evals             = {}
    TG_models_exp_act_pred_performance  = {}
    
    TG_models_act_exp_scores            = {}
    TG_models_act_exp_evals             = {}
    TG_models_act_exp_pred_performance  = {}

    
    ##  core EXP_EXP model
    a=datetime.utcnow()
    for target_gene in geneactivity_adata_i.var_names:#base_eGRN.TF_set: 
    # select 
        if target_gene not in geneactivity_adata_i.var_names:
            print(target_gene, 'not found')
            continue
        target = [target_gene]
        print(target)
        # get necessary data
        try:
            y, X, feature_map   = getTargetAndPredictorExpression(geneactivity_adata_i, base_eGRN, target, random_state=40, verbose=True)
        except:
            print('error with ',target_gene)
            continue
        #y, X, feature_map   = getTargetAndPredictorExpression(geneactivity_adata, base_eGRN, target, random_state=42, verbose=False)
        # # regression model
        if sum(y) < 500:
            print('skip',  target_gene, 'for number of counts')

        if len(X.colums) == 0:
            print('skip',  target_gene, 'for number of predictors')
            continue
        
        model = ModelGRN(X,y,feature_map, params)
        # # prepare train-test split for model
        model.makeTrainTestSplit()
        # # train model
        model.trainModelCV()
        TG_models_act_act_scores[target_gene]           = model.mapped_importance_scores
        TG_models_act_act_evals[target_gene]            = model.evals_result
        TG_models_act_act_pred_performance[target_gene] = (model.data_mean, model.train_mae, model.test_mae, model.test_mae_std, model.train_mae_std)
    b=datetime.utcnow()
    print(b-a)

    # Create the directory if it doesn't exist
    if not os.path.exists(base_result_path+run_name+'/model_act_act/'):
        os.makedirs(base_result_path+run_name+'/model_act_act/')
    
    path_to_save = base_result_path+run_name+'/model_act_act/baseline_model_1_scores.pkl'
    with gzip.open(path_to_save, "wb") as file:  # 'wb' means write in binary mode
        pickle.dump(TG_models_act_act_scores, file)
    path_to_save = base_result_path+run_name+'/model_act_act/baseline_model_1_evals.pkl'
    with gzip.open(path_to_save, "wb") as file:  # 'wb' means write in binary mode
        pickle.dump(TG_models_act_act_evals, file)    
    path_to_save = base_result_path+run_name+'/model_act_act/baseline_model_1_predictive_performance.pkl'
    with gzip.open(path_to_save, "wb") as file:  # 'wb' means write in binary mode
        pickle.dump(TG_models_act_act_pred_performance, file)    