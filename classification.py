import pickle
#import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import operator
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

GENERAL_DATASET_DIR = './datasets'                            # General directory with all datasets
NEW_DATASET_DIR = GENERAL_DATASET_DIR + '/5_filters'          # Directory of new dataset, without / at the end
N_SUBIMG = 16                                                 # Number of subsamples per sample
N_PROT = 11                                                   # Number of test subsamples per sample
M_LIST = [1, 4, 8]                                            # Neighborhood for ranking
CCM_DISTANCES = [1, 5]                                        # Distances for calculating CCM
QUANTIZATION = [16]                         # Quantization levels
CCMS_TYPES = ["cfa", "true"]                                  # Over what type of images are CCMs evaluated on
CLASSIFICATION_TYPES = ["Per image", "Per class"]             # Types of classification (how well we classify the specific sample and it's overall class (e.g. food, stone, wood, etc.))
EVALUATION_TYPES = ["eval_classification", "eval_retrieval"]  # Type of evaluation
N_TEST = N_SUBIMG - N_PROT

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def is_ccms_file(filename, ccms_type):
    return ccms_type in filename and ".pkl" in filename
    
def get_ccms_filters_paths(sample_path, ccms_type):
    ccms_filter_paths = []
    for root, _, fnames in sorted(os.walk(sample_path)):
        for fname in fnames:
            if is_ccms_file(fname, ccms_type) and NEW_DATASET_DIR in root:
                path = os.path.join(root, fname)
                ccms_filter_paths.append(path)
    return ccms_filter_paths

def intersection(ccm_test, ccm_tar):
    return (np.minimum(ccm_test, ccm_tar)).sum()

def similarity(ccms_test, ccms_tar):
    if len(ccms_test) != len(ccms_tar):
        raise ValueError("CCMs are not of the same size"
                         "Check if the folders are correct.")
    num_dist = ccms_test[0].shape[2]
    similarity_sum = [0] * num_dist
    for dist in range(num_dist):
        for idx, ccm_test in enumerate(ccms_test):
            similarity_sum[dist] += intersection(ccm_test[:, :, dist], ccms_tar[idx][:, :, dist])
    return [x / len(ccms_test) for x in similarity_sum]

def jeffrey_divergence(ccm_test, ccm_tar):
    ccm_test[np.isclose(ccm_test, 0.0)] = 1e-9
    ccm_tar[np.isclose(ccm_tar, 0.0)] = 1e-9
    average_ccm = (ccm_test + ccm_tar)/2
    test_kl_divergence = ccm_test*np.log(ccm_test/average_ccm)
    true_kl_divergence = ccm_tar*np.log(ccm_tar/average_ccm)
    return (test_kl_divergence + true_kl_divergence).sum()

def dissimilarity(ccms_test, ccms_tar):
    if len(ccms_test) != len(ccms_tar):
        raise ValueError("CCMs are not of the same size"
                         "Check if the folders are correct.")
    num_dist = ccms_test[0].shape[2]
    dissimilarity_sum = [0] * num_dist
    for dist in range(num_dist):
        for idx, ccm_test in enumerate(ccms_test):
            dissimilarity_sum[dist] += jeffrey_divergence(ccm_test[:, :, dist], ccms_tar[idx][:, :, dist])
    return [x / len(ccms_test) for x in dissimilarity_sum]

def calculate_metrics(ccms_test, ccms_tar):
    return similarity(ccms_test, ccms_tar), dissimilarity(ccms_test, ccms_tar)

def get_specific_quant_ccms_paths(paths, quant):
    return [path for path in paths if str(quant) + ".pkl" in path]

def calculate_rang_score(prototype_list):
    rang_dict = dict()
    for prot in prototype_list:
        rang = 0
        for idx, el in enumerate(prototype_list):
            if prot == el:
                rang += idx
        rang_dict[prot] = rang
    return rang_dict

def count_correct_classification(correct_metrics_image, correct_metrics_class, test_key_class, test_key, dist_idx, m_idx, metrics_classified):
    if metrics_classified == test_key:
        correct_metrics_image[dist_idx, m_idx] += 1
    else:
        print("PREDICTED IMAGE {}, TRUE IMAGE: {}".format(metrics_classified, test_key))
    if metrics_classified.split("/")[0] == test_key_class:
        correct_metrics_class[dist_idx, m_idx] += 1
    else:
        print("PREDICTED CLASS {}, TRUE CLASS: {}".format(metrics_classified.split("/")[0], test_key_class))
    return correct_metrics_image, correct_metrics_class

def quantization_loop(quant, ccms_paths, ccms_type):
    specific_quant_ccms_paths = get_specific_quant_ccms_paths(ccms_paths, quant)
    correct_inter_image = np.zeros((len(CCM_DISTANCES), len(M_LIST)))
    correct_inter_class = np.zeros((len(CCM_DISTANCES), len(M_LIST)))
    correct_dive_image = np.zeros((len(CCM_DISTANCES), len(M_LIST)))
    correct_dive_class = np.zeros((len(CCM_DISTANCES), len(M_LIST)))
    len_all_images = len(specific_quant_ccms_paths) * N_TEST
    start_time = time.time()
    # go through all test subsamples
    for idx_path, ccms_test_path in enumerate(specific_quant_ccms_paths):
        if idx_path % 10 == 0:
            print("Currently {}/{}".format(idx_path, len(specific_quant_ccms_paths)))
            print("--- Batch time: {} seconds ---".format((time.time() - start_time)))
            start_time = time.time()
        test_subsamples_ccms = load_obj(ccms_test_path)
        test_key_class = ccms_test_path.split("/")[-3]
        test_key = test_key_class + "/" + ccms_test_path.split("/")[-2]
        #start_time = time.time()
        # default fold 1 [:N_TEST], first N_TEST are going to be test
        # default fold 2 [N_PROT:], last N_TEST are going to be test
        for test_subsample in test_subsamples_ccms[N_PROT:]:
            intersection_list = []
            divergence_list = []

            # go through all target subsamples
            for ccms_tar_path in specific_quant_ccms_paths:
                tar_subsamples_ccms = load_obj(ccms_tar_path)
                tar_key_class = ccms_tar_path.split("/")[-3]
                tar_key = tar_key_class + "/" + ccms_tar_path.split("/")[-2]

                # default fold 1 [N_TEST:], first N_TEST are going to be test
                # default fold 2 [:N_PROT], last N_TEST are going to be test
                for tar_subsample in tar_subsamples_ccms[:N_PROT]:
                    inter, dive = calculate_metrics(test_subsample, tar_subsample)
                    intersection_list.append([tar_key, inter])
                    divergence_list.append([tar_key, dive])
            
            # go through different ccms distances
            for dist_idx, dist in enumerate(CCM_DISTANCES):
                intersection_dist_list = [[x[0], x[1][dist_idx]] for x in intersection_list]
                divergence_dist_list = [[x[0], x[1][dist_idx]] for x in divergence_list]
                intersection_dist_list.sort(key = lambda x: x[1], reverse = True) # desceding 1 is maximum
                divergence_dist_list.sort(key = lambda x: x[1]) # asceding 0 is the best

                # go through different number of neighbors
                for m_idx, m in enumerate(M_LIST):
                    interesction_list_prot = [sample[0] for sample in intersection_dist_list[:m]]
                    divergence_list_prot = [sample[0] for sample in divergence_dist_list[:m]]

                    # calculate best intersection rang
                    inter_rang_dict = calculate_rang_score(interesction_list_prot)
                    # calculate best divergence rang
                    dive_rang_dict = calculate_rang_score(divergence_list_prot)
                    
                    inter_classified = max(inter_rang_dict.items(), key=operator.itemgetter(1))[0]
                    dive_classified = max(dive_rang_dict.items(), key=operator.itemgetter(1))[0]

                    correct_inter_image, correct_inter_class = count_correct_classification(correct_inter_image, correct_inter_class, test_key_class, test_key, dist_idx, m_idx, inter_classified)
                    correct_dive_image, correct_dive_class = count_correct_classification(correct_dive_image, correct_dive_class, test_key_class, test_key, dist_idx, m_idx, dive_classified)

        #print("--- One image time: %s seconds ---" % (time.time() - start_time))
    percentage_classification_inter_image = (correct_inter_image/len_all_images)*100
    percentage_classification_inter_class = (correct_inter_class/len_all_images)*100

    percentage_classification_dive_image = (correct_dive_image/len_all_images)*100
    percentage_classification_dive_class = (correct_dive_class/len_all_images)*100

    np.savetxt(NEW_DATASET_DIR + "/" + EVALUATION_TYPES[0] + "/" + ccms_type + "/" + CLASSIFICATION_TYPES[0] + '/fold_2_inter_quant_{}.txt'.format(quant), percentage_classification_inter_image, fmt='%1.3f')
    np.savetxt(NEW_DATASET_DIR + "/" + EVALUATION_TYPES[0] + "/" + ccms_type + "/" + CLASSIFICATION_TYPES[1] + '/fold_2_inter_quant_{}.txt'.format(quant), percentage_classification_inter_class, fmt='%1.3f')

    np.savetxt(NEW_DATASET_DIR + "/" + EVALUATION_TYPES[0] + "/" + ccms_type + "/" + CLASSIFICATION_TYPES[0] + '/fold_2_dive_quant_{}.txt'.format(quant), percentage_classification_dive_image, fmt='%1.3f')
    np.savetxt(NEW_DATASET_DIR + "/" + EVALUATION_TYPES[0] + "/" + ccms_type + "/" + CLASSIFICATION_TYPES[1] + '/fold_2_dive_quant_{}.txt'.format(quant), percentage_classification_dive_class, fmt='%1.3f')
    
def texture_classification_evaluation(ccms_type):
    ccms_paths = get_ccms_filters_paths(GENERAL_DATASET_DIR, ccms_type)
    for quant in QUANTIZATION:
        print("--- Quantizaztion {} ---".format(quant))
        start_time = time.time()
        quantization_loop(quant, ccms_paths, ccms_type)
        print("--- Quantization {}: {} seconds ---".format(quant, (time.time() - start_time)))

def folder_initialization():
    for eval_type in EVALUATION_TYPES:
        for ccms_type in CCMS_TYPES:
            for class_type in CLASSIFICATION_TYPES:
                Path(NEW_DATASET_DIR + "/" + eval_type + "/" + ccms_type + "/" + class_type).mkdir(parents=True, exist_ok=True)

def classification_evaluation():
    folder_initialization()
    start_time = time.time()
    with mp.Pool(processes=len(CCMS_TYPES)) as pool:
        results = pool.map(texture_classification_evaluation, CCMS_TYPES)
    pool.close()
    print("--- Full loop: {} seconds ---".format((time.time() - start_time)))

if __name__ == "__main__":
    #subsample_dict = load_obj(SUBSAMPLE_DICT_FILE)
    classification_evaluation()
