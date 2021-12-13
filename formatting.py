import pandas as pd
import numpy as np
import os

GENERAL_DATASET_DIR = os.path.join('.', 'datasets')  # General directory with all datasets
NEW_DATASET_DIR = os.path.join(GENERAL_DATASET_DIR, '5_filters')  # Directory of new dataset, without \\ at the end
FOLD_TYPE = "fold_2"
M_LIST = [1, 4, 8]  # Neighborhood for ranking
CCM_DISTANCES = [1, 5]  # Distances for calculating CCM
QUANTIZATION = [16]  # Quantization levels
METRICS_TYPES = ["inter", "dive"]  # Metrics types
CCMS_TYPES = ["cfa", "true"]  # Over what type of images are CCMs evaluated on
CLASSIFICATION_TYPES = ["Per image",
                        "Per class"]  # Types of classification (how well we classify the specific sample and it's overall class (e.g. food, stone, wood, etc.))
EVAL_CLASSIFICATION_DIR = os.path.join(NEW_DATASET_DIR,
                                       "eval_classification")  # Evaluation of classification folder path
EVAL_DIR = EVAL_CLASSIFICATION_DIR  # Evaluation directory


def initialize_classification_dataframe():
    left_left = [str(dist) for dist in CCM_DISTANCES for neighbour in M_LIST for quant in QUANTIZATION]
    left_middle = [str(neighbour) for dist in CCM_DISTANCES for neighbour in M_LIST for quant in QUANTIZATION]
    left_right = [str(quant) for dist in CCM_DISTANCES for neighbour in M_LIST for quant in QUANTIZATION]
    left = [
        left_left,
        left_middle,
        left_right
    ]

    top_top = [metric for metric in METRICS_TYPES for ccms_type in CCMS_TYPES for class_type in CLASSIFICATION_TYPES]
    top_middle = [ccms_type for metric in METRICS_TYPES for ccms_type in CCMS_TYPES for class_type in
                  CLASSIFICATION_TYPES]
    top_bottom = [class_type for metric in METRICS_TYPES for ccms_type in CCMS_TYPES for class_type in
                  CLASSIFICATION_TYPES]
    top = [
        top_top,
        top_middle,
        top_bottom
    ]

    tuples_left = list(zip(*left))
    tuples_top = list(zip(*top))

    index_left = pd.MultiIndex.from_tuples(tuples_left, names=["Distance", "M", "q"])
    index_top = pd.MultiIndex.from_tuples(tuples_top)

    df = pd.DataFrame(np.zeros((len(tuples_left), len(tuples_top))), index=index_left, columns=index_top)
    return df

def is_quant(fname):
    for quant in QUANTIZATION:
        if str(quant) in fname:
            return True
    return False

def get_eval_txt_file_paths(eval_dir):
    eval_txt_file_paths = []
    for root, _, fnames in sorted(os.walk(eval_dir)):
        for fname in fnames:
            if ".txt" in fname and is_quant(fname) and FOLD_TYPE in fname:
                path = os.path.join(root, fname)
                eval_txt_file_paths.append(path)
    return eval_txt_file_paths

def load_eval_data():
    df = initialize_classification_dataframe()
    eval_txt_file_paths = get_eval_txt_file_paths(EVAL_DIR)
    for eval_txt_file_path in eval_txt_file_paths:
        path_split = eval_txt_file_path.split(os.sep)
        file_name = path_split[-1].split(".txt")[0]
        q = file_name.split("_")[-1]
        metric = file_name.split("_")[2]
        ccms_type = path_split[-3]
        class_type = path_split[-2]
        with open(eval_txt_file_path) as f:
            lines = f.readlines()
            for line_idx, line in enumerate(lines):
                parsed_line = line.split(" ")
                for pars_idx, pars in enumerate(parsed_line):
                    df.loc[(str(CCM_DISTANCES[line_idx]), str(M_LIST[pars_idx]), q), (
                    metric, ccms_type, class_type)] = float(pars)
    return df


if __name__ == "__main__":
    df = load_eval_data()
    print(df)
    df.to_excel(os.path.join(EVAL_CLASSIFICATION_DIR, "output_same_neighbourhood_{}.xlsx".format(FOLD_TYPE)))
