import numpy as np
import os
import csv
root_folder = 'D:/codee/data/abide1/'
data_folder = os.path.join(root_folder, 'filt_noglobal/filt_noglobal')
phenotype = os.path.join(root_folder, 'Phenotypic_V1_0b_preprocessed1.csv')

def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject.txt'), dtype=str)
    # os.path.join(data_folder, 'subject_IDs.txt') 输出 ABIDE_pcp/cpac/filt_noglobal\subject_IDs.txt

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs
def get_id1(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)
    # os.path.join(data_folder, 'subject_IDs.txt') 输出 ABIDE_pcp/cpac/filt_noglobal\subject_IDs.txt

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    # print('scores_dict:', scores_dict)
    return scores_dict