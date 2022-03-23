"""

Get inferences from GSSO annotations:
- From a class, all its superclasses [cls_entities_inf: IRI, cls_labels_inf: LABEL]
- From an individual, all its types (classes) [ind_entities_inf: IRI, ind_labels_inf: LABEL]


1 December 2021
"""
import os, time

import pandas as pd
from ast import literal_eval

from owlready2 import *

try:
    from gsso import load_gsso
except ModuleNotFoundError:
    from scripts.gsso import load_gsso

# Functions to query GSSO based on owlready2
try:
    from functionalities.sparql_owlready2 import get_superclasses, get_types
except ModuleNotFoundError:
    from scripts.functionalities.sparql_owlready2 import get_superclasses, get_types

PROJ_DIR = os.getcwd()
print("Hello from {}".format(PROJ_DIR))

DATA_DIR = os.path.join(PROJ_DIR, 'data')

# From all the data
N_PROT_ATTR_DICT = {'gender':88790, 'sexual_orientation':12713, 'race': 42906,
               'religion':70149, 'disability': 5559, 'none': 260337}
# or from a sample
sample = 5559 # take a sample
N_PROT_ATTR_DICT = {key:sample for key in N_PROT_ATTR_DICT}

PROT_ATTR_CONTEXT = list(N_PROT_ATTR_DICT.keys())


def import_onto_annotation_dict():
    onto_annotations = {}
    for S in PROT_ATTR_CONTEXT:
        fname = '{}_{}_data_splits_gsso.csv'.format(S, N_PROT_ATTR_DICT[S])
        try:
            onto_annotations[S] = pd.read_csv(os.path.join(DATA_DIR, 'gsso_annotations', fname))

            # import annotations as lists: will be working with labels
            annotation_cols = ['cls_labels', 'ind_labels', 'cls_entities', 'ind_entities']
            for col in annotation_cols:
                try:
                    onto_annotations[S][col] = onto_annotations[S][col].apply(lambda x: literal_eval(x))
                except ValueError:
                    print('Importing {} as string'.format(col))
        except FileNotFoundError:
            print('File not found: {}'.format(fname))
    return onto_annotations


def main():
    # Load gsso
    gsso = load_gsso()
    # Import ontology annotations
    print('Importing ontology annotations:')
    onto_annotations = import_onto_annotation_dict()

    # Export csv files with inferred facts
    for S in PROT_ATTR_CONTEXT:
        print('Inferring superclasses and types: {}'.format(S))
        df_asserted = onto_annotations[S]
        df_inferred = df_asserted.loc[:, ['id', 'comment_text']]

        # get all superclasses from the list of class labels
        print('Getting class superclasses')
        t0 = time.time()
        res_superclasses = df_asserted.apply(lambda row: get_superclasses(row['cls_entities'], gsso),axis=1)
        df_inferred['cls_entities_inf'] = res_superclasses.apply(lambda row: row[0])
        df_inferred['cls_labels_inf'] = res_superclasses.apply(lambda row: row[1])
        print("Executed in %s seconds." % str(time.time() - t0))

        # get all types from the list of individuals
        print('Getting individuals types')
        t1 = time.time()
        res_types = df_asserted.apply(lambda row: get_types(row['ind_entities'], gsso), axis=1)
        df_inferred['ind_entities_inf'] = res_types.apply(lambda row: row[0])
        df_inferred['ind_labels_inf'] = res_types.apply(lambda row: row[1])
        print("Executed in %s seconds." % str(time.time() - t1))

        # save df_inferred dataframe
        filename = '{}_{}_data_splits_gsso_infer.csv'.format(S, N_PROT_ATTR_DICT[S])
        o_file = os.path.join(DATA_DIR, filename)
        print('Exporting to: {}'.format(o_file))
        df_inferred.to_csv(o_file)


if __name__ == '__main__':
    main()