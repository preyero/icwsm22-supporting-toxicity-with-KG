"""
Annotate text from Civil Comments Identities dataset
using the knowledge in GSSO ontology

1 December 2021
"""

import os, time
import nltk
import numpy as np
import pandas as pd
from owlready2 import *

try:
    from dataCollect import get_tf_data, get_identity_data
except ModuleNotFoundError:
    from scripts.dataCollect import get_tf_data, get_identity_data

try:
    from gsso import load_gsso, create_gsso_dict, get_entity_annotation_dict
except ModuleNotFoundError:
    from scripts.gsso import load_gsso, create_gsso_dict, get_entity_annotation_dict

# try:
#     nltk.data.find('stopwords')
# except LookupError:
#     nltk.download('stopwords')

# Default variables

PROJ_DIR = os.getcwd()
print("Hello from {}".format(PROJ_DIR))

DATA_DIR = os.path.join(PROJ_DIR, 'data')

DEFAULT_N = 5559 # 260337 # len(data)= 1999514, len(data_identities)=
DEFAULT_SAMPLE_GROUP = 'none' # e.g., [None, 'gender', 'sexual_orientation', 'none']
# sampling from any group in each protected characteristic
DEFAULT_ANNOTATED_FILENAME = '{}_{}_data_splits_gsso.csv'.format(DEFAULT_SAMPLE_GROUP, DEFAULT_N)
print('... to export: {}'.format(DEFAULT_ANNOTATED_FILENAME))


def text_detection(onto_dict, text, verbose=False):
    # Export entities whose dict value matches the text. Returns list of entities (i.e, dict keys) and labels
    match = []
    for k, v in onto_dict.items():
        for vi in v: # for each property value of that entity
            if isinstance(vi, owlready2.entity.Thing) or \
                        isinstance(vi, owlready2.entity.ThingClass): # if entity, take its label
                vi = vi.label[0]
            # only if string appears in text, append the entity
            if isinstance(vi, owlready2.util.locstr):
                if re.search(r"\b" + re.escape(vi) + r"\b", text.lower()):
                    match.append(k)
    match_labels = [k.label[0] for k in match] # save in single list of strings
    if verbose:
        print(text)
        print()
        print([onto_dict[x] for x in match])

    match = [k.iri for k in match]  # save in a list of strings (corresponding to IRI)

    return match, match_labels


def get_text_annotations(onto_dict, texts, verbose=False):
    print('... {} text annotations'.format(len(texts)))
    t0 = time.time()
    entities, labels = [], []
    for text in texts:
        # Append to list the entities and label
        ent_i, lab_i = text_detection(onto_dict, text, verbose)
        entities.append(ent_i)
        labels.append(lab_i)
    t1 = time.time()
    print("Executed in %s seconds." % str(t1-t0))
    return entities, labels


def main():
    # load ontology and ontology dict of class and individuals
    gsso = load_gsso()
    gsso_cls_dict, gsso_indv_dict = create_gsso_dict(gsso)

    # annotate and export tf texts
    print('Importing tf dataset')
    data = get_tf_data()
    # Take sample of DEFAULT_N size to annotate
    if DEFAULT_SAMPLE_GROUP:
        print('... sampling {} with protected characteristic: {}'.format(DEFAULT_N,DEFAULT_SAMPLE_GROUP))
        data = get_identity_data(data)
        data_group = data[DEFAULT_SAMPLE_GROUP]
        data_group = data_group.apply(lambda y: np.nan if len(y) == 0 else y)
        data_sample = data.loc[data_group.notna(),:].sample(n=DEFAULT_N, random_state=1313)
    else:
        print('... sampling {} of any text'.format(DEFAULT_N))
        data_sample = data.sample(n=DEFAULT_N, random_state=1313)

    texts = data_sample.comment_text.to_list()

    texts = [t.replace('\n', ' ') for t in texts]
    texts = [t.replace('\n\n', ' ') for t in texts]
    print('... success: {}'.format(texts[:3]))

    # annotate a text (example)
    print('Testing text_detection function')
    example = texts[0]
    _, tmp_match_cls = text_detection(gsso_cls_dict, example, verbose=True)
    _, tmp_match_indv= text_detection(gsso_indv_dict, example, verbose=True)
    print('... result labels: \n classes: {} \n individuals: {}'.format(tmp_match_cls, tmp_match_indv))

    print('Getting class annotations')
    cls_entities, cls_labels = get_text_annotations(gsso_cls_dict, texts)#, verbose=True)
    # print('... result: \n entities: {} \n labels: {}'.format(cls_entities, cls_labels))

    print('Getting individuals annotations')
    ind_entities, ind_labels = get_text_annotations(gsso_indv_dict, texts)#, verbose=True)
    # print('... result: \n entities: {} \n labels: {}'.format(ind_entities, ind_labels))

    data.loc[data_sample.index, 'cls_entities'] = cls_entities
    data.loc[data_sample.index, 'cls_labels'] = cls_labels
    data.loc[data_sample.index, 'ind_entities'] = ind_entities
    data.loc[data_sample.index, 'ind_labels'] = ind_labels

    o_file = os.path.join(DATA_DIR, DEFAULT_ANNOTATED_FILENAME)
    print('Exporting to: {}'.format(o_file))
    data_out = data.loc[data_sample.index, ['id', 'comment_text', 'cls_entities', 'cls_labels', 'ind_entities', 'ind_labels']]
    # print(data_out)
    data_out.to_csv(o_file)


if __name__ == '__main__':
    main()