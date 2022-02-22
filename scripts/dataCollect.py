"""
Data collection: prepares dataset with Tensorflow (splits) and Kaggle (text context, counts)
for 1.8M and 448k identities (including protected group engineering).
"""

# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/civil_comments.py
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
import os, torch, time
from ast import literal_eval

PROJ_DIR = os.getcwd()
OUT_DIR = os.path.join(PROJ_DIR,'data')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
# Path to downloaded folder: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=all_data.csv
KAGGLE_DIR = '/Users/prl222/PycharmProjects/bias_datasets/civil_comments'
Y_LABEL = 'toxicity' # aggregated value to stratify splits, but not for training the model (toxic)
ID_LABEL = 'id'
TEXT_COL = 'text'
THR = 0.5 # transform identity values to binary if probability over thr

## Default column values of interest
TOXICITY_LABELS = ['identity_attack', 'insult', 'obscene', 'severe_toxicity',
                   'sexual_explicit', 'threat', 'toxicity']
TEXT_CONTEXT = ['id', 'comment_text', 'created_date', 'publication_id',
       'parent_id', 'article_id', 'rating', 'funny', 'wow', 'sad', 'likes', 'disagree'] # text columns of kaggle data
IDENTITY_CONTEXT = [ 'male', 'female', 'transgender',
       'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
       'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
       'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian',
       'latino', 'other_race_or_ethnicity', 'physical_disability',
       'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
       'other_disability']
ANNOTATION_COUNT = ['identity_annotator_count', 'toxicity_annotator_count']

ATTRIBUTES_DICT = {'gender': [ 'male', 'female', 'transgender', 'other_gender'],
              'sexual_orientation': ['heterosexual', 'homosexual_gay_or_lesbian',
                                     'bisexual','other_sexual_orientation'],
              'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist',
                           'atheist', 'other_religion'],
              'race': ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity'],
              'disability': ['physical_disability', 'intellectual_or_learning_disability',
                             'psychiatric_or_mental_illness','other_disability']}

PROT_ATTR_CONTEXT = list(ATTRIBUTES_DICT.keys())

print('Starting data collection from {}'.format(PROJ_DIR))

def get_tf_data(text_context=TEXT_CONTEXT, annotation_count=ANNOTATION_COUNT, text_col=TEXT_COL):
    # IMPORT TENSORFLOW DATASET TO GET SPLITS
    print('Importing data with all 1.8M comments')
    fname = os.path.join(OUT_DIR, 'all_data_splits.csv')

    t0 = time.time()
    if not os.path.exists(fname):
        print('... importing data from tensorflow datasets')
        splits = ['train', 'validation', 'test']
        ds, info = tfds.load('civil_comments', with_info=True)

        data_dict = {}

        for split in splits:
            df = tfds.as_dataframe(ds[split], info)
            if split == 'validation':
                df['split'] = ['dev']*df.shape[0]
            else:
                df['split'] = [split]*df.shape[0]
            data_dict[split] = df

        data = pd.concat([data_dict['train'], data_dict['validation'], data_dict['test']],
                         ignore_index=True)
        data.to_csv(fname)
    else:
        print('... importing from saved file')
        columns = pd.read_csv(fname, nrows=0).columns
        types_dict = {text_col: str, 'split': str}
        types_dict.update({col: float for col in columns if col not in types_dict})
        data = pd.read_csv(fname, index_col=0, dtype=types_dict)

    data.loc[:,'id'] = data['id'].astype(int)
    print(data.shape)
    print(data.dtypes)
    data.head()

    # MERGE WITH KAGGLE DATA TO GET TEXT CONTEXT
    print('Importing data from Kaggle dir to get context')

    data_identities_raw = pd.read_csv(os.path.join(KAGGLE_DIR, 'all_data.csv'))

    print(data_identities_raw.shape)
    print(data_identities_raw.dtypes)

    # Add text column to all_data
    data = pd.merge(data,data_identities_raw[text_context+annotation_count],
                    on='id',how='inner')

    t1 = time.time()
    print("Executed in %s seconds." % str(t1-t0))
    return data


def get_identity_data(data, identity_context=IDENTITY_CONTEXT, attributes=ATTRIBUTES_DICT, prot_attr_context=PROT_ATTR_CONTEXT,
                      text_context=TEXT_CONTEXT, annotation_count=ANNOTATION_COUNT, toxicity_labels=TOXICITY_LABELS, thr=THR):
    # IMPORT KAGGLE DATASET TO GET IDENTITIES
    print('Importing 448k data for evaluation with identity annotations')
    fname = os.path.join(OUT_DIR, 'identity_data_splits.csv')

    t0 = time.time()
    # data = get_tf_data()
    if not os.path.exists(fname):
        print('... importing data from Kaggle')
        data_identities_raw = pd.read_csv(os.path.join(KAGGLE_DIR, 'all_data.csv'))
        print(data_identities_raw.shape)
        print(data_identities_raw.dtypes)
        # Keep data only of identity annotations
        print('... keeping only identity annotations')
        print(data_identities_raw.shape)
        data_identities = data_identities_raw.dropna(subset=identity_context)
        print(data_identities.shape)

        print('... pre-processing group identity')

        def get_attr_list(row, attribute, thr, attributes=attributes):
            attribute_list = [cat for cat in attributes[attribute] if row[cat] >= thr]
            if len(attribute_list) == 0:
                attribute_list = []
            return attribute_list

        for attr in PROT_ATTR_CONTEXT:
            data_identities.loc[:, attr] = data_identities.apply(lambda row: get_attr_list(row, attr, thr),
                                                                 axis=1)
        # Column with none (text that doesn't correspond to any prot attribute.
        data_identities_all = data_identities[PROT_ATTR_CONTEXT[0]]
        for attr in PROT_ATTR_CONTEXT:
            data_identities_all = data_identities_all + data_identities[attr]
        data_identities.loc[:, 'none'] = data_identities_all.apply(lambda y: ['none'] if len(y) == 0 else [])


        # merge with kaggle data to get splits
        print('... merging with tf splits')
        data_identities = pd.merge(data[['id','text', 'split']+toxicity_labels],
                                   data_identities[text_context + identity_context + prot_attr_context + ['none']
                                                   + annotation_count],
                     on=['id'], how='inner')
        print(data_identities.shape)  #  should be equal to data_identities shape, i.e., (448000, 46)
        data_identities.reset_index().to_csv(fname)
        print('... file saved: {}'.format(fname))
    else:
        print('... importing from saved file')
        data_identities = pd.read_csv(fname, index_col=0)
        print(data_identities.dtypes)

        # read group columns as list
        for k,v in attributes.items():
            data_identities[k] = data_identities[k].apply(lambda x: literal_eval(x))
        data_identities['none'] = data_identities['none'].apply(lambda x: literal_eval(x))

    # ... check type of engineered features
    print('Check correct engineered features')
    for i, l in enumerate(data_identities['gender'].head(10)):
        print("list {} is a {}".format(l, type(l)))
    print('... should print <class \'list\'>')

    # ... check examples of final group identities
    for k, v in attributes.items():
        print(k.upper())
        d_k = data_identities[k]
        d_k = d_k.apply(lambda y: np.nan if len(y) == 0 else y)
        print(data_identities.loc[d_k.notna(), [k] + v].head(3))
    print('... should print protected groups examples of not empty lists')

    t1 = time.time()
    print("Executed in %s seconds." % str(t1-t0))
    return data_identities


def main():
    # Import data
    print('Importing tf dataset')
    data = get_tf_data()

    data = get_identity_data(data)


if __name__ == '__main__':
    main()