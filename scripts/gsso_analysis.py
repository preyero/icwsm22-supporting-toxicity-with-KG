"""
Analysis of GSSO ontology term coverage
in hate speech detection data

25 November 2021
"""
import os, time
import numpy as np
import pandas as pd
from ast import literal_eval

try:
    from gsso import load_gsso, collect_gsso_dict, create_iri2label_from_dict, get_label_from_iri_list, get_iri_from_label_list
except ModuleNotFoundError:
    from scripts.gsso import load_gsso, collect_gsso_dict, create_iri2label_from_dict, get_label_from_iri_list, get_iri_from_label_list

try:
    from utils import flatten, select_keys_from_dict
except ModuleNotFoundError:
    from scripts.utils import flatten, select_keys_from_dict
try:
    from plots import export_freq_plot
except ModuleNotFoundError:
    from scripts.plots import export_freq_plot


PROJ_DIR = os.getcwd()
print("Hello from {}".format(PROJ_DIR))

DATA_DIR = os.path.join(PROJ_DIR, 'data')
RES_DIR = os.path.join(PROJ_DIR, 'results')
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

USE_ALL_DATA = False
thr_long_analysis=600
# over that, only use all entities (cls+ind) and all (i.e. asserted+inferred)

# in all samples: analyse for onto annotations of all_entities (assert+inf) and all (cls+ind)
N_PROT_ATTR_DICT = {'gender':88790, 'sexual_orientation':12713, 'race': 42906,
                   'religion':70149, 'disability': 5559, 'none': 260337}

if not USE_ALL_DATA:
    # Use samples for equal group representation
    sample = 5559
    N_PROT_ATTR_DICT = {key:sample for key in N_PROT_ATTR_DICT}

PROT_ATTR_CONTEXT = list(N_PROT_ATTR_DICT.keys())


# Import a dictionary of PROT_ATTR keys with their annotated dataframe
def import_onto_annotation_dict(fname_label, dirname, annotation_cols):
    onto_annotations = {}
    for S in PROT_ATTR_CONTEXT:
        fname = fname_label.format(S, N_PROT_ATTR_DICT[S])
        try:
            onto_annotations[S] = pd.read_csv(os.path.join(DATA_DIR, dirname, fname))

            # import annotations as lists: will be working with labels
            for col in annotation_cols:
                try:
                    onto_annotations[S][col] = onto_annotations[S][col].apply(lambda x: literal_eval(x))
                except ValueError:
                    print('Importing {} as string'.format(col))
        except FileNotFoundError:
            print('File not found: {}'.format(fname))
    return onto_annotations


# create string of S_n to export files
def get_tag_n_prot_attr(n_dict = N_PROT_ATTR_DICT):
    n_tag = ''
    for S in N_PROT_ATTR_DICT.keys():
        n_tag = '{}_{}_'.format(S, N_PROT_ATTR_DICT[S]) + n_tag
    return n_tag


# print a list of entity labels from a IRI
def print_labels(iri_list, iri2label_dict):
    label_list = [iri2label_dict[iri] for iri in iri_list]
    label_str = ', '.join(label_list)
    print(label_str)
    return label_list


# Wrapper function
def collect_data_analysis():
    t0 = time.time()
    # Import ontology annotations: asserted
    print('Importing ontology annotations:')
    cols = ['cls_labels', 'ind_labels', 'cls_entities', 'ind_entities']
    assert_dict = import_onto_annotation_dict('{}_{}_data_splits_gsso.csv',
                                                   'gsso_annotations', cols)
    # ... create column to analyse all gsso annotations
    for S in PROT_ATTR_CONTEXT:
        assert_dict[S]['all_entities'] = assert_dict[S]['cls_entities'] + assert_dict[S]['ind_entities']
        assert_dict[S]['all_labels'] = assert_dict[S]['cls_labels'] + assert_dict[S]['ind_labels']

    # Import ontology annotations: inferred
    cols_inf0 = ['cls_labels_inf', 'ind_labels_inf', 'cls_entities_inf', 'ind_entities_inf']
    inf_dict0 = import_onto_annotation_dict('{}_{}_data_splits_gsso_infer.csv',
                                                   'gsso_annotations_inferred', cols_inf0)

    # ... transform to same format as asserted
    inf_dict = {}
    for S in PROT_ATTR_CONTEXT:
        # cls and ind col: rename and flatten list of list to a single list
        inf_dict[S] = inf_dict0[S].rename(columns={old:new for old, new in zip(cols_inf0, cols)})
        for col_i in cols:
            inf_dict[S][col_i] = inf_dict[S].apply(lambda row: flatten(row[col_i]), axis=1)
        # create an all col
        inf_dict[S]['all_entities'] = inf_dict[S]['cls_entities'] + inf_dict[S]['ind_entities']
        inf_dict[S]['all_labels'] = inf_dict[S]['cls_labels'] + inf_dict[S]['ind_labels']

    # Create dict of asserted and inferred
    all_ent_dict = {}
    for S in PROT_ATTR_CONTEXT:
        all_ent_dict[S] = assert_dict[S].copy()
        for tag in ['cls', 'ind', 'all']:
            tag1, tag2 = '{}_entities'.format(tag), '{}_labels'.format(tag)
            all_ent_dict[S][tag1] = assert_dict[S][tag1] + inf_dict[S][tag1]
            all_ent_dict[S][tag2] = assert_dict[S][tag2] + inf_dict[S][tag2]
    print("Executed in %s seconds." % str(time.time()-t0))

    return assert_dict, inf_dict, all_ent_dict, inf_dict0


# Create np.array matrix of occurrences of entities in gsso_dict
def _get_tf_matrix(annotation_col, gsso_dict):
    """
    Term-occurrence matrix N (texts) x M (entities) given a:
     - M entities (i.e., gsso_dict from gsso.get_entity_annotation_dict)
     - N annotations of the texts (i.e, annotation_col Mx1) """

    iri_list = [k.iri for k in gsso_dict.keys()]
    # TF-matrix as a NxM dataframe (N texts, M entities)
    tf_matrix = pd.DataFrame(columns=iri_list)

    # Fill occurrences:
    for i, annotation_list in enumerate(annotation_col):
        entities_found = list(set(annotation_list))
        # print(entities_found)
        # Intersection with M entities list.
        if USE_ALL_DATA or sample > thr_long_analysis:
            # computing analysis with all_ent, all
            entities_observed = entities_found
        else:
            entities_observed = [ent for ent in entities_found if ent in iri_list]
        # print('\n *** {}'.format(entities_observed))
        tf_matrix.loc[i, entities_observed] = 1
    print('... filled for samples: {}'.format(tf_matrix.shape))

    # Fill nan with 0
    tf_matrix.fillna(0, inplace=True)
    return tf_matrix


# export document-term matrix
def create_tf_matrixes_dict(annotation_dict, gsso_dict):
    """ Create dict for gsso_dict keys (e.g., cls, ind, cls+ind=all) for each S"""
    print('Creating dict of TF matrixes')
    t0 = time.time()
    tf = {}
    for subonto in gsso_dict.keys():
        # Select entries to analyse distribution
        tf_subonto = {}
        print('... TF matrixes: {}'.format(subonto))
        # For each attribute, create TF matrix with each gsso_dict key annotation column
        for S in PROT_ATTR_CONTEXT:
            tf_subonto[S] = _get_tf_matrix(annotation_dict[S]['{}_entities'.format(subonto)], gsso_dict[subonto])
            print('... ** filled {}: {}'.format(S, tf_subonto[S].shape))
        tf[subonto] = tf_subonto
    print("Executed in %s seconds." % str(time.time()-t0))

    return tf


def save_tf_dict(title_tag, subannot_tag, dict_of_df):
    """ Save a copy of dict of df: tf matrixes"""
    o_dir = os.path.join(RES_DIR, 'saved_dict')
    n_tag = get_tag_n_prot_attr()
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)

    for subonto in dict_of_df.keys():
        for S in PROT_ATTR_CONTEXT:
            o_file = os.path.join(o_dir,'_'.join([title_tag,S, subannot_tag, subonto,n_tag])+'.csv')
            dict_of_df[subonto][S].to_csv(o_file)


def open_tf_dict(title_tag, subannot_tag, gsso_dict):
    """ Import saved dict: tf matrixes.
    Freq would have to be imported as pd.Series
    and doesn't take that long to compute"""
    o_dir = os.path.join(RES_DIR, 'saved_dict')
    n_tag = get_tag_n_prot_attr()
    dict_subonto = {}
    for subonto in gsso_dict.keys():
        dict_subonto_S = {}
        for S in PROT_ATTR_CONTEXT:
            in_file = os.path.join(o_dir, '_'.join([title_tag, S, subannot_tag, subonto,n_tag]) + '.csv')
            dict_subonto_S[S] = pd.read_csv(in_file, index_col=0)
        dict_subonto[subonto] = dict_subonto_S
    return dict_subonto


# export frequency from DT matrix
def compute_freq_from_tf_matrixes(tf_dict):
    print('Creating dict of Frequencies')
    t0 = time.time()
    freq = {}
    for subonto in tf_dict.keys():
        freq_subonto = {}
        for S in PROT_ATTR_CONTEXT:
            N = tf_dict[subonto][S].shape[0] # N texts (same for all S of each subonto)
            freq_subonto[S] = tf_dict[subonto][S].sum(axis=0)/N
        freq[subonto]= freq_subonto
    print("Executed in %s seconds." % str(time.time()-t0))

    return freq


# export list of entities with freq higher than thr in all S
def get_common_entities(freq_dict, freq_args={'subannot': 'all_ent', 'subonto': 'all'}, thr=0.95):
    """ List of IRIs from freq_dict if frequency is >= to threshold thr in all S"""
    subannot, subonto = freq_args['subannot'], freq_args['subonto']
    print('Computing LIST COMMON ENTITIES:\n Subannotations, i.e., linked facts: {}'
          '\n Subonto, i.e., accounting for entities: {}'.format(subannot, subonto))
    freq_df = pd.DataFrame(freq_dict[subannot][subonto])
    most_common_list = list(freq_df.loc[(freq_df>=thr).all(axis=1)].index)
    print('... {} found.'.format(len(most_common_list)))
    return most_common_list


# create table with top N entities of a freq_dict
def create_top_freq_table(freq_dict, iri2label_dict, n, o_filename=None, stopwords=None):
    print('Creating tables with top-{} frequency terms'.format(n))
    t0 = time.time()
    table_dict = {}

    rows = []
    for subonto in freq_dict.keys():
        # In each subontology (i.e., cls, ind, all)
        table_subonto_dict = {}
        for S in PROT_ATTR_CONTEXT:
            # ... get the list of the top n most frequent entities of each S
            freq_df_S = freq_dict[subonto][S]
            if stopwords:
                top_iri_list = list(freq_df_S.loc[~freq_df_S.index.isin(stopwords)].sort_values(ascending=False)[:n].index)
            else:
                top_iri_list = list(freq_df_S.sort_values(ascending=False)[:n].index)
            table_subonto_dict[S] = top_iri_list
        table_dict[subonto] = table_subonto_dict
        rows.append(table_subonto_dict)

    # export table of IRI's label
    table_df = pd.DataFrame(rows)
    table_df.index = freq_dict.keys()
    for col in table_df.columns:
        table_df[col] = table_df.apply(lambda row: get_label_from_iri_list(row[col], iri2label_dict), axis=1)
    if o_filename:
        o_path = os.path.join(RES_DIR,'freq_tables')
        if not os.path.isdir(o_path):
            os.mkdir(o_path)
        table_df.to_csv(path_or_buf=os.path.join(o_path,o_filename))
        with pd.option_context("max_colwidth", 1000), open(os.path.join(o_path, o_filename.split('.')[0]+'.tex'), 'w') as tf:
            tf.write(table_df.to_latex())

    print("Executed in %s seconds." % str(time.time()-t0))
    return table_dict


# create heatmap of entities in top N (table_dict)
def create_top_freq_plots(table_dict, freq_dict, iri2label_dict, args={'subonto':'all', 'S': None, 'sw_tag': None}):
    """ Create heatmap of top-N frequencies of (sub)ontology
    For each top-N in Si, create heatmap to compare their freq in all other S"""
    print('Creating plots Si top-N frequency terms in all S: {}'.format(args['S']))
    t0 = time.time()

    o_path = os.path.join(RES_DIR, 'freq_plots')
    n_tag = get_tag_n_prot_attr()
    if not os.path.isdir(o_path):
        os.mkdir(o_path)

    subonto, sw, S = args['subonto'], args['sw_tag'], args['S']
    for subannot in table_dict.keys():
        if sw:
            title = '{}_{}_{}'.format(sw, subannot, subonto)
        else:
            title = '{}_{}'.format(subannot, subonto)

        print('... exporting uniplot: {}'.format(S))
        # single freq plot of topN for each Si
        table_dict_S = select_keys_from_dict(table_dict[subannot][subonto], [S])
        export_freq_plot(table_dict_S, freq_dict[subannot][subonto],iri2label_dict,
                              o_path,title, n_tag)
    print("Executed in %s seconds." % str(time.time()-t0))
    return


# create heatmap of the entities corresponding to annotations categories
def create_category_freq_plots(freq_dict, iri2label_dict, args={'subonto':'all'},
                               categories=['gender', 'sexual orientation', 'race', 'disability', 'religion']):
    """ Create heatmap of annotation categories frequencies of (sub)ontology: subannot - all_ent
    For each category class, create heatmap to compare their freq in all other S"""
    print('Creating plots of category frequency terms in all S')
    t0 = time.time()

    o_path = os.path.join(RES_DIR, 'freq_plots_category')
    n_tag = get_tag_n_prot_attr()
    if not os.path.isdir(o_path):
        os.mkdir(o_path)

    IRI_categories = get_iri_from_label_list(categories, iri2label_dict)
    index_dict = {'categories':IRI_categories}

    subonto = args['subonto']
    for subannot in freq_dict.keys():
        title = '{}_{}'.format(subannot, subonto)
        export_freq_plot(index_dict, freq_dict[subannot][subonto], iri2label_dict, o_path,title, n_tag)
    print("Executed in %s seconds." % str(time.time()-t0))
    return


# export two df with onto annotations + tf matrix of a pos/neg space
def get_annotation_space(full_annotation_dict, full_tf_dict, subannot='all_ent', subonto='all',
                         args_space={'space_fp': ['gender', 'sexual_orientation'],
                                     'space_fn':['religion', 'race', 'disability', 'none']}):
    """
    Get pos (i.e., the considered related) and negative data
    """
    print('Creating df with onto annotation + tf data in pos, neg space')
    t0 = time.time()

    annotation_dict, tf_dict = full_annotation_dict[subannot], full_tf_dict[subannot][subonto]

    pos_attr, neg_attr = args_space['space_fp'], args_space['space_fn']

    # Positive data: all samples with a pos_attrs: 9 + 14280
    pos_frames = [pd.concat([annotation_dict[S], tf_dict[S]], axis=1) for S in pos_attr]
    data_pos = pd.concat(pos_frames, ignore_index=True)

    # Negative data: all samples in neg_attrs not in pos_attrs
    neg_frames = [pd.concat([annotation_dict[S], tf_dict[S]], axis=1) for S in neg_attr]
    data_neg = pd.concat(neg_frames, ignore_index=True)
    data_neg = data_neg.loc[~data_neg.id.isin(data_pos.id),:]
    print("Executed in %s seconds." % str(time.time()-t0))

    return data_pos, data_neg


# compute onto liability score, for each row
def _compute_annot_reliability_score(tf_row, entities, freq_matrix):
    """
    Compute likelihood of neg samples belonging to pos space based on onto annotation freqs
    :return:
    """
    # Score is the sum of (freq in pos)-(freqs in neg) of the entities occurring in that text
    n_ent_found = sum(tf_row)
    if n_ent_found > 0:
        score = sum(tf_row * freq_matrix[entities])/n_ent_found
    else:
        score = 0
    return score


def compute_annot_reliability_score(search_df, full_freq_dict, subannot, subonto, S_pos, S_neg, entities_dict=None):
    print('Computing reliability score')
    t0 = time.time()
    # Select columns in search df corresponding to TF
    freqs = full_freq_dict[subannot][subonto]
    if entities_dict:
        # ... compute score only with certain entities
        IRI_l = get_iri_from_label_list(entities_dict['entities'], entities_dict['iri2label'])
        tf_df_pos = search_df[IRI_l]
    else:
        tf_df_pos = search_df[freqs[list(freqs.keys())[0]].index]

    # Compute probability values of each entities (avg freq in pos - avg freq in neg space)
    prob_pos, prob_neg = freqs[S_pos[0]], freqs[S_neg[0]]
    n_pos, n_neg = len(S_pos), len(S_neg)
    # ... add probabilities of entities on each S
    for S_pos_i in S_pos[1:]:
        prob_pos = prob_pos.add(freqs[S_pos_i])
    for S_neg_i in S_neg[1:]:
        prob_neg += freqs[S_neg_i]
    # ... get the avg
    prob_pos, prob_neg = prob_pos/n_pos, prob_neg/n_neg
    freq_avg_matrix = prob_pos.subtract(prob_neg)

    # compute score for each row
    scores = tf_df_pos.apply(lambda tf_row: _compute_annot_reliability_score(tf_row, tf_df_pos.columns,freq_avg_matrix),
                                 axis=1)

    print("Executed in %s seconds." % str(time.time()-t0))

    return scores


# add human annotation data, reorder, and export csv
def export_error_candidates(search_dict, args_space):
    """
    Export id, text, score_categories, score, human_prot list, human_probs, human score
    Order descending by higher score, then lower prob of human annot (e.g., sum of all categories of gender and so if
    looking for fn)
    :return:
    """
    print('Exporting csv files of FP, FN candidates (acc. to onto and human scores)')
    t0 = time.time()
    neg_keys, pos_keys = 'space_fn', 'space_fp'

    # ... import data with identity annotations
    try:
        from scripts.dataCollect import get_tf_data, get_identity_data
    except ModuleNotFoundError:
        from dataCollect import get_tf_data, get_identity_data

    data0 = get_identity_data(get_tf_data())
    DATA_ATTRIBUTES_DICT = {'gender': ['male', 'female', 'transgender', 'other_gender'],
                       'sexual_orientation': ['heterosexual', 'homosexual_gay_or_lesbian',
                                              'bisexual', 'other_sexual_orientation'],
                       'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist',
                                    'atheist', 'other_religion'],
                       'race': ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity'],
                       'disability': ['physical_disability', 'intellectual_or_learning_disability',
                                      'psychiatric_or_mental_illness', 'other_disability'],
                       'none' : []}
    # ... compute annotation score in fn,fp space
    data = {}
    # ... 1. compute score of FN candidates
    cols_labels = args_space[pos_keys] # prot attr of the other space
    cols_probs = [vi for k,v in select_keys_from_dict(DATA_ATTRIBUTES_DICT,cols_labels).items() for vi in v] # prot attr val columns
    cols = ['id'] + cols_labels + cols_probs
    if not USE_ALL_DATA:
        # some samples of the other space may not be part of the drawn sample so check all human annot
        cols = ['id'] + PROT_ATTR_CONTEXT + cols_probs
    data['space_fn'] = data0.loc[data0.id.isin(search_dict['space_fn'].id),cols]
    # sum score of being from the other space given by the humans
    data['space_fn']['human_score'] = data['space_fn'][cols_probs].sum(axis=1)
    # consider subtracting probs of the other space to the score (i.e., some annotators found that relation).

    # ... 2. compute score of FP candidates
    cols_labels = args_space[neg_keys] # prot attr of the other space
    cols_probs = [vi for k,v in select_keys_from_dict(DATA_ATTRIBUTES_DICT,cols_labels).items() for vi in v] # prot attr val columns
    cols = ['id'] + cols_labels + cols_probs
    if not USE_ALL_DATA:
        # some samples of the other space may not be part of the drawn sample so check all human annot
        cols = ['id'] + PROT_ATTR_CONTEXT + cols_probs
    data['space_fp'] = data0.loc[data0.id.isin(search_dict['space_fp'].id),cols]
    # sum score of being from the other space given by the humans
    data['space_fp']['human_score'] = data['space_fp'][cols_probs].sum(axis=1)

    # ... export csv candidates
    o_path = os.path.join(RES_DIR, 'candidate_scores')
    n_tag = get_tag_n_prot_attr()
    if not os.path.isdir(o_path):
        os.mkdir(o_path)

    cols2export = ['id', 'comment_text', 'all_entities', 'all_labels', 'score_categories', 'score']
    fn_candidates = pd.merge(search_dict['space_fn'][cols2export], data['space_fn'], how='inner', on='id')
    # ... sort by samples with highest prob to being in the other space acc to the onto and lower prob of the human annot
    fn_candidates.sort_values(by=['score','human_score'], ascending=[False, True], inplace=True)
    fn_candidates.to_csv(os.path.join(o_path, 'fn_by_score_{}.csv'.format(n_tag)))
    fn_candidates.sort_values(by=['score_categories','human_score'], ascending=[False, True], inplace=True)
    fn_candidates.to_csv(os.path.join(o_path, 'fn_by_score_categories_{}.csv'.format(n_tag)))

    fp_candidates = pd.merge(search_dict['space_fp'][cols2export], data['space_fp'], how='inner', on='id')
    fp_candidates.sort_values(by=['score','human_score'], ascending=[False, True], inplace=True)
    fp_candidates.to_csv(os.path.join(o_path, 'fp_by_score_{}.csv'.format(n_tag)))
    fp_candidates.sort_values(by=['score_categories','human_score'], ascending=[False, True], inplace=True)
    fp_candidates.to_csv(os.path.join(o_path, 'fp_by_score_categories_{}.csv'.format(n_tag)))
    print("Executed in %s seconds." % str(time.time()-t0))

    return data, fn_candidates, fp_candidates



def main():
    print('Starting gsso annotation analysis: \n In all data: {} \n'.format(USE_ALL_DATA))
    if not USE_ALL_DATA:
        print('... sample: {}'.format(sample))

    # Import ontology annotations: asserted, inferred (inc. transform to match asserted format), all_ent (merge both)
    # ... _ to import raw version of inferred csv file
    annot_dict = {}
    annot_dict['assert'], annot_dict['inf'], annot_dict['all_ent'], _ = collect_data_analysis()
    if USE_ALL_DATA or sample>thr_long_analysis:
        annot_dict = select_keys_from_dict(annot_dict, ['all_ent'])

    # Load ontology and ontology dict of class, individuals, and both (i.e., all)
    gsso = load_gsso()
    gsso_dict = {}
    gsso_dict['cls'], gsso_dict['ind'], gsso_dict['all'] = collect_gsso_dict(gsso)
    if USE_ALL_DATA or sample>thr_long_analysis:
        gsso_dict = select_keys_from_dict(gsso_dict, ['all'])
    # ... to map IRI to primary label, to analyse annotations in different ontology subsets
    gsso_iri2label_dict = create_iri2label_from_dict(gsso_dict['all'])

    # 1. Create Term-Occurrence matrixes: asserted, inferred, all_entities (asserted+inferred)
    # ... for each analysing annotations of subonto (classes, individuals, and all)
    o_dir = os.path.join(RES_DIR, 'saved_dict')
    n_tag = get_tag_n_prot_attr()
    title_tag = 'tf_dict'

    # subannot (asserted, inferred, and both), subonto (classes, indv, and both)
    tf_dict = {}
    for subannot in annot_dict.keys():
        filename = os.path.join(o_dir,'_'.join([title_tag,PROT_ATTR_CONTEXT[0],subannot, 'all',n_tag])+'.csv')
        if not os.path.exists(filename) or not (USE_ALL_DATA or sample>thr_long_analysis):
            # compute and save
            tf_dict[subannot] = create_tf_matrixes_dict(annot_dict[subannot], gsso_dict)
            if USE_ALL_DATA or sample>thr_long_analysis:
                save_tf_dict(title_tag, subannot, tf_dict[subannot])
        else:
            # import
            print('Importing saved tf matrixes')
            tf_dict[subannot] = open_tf_dict(title_tag, subannot, gsso_dict)
    # ... check example -> occurrences of woman: tf['assert']['cls']['gender']['http://purl.obolibrary.org/obo/GSSO_000369']


    # 2. Create frequency (#Total occurrences/#Total texts of each S)
    freq_dict = {}
    for subannot in annot_dict.keys():
        freq_dict[subannot] = compute_freq_from_tf_matrixes(tf_dict[subannot])


    # IRI_STOPWORD list: i.e., over a freq (e.g, 0.50) in all S, for plotting
    IRI_COMMON = get_common_entities(freq_dict, thr=0.50)
    if len(IRI_COMMON)<30:
        print([gsso_iri2label_dict[iri] for iri in IRI_COMMON])

    # 3. Collect top-N in tables: not using STOPWORDS
    table_dict, n = {}, 30
    for subannot in annot_dict.keys():
        n_tag = get_tag_n_prot_attr()
        o_filename = '{}_{}_top{}.csv'.format(subannot, n_tag, n)

        table_dict[subannot] = create_top_freq_table(freq_dict[subannot], gsso_iri2label_dict,
                                                     n, o_filename, stopwords=IRI_COMMON)

    # plot top-N frequencies
    for S in PROT_ATTR_CONTEXT:
        args_plot = {'subonto':'all', 'S': S, 'sw_tag':None}
        create_top_freq_plots(table_dict, freq_dict, gsso_iri2label_dict, args=args_plot)

    # 4. Plot frequencies in all_ent of the classes that represent the 6 annotation categories
    S_ATTR_categories = ['gender', 'sexual orientation', 'race', 'disability', 'religion']
    create_category_freq_plots(freq_dict, gsso_iri2label_dict, categories=S_ATTR_categories)

    # 5. Export human annotation error candidates (ie, FPs in neg space and FNs in pos space).

    # ... prepare df with samples of positive (i.e., related to g, so), and neg.
    args_df_space = {'space_fp': ['gender', 'sexual_orientation'],
                     'space_fn':['religion', 'race', 'disability', 'none']}
    search_dict = {}
    search_dict['space_fp'] , search_dict['space_fn'] = get_annotation_space(annot_dict, tf_dict, args_space=args_df_space)
    assert pd.merge(search_dict['space_fn'] , search_dict['space_fp'], how='inner', on=['id']).shape[0] == 0 # that there is no intersection

    # ... 5.1) export top-X FN candidates (i.e., samples in negative space with highest freq of g, so classes
    args_err_search = {'search': ['gender', 'sexual_orientation'],
                       'filter': ['religion', 'race', 'disability', 'none']}
    # a. only using frequencies of annotation categories
    entities_dict = {'entities': PROT_ATTR_CONTEXT, 'iri2label': gsso_iri2label_dict}
    search_dict['space_fn']['score_categories'] = compute_annot_reliability_score(search_dict['space_fn'], freq_dict, subannot='all_ent', subonto='all', S_pos=args_err_search['search'],S_neg=args_err_search['filter'], entities_dict=entities_dict)
    # b. using frequencies of all possible entities
    search_dict['space_fn']['score'] = compute_annot_reliability_score(search_dict['space_fn'], freq_dict, subannot='all_ent', subonto='all', S_pos=args_err_search['search'],S_neg=args_err_search['filter'])

    # ... 5.2) export top-X FP candidates (i.e., samples in positive space with lowest freq of g, so classes)
    # a. only using frequencies of annotation categories
    entities_dict = {'entities': PROT_ATTR_CONTEXT, 'iri2label': gsso_iri2label_dict}
    search_dict['space_fp']['score_categories'] = compute_annot_reliability_score(search_dict['space_fp'], freq_dict,subannot='all_ent', subonto='all',S_pos=args_err_search['search'],S_neg=args_err_search['filter'],entities_dict=entities_dict)
    # b. using frequencies of all possible entities
    search_dict['space_fp']['score'] = compute_annot_reliability_score(search_dict['space_fp'], freq_dict,subannot='all_ent', subonto='all',S_pos=args_err_search['search'],S_neg=args_err_search['filter'])


    # ... 5.3) export csv files to identify misannotations: candidates are at the bottom (lowest score and highest human score)
    search_cand_dict = {}
    search_human_info_dict, search_cand_dict['space_fn'], search_cand_dict['space_fp'] = export_error_candidates(search_dict, args_df_space)  # inc. human annot




if __name__ == '__main__':
    main()