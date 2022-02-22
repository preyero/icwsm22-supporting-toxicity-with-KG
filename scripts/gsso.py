"""
Functions to use GSSO ontology in Python

8 December 2021
"""
import time

from owlready2 import *

PROJ_DIR = os.getcwd()
print("Hello from {}".format(PROJ_DIR))


def load_gsso():
    O_L_PATH = os.path.join(PROJ_DIR, "owl")
    O_URL = 'https://raw.githubusercontent.com/Superraptor/GSSO/master/gsso.owl'
    print('... ontology local path: {}'.format(O_L_PATH))
    # glob var of local copies of ontologies
    onto_path.append(O_L_PATH)
    gsso = get_ontology(O_URL).load()
    return gsso


def get_entity_annotation_dict(subonto):
    """ Dictionary for text annotation (linking ontology to text) """
    # Create a dictionary with annotation attributes: ['label', 'alternate_name', 'short_name', 'has_synonym', 'has_exact_synonym',
    #                        'has_broad_synonym', 'has_narrow_synonym', 'has_related_synonym', 'replaces']
    subonto_dict = {k: k.label + k.alternateName + k.short_name + k.hasSynonym + k.hasExactSynonym + k.hasBroadSynonym +
                       k.hasNarrowSynonym + k.hasRelatedSynonym + k.replaces + k.isReplacedBy
                    for k in subonto}
    print(len(subonto_dict))

    # drop entries without a label
    subonto_dict = {k:v for k,v in subonto_dict.items() if len(v)!=0}
    print('... after dropping missing labels: {}'.format(len(subonto_dict)))


    return subonto_dict


def create_gsso_dict(gsso):
    """ Store ontology as dictionary for ontology manipulation """
    gsso_cls, gsso_indv = list(gsso.classes()), list(gsso.individuals())
    print('Retrieved {} classes and {} individuals\n'.format(len(gsso_cls), len(gsso_indv)))

    print('Creating dict of key labels:')
    gsso_cls_dict, gsso_indv_dict = get_entity_annotation_dict(gsso.classes()), get_entity_annotation_dict(gsso.individuals())
    return gsso_cls_dict, gsso_indv_dict


def create_gsso_dict_all(gsso):
    """ Store ontology as dict for ontology manipulation"""
    gsso_all = list(gsso.classes()) + list(gsso.individuals())
    print('Retrieved {} entities \n'.format(len(gsso_all)))

    print('Creating dict of key labels:')
    gsso_dict = get_entity_annotation_dict(gsso_all)
    return gsso_dict


def collect_gsso_dict(gsso):
    """ Export gsso as a dict: keys are cls, ind, all (ie cls+ind)"""
    print('Importing gsso as dict')
    t0 = time.time()
    gsso_cls_dict, gsso_ind_dict = create_gsso_dict(gsso)
    gsso_all_dict = create_gsso_dict_all(gsso)
    print("Executed in %s seconds." % str(time.time()-t0))

    return gsso_cls_dict, gsso_ind_dict, gsso_all_dict


# Using GSSO with IRI and main label (text detection is done with gsso_dict
def create_iri2label_from_dict(gsso_dict):
    """ Creating a dict of IRI: label[0] using a (sub)ontology dict"""
    gsso_iri2label_dict= {k.iri:v[0] for k, v in gsso_dict.items()}

    return gsso_iri2label_dict


def get_label_from_iri_list(iri_list, iri2label_dict):
    """ Helper for getting labels of a df with IRI lists as values using an iri2label dict"""
    label_list = [iri2label_dict[iri] for iri in iri_list]
    return label_list


def get_iri_from_label_list(label_list, iri2label_dict):
    # match = [iri for iri,labels in iri2label_dict.items()
    #          for lab in labels
    #          if lab.lower() in label_list ]
    match = [iri for iri,label in iri2label_dict.items() if label in label_list]
    return match




def main():
    # load ontology
    gsso = load_gsso()
    gsso_cls, gsso_indv = list(gsso.classes()), list(gsso.individuals())
    print('Retrieved {} classes and {} individuals\n'.format(len(gsso_cls), len(gsso_indv)))

    # create/load entity dictionaries (classes, individuals)
    print('Creating dict of key labels:')
    gsso_cls_dict, gsso_indv_dict = get_entity_annotation_dict(gsso.classes()), get_entity_annotation_dict(gsso.individuals())


if __name__ == '__main__':
    main()