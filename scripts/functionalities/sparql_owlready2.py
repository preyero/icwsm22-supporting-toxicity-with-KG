"""

Examples of SPARQL queries in GSSO ontology
using owlready2:

- get a concept
- from a concept, get its superclasses
- from a concept, get all individuals

29th November 2021

"""

import os

import owlready2
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


def example_get_concept():
    r = list(default_world.sparql("""
                SELECT ?x
                { ?x rdfs:label "exclusive gender identity" . }
        """))
    print(r)
    # r = gsso.search_one(label = "exclusive gender identity")
    return r


def example_get_superclasses():
    " Get all superclasses from a class "
    r = list(default_world.sparql("""
            SELECT ?y
            { ?x rdfs:label "exclusive gender identity" .
              ?x rdfs:subClassOf* ?y
            }
            """))
    return r


def example_get_individuals():
    " Get individuals of a class "
    r = list(default_world.sparql("""
            SELECT ?y
            { ?x rdfs:label "initialism" .
              ?y a/rdfs:subClassOf* ?x
            }
            """))
    return r


def example_get_types():
    " Get types of an individual "
    r = list(default_world.sparql("""
            SELECT ?y
            { ?x rdfs:label "it" .
              ?x rdf:type ?y
            }
            """))
    r = [vi for vi in r
         if isinstance(vi[0], owlready2.entity.Thing) or isinstance(vi[0], owlready2.entity.ThingClass)]
    return r


def get_superclasses(cls_iris, gsso):
    " Get all superclasses from a list of class labels"
    cls_entities_inf = []
    cls_labels_inf = []
    for cls_iri in cls_iris:
        # Search concept in ontology with that label
        concept = gsso.search_one(iri=cls_iri)
        # Get all superclasses
        if isinstance(concept, owlready2.entity.ThingClass):
            r = list(default_world.sparql("""
                    SELECT ?y
                    {   ?? rdfs:subClassOf* ?y
                    }
                    """, [concept]))
            # append list: inferred IRI and Labels from each concept
            cls_entities_inf.append([k[0].iri for k in r if k[0].label])
            cls_labels_inf.append([k[0].label[0] for k in r if k[0].label]) # utils.flatten to get list of sublists
    return cls_entities_inf, cls_labels_inf


def get_types(ind_iris, gsso):
    " Get types from a list of individual labels"
    ind_entities_inf = []
    ind_labels_inf = []
    for ind_iri in ind_iris:
        # Search concept in ontology with that label
        concept = gsso.search_one(iri=ind_iri)
        # print(concept)
        # Get all types (from the individual, i.e., Thing)
        if isinstance(concept, owlready2.entity.Thing):
            ri = list(default_world.sparql("""
                    SELECT ?y
                    { ?? rdf:type ?y
                    }
                    """, [concept]))
            # types are classes (i.e., ThingClass)
            ri = [vi for vi in ri if isinstance(vi[0], owlready2.entity.ThingClass)]

            ind_entities_inf.append([k[0].iri for k in ri if k[0].label])
            ind_labels_inf.append([k[0].label[0] for k in ri if k[0].label])
        else:
            print('ind IRI not found: {} \n {}'.format(concept, concept.iri))
    return ind_entities_inf, ind_labels_inf


def main():
    # Load one or more ontologies
    gsso = load_gsso()

    print(list(default_world.sparql("""
               SELECT (COUNT(?x) AS ?nb)
               { ?x a owl:Thing . }
        """)))

    example_concept = example_get_concept()[0][0]
    print('Concept: {}'.format(example_concept))

    example_superclasses = example_get_superclasses()
    example_superclasses_labels = [k[0].label[0] for k in example_superclasses]
    print(example_superclasses_labels)

    example_individuals = example_get_individuals()
    print([k[0].label[0] for k in example_individuals])

    example_types = example_get_types()
    print([k[0].label[0] for k in example_types])

    # Load class, ind annotation labels
    cls_labels_i = ['female gender identity', 'woman', 'video', 'submission', 'face', 'interest']
    ind_labels_i = ['it', 'out', 'outing', 'it']

    # Load class, ind entities
    cls_entities_i = ['http://purl.obolibrary.org/obo/GSSO_000089','http://purl.obolibrary.org/obo/GSSO_000369',
                      'http://purl.obolibrary.org/obo/NCIT_C96985','http://purl.obolibrary.org/obo/GSSO_001306',
                      'http://purl.org/sig/ont/fma/fma24728','http://semanticscience.org/resource/SIO_000848']
    ind_entities_i = ['http://purl.obolibrary.org/obo/GSSO_002441','http://purl.obolibrary.org/obo/GSSO_009441',
                      'http://purl.obolibrary.org/obo/GSSO_009442','http://purl.obolibrary.org/obo/GSSO_010937']

    # Get superclasses
    cls_entities_i_superclasses, cls_labels_i_superclasses = get_superclasses(cls_labels_i, gsso)
    print('{}: \n \n {} \n \n {}'.format(cls_labels_i, cls_entities_i_superclasses,
                                         cls_labels_i_superclasses))

    # Get types
    ind_entities_i_types, ind_labels_i_types = get_types(ind_entities_i, gsso)
    print('{}: \n \n {} \n \n {}'.format(ind_labels_i, ind_entities_i_types,
                                         ind_labels_i_types))

if __name__ == '__main__':
    main()

