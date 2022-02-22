## dataCollect.py

Helper functions to import data:

* get_tf_data: returns Jigsaw Toxicity dataset (1.8 M)

* get_identity_data(tf_data): merges Jigsaw Toxicity dataset with identify annotations (445k).


## gsso.py

Helper functions to use gsso ontology:

* `load_gsso()`: returns Ontology object (owlready2)
* `get_entity_annotation_dict(subonto)`: from subonto (Ontology .classes(), .individuals(), or both).
Wrapper function (`create_gsso_dict(gsso)` to get cls, ind; and `create_gsso_dict_all(gsso)` for both).
Wrapper function (`collect_gsso_dict(gsso)` to perform all previous functions).

* `create_iri2label_from_dict(gsso_dict)`: dictionary with all entities {IRI:[label[0]}
To use with `get_label_from_iri_list(iri_list, iri2label_dict)`, `get_iri_from_label_list(label_list, iri2label_dict)` functions.


## gsso_annotate.py

Script to detect *class* and *individual* entities of the GSSO in text.

Parameters:
* `DEFAULT_SAMPLE_GROUP`: String. Protected group (from dataCollect categories) if annotating a sample from them. 
Otherwise, from Jigsaw Toxicity 1.8M.
* `DEFAULT_N`: Int. Number of texts to sample randomly.

## gsso_annotate_inferred.py

Script to get superclasses of class entities (`'cls_entities'`) and types of individual entities (`'ind_entities'`)

Input: exported annotated files using 'gsso_annotate.py' and copied to `gsso_annotations` folder.

Dependency: owlready2 functions from script (/functionalities/sparql_owlready2.py)

Parameters:
* `sample`: if using a sample `gsso_annotate.py` .

## gsso_analysis.py

Script to export tables with top N most frequent entities for each protected group category (`freq_tables`) 
and their heatmap with the frequencies in all protected groups of each top N (`freq_plots`),

Heatmaps of frequencies of protected group categories of asserted and inferred facts(`freq_plot_categories`)

And a sorted list of most likely human annotation errors, i.e., texts not annotated as related to gender or sexual 
orientation but with entities that frequently appear in gender and so related language (`FN`).

Dependency: functions from the `plots.py` helper script.

Parameters:
* `USE_STOP_WORDS`: Bool. remove entities whose label matches words in gensim, nltk, and sklearn stopword lists, indepedently.

The analysis will be restricted to only inferred and asserted entities (i.e. `all_ent`) and both classes and individuals (i.e.`all`)
to reduce computational time if:
* `USE_ALL_DATA`: Bool. We take the total number of samples of each protected group (defined in `N_PROT_ATTR_DICT`)
* `thr_long_analysis`: Int. We define a threshold sample size (with `sample` line 48) over which the analysis will be restricted to those two.

