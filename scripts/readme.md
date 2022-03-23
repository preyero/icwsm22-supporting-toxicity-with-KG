## dataCollect.py

Helper functions to import data:

* `get_tf_data()`: returns Jigsaw Toxicity dataset (1.8 M)

* `get_identity_data(tf_data)`: merges Jigsaw Toxicity dataset with identify annotations (445k).


## gsso.py

Helper functions to use gsso ontology:

* `load_gsso()`: returns Ontology object (owlready2)
* `create_gsso_dict_all(gsso)`: create dict from subonto (i.e. {IRI:label, alternateName, [...]}) for text detection.
* `create_iri2label_from_dict(gsso_dict)`: dict with all entities (i.e. {IRI:[label[0]}) for analysis.

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
* `sample`: if used a sample in `gsso_annotate.py` .

## gsso_analysis.py

Script to export tables with top N most frequent entities for each protected group category (`freq_tables`) 
and their heatmap with the frequencies in all protected groups of each top N (`freq_plots`),

Heatmaps of frequencies of protected group categories of asserted and inferred facts(`freq_plot_categories`)

And a sorted list of most likely human annotation errors, i.e., texts not annotated as related to gender or sexual 
orientation but with entities that frequently appear in gender and so related language (`FN`).

Dependency: functions from the `plots.py` helper script.

Parameters:

The analysis will be restricted to only inferred and asserted entities (i.e. `'all_ent'`) and both classes and individuals (i.e.`'all'`)
to reduce computational time if:
* `USE_ALL_DATA`: Bool. If we take the total number of samples of each protected group (defined in `N_PROT_ATTR_DICT`),
* `thr_long_analysis`: Int. Or if we define a threshold sample size (with `sample`) over which the analysis will be restricted to those two.

