"""

Helper functions

"""

def flatten(t):
    """ Helper function to make a list from a list of lists """
    return [item for sublist in t for item in sublist]

def select_keys_from_dict(dict, key_l):
    """ Return a dict with only keys in key_l list"""
    dict_sel = {}
    for key in key_l:
        dict_sel[key] = dict[key]
    return dict_sel
