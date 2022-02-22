"""

Helper functions

"""

def flatten(t):
    """ Helper function to make a list from a list of lists """
    return [item for sublist in t for item in sublist]


def get_stop_words_list(library='nltk'):
    """ Export list of stopwords. Currently spacy not working locally """
    sw = []
    if library == 'nltk':
        import nltk
        from nltk.corpus import stopwords
        sw = stopwords.words('english')
    elif library == 'spacy':
        import spacy
        # loading the english language small model of spacy
        try:
            en = spacy.load('en_core_web_sm')
            sw = en.Defaults.stop_words
        except:
            raise Exception('Try from terminal: python3 -m spacy download en_core_web_sm')
    elif library == 'gensim':
        import gensim
        from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
        sw = STOPWORDS
    elif library == 'sklearn':
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        sw = ENGLISH_STOP_WORDS
    else:
        raise Exception('Library {} not supported.'.format(library))
    return sw


def select_keys_from_dict(dict, key_l):
    """ Return a dict with only keys in key_l list"""
    dict_sel = {}
    for key in key_l:
        dict_sel[key] = dict[key]
    return dict_sel
