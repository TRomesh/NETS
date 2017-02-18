import string
import re
from gensim.models.doc2vec import TaggedDocument

regex_newline_tag = re.compile(r'(<br(>|/>)|\r\n)')
regex_with_slash = re.compile(r'(^| )w/\S')
regex_vt = re.compile(r'[\x0b]')


# modified version of https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
def normalize_text(text, _regex_newline_tag=regex_newline_tag, _regex_with_slash=regex_with_slash, _regex_vt=regex_vt):
    norm_text = text.lower()

    # Replace VT with spaces
    norm_text = re.sub(_regex_vt, ' ', norm_text)

    # Remove non-printable characters
    printable_chars = list()
    for x in norm_text:
        printable_chars.append(x if x in string.printable else ' ')
    norm_text = ''.join(printable_chars)

    # Replace breaks with spaces
    '''
    norm_text = norm_text.replace('<br />', ' ')
    norm_text = norm_text.replace('<br>', ' ')  # additional rule
    norm_text = norm_text.replace('<br/>', ' ')  # additional rule 1
    norm_text = norm_text.replace('\r\n', ' ')  # additional rule 2
    '''
    norm_text = re.sub(_regex_newline_tag, ' ', norm_text)

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':', '@', '’', '-', '\'', '&', '“', '”', '+', '#']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    # pad after space + w/
    norm_text = re.sub(_regex_with_slash, ' w/ ', norm_text)

    def pad_slash_with_spaces(_text):
        c_walk = 0
        while c_walk < len(_text):
            if '/' == _text[c_walk]:
                if c_walk == 0:
                    _text = _text.replace('/', ' / ', 1)
                    c_walk = 3
                elif c_walk == 1:  # ^w/ (it can't reach here because re.sub added a space before ^w/)
                    c_walk += 1
                else:
                    if not ' w' == _text[c_walk - 2:c_walk]:  # ' w/'
                        _text = _text[:c_walk] + _text[c_walk:].replace('/', ' / ', 1)
                        c_walk += 3
                    else:
                        c_walk += 1
            else:
                c_walk += 1
        return _text

    norm_text = pad_slash_with_spaces(norm_text)

    return norm_text.strip()


class EventTitleCorpus(object):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        for week in self.source:
            for event in week:
                # 3: title
                words = event[3]  # already tokenized
                # 0: user, 1: week, 5: sequence
                yield TaggedDocument(words, [event[0] + '_' + event[1] + '_' + event[5]])

