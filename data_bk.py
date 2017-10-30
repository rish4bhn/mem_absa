import os
import pdb
import xml.etree.ElementTree as ET
from collections import Counter

import spacy

en_nlp = spacy.load("en")


# def _get_abs_pos(cur, ids):
#     min_dist = 1000
#     for i in ids:
#         if abs(cur - i) < min_dist:
#             min_dist = abs(cur - i)
#     if min_dist == 1000:
#         raise ("[!] ids list is empty")
#     return min_dist
#
#
# def _count_pre_spaces(text):
#     count = 0
#     for i in range(len(text)):
#         if text[i].isspace():
#             count = count + 1
#         else:
#             break
#     return count
#
#
# def _count_mid_spaces(text, pos):
#     count = 0
#     for i in range(len(text) - pos):
#         if text[pos + i].isspace():
#             count = count + 1
#         else:
#             break
#     return count
#
#
# def _check_if_ranges_overlap(x1, x2, y1, y2):
#     return x1 <= y2 and y1 <= x2


def _get_data_tuple(sptoks, asp_term, from_idx, to_idx, label, word2idx):
    # Find the ids of aspect term
    aspect_is = []
    for sptok in sptoks:
        # if sptok.idx >= from_idx and sptok.idx + len(sptok.text) <= to_idx:
        if sptok.idx<to_idx and sptok.idx + len(sptok.text) > from_idx:
            aspect_is.append(sptok.i)

    assert aspect_is, pdb.set_trace()
    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = 0
    elif label == 'neutral':
        lab = 1
    elif label == "positive":
        lab = 2
    else:
        raise ValueError("Unknown label: %s" % lab)

    return pos_info, lab


# def old_get_data_tuple(sptoks, asp_term, fro, to, label, word2idx):
#     # Find the ids of aspect term
#     ids, st, i = [], _count_pre_spaces(sptoks.text), 0
#     for sptok in sptoks:
#         if _check_if_ranges_overlap(st, st + len(sptok.text) - 1, fro, to - 1):
#             ids.append(i)
#         st = st + len(sptok.text) + _count_mid_spaces(sptok.text.lower(), st + len(sptok.text))
#         i = i + 1
#     pos_info, i = [], 0
#     for sptok in sptoks:
#         pos_info.append(_get_abs_pos(i, ids))
#         i = i + 1
#
#     lab = None
#     if label == 'negative':
#         lab = -1
#     elif label == 'neutral':
#         lab = 0
#     elif label == "positive":
#         lab = 1
#     else:
#         raise ValueError("Unknown label: %s" % lab)
#
#     return pos_info, lab


def read_data(fname, source_count, source_word2idx):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    tree = ET.parse(fname)
    root = tree.getroot()

    source_words, target_words, max_sent_len = [], [], 0
    for sentence in root:
        sptoks = en_nlp(sentence.find('text').text)
        source_words.extend([sp.text.lower() for sp in sptoks])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.findall('aspectTerm'):
                if asp_term.get("polarity") == "conflict": continue  # TODO:
                t_sptoks = en_nlp(asp_term.get('term'))
                target_words.extend([sp.text.lower() for sp in t_sptoks])
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()
    for sentence in root:
        sptoks = en_nlp(sentence.find('text').text)
        if len(sptoks.text.strip()) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[sptok.text.lower()])
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    if asp_term.get("polarity") == "conflict": continue  # TODO:
                    t_sptoks = en_nlp(asp_term.get('term'))
                    source_data.append(idx)
                    pos_info, lab = _get_data_tuple(sptoks, t_sptoks, int(asp_term.get('from')),
                                                    int(asp_term.get('to')), asp_term.get('polarity'), source_word2idx)
                    source_loc_data.append(pos_info)
                    target_data.append([source_word2idx[sp.text.lower()] for sp in t_sptoks])
                    target_label.append(lab)

    print("Read %s aspects from %s" % (len(source_data), fname))
    return source_data, source_loc_data, target_data, target_label, max_sent_len
