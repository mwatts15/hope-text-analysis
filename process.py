#!/usr/bin/env python
from __future__ import print_function

import sys

import binascii
import struct
import json
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize.casual import casual_tokenize
from nltk.stem.porter import PorterStemmer
import nltk.data

THRESHOLD = 1
WORD_THRESHOLD = 1

DISPLAY_LIMIT = 20
DISPLAY_WORD_LIMIT = 40

COMBINE_COLOCS_FOR_TALK = True

r = open("schedule.html").read()
soup = BeautifulSoup(r, 'html.parser')

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

k_start = 12
p_start = 13 - k_start

paras = list(soup.find_all('p'))[k_start:]

sw = set(stopwords.words('english')) | \
        set(('use', 'talk', 'present', 'year', 'new', 'discuss', 'two',
             "it'", 'like', 'get', 'make', 'technolog', 'also', 'need',
             'well',  # rather extensive use of 'as well as' in descriptions
             'way',  # 'along the way', 'the ways', 'new ways', etc.
             'panel',  # all about the format of the talk
             'take',
             'time',  # mostly used in common phrases/idosyncracies of author
             'one', 'includ')) | \
        set(':(/+=!*)"?.-,[]' + "'")

descriptions = []
speakers = []


def add(x, last):
    global descriptions
    descriptions.append(x)
    speakers.append(last)

last = None
for i, elt_contents in enumerate(paras):
    m = ''
    for z in elt_contents.contents:
        if isinstance(z, basestring):
            m += unicode(z)
        else:
            m += z.get_text()

    if i < 60 + p_start:
        if i % 4 == (0 + p_start) % 4:
            add(m, last)
    elif i == 60 + p_start:
        add(m, last)
    elif i == 63 + p_start:
        pass
    elif i < 151 + p_start:
        if i % 4 == (3 + p_start) % 4:
            add(m, last)
    elif i < 184 + p_start:
        # This skips the keynote -- this is what we want! There's no content
        # that we want there since the description definitely isn't the only
        # stuff Doctorow talks about
        if i % 4 == (0 + p_start) % 4:
            add(m, last)
    else:
        if i % 4 == (3 + p_start) % 4:
            add(m, last)

    last = m

stemmer = PorterStemmer()

all_pairs = Counter()
all_words = Counter()
pairs_concordance = dict()
sents = dict()
stem_record = dict()

def mhash(s):
    return binascii.hexlify(struct.pack("q", hash(s)))

with open("hope-descriptions.json", 'w') as f:
    json.dump([{'desc': m,
                'speakers': au,
                'tag':  mhash(m)}
               for au, m in zip(speakers, descriptions)],
              f, indent=2)

def process_text(text, adds=None, removals=None):
    if adds is None:
        adds = set([])
    if removals is None:
        removals = set([])
    words = casual_tokenize(text, preserve_case=False)
    filtered = set([])
    go_words = set([])
    normed_go_words = set([])
    for x in words:
        if x in sw:
            filtered.add(x)
        else:
            go_words.add(x)

    for x in go_words:
        nw = stemmer.stem(x)

        stem_record.setdefault(nw, set([]))
        stem_record[nw].add(x)

        if nw in sw:
            filtered.add(nw)
        else:
            normed_go_words.add(nw)
    normed_go_words = (normed_go_words | adds) - removals
    return normed_go_words

def make_sentences(text):
    return sent_detector.tokenize(text.strip())

for i, descr in enumerate(descriptions):
    removals = set([])  # Words that must be removed before further processing
    adds = set([])  # Words that the tokenizer misses that should be added

    if descr.startswith('This spring, the FCC'):
        # Proper name 'May First'
        removals.add('first')

    if 'Clipper Chip' in descr:
        adds.add('surveil')

    normed_go_words = process_text(descr, adds, removals)

    all_words.update(normed_go_words)

    pairs = set([])
    for sent in make_sentences(descr):
        sent_pairs = set([])
        sent_hash = mhash(sent)
        sent_words = process_text(sent)
        print_pairs = False
        for x in sent_words:
            for y in sent_words:
                key = None
                if x > y:
                    key = (x, y)
                elif y > x:
                    key = (y, x)
                else:
                    continue

                if key in (('role', 'play'), # A role to play
                           ('question', 'ask'), # Ask the question
                           ('role', 'import')): # Important role
                    continue

                sent_pairs.add(key)
                pairs_concordance.setdefault(key, set([]))
                pairs_concordance[key].add(sent_hash)
        sents[sent_hash] = sent
        if COMBINE_COLOCS_FOR_TALK:
            pairs.update(sent_pairs)
        else:
            all_pairs.update(sent_pairs)

    if COMBINE_COLOCS_FOR_TALK:
        all_pairs.update(pairs)

sig_pairs = []
for x in all_pairs:
    if all_pairs[x] > THRESHOLD:
        sig_pairs.append((all_pairs[x], x[0], x[1]))

sig_words = []
for x in all_words:
    if all_words[x] > WORD_THRESHOLD:
        sig_words.append((all_words[x], x))

with open('sig_words.json', 'w') as f:
    json.dump(sorted(sig_words, reverse=True), f, indent=2)

with open('sig_pairs.json', 'w') as f:
    json.dump(sorted(sig_pairs, reverse=True), f, indent=2)

with open('sentences.json', 'w') as f:
    json.dump(sents, f, indent=2)

with open('pairs_concordance.json', 'w') as f:
    json.dump([(k[1], k[2],
                tuple(pairs_concordance[(k[1], k[2])]))
               for k in sorted(sig_pairs, reverse=True)], f, indent=2)


def klap(sig_words, display_limit, show_unstemmed=False, show_context=False):
    print("----")
    sorted_sig_words = sorted(sig_words, reverse=True)
    sig_word_nums = [x[0] for x in sorted_sig_words]

    print('median', sorted_sig_words[len(sorted_sig_words) / 2])
    print('average', sum(sig_word_nums) / len(sig_word_nums))

    top = sorted_sig_words[display_limit - 1]
    below = len([x for x in sorted_sig_words if x[0] < top[0]])
    same = len([x for x in sorted_sig_words if x[0] == top[0]])
    print('top entry percentile',  (below + same * .5) / len(sorted_sig_words))
    for x in sorted_sig_words[:display_limit]:
        print(*x, end=" ")
        if show_context:
            print()
            for skey in pairs_concordance[x[1:]]:
                print('   ', sents[skey])

        if show_unstemmed:
            print(*tuple(str(m) for m in stem_record[x[1]]))
        else:
            print()
    print("----")

klap(sig_words, DISPLAY_WORD_LIMIT, show_unstemmed=True)
klap(sig_pairs, DISPLAY_LIMIT, show_context=True)
