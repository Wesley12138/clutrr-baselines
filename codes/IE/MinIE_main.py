#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import csv
from tqdm import tqdm
import os
import pandas as pd
from ast import literal_eval

os.environ['CLASSPATH'] = "../../../../../target/minie-0.0.1-SNAPSHOT.jar"

# Uncomment to point to your java home (an example is given for arch linux)
# if you don't have it as an environment variable already.
# os.environ['JAVA_HOME'] = '/usr/lib/jvm/default'

# Import java classes in python with pyjnius' autoclass (might take some time)
from jnius import autoclass

CoreNLPUtils = autoclass('de.uni_mannheim.utils.coreNLP.CoreNLPUtils')
AnnotatedProposition = autoclass('de.uni_mannheim.minie.annotation.AnnotatedProposition')
MinIE = autoclass('de.uni_mannheim.minie.MinIE')
StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
String = autoclass('java.lang.String')

# Dependency parsing pipeline initialization
parser = CoreNLPUtils.StanfordDepNNParser()


def get_closest_name(rel, lst, text, direction, rel_id=None, ignore_name=None):
    """
    # guest the object is the [ahead/next] closest NAME
    :param rel: relation
    :param lst: now we are in which lst, sub_lst or obj_lst
    :param text: story text
    :param direction: ahead or next
    :param rel_id: the ind of rel in the text, default=None
    :param ignore_name: ignore particular name, default=None
    :return: Name
    """
    dire = 0 if direction == 'ahead' else -1
    text = text.replace(',', '').replace('.', '').replace("'", " ")

    for h in ['his', 'her']:  # lots of situation with 'his/her'
        if h in lst:
            new_text = text.split(f'{h} {rel}')[dire].split(' ')
            if not dire:
                new_text = new_text[::-1]
            for w in new_text:
                if w.istitle():
                    return w.rstrip(',').rstrip('.')

    # if no 'his/her', we try to look around
    text = text.split(' ')
    if not rel_id:
        rel_id = text.index(rel)
    new_text_1, new_text_2 = text[:rel_id], text[rel_id + 1:]
    if not dire:
        for w in new_text_1[::-1]:
            if w.istitle() and (w != ignore_name):
                return w
        for w in new_text_2:
            if w.istitle() and (w != ignore_name):
                return w
    else:
        for w in new_text_2:
            if w.istitle() and (w != ignore_name):
                return w
        for w in new_text_1[::-1]:
            if w.istitle() and (w != ignore_name):
                return w
    return


def main(argv):
    argparser = argparse.ArgumentParser('MINIE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    extract_path = "corpus/6.2_train.csv"   # extract file

    argparser.add_argument('--extract', action='store', type=str, default=extract_path)

    argparser.add_argument('--concat', action='store_true', default=False)
    argparser.add_argument('--random', action='store_true', default=False)
    argparser.add_argument('--guess', action='store_true', default=False)

    # Normal: te=0;  Test: te=1, te_ind=[...]
    argparser.add_argument('--te', action='store_true', default=False)
    argparser.add_argument('--te_ind', nargs='+', type=int, default=0)

    args = argparser.parse_args(argv)

    extract_path = args.extract
    concat = args.concat
    random = args.random
    guess = args.guess
    te = args.te
    te_ind = args.te_ind

    # get test data
    with open(extract_path, newline='') as ft:
        reader = csv.reader(ft)
        story_triples = []
        for i, row in enumerate(tqdm(reader)):
            if i == 0:
                continue
            triple = []
            state = row[8]  # "proof-state"
            state = literal_eval(state)  # str->dict
            tmp0 = [v for val in state[0].values() for v in val]   # top term's value (keep)
            if len(state) == 1:
                triple.extend(tmp0)
            else:
                tmp1 = [key for s in state[1:] for key in s.keys()]  # rest terms' key (remove)
                tmp2 = [v for s in state[1:] for val in s.values() for v in val]  # rest terms' value (keep)
                tmp3 = [value for value in set(tmp0 + tmp2) if value not in set(tmp1)]
                triple.extend(tmp3)
            story_triples.append(list(set(triple)))
        print('Extraction of test triple list done!')

    kinship = ['aunt', 'brother', 'brother-in-law', 'daughter', 'daughter-in-law', 'father',
               'father-in-law', 'granddaughter', 'grandfather', 'grandmother', 'grandson', 'husband',
               'mother', 'mother-in-law', 'nephew', 'niece', 'sister', 'sister-in-law', 'son',
               'son-in-law', 'uncle', 'wife', 'mon', 'dad']
    ex = {'aunt': ['nephew', 'niece', 'uncle'], 'brother': ['sister'], 'brother-in-law': ['sister-in-law'],
          'daughter': ['father', 'mother', 'son'], 'daughter-in-law': ['father-in-law', 'mother-in-law', 'son-in-law'],
          'father': ['son', 'daughter', 'mother'], 'father-in-law': ['son-in-law', 'daughter-in-law', 'mother-in-law'],
          'granddaughter': ['grandfather', 'grandmother', 'grandson'],
          'grandfather': ['grandson', 'granddaughter', 'grandmother'],
          'grandmother': ['grandson', 'granddaughter', 'grandfather'],
          'grandson': ['grandfather', 'grandmother', 'granddaughter'],
          'husband': ['wife'], 'mother': ['son', 'daughter', 'father'],
          'mother-in-law': ['son-in-law', 'daughter-in-law', 'father-in-law'],
          'nephew': ['uncle', 'aunt', 'niece'], 'niece': ['uncle', 'aunt', 'nephew'], 'sister': ['brother'],
          'sister-in-law': ['brother-in-law'], 'son': ['father', 'mother', 'daughter'],
          'son-in-law': ['father-in-law', 'mother-in-law', 'daughter-in-law'], 'uncle': ['nephew', 'niece', 'aunt'],
          'wife': ['husband'],
          'mon':['mother', 'son', 'daughter', 'father'],
          'dad':['father', 'son', 'daughter', 'mother']}  # degree + gender exchangeable

    with open(extract_path, newline='') as f:
        reader = csv.reader(f)
        failed = []
        err = []
        tri_lst = []  # save tri for all stories
        for ind, row in enumerate(tqdm(reader)):
            # ind, story_, cstory_ = row[0], row[2], row[7]
            if random:
                cstory_ = row[2]
            else:
                cstory_ = row[7]
            tri_one = []  # save tri for each story

            if ind > 0:
                # Remove '[' and ']', and remove comma ',' to integrate two sentences.
                cstory_ = cstory_.rstrip(' ')
                if cstory_[-1] != '.':
                    cstory_ += '.'
                cstory_ = cstory_.replace('[', '').replace(']', '').replace('  ', ' ').replace(' He ', ' he ').replace(' She ', ' she ')  # .replace('.', ',', 1)
                if concat:
                    cstory_ = ','.join(cstory_.split('.')[:-1]) + '.'    # concat sentences
                if te and (ind in te_ind):
                    print(f'No.{ind}; Story: {cstory_}')
                elif te and te_ind and (ind > max(te_ind)):  # exit after run all ind in te_ind
                    exit()
                elif te and (ind not in te_ind):   # only run ind in te_ind
                    continue

                # Generate the extractions (With SAFE mode (mode = 2))
                # NOTE: sentence must be wrapped into String, else it won't work.
                minie = MinIE(String(cstory_), parser, 2)

                for ap in minie.getPropositions().elements():
                    if ap:   # Some elements might be null
                        tri_str = ap.getTripleAsString().replace('"', '').split('\t')[:-1]
                        if len(tri_str) == 3:
                            s, r, o = tri_str
                        else: continue
                        triple = {'subject': s, 'relation': r, 'object': o}
                        if te and (ind in te_ind): print('|-', triple)
                    else: continue
                    flag = 0

                    sub_lst = triple['subject'].split(' ')
                    rel_lst = triple['relation'].split(' ')
                    obj_lst = triple['object'].split(' ')

                    name = []
                    for w_o in obj_lst:
                        if w_o.istitle():
                            name.append(w_o)
                    if len(name) == 2:  # two name in obj_lst
                        if len(set(obj_lst) & set(kinship)) == 1:
                            sub1 = name[1]
                            rel1 = list(set(obj_lst) & set(kinship))[0]
                            obj1 = name[0]  # 'Chong', 'is grandmother of', "[Margaret] [Norma] 's [daughter]"
                            tri1 = (sub1, rel1, obj1)
                            tri_one.append(tri1)
                            if te and (ind in te_ind): print(f'(1) two name get 1 !!! {tri1}')
                            flag = 1
                        if sub_lst[0].istitle() and (len(set(rel_lst) & set(kinship)) == 1):
                            sub2 = sub_lst[0]
                            rel2 = list(set(rel_lst) & set(kinship))[0]
                            obj2 = name[0]   # '[Chong]', 'is [grandmother] of', "[Margaret] Norma 's daughter"
                            tri2 = (sub2, rel2, obj2)
                            tri_one.append(tri2)
                            if te and (ind in te_ind): print(f'(2) two name get 2 !!! {tri2}')
                            flag = 1
                        else: name = None
                    elif len(name) == 1: name = name[0]
                    else: name = None
                    if flag == 1: continue

                    if name and (len(set(rel_lst) & set(kinship)) == 2):
                        mid_name = None
                        rel_set = []
                        for w_r in rel_lst:
                            if w_r.istitle(): mid_name = w_r
                            if w_r in kinship: rel_set.append(w_r)
                        if mid_name:
                            sub1 = mid_name
                            rel1 = rel_set[1]
                            obj1 = name
                            tri1 = (sub1, rel1, obj1)
                            tri_one.append(tri1)   # 'Grace', 'is daughter of [James] [father] of', '[Richard]'
                            if te and (ind in te_ind): print(f'(3) two rel get 1 !!! {tri1}')
                            flag = 1
                        elif guess:
                            rel1 = rel_set[1]
                            obj1 = name
                            sub1 = get_closest_name(rel=rel1, lst=[], text=cstory_, direction='ahead')
                            if sub1:
                                tri1 = (sub1, rel1, obj1)
                                tri_one.append(tri1)  # 'Grace', 'is daughter [father] of', '[Richard]'
                                if te and (ind in te_ind): print(f'(4) two rel & guess !!! {tri1}')
                                flag = 1

                        if sub_lst[0].istitle():
                            if mid_name:
                                obj2 = mid_name
                            elif guess and sub1:   # utilise the first guess sub
                                obj2 = sub1
                            if obj2:
                                sub2 = sub_lst[0]
                                rel2 = rel_set[0]
                                tri2 = (sub2, rel2, obj2)
                                tri_one.append(tri2)  # '[Grace]', 'is [daughter] (of [James]) father of', 'Richard'
                                if te and (ind in te_ind): print(f'(5) two rel get 2 !!! {tri2}')
                                flag = 1
                    if flag == 1: continue

                    if name and ('of' in rel_lst or rel_lst == ['is']) and \
                            (len((set(rel_lst) & set(kinship)) | (set(obj_lst) & set(kinship)))==1):
                        # is ... of situation, need exchange sub and obj
                        flag = 1
                        sub = name
                        rel = list((set(rel_lst) & set(kinship)) | (set(obj_lst) & set(kinship)))[0]
                        if sub_lst[0].istitle():
                            obj = sub_lst[0]
                            if sub == obj: flag = 0; continue
                            tri = (sub, rel, obj)
                            tri_one.append(tri)
                            if te and (ind in te_ind): print(f'of/is !!! {tri}')
                        elif guess:  # if no name in sub_lst, try to guest
                            obj = get_closest_name(rel=rel, lst=obj_lst, text=cstory_, direction='ahead')
                            if obj:
                                if sub == obj: flag = 0; continue
                                tri = (sub, rel, obj)
                                tri_one.append(tri)
                                if te and (ind in te_ind): print(f'(6) of/is & guest !!! {tri}')
                            else:
                                flag = 0

                    elif name and (len((set(rel_lst) & set(kinship)) | (set(obj_lst) & set(kinship)))==1):
                        flag = 1
                        rel = list((set(rel_lst) & set(kinship)) | (set(obj_lst) & set(kinship)))[0]
                        obj = name
                        if sub_lst[0].istitle():
                            sub = sub_lst[0]
                            if sub == obj: flag = 0; continue
                            tri = (sub, rel, obj)
                            tri_one.append(tri)
                            if te and (ind in te_ind): print(f'(7) obj !!! {tri}')
                        elif guess:  # if no name in sub_lst, try to guest
                            sub = get_closest_name(rel=rel, lst=obj_lst, text=cstory_, direction='ahead')
                            if sub:
                                if sub == obj: flag = 0; continue
                                tri = (sub, rel, obj)
                                tri_one.append(tri)
                                if te and (ind in te_ind): print(f'(8) obj & guest !!! {tri}')
                            else:
                                flag = 0

                    elif rel_lst == ['has'] and ('called' in obj_lst or 'named' in obj_lst)\
                            and (len((set(rel_lst) & set(kinship)) | (set(obj_lst) & set(kinship)))==1):
                        # if no name appear in obj_lst, try to guest  Todo: (may be discard) A is father: father A / A is father of
                        flag = 1
                        rel = list((set(rel_lst) & set(kinship)) | (set(obj_lst) & set(kinship)))[0]
                        if sub_lst[0].istitle():
                            sub = sub_lst[0]
                            obj = get_closest_name(rel=rel, lst=obj_lst, text=cstory_, direction='next')
                            if obj:
                                if sub == obj: flag = 0; continue
                                tri = (sub, rel, obj)
                                tri_one.append(tri)
                                if te and (ind in te_ind): print(f'(9) keyword & guest !!! {tri})')
                            else:
                                flag = 0
                        else:
                            flag = 0

                    elif name and guess and (len(set(rel_lst) & set(kinship)) == 1 and len(set(obj_lst) & set(kinship)) == 1):
                        rel1 = list((set(obj_lst) & set(kinship)))[0]
                        obj1 = get_closest_name(rel=rel1, lst=[], text=cstory_, direction='ahead', ignore_name=name)
                        if obj1:
                            sub1 = name
                            tri1 = (sub1, rel1, obj1)
                            tri_one.append(tri1)  # 'Norma', 'has brother named', "[Susan] 's [mother]"
                            if te and (ind in te_ind): print(f'(10) two rel & guess 1 !!! {tri1}')
                            flag = 1
                        if sub_lst[0].istitle():
                            sub2 = sub_lst[0]
                            rel2 = list((set(rel_lst) & set(kinship)))[0]
                            obj2 = get_closest_name(rel=rel2, lst=[], text=cstory_, direction='next', ignore_name=name)
                            tri2 = (sub2, rel2, obj2)
                            tri_one.append(tri2)  # '[Norma]', 'has [brother] named', "Susan 's mother"
                            if te and (ind in te_ind): print(f'(11) tow rel & guess 2 !!! {tri2}')
                            flag = 1


                    if not flag and guess:  # if based on obj_lst cannot get triples, we try to look at sub_lst
                        name = []
                        for w_s in sub_lst:
                            if w_s.istitle():
                                name.append(w_s)
                        if len(name) > 1 or len(name) == 0:
                            name = None
                        else:
                            name = name[0]

                        if name and len(set(sub_lst) & set(kinship)) == 1:
                            rel = list(set(sub_lst) & set(kinship))[0]
                            if "'s" or "of" in sub_lst:  # i.e. Guillermina 's father; father of Guillermina
                                sub = name
                                if obj_lst[0].istitle():
                                    obj = obj_lst[0]
                                elif guess: obj = get_closest_name(rel=rel, lst=sub_lst, text=cstory_, direction='next')
                                if obj:
                                    if sub == obj: continue
                                    tri = (sub, rel, obj)
                                    tri_one.append(tri)
                                    if te and (ind in te_ind): print(f'(12) sub & guest 1 !!! {tri}')
                            elif guess:
                                obj = name
                                sub = get_closest_name(rel=rel, lst=sub_lst, text=cstory_, direction='ahead')
                                if sub:
                                    if sub == obj: continue
                                    tri = (sub, rel, obj)
                                    tri_one.append(tri)
                                    if te and (ind in te_ind): print(f'(13) sub & guest 2 !!! {tri}')

                # true value
                target = story_triples[ind - 1]

                # guest triple from story if triples num < triple_num(1.2,1.3_train.csv at least two triples)
                # aaaaa = len(set(tri_one))
                # bbbbb = len(target)
                # if ind == 118:
                #     print('HI')
                if (len(set(tri_one)) < len(target)) and guess:
                    if te and (ind in te_ind): print(
                        f'Triple num less than {len(target)}, now only: {len(list(set(tri_one)))}')
                    for i, word in enumerate(cstory_.replace(',', '').replace('.', '').replace("'", " ").split(' ')):
                        # based on the kinship position to find name around
                        if word in kinship:
                            rel = word
                            sub = get_closest_name(rel=rel, lst=[], text=cstory_, direction='ahead', rel_id=i)
                            obj = get_closest_name(rel=rel, lst=[], text=cstory_, direction='next', rel_id=i, ignore_name=sub)
                            if sub and obj:
                                tri = (sub, rel, obj)
                                tri_one.append(tri)
                                if te and (ind in te_ind): print(f'(14) last guest !!! {tri}')

                # Test result
                if sorted(list(set(tri_one))) != sorted(target):
                    aa = sorted(list(set(tri_one)))
                    bb = sorted(target)
                    if len(aa) < len(bb):
                        if te and (ind in te_ind): print(
                            f'(1) Error term: {sorted(list(set(tri_one)))}; True: {sorted(target)}')
                        err.append(ind)
                    else:
                        cor = 0
                        for a in aa:
                            for b in bb:
                                if sorted([a[0], a[2]]) == sorted([b[0], b[2]]):
                                    if a[1] == b[1]:
                                        cor += 1; break
                                    elif b[1] in ex[a[1]]:
                                        cor += 1; break
                        if cor < len(bb):
                            if te and (ind in te_ind): print(
                                f'(2) Error term: {sorted(list(set(tri_one)))}; True: {sorted(target)}')
                            err.append(ind)

                    # print('Triples careful checking end!')

                if len(set(tri_one)) >= len(target):  # 1.2,1.3_train.csv at least two triples
                    if te and (ind in te_ind): print(f'Total triple num: {len(list(set(tri_one)))}')
                    tri_lst.append(list(set(tri_one)))
                else:
                    if te and (ind in te_ind): print('Failed to get more triple')
                    failed.append(ind)

                if ind % 1000 == 0:
                    print(f'example:{ind}, got:{ind - len(failed)}, accuracy:{1 - len(err) / ind}')

        print(f'Example:{ind}; Got:{ind - len(failed)}; Accuracy:{1 - len(err) / ind}')

    print(f'Failed (index): {failed}')  # failed to get enough triple
    print(f'Error (index): {err}')


if __name__ == '__main__':
    print(' '.join(sys.argv))
    main(sys.argv[1:])

# python main.py --extract corpus/6.2_train.csv > output.txt
# --concat
# --random
# --guess
# --te --te_ind 1 2 3
