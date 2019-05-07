import errno
import itertools
import os
import random
from shutil import copyfile
from shutil import rmtree

import networkx as nx
import nltk
from sklearn.model_selection import train_test_split

from constants import *


# apply syntactic filters based on POS tags
def filter_for_tags(tagged, tags=None):
    if tags is None:
        tags = ['NN', 'JJ', 'NNP']
    return [item for item in tagged if item[1] in tags]


def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]


def levenshtein_distance(first_string, second_string):
    """Function to find the Levenshtein distance between two words/sentences"""
    if len(first_string) > len(second_string):
        temp = first_string
        first_string = second_string
        second_string = temp
    distances = range(len(first_string) + 1)
    for index2, char2 in enumerate(second_string):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first_string):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]


def build_graph(nodes):
    graph = nx.Graph()  # initialize an undirected graph
    graph.add_nodes_from(nodes)
    node_pairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in node_pairs:
        first_element = pair[0]
        second_element = pair[1]
        lev_distance = levenshtein_distance(first_element, second_element)
        graph.add_edge(first_element, second_element, weight=lev_distance)

    return graph


def extract_key_phrases(doc_text):
    # tokenize the text using nltk
    word_tokens = nltk.word_tokenize(doc_text)

    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    text_list = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    words_list = list(set([x[0] for x in tagged]))

    graph = build_graph(words_list)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    key_phrases = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    # the number of key phrases returned will be relative to the size of the text (a third of the number of vertices)
    a_third = int(len(words_list) / 3)
    key_phrases = key_phrases[0:a_third + 1]

    # take key phrases with multiple words into consideration as done in the paper -
    #   if two words are adjacent in the text and are selected as keywords, join them together
    modified_key_phrases = set([])
    dealt_with = set([])  # keeps track of individual keywords that have been joined to form a key phrase
    x = 0
    y = 1
    while y < len(text_list):
        first_word = text_list[x]
        second_word = text_list[y]
        if first_word in key_phrases and second_word in key_phrases:
            key_phrase = first_word + ' ' + second_word
            modified_key_phrases.add(key_phrase)
            dealt_with.add(first_word)
            dealt_with.add(second_word)
        else:
            if first_word in key_phrases and first_word not in dealt_with:
                modified_key_phrases.add(first_word)

            # if this is the last word in the text, and it is a keyword,
            # it definitely has no chance of being a key phrase at this point
            if y == len(text_list) - 1 and second_word in key_phrases and second_word not in dealt_with:
                modified_key_phrases.add(second_word)

        x = x + 1
        y = y + 1

    return list(modified_key_phrases)


def get_label(document):
    global label_input
    invalid_input = True
    while invalid_input:
        print("************************")
        for label in LABEL_REV_DICT.keys():
            print("-> For " + label.upper() + ", Enter " + str(LABEL_REV_DICT[label]))
        print("************************")
        label_input = int(input("Enter label for " + document + " : "))
        if label_input != NO_LABEL and label_input in LABEL_REV_DICT.values():
            print("Added " + document + " as a " + LABEL_DICT[label_input] + " document")
            invalid_input = False
        else:
            print("Invalid Label entered : " + str(label_input))
    return label_input


def to_lower(list1):
    count = len(list1)
    for x in range(0, count):
        list1[x] = list1[x].lower()
    return list1


# Recreates new list by splitting the double word keywords that were treated as single word.
def split(list1):
    new_list = []
    count = len(list1)
    for x in range(0, count):
        temp = list1[x].split(' ')
        temp_count = len(temp)
        for y in range(0, temp_count):
            new_list.append(temp[y])
    return new_list


# Remove special symbols from the keywords
def remove_symbols(list1):
    count = len(list1)
    for x in range(0, count):
        list1[x] = list1[x].replace("-", "")
    for x in range(0, count):
        temp_length = len(list1[x])
        if list1[x][temp_length - 2] == '\'' and list1[x][temp_length - 1] == 's':
            list1[x] = list1[x][:-2]
    return list1


def process(words):
    words = to_lower(words)
    words = split(words)
    words = remove_symbols(words)
    return words


def add_to_knowledge(dic, key_words, doc_label):
    for x in key_words:
        # check if the value is already in the list. If its label changes, mark the
        # label as -1 indicating that it will no longer be used for classification.
        if x in dic:
            if dic[x] != doc_label:
                dic[x] = NO_LABEL
        else:
            dic[x] = doc_label
    return


def prepare_fresh_train_test_data(affirmative):
    if affirmative:
        print("Creating fresh Train/Test data")
        delete_directory_tree(TRAIN_DATA_DIR)
        delete_directory_tree(TEST_DATA_DIR)
        for label_value in LABEL_DIR_NAMES:
            create_directories(os.path.join(LABELLED_TRAIN_DATA_DIR, label_value))
            create_directories(os.path.join(UNLABELLED_TRAIN_DATA_DIR))
            create_directories(os.path.join(TEST_DATA_DIR))

            total_label_articles = os.listdir(os.path.join(BBC_DATA_DIR, label_value))
            total_articles = len(total_label_articles)
            total_label_values = [label_value] * total_articles
            x = train_test_split(total_label_articles,
                                 total_label_values,
                                 train_size=TRAIN_LABELLED_DOCUMENTS_FRACTION)
            train_articles = x[0]
            test_articles = x[1]

            for i in range(len(train_articles)):
                copyfile(os.path.join(BBC_DATA_DIR, label_value, train_articles[i]),
                         os.path.join(LABELLED_TRAIN_DATA_DIR, label_value, train_articles[i]))

            for i in range(len(test_articles)):
                copyfile(os.path.join(BBC_DATA_DIR, label_value, test_articles[i]),
                         os.path.join(TEST_DATA_DIR, label_value + "_" + test_articles[i]))

            unlabelled_train_articles = random.sample(total_label_articles,
                                                      int(TRAIN_UNLABELLED_DOCUMENTS_FRACTION * total_articles))

            for i in range(len(unlabelled_train_articles)):
                copyfile(os.path.join(BBC_DATA_DIR, label_value, unlabelled_train_articles[i]),
                         os.path.join(UNLABELLED_TRAIN_DATA_DIR, label_value + "_" + unlabelled_train_articles[i]))
        else:
            print("Not creating fresh Train/Test data")
    return


def create_intermediate_directories(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return


def create_directories(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return


def delete_directory_tree(directory_path):
    if os.path.exists(directory_path):
        try:
            rmtree(directory_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return
