import io

from news_util import *


# Get probability that it belongs to new category with given label
def get_prob(key_words_list, label_value_for_doc):
    keywords_not_in_label = 0
    keywords_in_label = 0
    probability = DEFAULT_PROBABILITY
    for word in key_words_list:
        if word in knowledge_base:
            if knowledge_base.get(word) == label_value_for_doc:
                keywords_in_label += 1
            else:
                keywords_not_in_label += 1

    # if less matches are found, get the article's label and add that to knowledge base
    if keywords_in_label != 0 or keywords_not_in_label != 0:
        probability = keywords_in_label * 1.0 / (keywords_not_in_label + keywords_in_label)
    return probability


def predict_label(key_words_list):
    max_probability = 0.0
    resulting_label = NO_LABEL
    for current_label in LABEL_DICT.keys():
        if current_label != NO_LABEL:
            current_label_probability = get_prob(key_words_list, current_label)
            if current_label_probability > max_probability:
                resulting_label = current_label
                max_probability = current_label_probability
    return resulting_label, max_probability


# ----------------------------
# PREPARE TRAIN and TEST DATA
# ----------------------------
prepare_fresh_train_test_data(CREATE_FRESH_DATA)

# ----------------------------
#     CODE STARTS HERE
# ----------------------------

# initialize global variables
knowledge_base = {}
training_labels = {}
test_labels = {}
label_human_input_dict = {}

# Process Labelled Training Data
print("********************************")
print("Training Using Labelled Data")
print("********************************")
for label_from_training_data in LABEL_DIR_NAMES:
    print('-> Loading docs from ' + label_from_training_data.upper() + ' category (label = ' + str(
        LABEL_REV_DICT[label_from_training_data]) + ')')
    articles = os.listdir(os.path.join(LABELLED_TRAIN_DATA_DIR, label_from_training_data))
    for article in articles:
        print('Reading ' + article)
        article_path = os.path.join(LABELLED_TRAIN_DATA_DIR, label_from_training_data, article)
        article_file = io.open(article_path, 'r', encoding=TEXT_FILE_ENCODING)
        text = article_file.read().strip()
        key_words = extract_key_phrases(text)

        key_words = process(key_words)
        # removes duplicate words from list
        key_words = list(set(key_words))
        add_to_knowledge(knowledge_base, key_words, LABEL_REV_DICT[label_from_training_data])

# Process Unlabelled Training Data
print("********************************")
print("Training Using Labelled Data")
print("********************************")
iteration = 0
while iteration < MAX_ITERATIONS:
    iteration += 1
    human_help_taken = 0
    print("************************")
    print("ITERATION : " + str(iteration) + "/" + str(MAX_ITERATIONS))
    articles = os.listdir(UNLABELLED_TRAIN_DATA_DIR)
    for article in articles:
        article_path = os.path.join(UNLABELLED_TRAIN_DATA_DIR, article)
        article_file = io.open(article_path, 'r', encoding=TEXT_FILE_ENCODING)
        text = article_file.read().strip()
        key_words = extract_key_phrases(text)
        key_words = process(key_words)
        label, prob = predict_label(key_words)
        print('Article Read : ' + article + ", Predicted Label : " + LABEL_DICT[label].upper() +
              " with Probability : " + str(prob))

        # probability threshold for classification
        if prob < PROBABILITY_THRESHOLD:
            label_input_received = NO_LABEL
            human_help_taken += 1
            # Don't ask Human if he has already provided label for a particular article
            if article_path in label_human_input_dict.keys():
                label_input_received = label_human_input_dict[article_path]
                print("HUMAN HELP (from cache) : " + article + " -> " + LABEL_DICT[label_input_received].upper())
            else:
                if TAKE_HUMAN_HELP:
                    label_input_received = get_label(article_path)
                else:
                    label_input_received = LABEL_REV_DICT[article.split("_", 1)[0]]
                    print("HUMAN HELP SIMULATED : " + article + " -> " + LABEL_DICT[label_input_received].upper())
                label_human_input_dict[article_path] = label_input_received
            add_to_knowledge(knowledge_base, key_words, label_input_received)
            training_labels[article_path] = label_input_received
        else:
            training_labels[article_path] = label
    print("Iteration " + str(iteration) + "/" + str(MAX_ITERATIONS) + " completed.")

# Process Test Data
print("********************************")
print("Predicting Test Data")
print("********************************")
articles = os.listdir(TEST_DATA_DIR)
total_test_articles = len(articles)
true_positives = 0
for article in articles:
    article_path = os.path.join(TEST_DATA_DIR, article)
    article_file = io.open(article_path, 'r', encoding=TEXT_FILE_ENCODING)
    text = article_file.read().strip()
    key_words = extract_key_phrases(text)
    key_words = process(key_words)
    label, prob = predict_label(key_words)
    print('Article Read : ' + article + ", Predicted Label : " + LABEL_DICT[label].upper() +
          " with Probability : " + str(prob))

    test_labels[article] = label
    actual_label = LABEL_REV_DICT[article.split("_", 1)[0]]
    if actual_label == label:
        true_positives += 1

print("********************************")
print("Total Test Documents : " + str(total_test_articles))
print("True Positives : " + str(true_positives))
print("Algorithm Accuracy : " + str(100.0 * true_positives / total_test_articles))
print("********************************")
print("Test document labels : \n" + str(test_labels))
