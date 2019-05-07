# active-learning-classification
Active Learning is an emerging field of machine leraning technique which leverages oracle (human support) to learn the trends/patterns in the data faster and more intelligently than a standalone machine learning alogorithm.
This project aims at demonstrating the power of Active Learning as compared to other Passive Machine Learning Algorithms used of classification of any sort.

## Dataset
I have used the BBC dataset which contains 2225 documents from the BBC news website corresponding to stories in five topical areas in 2004-2005.
Natural Classes: 5 (business, entertainment, politics, sports, tech)

    If you make use of the dataset, please consider citing the publication: 
    - D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

    All rights, including copyright, in the content of the original articles are owned by the BBC.

    Contact Derek Greene <derek.greene@ucd.ie> for further information.
    http://mlg.ucd.ie/datasets/bbc.html

## Steps

1. Extracting Keywords (Preprocessing)
    The model extracts the keywords from the existing documents which are further processed and stored in a global dictionary (knowledge base of the model) with assigned label as news document class.
    We discard the words from the dictionary which are ambiguous, i.e. are found in multiple documents, hence they won't play a role in clasiification of documents.
    The keyword extraction is done using TextRank algorithm, which represents each word as a node in a graph and is connected to words in its vicinity. Then we use well-known PageRank algorithm to rank the words based on their importance.
    We then consider only Nouns and Adjectives as possible candidates for training model.
    Some other steps taken for preprocessing:
        a) Convert keywords to lowercase
        b) Split the words by space if any
        c) Remove special characters like 's etc.

2. Global Dictionary
    The global dicctionary contains keywords as keys and their corresponding values as their label.
    
    Labels:
    0 : business
    1 : entertainment
    2 : politics
    3 : sports
    4 : tech
    -1: None (ambiguous case, where a keywords is found in multiple documents belonging different new categories)
    
        Example:
        Doc1: {“baseball”, “winner”, “rate”, “progress”, “award”} ɛ SPORTS NEWS
        Doc2: {“rate”, “progress”, “economy”, “stock”, “market”} ɛ BUSINESS NEWS
    
        Dictionary: { ( “baseball” , 3 ) , ( “winner” , 3 ) , ( “rate” , -1 ) , ( “progress” , -1 ) , ( “award” , 3 ) , ( “economy” , 0 ) , ( “ stock” , 0 ) , ( “market” , 0 ) }
        Where 3 means that the keyword belongs to the SPORTS NEWS ARTICLE, and 0 means that the keyword belongs to the BUSINESS NEWS ARTICLE, and -1 means that the keyword is found in both kind of ARTICLES and hence we should not use it for classifying any article having based on this keyword.
    
3. Learning
    The algorithm populates the global dictionary and classifies the documents based on the score received for the document corresponding to each class.
    We use maximum matching criteria to decide the classification of unlabelled documents in the training set.
    We find the probability of document belonging to each category of news article.
    
        prob( doc, category_i ) = score(doc, category_i ) /  for x in all categories ∑ score(doc, x)
    
    Check the category for which document is similar with maximum probability. 

        d(doc, prob) =     ASSIGN CATEGORY if prob >= 0.65 esle ASK ORACLE FOR LABEL

## Conclusion
    We observe that the accuracy of classification is quite appreciable when we use Active Learning to when compared to other techniques.

## References
    1. TextRank: Bringing Order into Texts, Rada Mihalcea and Paul Tarau, Department of Computer Science, University of North Texas
    2. http://mlg.ucd.ie/datasets/bbc.html
    3. http://www.nltk.org/book/
