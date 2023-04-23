from getVocabList import get_vocab_list
import re
from nltk.stem import PorterStemmer
import numpy as np


def process_email(email_contents):
    # process_email preprocesses the body of an email and
    # returns a list of word_indices
    #   word_indices = process_email(email_contents) preprocesses
    #   the body of an email and returns a list of indices of the
    #   words contained in the email.

    # Load Vocabulary
    vocab_list = get_vocab_list()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    #
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    #
    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://\S*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('\S+@\S+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')

    # Process file
    l = 0

    # create a stemmer object
    stemmer = PorterStemmer()

    # Tokenize email_contents and remove any punctuation
    punctuation = r'[@$/#.-:&*+=\[\]?!(){},\'\">_<;%\r]'
    tokens = re.sub(punctuation, '', email_contents).replace('\n', ' ').strip().split()

    for token in tokens:
        # Remove any non-alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)

        # Stem the word using PorterStemmer
        try:
            token = stemmer.stem(token.strip())
        except:
            token = ''
            continue

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(token) + 1) > 78:
            print('\n', end='')
            l = 0

        print(token, end=' ')
        l = l + len(token) + 1

        # Skip the word if it is too short
        if len(token) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices
        #               vector (e.g., word_indices = [word_indices ; 18]; ).
        #
        # Note: vocabList{idx} returns the word with index idx in the
        #       vocabulary list.
        #
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.

        try:
            index = vocab_list.index(token)
        except:
            continue
        word_indices.append(index + 1)

        # =============================================================

    print("\n\n=========================\n")

    return np.array(word_indices)
