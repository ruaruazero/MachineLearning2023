import numpy as np
from processEmail import process_email
from emailFeatures import email_features
from scipy.io import loadmat
from sklearn.svm import SVC
from getVocabList import get_vocab_list


def main():
    # ==================== Part 1: Email Preprocessing ====================
    #  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    #  to convert each email into a vector of features. In this part, you will
    #  implement the preprocessing steps for each email. You should
    #  complete the code in processEmail.m to produce a word indices vector
    #  for a given email.

    print('Preprocessing sample email (emailSample1.txt)\n')

    # Extract Features
    file_contents = open('emailSample1.txt').read()
    word_indices = process_email(file_contents)

    # Print Stats
    print('Word indices: \n')
    for i in range(len(word_indices)):
        if (i + 1) % 5 == 0:
            print('\n', end='')
        print(word_indices[i], end=' ')
    print('\n')

    input("Program paused. Press enter to continue.")

    # ==================== Part 2: Feature Extraction ====================
    # Now, you will convert each email into a vector of features in R^n.
    # You should complete the code in emailFeatures.m to produce a feature
    # vector for a given email.

    print("nExtracting features from sample email (emailSample1.txt)\n")

    # Extract Features
    file_contents = open('emailSample1.txt').read()
    word_indices = process_email(file_contents)
    features = email_features(word_indices)

    # Print Stats
    print('Length of feature vector: {}'.format(len(features)))
    print('Number of non-zero entries: {}'.format(np.sum(features > 0)))

    input("Program paused. Press enter to continue.")

    # =========== Part 3: Train Linear SVM for Spam Classification ========
    # In this section, you will train a linear classifier to determine if an
    # email is Spam or Not-Spam.

    # Load the Spam Email dataset
    # You will have X, y in your environment
    data = loadmat("spamTrain.mat")
    X = data["X"]
    y = data["y"]

    print('\nTraining Linear SVM (Spam Classification)')
    print('(this may take 1 to 2 minutes) ...')

    C = 0.03
    model = SVC(C=C, kernel='linear')
    model.fit(X, y.flatten())

    print('Training Accuracy: %.3f' % model.score(X, y.flatten()))

    # =================== Part 4: Test Spam Classification ================
    # After training the classifier, we can evaluate it on a test set. We have
    # included a test set in spamTest.mat

    # load the test dataset
    data = loadmat('spamTest.mat')
    Xtest = data['Xtest']
    ytest = data['ytest']

    print('\nEvaluating the trained Linear SVM on a test set ...')

    print('Training Accuracy: %.3f' % model.score(Xtest, ytest.flatten()))

    input("Program paused. Press enter to continue.")

    # ================= Part 5: Top Predictors of Spam ====================
    # Since the model we are training is a linear SVM, we can inspect the
    # weights learned by the model to understand better how it is determining
    # whether an email is spam or not. The following code finds the words with
    # the highest weights in the classifier. Informally, the classifier
    # 'thinks' that these words are the most likely indicators of spam.

    # Sort the weights and obtain the vocabulary list
    weights = model.coef_.flatten()
    idx = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)
    vocab_list = get_vocab_list()

    print('\nTop predictors of spam: ')
    for i in range(15):
        print(f' {vocab_list[idx[i]]:15} ({weights[i]})')

    print('\n')

    input("Program paused. Press enter to continue.")

    # =================== Part 6: Try Your Own Emails =====================
    # Now that you've trained the spam classifier, you can use it on your own
    # emails! In the starter code, we have included spamSample1.txt,
    # spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
    # The following code reads in one of these emails and then uses your
    # learned SVM classifier to determine whether the email is Spam or
    # Not Spam
    #
    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!

    filename = 'spamSample1.txt'

    # Read and predict
    file_contents = open(filename).read()
    word_indices = process_email(file_contents)
    x = email_features(word_indices)
    p = model.predict(x)

    print(f'\nProcessed {filename}\n\nSpam Classification: {p}')
    print('(1 indicates spam, 0 indicates not spam)\n')


if __name__ == '__main__':
    main()
