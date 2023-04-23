def get_vocab_list():
    # get_vocab_list reads the fixed vocabulary list in vocab.txt and returns a
    # cell array of the words
    #    vocabList = get_vocab_list() reads the fixed vocabulary list in vocab.txt
    #    and returns a cell array of the words in vocabList.

    # Read the fixed vocabulary list
    with open("vocab.txt", 'r') as f:

        # Store all dictionary words in cell array vocab{}
        vocab_list = []
        for line in f:
            word = line.strip().split()[1]
            vocab_list.append(word)

        return vocab_list


if __name__ == '__main__':
    a = get_vocab_list()
    print(a)
