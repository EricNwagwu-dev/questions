import nltk
import sys
import os
import string 
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filesDict = dict()
    filesInDire = os.listdir(directory)
    for file in filesInDire:
        with open(os.path.join(directory,file), encoding="utf8") as filePointer:
            filesDict[file] = filePointer.read()
    return filesDict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctFreeDoc = ""
    filteredDocList = []
    punctuation = set(string.punctuation)
    for char in document.lower():
        if not char in punctuation:
            punctFreeDoc += char
    documentAsList = nltk.word_tokenize(punctFreeDoc)
    for word in documentAsList:
        if not word in nltk.corpus.stopwords.words("english"):
            filteredDocList.append(word)
    return filteredDocList


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfDict = dict()
    numOfTotalDocs = len(documents)
    for documentKey in documents:
        wordsInDoc = documents[documentKey]
        seenInDoc = set()
        for word in wordsInDoc:
            if not word in idfDict:
                idfDict[word] = 1
                seenInDoc.add(word)
            elif not word in seenInDoc:
                idfDict[word] += 1
                seenInDoc.add(word)
    for word in idfDict:
        idfDict[word] = math.log(numOfTotalDocs/idfDict[word])
    return idfDict

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfIdfs = dict()
    for documentName in files:
        tfIdfs[documentName] = 0
        for qWord in query:
            tf = 0
            for word in files[documentName]:
                if word == qWord:
                    tf += 1
            tfIdfs[documentName] += tf*idfs[qWord]
    topFiles = [tup[0] for tup in sorted(tfIdfs.items(),key=sortBy, reverse = 1)]
    return topFiles[:n]

def sortBy(t):
    return t[1]

def sortBySent(t):
    return t[1][0]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentenceIdfs = dict()
    for sentenceKey in sentences:
        sentenceIdfs[sentenceKey] = [0,0]
        numMatchQuery = 0
        for word in query:
            if word in sentences[sentenceKey]:
                sentenceIdfs[sentenceKey][0] += idfs[word]
                numMatchQuery += 1
        sentenceIdfs[sentenceKey][1] = numMatchQuery/len(sentences[sentenceKey])
    topSentences = sorted(sentenceIdfs.items(),key=lambda sent: (sent[1][0], sent[1][1]), reverse = 1)
    return [result[0] for result in topSentences[:n]]

if __name__ == "__main__":
    main()
