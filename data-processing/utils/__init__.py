"""
Some general utils.

"""

#=============================================================================
# Standard library imports
#=============================================================================
import string 
import re
import os
import errno
import hashlib

#================================ End Imports ================================

def deletechars(s, exclude_chars):
    ''' Fast deletion of characters from string.
    It uses a dummy translation table, and so no mapping is applied, and we
    just delete the exclude_chars characters.
    '''
    phony_translate_table = string.maketrans("","")
    return s.translate(phony_translate_table, exclude_chars)


def deletepunctuation(s):
    ''' Fast deletion of punctuation from string'''
    return deletechars(s,string.punctuation)


def tokenize(text, foldcase=True):
    ''' 
    A very cheap and easy tokenization.
    First, remove "'s". For example, "dog's" becomes "dog".
    Second, zap utf-8 chars.
    Then, remove all punctuation and, by default, fold upper and lower case words
    and then split by whitespace.
    '''

    text = re.sub(r'\'s','', text)
    s = ''.join([s for s in text if s in string.printable])

    s = str(s) # Got to convert it to str.
    s = deletepunctuation(s)

    if foldcase:
        s = s.lower()
    return s.split()


def mkdir_p(path):
    '''
    Make a directory, making parents if necessary.
    Taken verbatim from
    http://stackoverflow.com/a/600612
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def checksum(argument, algorithm='sha256'):
    '''
    Returns the hash checksum of `argument'.
    If `argument' is a name of a file, then perform the checksum on the file.
    Otherwise, the checksum is of the string `argument'.
    By default, it will be the sha1 checksum (and so equivalent to linux's
    sha1sum). Alternatively, the algorithm could be md5 (equivalent to linux's
    md5sum), or else sha224, sha256, sha384, sha512.
    '''

    h = hashlib.new(algorithm)

    if os.path.exists(argument) and os.path.isfile(argument):
        argument = open(argument,'rb').read()

    h.update(argument)

    return h.hexdigest()
