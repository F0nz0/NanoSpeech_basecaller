# importing needed modules

import os, pysam, sys
import tensorflow as tf
tf.autograph.set_verbosity(2)
from joblib import load
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from ont_fast5_api.fast5_interface import get_fast5_file
from math import ceil

############
# Code taken, adapted and modified from: 
# https://github.com/keras-team/keras-io/blob/master/examples/audio/transformer_asr.py
# Copyright 2020. The Keras authors and Apoorv Nandan
# Released under Apache 2.0 License.
#############

# some needed functions
def generate_chunks(pA_data, chunks_len, shift=None):
    length = pA_data.shape[0]
    if shift == None:
        shift=chunks_len
    n_chunks = ceil(length/chunks_len)
    start = 0
    for n,w in enumerate(range(n_chunks)):
        chunk = pA_data[start:start+chunks_len]
        start = start+chunks_len
        if chunk.shape[0] == chunks_len:
            if n == 0:
                X = chunk
            else:
                X = np.vstack((X,chunk))
    return X


def generator_consumer(X, pad_len = 971):
    for c,audio in enumerate(X):
        try:
            audio_ds = pA_to_audio(audio, pad_len)
        except Exception as e:
            print(f"[generator_consumer message] EXCEPTION during conversion of audio via FT at chunk n° {c+1}/{X.shape[0]} (it will be skipped):", len(audio), file=sys.stderr, flush=True)
            print(f"[generator_consumer message] Exception --> {e}", file=sys.stderr, flush=True)
            continue
        yield audio_ds



def pA_to_audio(pA_chunk, pad_len = 971):
    # eliminate nan before stft operation
    pA_chunk = pA_chunk[~tf.math.is_nan(pA_chunk)]
    # spectrogram using stft starting from chunks of pA currents measurements
    stfts = tf.signal.stft(pA_chunk, frame_length=150, frame_step=5, fft_length=250)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to a fixed length
    #pad_len = 971
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(X):
    audio_ds = tf.data.Dataset.from_tensor_slices(X)
    audio_ds = audio_ds.map(
        pA_to_audio, num_parallel_calls=tf.data.AUTOTUNE
    )
    return audio_ds


def create_text_ds(y, vectorizer):
    texts = [_ for _ in y]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def create_tf_dataset(X, y, vectorizer, bs=4):
    audio_ds = create_audio_ds(X)
    text_ds = create_text_ds(y, vectorizer)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_tf_dataset_basecaller(X, bs=4):
    audio_ds = create_audio_ds(X)
    ds = audio_ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# defining useful functions to handle with fast5 files
def raw_to_pA(f5):
    '''
    Function to transform back from raw signal to pA scale.
    '''
    try:
        raw_unit = f5.get_channel_info()["range"] / f5.get_channel_info()["digitisation"]
        offset = f5.get_channel_info()["offset"]
        pA_signal = (f5.get_raw_data() + offset) * raw_unit
        return pA_signal
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e, flush=True)


def retrieve_read_pA_fastq_from_fast5(fast5_fullpath, read_name_id):
    '''
    Retrieve fastq and pA converted data related to a given readname_id from a fast5 file 
    '''
    with get_fast5_file(fast5_fullpath) as f5:
        r = f5.get_read(read_name_id)
        if r.read_id == read_name_id:
            read_name = r.read_id
            print(f"One putative read example found with id {read_name} in fast5 file: {fast5_fullpath}", flush=True)
            pA_data = raw_to_pA(r)
            fastq = r.get_analysis_dataset("Basecall_1D_000/BaseCalled_template", "Fastq")
    return pA_data, fastq


def generator(pA_data, chunks_len=3500, shift=None):
    '''
    A function to iterate over a pA converted data retrieved from a fasta file
    '''
    if shift == None:
        shift=chunks_len
    X = []
    dataset =  tf.data.Dataset.from_tensor_slices(pA_data)
    for w in dataset.window(chunks_len, shift=shift, drop_remainder=False):
        chunk = list(w.as_numpy_iterator())
        if len(chunk) == chunks_len:
            X.append(chunk)
        else:
            chunk_padding = chunks_len - len(chunk)
            chunk = chunk + [np.nan] * chunk_padding
            X.append(chunk)
    return np.array(X)


# convert Inosine to Adenosine and save idexes
def convert_ItoA(sequence):
    '''
    A function which take in input a nucleotide sequence with Inosines and convert
    the sequence in only giving back the indices of Inosines (modified Adenosines).
    '''
    Is_idx = []  # a list of the indices (0-based) of the modified Adenosines (Inosines)
    As_count = 0 # count of modified and unmodified Adenosines
    seq_conv = ""
    for i,b in enumerate(sequence):
        if b == "A":
            As_count += 1
            seq_conv += b
        elif b == "I":
            Is_idx.append(i)
            As_count += 1
            seq_conv += "A"
        else:
            seq_conv += b
    return seq_conv, Is_idx

# convert Modified nucleotide to Canonical version and save idexes
def convert_MODtoCAN(sequence, mods_dict):
    '''
    A function which take in input a nucleotide sequence with Inosines and convert
    the sequence in only giving back the indices of Inosines (modified Adenosines).
    '''
    seq_conv = ""
    ModS_idxs = {}
    # inizialize ModS_idxs
    for k in mods_dict.keys():
        ModS_idxs[k] = []
    for i,b in enumerate(sequence):
        if b in mods_dict.keys():
            seq_conv += mods_dict[b]
            ModS_idxs[b] += [i]
        else:
            seq_conv += b
    return seq_conv, ModS_idxs

def phred_score_to_symbol(phred_score):
    '''
    A function to convert the phred score to ascii symbol or vice versa.
    '''
    q_table = {   
        '!': 0,
        '"': 1,
        '#': 2,
        '$': 3,
        '%': 4,
        '&': 5,
        "'": 6,
        '(': 7,
        ')': 8,
        '*': 9,
        '+': 10,
        ',': 11,
        '-': 12,
        '.': 13,
        '/': 14,
        '0': 15,
        '1': 16,
        '2': 17,
        '3': 18,
        '4': 19,
        '5': 20,
        '6': 21,
        '7': 22,
        '8': 23,
        '9': 24,
        ':': 25,
        ';': 26,
        '<': 27,
        '=': 28,
        '>': 29,
        '?': 30,
        '@': 31,
        'A': 32,
        'B': 33,
        'C': 34,
        'D': 35,
        'E': 36,
        'F': 37,
        'G': 38,
        'H': 39,
        'I': 40,
        'J': 41,
        'K': 42,
        'L': 43,
        'M': 44,
        'N': 45,
        'O': 46,
        'P': 47,
        'Q': 48,
        'R': 49,
        'S': 50,
        'T': 51,
        'U': 52,
        'V': 53,
        'W': 54,
        'X': 55,
        'Y': 56,
        'Z': 57,
        '[': 58,
        '\\': 59,
        ']': 60,
        '^': 61,
        '_': 62,
        '`': 63,
        'a': 64,
        'b': 65,
        'c': 66,
        'd': 67,
        'e': 68,
        'f': 69,
        'g': 70,
        'h': 71,
        'i': 72,
        'j': 73,
        'k': 74,
        'l': 75,
        'm': 76,
        'n': 77,
        'o': 78,
        'p': 79,
        'q': 80,
        'r': 81,
        's': 82,
        't': 83,
        'u': 84,
        'v': 85,
        'w': 86,
        'x': 87,
        'y': 88,
        'z': 89,
        '{': 90,
        '|': 91,
        '}': 92,
        '~': 93,
        0: '!',
        1: '"',
        2: '#',
        3: '$',
        4: '%',
        5: '&',
        6: "'",
        7: '(',
        8: ')',
        9: '*',
        10: '+',
        11: ',',
        12: '-',
        13: '.',
        14: '/',
        15: '0',
        16: '1',
        17: '2',
        18: '3',
        19: '4',
        20: '5',
        21: '6',
        22: '7',
        23: '8',
        24: '9',
        25: ':',
        26: ';',
        27: '<',
        28: '=',
        29: '>',
        30: '?',
        31: '@',
        32: 'A',
        33: 'B',
        34: 'C',
        35: 'D',
        36: 'E',
        37: 'F',
        38: 'G',
        39: 'H',
        40: 'I',
        41: 'J',
        42: 'K',
        43: 'L',
        44: 'M',
        45: 'N',
        46: 'O',
        47: 'P',
        48: 'Q',
        49: 'R',
        50: 'S',
        51: 'T',
        52: 'U',
        53: 'V',
        54: 'W',
        55: 'X',
        56: 'Y',
        57: 'Z',
        58: '[',
        59: '\\',
        60: ']',
        61: '^',
        62: '_',
        63: '`',
        64: 'a',
        65: 'b',
        66: 'c',
        67: 'd',
        68: 'e',
        69: 'f',
        70: 'g',
        71: 'h',
        72: 'i',
        73: 'j',
        74: 'k',
        75: 'l',
        76: 'm',
        77: 'n',
        78: 'o',
        79: 'p',
        80: 'q',
        81: 'r',
        82: 's',
        83: 't',
        84: 'u',
        85: 'v',
        86: 'w',
        87: 'x',
        88: 'y',
        89: 'z',
        90: '{',
        91: '|',
        92: '}',
        93: '~'
        }
    if type(phred_score) == int:
        if phred_score > 93:
            phred_score = 93
    return q_table[phred_score]