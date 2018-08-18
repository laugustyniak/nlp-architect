# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from nlp_architect.utils.attr_dict import AttrDict
from nlp_architect.utils.embedding_augmentation import most_similar, filter_words_to_augment
from nlp_architect.utils.text import Vocabulary


class SequentialTaggingDataset(object):
    """
    Sequential tagging dataset loader.
    Loads train/test files with tabular separation.

    Args:
        train_file (str): path to train file
        test_file (str): path to test file
        max_sentence_length (int, optional): max sentence length
        max_word_length (int, optional): max word length
        tag_field_no (int, optional): index of column to use a y-samples
        augment_data (bool, optional): do we want to augment data with word embeddings similarity
    """

    def __init__(
            self,
            train_file,
            test_file,
            max_sentence_length=30,
            max_word_length=20,
            tag_field_no=4,
            augment_data=False,
            similarity_threshold=0.8,
    ):
        self.similarity_threshold = similarity_threshold
        self.augment_data = augment_data
        self.files = {
            'train': train_file,
            'test': test_file
        }
        self.max_sent_len = max_sentence_length
        self.max_word_len = max_word_length
        self.tf = tag_field_no

        self.vocabs = {
            'token': Vocabulary(2),  # 0=pad, 1=unk
            'char': Vocabulary(2),  # 0=pad, 1=unk
            'tag': Vocabulary(1)  # 0=pad
        }

        self.data = {}
        self.data_augmentation = AttrDict

        train_file_name = Path(train_file).stem.split('-')[0]
        features_file_name = (
                train_file_name +
                '-similarity-' + str(similarity_threshold) +
                '-max-sent-len-' + str(max_sentence_length) +
                '-max-word-len-' + str(max_word_length)
        )
        features_path = Path(train_file).parent / 'features' / features_file_name

        for f in self.files:

            raw_sentences = self._read_file(self.files[f])
            self.word_vecs = []
            self.char_vecs = []
            self.tag_vecs = []

            if features_path.exists() and self.augment_data and f == 'train':
                features_path.parent.mkdir(exist_ok=True)
                with open(features_path.as_posix(), 'rb') as features_file:
                    self.word_vecs, self.char_vecs, self.tag_vecs, self.data[f] = pickle.load(features_file)
                    print(features_path.as_posix() + ' has bee loaded')
                    continue

            for tokens_original, tags in tqdm(raw_sentences, desc='Sentences for ' + train_file + ' -> ' + f):
                if self.augment_data and f == 'train':
                    tokens = tokens_original.copy()
                    self.data_augmentation.words = defaultdict(list)
                    self.data_augmentation.sentences = defaultdict(list)
                    n_tokens = len(tokens)
                    for token, augmentation, token_idx in filter_words_to_augment(tokens):
                        if token_idx < n_tokens:
                            for token_augmentation in most_similar(word=token, threshold=self.similarity_threshold):
                                self.data_augmentation.words[tokens[token_idx]].append(token_augmentation.text)
                                self.data_augmentation.sentences[
                                    tokens[token_idx] + '->' + token_augmentation.text].append(' '.join(tokens))
                                tokens[token_idx] = token_augmentation.text
                                self._featurize_data(tokens, tags)
                else:
                    self._featurize_data(tokens_original, tags)

            self.word_vecs = pad_sequences(self.word_vecs, maxlen=self.max_sent_len)
            self.char_vecs = np.asarray(self.char_vecs)
            self.tag_vecs = pad_sequences(self.tag_vecs, maxlen=self.max_sent_len)
            self.data[f] = self.word_vecs, self.char_vecs, self.tag_vecs

            if self.augment_data and f == 'train':
                with open(features_path.as_posix(), 'wb') as features_file:
                    pickle.dump((self.word_vecs, self.char_vecs, self.tag_vecs, self.data['train']), features_file)
                    print(features_path.as_posix() + ' has bee pickled!')

    def _featurize_data(self, tokens, tags):
        self.word_vecs.append(np.array([self.vocabs['token'].add(t) for t in tokens]))
        word_chars = []
        for token in tokens:
            word_chars.append(np.array([self.vocabs['char'].add(char) for char in token]))
        word_chars = pad_sequences(word_chars, maxlen=self.max_word_len)
        if self.max_sent_len > len(tokens):
            char_padding = self.max_sent_len - len(word_chars)
            self.char_vecs.append(np.concatenate((np.zeros((char_padding, self.max_word_len)), word_chars), axis=0))
        else:
            self.char_vecs.append(word_chars[-self.max_sent_len:])
        self.tag_vecs.append(np.array([self.vocabs['tag'].add(t) for t in tags]))

    @property
    def y_labels(self):
        """return y labels"""
        return self.vocabs['tag'].vocab

    @property
    def word_vocab(self):
        """words vocabulary"""
        return self.vocabs['token'].vocab

    @property
    def char_vocab(self):
        """characters vocabulary"""
        return self.vocabs['char'].vocab

    @property
    def word_vocab_size(self):
        """word vocabulary size"""
        return len(self.vocabs['token']) + 2

    @property
    def char_vocab_size(self):
        """character vocabulary size"""
        return len(self.vocabs['char']) + 2

    @property
    def train(self):
        """Get the train set"""
        return self.data['train']

    @property
    def test(self):
        """Get the test set"""
        return self.data['test']

    def _read_file(self, path):
        with open(path, encoding='utf-8') as fp:
            data = fp.readlines()
            data = [d.strip() for d in data]
            data = [d for d in data if 'DOCSTART' not in d]
            sentences = self._split_into_sentences(data)
            parsed_sentences = [self._parse_sentence(s) for s in sentences if len(s) > 0]
        return parsed_sentences

    def _parse_sentence(self, sentence):
        tokens = []
        tags = []
        for line in sentence:
            fields = line.split()
            assert len(fields) >= self.tf, 'tag field exceeds number of fields'
            if 'CD' in fields[1]:
                tokens.append('0')
            else:
                tokens.append(fields[0])
            tags.append(fields[self.tf - 1])
        return tokens, tags

    @staticmethod
    def _split_into_sentences(file_lines):
        sents = []
        s = []
        for line in file_lines:
            line = line.strip()
            if len(line) == 0:
                sents.append(s)
                s = []
                continue
            s.append(line)
        sents.append(s)
        return sents
