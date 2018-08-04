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

from typing import Dict

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dropout, TimeDistributed, Bidirectional, LSTM, concatenate, \
    Dense, Activation
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from nlp_architect.utils.embedding import load_word_embeddings


class AspectExtraction(object):
    """ NER model with CRF classification layer (Keras/TF implementation) """

    def __init__(self):
        self.model = None

    def build(
            self,
            sentence_length: int,
            word_length: int,
            target_label_dims: int,
            word_vocab: Dict,
            word_vocab_size: int,
            char_vocab_size: int,
            word_embedding_dims: int = 100,
            char_embedding_dims: int = 25,
            word_lstm_dims: int = 25,
            tagger_lstm_dims: int = 100,
            tagger_fc_dims: int = 100,
            dropout: float = 0.2,
            external_embedding_model: str = None,
            crf_layer: bool = True,
            bilstm_layer: bool = True,
            word_embedding_layer: bool = True,
            char_embedding_layer: bool = True,
    ):
        """
        Build a LSTM-based aspect extraction model, bye dafault it it BILSTM + CRF

        Args:
            sentence_length (int): max sentence length
            word_length (int): max word length in characters
            target_label_dims (int): number of entity labels (for classification)
            word_vocab (dict): word to int dictionary
            word_vocab_size (int): word vocabulary size
            char_vocab_size (int): character vocabulary size
            word_embedding_dims (int): word embedding dimensions
            char_embedding_dims (int): character embedding dimensions
            word_lstm_dims (int): character LSTM feature extractor output dimensions
            tagger_lstm_dims (int): word tagger LSTM output dimensions
            tagger_fc_dims (int): output fully-connected layer size
            dropout (float): dropout rate
            external_embedding_model (str): path to external word embedding model
            bilstm_layer:
            crf_layer:
        """
        all_inputs = []

        # build word input
        words_input = Input(shape=(sentence_length,), name='words_input')
        all_inputs.append(words_input)

        all_features = []

        if word_embedding_layer:
            if external_embedding_model is not None:
                # load and prepare external word embedding
                external_emb, ext_emb_size = load_word_embeddings(external_embedding_model)

                embedding_matrix = np.zeros((word_vocab_size, ext_emb_size))
                for word, i in word_vocab.items():
                    embedding_vector = external_emb.get(word.lower())
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[i] = embedding_vector

                # load pre-trained word embeddings into an Embedding layer
                # note that we set trainable = False so as to keep the embeddings fixed
                embedding_layer = Embedding(
                    word_vocab_size,
                    ext_emb_size,
                    weights=[embedding_matrix],
                    input_length=sentence_length,
                    trainable=False
                )
            else:
                # learn embeddings ourselves
                embedding_layer = Embedding(
                    word_vocab_size,
                    word_embedding_dims,
                    input_length=sentence_length
                )

            word_embeddings = embedding_layer(words_input)
            word_embeddings = Dropout(dropout)(word_embeddings)
            all_features.append(word_embeddings)

        # create word character embeddings
        if char_embedding_layer:
            word_chars_input = Input(shape=(sentence_length, word_length), name='word_chars_input')
            all_inputs.append(word_chars_input)
            char_embedding_layer = Embedding(char_vocab_size, char_embedding_dims, input_length=word_length)
            char_embeddings = TimeDistributed(char_embedding_layer)(word_chars_input)
            if bilstm_layer:
                char_embeddings = TimeDistributed(Bidirectional(LSTM(word_lstm_dims)))(char_embeddings)
            char_embeddings = Dropout(dropout)(char_embeddings)
            all_features.append(char_embeddings)

        # create the final feature vectors
        features = concatenate(all_features, axis=-1)

        # encode using a bi-lstm
        if bilstm_layer:
            lstm_layers = Bidirectional(LSTM(tagger_lstm_dims, return_sequences=True))(features)
            lstm_layers = Dropout(dropout)(lstm_layers)
            after_lstm_hidden = TimeDistributed(Dense(tagger_fc_dims))(lstm_layers)
        else:
            after_lstm_hidden = TimeDistributed(LSTM(tagger_lstm_dims, return_sequences=True))(features)

        # classify the dense vectors
        if crf_layer:
            crf = CRF(target_label_dims, sparse_target=False)
            predictions = crf(after_lstm_hidden)
            # compile the model
            model = Model(inputs=all_inputs, outputs=predictions)
            model.compile(
                loss=crf.loss_function,
                optimizer='adam',
                metrics=[crf.accuracy]
            )
        else:
            model = Activation('softmax')(after_lstm_hidden)
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy']
            )

        self.model = model

    def fit(self, x, y, epochs=1, batch_size=1, callbacks=None, validation=None):
        """
        Train a model given input samples and target labels.

        Args:
            x (numpy.ndarray or :obj:`numpy.ndarray`): input samples
            y (numpy.ndarray): input sample labels
            epochs (:obj:`int`, optional): number of epochs to train
            batch_size (:obj:`int`, optional): batch size
            callbacks(:obj:`Callback`, optional): Keras compatible callbacks
            validation(:obj:`list` of :obj:`numpy.ndarray`, optional): optional validation data
                to be evaluated when training
        """
        assert self.model, 'Model was not initialized'
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True,
                       validation_data=validation,
                       callbacks=callbacks)

    def predict(self, x, batch_size=1):
        """
        Get the prediction of the model on given input

        Args:
            x (numpy.ndarray or :obj:`numpy.ndarray`): input samples
            batch_size (:obj:`int`, optional): batch size

        Returns:
            numpy.ndarray: predicted values by the model
        """
        assert self.model, 'Model was not initialized'
        return self.model.predict(x, batch_size=batch_size)

    def save(self, path):
        """
        Save model to path

        Args:
            path (str): path to save model weights
        """
        save_load_utils.save_all_weights(self.model, path)

    def load(self, path):
        """
        Load model weights

        Args:
            path (str): path to load model from
        """
        save_load_utils.load_all_weights(self.model, path, include_optimizer=False)
