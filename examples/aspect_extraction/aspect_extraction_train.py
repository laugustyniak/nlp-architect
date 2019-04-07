from __future__ import division, print_function, unicode_literals, absolute_import

import pickle
from os.path import basename
from pathlib import Path

import click
from keras.utils import to_categorical

from nlp_architect.data.sequential_tagging import SequentialTaggingDataset
from nlp_architect.models.aspect_extraction import AspectExtraction


@click.command()
@click.option('--dataset-path', required=True, type=Path, help='Path to the training dataset')
@click.option('--word-embedding-path', required=True, type=Path, help='Path to the embeddings')
@click.option('--word-embedding-dim', required=False, type=int, default=300, help='Len of word embedding vectors')
@click.option('--char-embedding-dim', required=False, type=int, default=25, help='Len of character embedding vectors')
@click.option('--epochs', required=False, default=25, help='Number of epochs to calculate')
@click.option('--tag-number', required=False, default=3, help='Number of column with tag to classify')
@click.option('--sentence-length', required=False, type=int, default=30)
@click.option('--word-length', required=False, type=int, default=20)
@click.option('--batch-size', required=False, type=int, default=3)
@click.option('--dropout', required=False, type=float, default=0.5)
def train_aspect_extractor(
        dataset_path: Path,
        word_embedding_path: Path,
        word_embedding_dim: int,
        char_embedding_dim: int,
        epochs: int,
        tag_number: int,
        sentence_length: int,
        word_length: int,
        batch_size: int,
        dropout: float
):
    click.echo('Word embedding: ' + word_embedding_path.as_posix())

    bilstm_layer = True
    crf_layer = True
    word_embedding_flag = True
    char_embedding_flag = True

    network_params = [
        ('char', char_embedding_flag),
        ('word', word_embedding_flag),
        ('bilstm', bilstm_layer),
        ('lstm', not bilstm_layer),
        ('crf', crf_layer),
        (str(epochs) + 'epochs', True),
    ]

    network_params_string = '-'.join([param for param, flag in network_params if flag])

    logs_path = dataset_path.parent / 'logs'
    logs_path.mkdir(exist_ok=True, parents=True)

    # load dataset and parameters
    model_name = 'model-info' + '-' + network_params_string + '-' + basename(dataset_path.stem) + '.info'
    model_path = dataset_path.parent / model_name

    dataset_path = dataset_path.as_posix()

    dataset = SequentialTaggingDataset(
        train_file=dataset_path,
        test_file=dataset_path,  # only as placeholder, never used
        max_sentence_length=sentence_length,
        max_word_length=word_length,
        tag_field_no=tag_number
    )

    # get the train and test data sets
    x_train, x_char_train, y_train = dataset.data['train']

    if word_embedding_flag and char_embedding_flag:
        x_train = [x_train, x_char_train]
    elif word_embedding_flag and not char_embedding_flag:
        x_train = x_train
    elif not word_embedding_flag and char_embedding_flag:
        x_train = x_char_train
    else:
        raise Exception('Wrong features')

    num_y_labels = len(dataset.y_labels) + 1
    vocabulary_size = dataset.word_vocab_size + 1
    char_vocabulary_size = dataset.char_vocab_size + 1

    y_train = to_categorical(y_train, num_y_labels)

    aspect_model = AspectExtraction()
    aspect_model.build(
        sentence_length=sentence_length,
        word_length=word_length,
        target_label_dims=num_y_labels,
        word_vocab=dataset.word_vocab,
        word_vocab_size=vocabulary_size,
        char_vocab_size=char_vocabulary_size,
        word_embedding_dims=word_embedding_dim,
        char_embedding_dims=char_embedding_dim,
        word_lstm_dims=char_embedding_dim,
        tagger_lstm_dims=word_embedding_dim + char_embedding_dim,
        tagger_fc_dims=word_embedding_dim + char_embedding_dim,
        dropout=dropout,
        external_embedding_model=word_embedding_path.as_posix(),
        bilstm_layer=bilstm_layer,
        crf_layer=crf_layer,
        word_embedding_flag=word_embedding_flag,
        char_embedding_flag=char_embedding_flag,
    )

    # Set callback functions to early stop training and save the best model so far
    tensorboard_path = (logs_path / ('tensorboard-' + model_name)).as_posix()
    print('Tensorboard: ' + tensorboard_path)

    aspect_model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
    )

    aspect_model.save(model_path.with_suffix('.h5').as_posix())

    # saving model info
    with open('{}-info'.format(model_path), 'wb') as fp:
        info = {
            'sentence_len': sentence_length,
            'word_len': word_length,
            'num_of_labels': num_y_labels,
            'labels_id_to_word': {v: k for k, v in dataset.y_labels.vocab.items()},
            'epoch': epochs,
            'word_vocab': dataset.word_vocab,
            'vocab_size': vocabulary_size,
            'char_vocab_size': char_vocabulary_size,
            'char_vocab': dataset.char_vocab,
            'word_embedding_dims': word_embedding_dim,
            'char_embedding_dims': char_embedding_dim,
            'word_lstm_dims': char_embedding_dim,
            'tagger_lstm_dims': word_embedding_dim + char_embedding_dim,
            'dropout': dropout,
            'external_embedding_model': word_embedding_path,
            'train_file': dataset_path,
            'eval': eval,
            'bilstm_layer': bilstm_layer,
            'crf_layer': crf_layer,
            'word_embedding_layer': word_embedding_flag,
            'char_embedding_layer': char_embedding_flag,
            'y_labels': dataset.y_labels
        }
        pickle.dump(info, fp)


if __name__ == '__main__':
    train_aspect_extractor()
