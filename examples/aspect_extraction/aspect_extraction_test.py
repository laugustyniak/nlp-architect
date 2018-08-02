from collections import namedtuple
from pathlib import Path
from typing import Iterable

import click
from tqdm import tqdm
from train import run_aspect_sequence_tagging

DatasetFiles = namedtuple('Dataset', ['train_file', 'test_file'])

EMBEDDINGS = [
    'glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt',
    'glove.twitter.27B.25d.txt', 'glove.twitter.27B.50d.txt', 'glove.twitter.27B.100d.txt',
    'glove.twitter.27B.200d.txt',
]
EMBEDDINGS_PATH = Path('/home/lukasz/data/')
CONLL_FILES_PATH = '/home/lukasz/github/phd/sentiment-backend/aspects/data/aspects/bing_liu/bio_tags'
SEMEVAL_FILES_PATH = '/home/lukasz/github/nlp/nlp-architect/examples/aspect_extraction/semeval/2014'


@click.command()
@click.argument('conll_files', default=CONLL_FILES_PATH, type=click.Path(exists=True))
@click.argument('embedding_model', default='/home/lukasz/data/glove.6B.50d.txt', type=click.Path(exists=True))
@click.argument(
    'models_output',
    default='/home/lukasz/github/nlp/nlp-architect/examples/aspect_extraction/models-tmp',
    type=click.Path(exists=False)
)
@click.argument('augment_data', default=False, type=bool)
@click.argument('similarity_threshold', default=0.8, type=float)
def run_evaluation_multi_datasets(conll_files, embedding_model, models_output, augment_data, similarity_threshold):
    Path(models_output).mkdir(parents=True, exist_ok=True)
    datasets_path = Path(conll_files)
    conll_train_files = list(datasets_path.glob('*train.conll'))
    conll_test_files = list(datasets_path.glob('*test.conll'))

    for train_file in conll_train_files:
        test_file = [f for f in conll_test_files if train_file.stem.replace('train', '') in f.as_posix()][0]
        click.echo('Dataset: ' + train_file.as_posix())
        run_aspect_sequence_tagging(
            train_file=train_file.as_posix(),
            test_file=test_file.as_posix(),
            embedding_model=embedding_model,
            models_output=models_output,
            tag_num=2,
            epoch=20,
            augment_data=augment_data,
            similarity_threshold=similarity_threshold,
        )


def run_evaluation_multi_datasets_and_multi_embeddings(
        models_output: str='/home/lukasz/github/nlp/nlp-architect/examples/aspect_extraction'):
    for embedding in tqdm(EMBEDDINGS, desc='Embeddings progress'):
        click.echo('Embedding: ' + embedding)
        embedding_model = (EMBEDDINGS_PATH / embedding).as_posix()
        embedding_name = Path(embedding).stem
        models_output = (Path(models_output) / ('models-' + embedding_name)).as_posix()
        Path(models_output).mkdir(parents=True, exist_ok=True)

        for dataset_file in tqdm(get_aspect_datasets(), desc='Datasets progress'):
            click.echo('Dataset: ' + dataset_file.train_file.as_posix())
            run_aspect_sequence_tagging(
                train_file=dataset_file.train_file.as_posix(),
                test_file=dataset_file.test_file.as_posix(),
                embedding_model=embedding_model,
                models_output=models_output,
                tag_num=2,
                epoch=20,
                augment_data=False,
            )


def get_aspect_datasets() -> Iterable[DatasetFiles]:
    datasets = []
    for datasets_path in tqdm([CONLL_FILES_PATH, SEMEVAL_FILES_PATH]):
        datasets_path = Path(datasets_path)
        train_files = list(datasets_path.glob('*train.conll'))
        test_files = list(datasets_path.glob('*test.conll'))
        for train_file in tqdm(train_files, desc='Datasets progress'):
            test_file = [f for f in test_files if train_file.stem.replace('train', 'test') == f.stem][0]
            datasets.append(DatasetFiles(train_file=train_file, test_file=test_file))
            click.echo('Dataset: ' + train_file.as_posix())
    return datasets


if __name__ == '__main__':
    run_evaluation_multi_datasets_and_multi_embeddings()
