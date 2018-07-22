from pathlib import Path

import click
from train import run_aspect_sequence_tagging


@click.command()
@click.argument(
    'conll_files',
    default='/home/lukasz/github/phd/sentiment-backend/aspects/data/aspects/bing_liu/bio_tags',
    type=click.Path(exists=True)
)
@click.argument('embedding_model', default='/home/lukasz/data/glove.6B.50d.txt', type=click.Path(exists=True))
@click.argument('models_output', default='models-temp', type=click.Path(exists=False))
def run_evaluation_multi_datasets(conll_files, embedding_model, models_output):
    Path(models_output).mkdir(exist_ok=True)
    datasets_path = Path(conll_files)
    conll_train_files = list(datasets_path.glob('*train.conll'))
    conll_test_files = list(datasets_path.glob('*test.conll'))

    for train_file in conll_train_files:
        test_file = [f for f in conll_test_files if train_file.stem.replace('train', '') in f.as_posix()][0]
        run_aspect_sequence_tagging(
            train_file=train_file.as_posix(),
            test_file=test_file.as_posix(),
            embedding_model=embedding_model,
            tag_num=2,
            epoch=50,
            models_output=models_output
        )


if __name__ == '__main__':
    run_evaluation_multi_datasets()
