from collections import namedtuple

from pathlib import Path

from aspects.analysis.nlp_architect import get_models_attributes
from nlp_architect.utils.metrics import get_y_label_and_prediction_tuples, get_conll_scores

Evaluation = namedtuple('Evaluation', 'y, prediction, labels')

MODELS_PATH = Path('/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/models/glove.840B.300d/')

pred_y_ylables = get_models_attributes(MODELS_PATH.glob('*model-info-char-word-bilstm-crf-10epochs-Laptops_poria-train.conll.info*'), 'predictions', 'y_test', 'y_labels', 'test_raw_sentences')

for dataset, item in list(pred_y_ylables.items()):
    print(dataset)
    args = (item['predictions'], item['y_test'],
        {
            v: k
            for k, v
            in item['y_labels'].items()
        })

    scores = get_conll_scores(*args)
    true_labels_and_predictions = get_y_label_and_prediction_tuples(*args)

    errors = []
    for true_predictions, raw_sentences in zip(true_labels_and_predictions, item['test_raw_sentences']):
        if not raw_sentences[1] == true_predictions[2]:
            errors.append(Evaluation(raw_sentences[1], true_predictions[2], raw_sentences[0]))

    for error in errors:
        if error.y != error.prediction:
            print(list(zip(error.y, error.prediction, error.labels)), '\n')
