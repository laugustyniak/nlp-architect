from pathlib import Path

from aspects.analysis.nlp_architect import get_models_attributes
from nlp_architect.utils.metrics import get_conll_scores

MODELS_PATH = Path('/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/models/glove.840B.300d/')

pred_y_ylables = get_models_attributes(MODELS_PATH.glob('*'), 'predictions', 'y_test', 'y_labels')

for item in list(pred_y_ylables.values()):
    scores = get_conll_scores(item['predictions'], item['y_test'], {v: k for k, v in item['y_labels'].items()})
    break

pass

