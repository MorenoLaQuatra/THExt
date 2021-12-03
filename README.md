# THExt

Transformer-based Highlights Extraction (THExt)

## Installation

Run the following to install

```python
pip install git+LINK_TO_THIS_REPO
```

## Usage
### Using pretrained models
```python
from THExt.SentenceRankerPlus import SentenceRankerPlus
from THExt.RedundancyManager import RedundancyManager
from THExt.Highlighter import Highlighter

sr = SentenceRankerPlus()
sr.load_model(base_model_name, model_name_or_path=checkpoint_dir)
rm = RedundancyManager()
h = Highlighter(sr, rm)

# Define a set of sentences
sentences = ["The pen is on the table", "I love cats"]
abstract = "text of the abstract 1"
number_extracted_highlights = 1

highlights = h.get_highlights_simple(sentences, abstract, rel_w=1.0, pos_w=0.0, red_w=0.0, prefilter=False, NH = number_extracted_highlights)

for h in highlights:
    print (h)

```

## Developing THExt
To install THExt, along with the tools you need to develop and run tests, run the following in your virtualenv

```bash
$ pip install -e .[dev]
```