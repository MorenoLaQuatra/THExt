# THExt

Transformer-based Highlights Extraction (THExt)

### Examples and demo

- Highlights for [ACL 2021 conference papers](https://aclanthology.org/volumes/2021.acl-long/) available here: [AI model](demos/acl_highlights_ai.md) - [CS model](demos/acl_highlights_cs.md)
- Highlights for [Journal of Machine Learning Research (Volume 22)](https://jmlr.org/papers/v22/) available here: [AI model](demos/jmlr_highlights_ai.md) - [CS model](demos/jmlr_highlights_cs.md)

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
sentences = ["This is the first sentence ...", "This is the Nth sentence..."]
abstract = "Text of the abstract"
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
