# THExt

**T**ransformer-based **H**ighlights **Ext**raction from scientific papers (**THExt**)

### Examples and demo

All examples provided below have been extracted using the best-performing model reported in the paper. No manual pre- or post- processing has been applied for highlights extraction. The text of the papers has been parsed from PDF files using [GROBID](https://grobid.readthedocs.io/en/latest/).

- Highlights for [ACL 2021 conference papers](https://aclanthology.org/volumes/2021.acl-long/) available here: [AI model](demos/acl_highlights_ai.md) - [CS model](demos/acl_highlights_cs.md)
- Highlights for [Journal of Machine Learning Research (Volume 22)](https://jmlr.org/papers/v22/) available here: [AI model](demos/jmlr_highlights_ai.md) - [CS model](demos/jmlr_highlights_cs.md)

Pre-trained models will be released after paper revision process.


## Installation

Run the following to install

```python
pip install thext
python -m spacy download en_core_web_lg
```

## Usage

### Pretrained models on 🤗 Hub:

- Computer Science: https://huggingface.co/morenolq/thext-cs-scibert `morenolq/thext-cs-scibert`
- Artificial Intelligence: https://huggingface.co/morenolq/thext-cs-scibert `morenolq/thext-ai-scibert`
- Biology and Medicine: https://huggingface.co/morenolq/thext-cs-scibert `morenolq/thext-bio-scibert`

### Using pretrained models
```python
from thext import SentenceRankerPlus
from thext import RedundancyManager
from thext import Highlighter

base_model_name = "morenolq/thext-cs-scibert"
model_name_or_path = "morenolq/thext-cs-scibert"
sr = SentenceRankerPlus()
sr.load_model(base_model_name=base_model_name, model_name_or_path=model_name_or_path)
h = Highlighter(sr)

# Define a set of sentences
sentences = [
    "We propose a new approach, based on Transformer-based encoding, to highlight extraction. To the best of our knowledge, this is the first attempt to use transformer architectures to address automatic highlight generation.", 
    "We design a context-aware sentence-level regressor, in which the semantic similarity between candidate sentences and highlights is estimated by also attending the contextual knowledge provided by the other paper sections.",
    "Fig. 2, Fig. 3, Fig. 4 show the effect of varying the number K of selected highlights on the extraction performance. As expected, recall values increase while increasing the number of selected highlights, whereas precision values show an opposite trend.",
]
abstract = "Highlights are short sentences used to annotate scientific papers. They complement the abstract content by conveying the main result findings. To automate the process of paper annotation, highlights extraction aims at extracting from 3 to 5 paper sentences via supervised learning. Existing approaches rely on ad hoc linguistic features, which depend on the analyzed context, and apply recurrent neural networks, which are not effective in learning long-range text dependencies. This paper leverages the attention mechanism adopted in transformer models to improve the accuracy of sentence relevance estimation. Unlike existing approaches, it relies on the end-to-end training of a deep regression model. To attend patterns relevant to highlights content it also enriches sentence encodings with a section-level contextualization. The experimental results, achieved on three different benchmark datasets, show that the designed architecture is able to achieve significant performance improvements compared to the state-of-the-art."

num_highlights = 1

highlights = h.get_highlights_simple(sentences, abstract,
                rel_w=1.0, 
                pos_w=0.0, 
                red_w=0.0, 
                prefilter=False, 
                NH = num_highlights)

for i, h in enumerate(highlights):
    print (f"{i}\t{h}")

# 0	We propose a new approach, based on Transformer-based encoding, to highlight extraction. To the best of our knowledge, this is the first attempt to use transformer architectures to address automatic highlight generation.

```

## References:

If you find it useful, please cite the following paper:

```bibtex
@article{thext,
  title={Transformer-based highlights extraction from scientific papers},
  author={La Quatra, Moreno and Cagliero, Luca},
  journal={Knowledge-Based Systems},
  pages={109382},
  year={2022},
  publisher={Elsevier}
}
```


## Developing THExt
To install THExt, along with the tools you need to develop and run tests, run the following in your virtualenv

```bash
$ pip install -e .[dev]
```


