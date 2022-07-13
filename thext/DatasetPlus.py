import glob, os, re
from tqdm import tqdm
import nltk
import spacy
from joblib import Parallel, delayed

from multiprocess import Process, Manager, Pool
import rouge #py-rouge

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


class DatasetPlus():
    """
    Dataset class containing all the utility methods to parse, structure and store data.
    ...

    Attributes
    ----------
    list_text : str
        A list of documents to be summarized:
        [
            "This is the first document text...", 
            "This is the second document text...", 
            ...
        ]
    list_highlights : str
        A nested list of highlights that can be used to compute the ground_truth labels.
        [ 
            [summary_1_text_1, summary_2_text_1, ...], 
            [summary_1_text_2, summary_2_text_2, ...], 
            ...
        ]
        **
        If list_highlights is not set, the default None parameter indicate a blind test dataset (no label computed)

    Methods
    -------
    parse_data()
        It parses training data organizing them into a dictionary.
    """

    def __init__(self, list_text, list_abstract, list_highlights=None, 
                    n_jobs = 4, 
                    aggregation_method="max", 
                    spacy_modelname="en_core_web_lg",
                    normalization_score=1):
        self.list_text = list_text
        self.list_abstract = list_abstract
        self.list_highlights = list_highlights
        self.dataset = {}
        self.n_jobs = n_jobs
        self.normalization_score = normalization_score
        self.aggregation_method = aggregation_method
        
        self.nlp = spacy.load(spacy_modelname, disable = ['ner'])
        self.nlp.max_length = 1000000
        

        self.parse_data()

    def is_test_set(self):
        if self.list_highlights != None:
            return True
        else:
            return False

    def parse_data(self):
        logging.info("Dataset - Parsing Data - Train? " + str(self.is_test_set()))
        manager = Manager()
        self.dataset = manager.dict()
        self.list_keys = range(0, len(self.list_text))
        p = Pool(self.n_jobs)

        if self.is_test_set():
            p.map(self._job_parse_train_eval, self.list_keys)
        else:
            p.map(self._job_parse_test, self.list_keys)

        self.dataset = dict(self.dataset)
        


    def _job_parse_train_eval(self, k):
        logging.info(str(len(self.dataset.keys())) + "/" + str(len(self.list_keys)))
        d = {}
        d["key"] = k
        raw_text = self.list_text[k]
        d["raw_text"] = raw_text
        clean_text = self.clean_text(raw_text)
        d["clean_text"] = clean_text
        d["raw_abstract"] = self.list_abstract[k]

        self.fill_sentences(d)

        if len(d["clean_sentences"].items()) <= 1:
            logging.error("Paper "+str(k)+" do not contain enough sentences.")
            return

        d["highlights"] = []
        d["highlights"] = self.list_highlights[k]

        self.fill_regression_labels(d)
        self.dataset[k] = d


    def _job_parse_test(self, k):
        logging.info(str(len(self.dataset.keys())) + "/" + str(len(self.list_keys)))
        d = {}
        d["key"] = k
        raw_text = self.list_text[k]
        d["raw_text"] = raw_text
        clean_text = self.clean_text(raw_text)
        d["clean_text"] = clean_text
        d["raw_abstract"] = self.list_abstract[k]

        self.fill_sentences(d)
        self.dataset[k] = d

    def clean_text(self, text):
        clean = text.replace('\n',' ')
        clean = re.sub(' +', ' ', clean)
        return clean

    def clean_sentence(self, s, remove_punct=True, remove_sym = True, remove_stop=True):
        
        analyzed_sentence = self.nlp(s)
        clean_token = []

        for token in analyzed_sentence:
            if token.pos_ != "PUNCT":
                clean_token.append(token)

        if remove_punct:
            ct = []
            for token in clean_token:
                if token.pos_ != "PUNCT":
                    ct.append(token)
            clean_token = ct

        if remove_sym:
            ct = []
            for token in clean_token:
                if token.is_stop == False:
                    ct.append(token)
            clean_token = ct

        if remove_stop:
            ct = []
            for token in clean_token:
                if token.pos_ != "SYM":
                    ct.append(token)
            clean_token = ct

        return ' '.join(word.text for word in clean_token)

    def fill_sentences(self, d):
        d["raw_sentences"] = {}
        d["clean_sentences"] = {}
        doc = self.nlp(d["clean_text"])
        sentences = doc.sents
        for i, s in enumerate(sentences):
            clean_s = self.clean_sentence(s.text, self.nlp)
            d["raw_sentences"][i] = s.text
            d["clean_sentences"][i] = clean_s

    def fill_regression_labels(self, d):
        d["r2p_labels"] = {}
        d["r2r_labels"] = {}
        d["r2f_labels"] = {}
        d["rlp_labels"] = {}
        d["rlr_labels"] = {}
        d["rlf_labels"] = {}


        for i, s in d["raw_sentences"].items():
            r2p, r2r, r2f, rlp, rlr, rlf = self.get_regression_labels(s, d["highlights"])
            d["r2p_labels"][i] = r2p
            d["r2r_labels"][i] = r2r
            d["r2f_labels"][i] = r2f
            d["rlp_labels"][i] = rlp
            d["rlr_labels"][i] = rlr
            d["rlf_labels"][i] = rlf

        if self.normalization_score != None:
            # normalizing between 0 and 1
            for type_score in ["r2p_labels", "r2r_labels", "r2f_labels", "rlp_labels", "rlr_labels", "rlf_labels"]:
                try:
                    values = list(d[type_score].values())
                    max_value = max(values)
                    min_value = min(values)

                    for k, v in d[type_score].items():
                        d[type_score][k] = ((d[type_score][k] - min_value) / (max_value - min_value)) * self.normalization_score
                except Exception as e:
                    logging.error(e)
                    for k, v in d[type_score].items():
                        d[type_score][k] = 0.0
        '''
        for i, s in d["raw_sentences"].items():
            print(d["r2f_labels"][i], s)
        '''
        return d

    def get_regression_labels(self, sentence, list_highlights, aggregation="max"):
        aggregation = self.aggregation_method
        r_computer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], limit_length=False, max_n=2, alpha=0.5, stemming=False)
        r2p = 0
        r2r = 0
        r2f = 0
        rlp = 0
        rlr = 0
        rlf = 0
        if aggregation == "max":
            for summ in list_highlights:
                score = r_computer.get_scores(sentence, summ)
                if (score["rouge-2"]["p"] > r2p):
                    r2p = score["rouge-2"]["p"]
                if (score["rouge-2"]["r"] > r2r):
                    r2r = score["rouge-2"]["r"]
                if (score["rouge-2"]["f"] > r2f):
                    r2f = score["rouge-2"]["f"]
                
                if (score["rouge-l"]["p"] > rlp):
                    rlp = score["rouge-l"]["p"]
                if (score["rouge-l"]["r"] > rlr):
                    rlr = score["rouge-l"]["r"]
                if (score["rouge-l"]["f"] > rlf):
                    rlf = score["rouge-l"]["f"]
        else:
            print ("Aggregation type: " + aggregation + " not supported yet")
            exit()

        return r2p, r2r, r2f, rlp, rlr, rlf
