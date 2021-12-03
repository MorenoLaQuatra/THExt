from THExt.RedundancyManager import RedundancyManager
from THExt.SentenceRanker import SentenceRanker
from THExt.SentenceRankerPlus import SentenceRankerPlus
import spacy
import rouge #py-rouge
import numpy as np

class Highlighter():

    def __init__(self, sentence_ranker, redundancy_manager, sent_min_length=5, sent_max_length=40, spacy_modelname="en_core_web_lg"):
        print ("Full implementation release upon paper acceptance")






