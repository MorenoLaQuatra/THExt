from thext import RedundancyManager
from thext import SentenceRankerPlus
import spacy
import rouge #py-rouge
import numpy as np

class Highlighter():

    def __init__(self, sentence_ranker, 
            redundancy_manager=None, 
            sent_min_length=5, 
            sent_max_length=40, 
            spacy_modelname="en_core_web_lg", 
            batch_size = 8
        ):
        self.sr = sentence_ranker
        self.rm = redundancy_manager
        self.sent_min_length = sent_min_length
        self.sent_max_length = sent_max_length
        self.nlp = spacy.load(spacy_modelname, disable = ['ner'])
        self.nlp.max_length = 10000000
        self.batch_size = batch_size

    def get_highlights(self, sentences_list, rel_w=1.0, pos_w=0.5, red_w=0.5, NH=3, prefilter=True):

        # Remove duplicates
        sentences_list = list(set(sentences_list))

        highlights = []
        if prefilter:
            sentences_list = self.prefiltering_sentences(sentences_list)

        rank_scores = self.sr.get_scores(sentences_list, batch_size=self.batch_size)

        position_scores = np.linspace(1.0, 0.0, num=len(sentences_list), endpoint=True)
        #normalize rank scores
        min_value = min(rank_scores)
        max_value = max(rank_scores)

        index_max = rank_scores.index(max_value)
        rank_scores[index_max] = -1
        highlights.append(sentences_list[index_max])

        while len(highlights) < NH:

            if red_w != 0.0:
                # redundancy_scores
                redundancy_scores = self.rm.batch_redundancy_score_aggregated(sentences_list, highlights)
            else:
                redundancy_scores = [0.0] * len(rank_scores)

            overall_scores = [0.0] * len(rank_scores)
            for i in range(0, len(rank_scores)):
                if rank_scores[i] < 0:
                    overall_scores[i] = 0.0
                else:
                    overall_scores[i] = rel_w * rank_scores[i] + pos_w * position_scores[i] + red_w * (1-redundancy_scores[i])

            max_value = max(overall_scores)
            index_max = overall_scores.index(max_value)
            rank_scores[index_max] = -1
            highlights.append(sentences_list[index_max])


        return highlights

    def get_highlights_simple(self, sentences_list, abstract=None, rel_w=1.0, pos_w=0.5, red_w=0.5, NH=3, prefilter=True, rerank = False):

        # Remove duplicates
        sentences_list = list(set(sentences_list))

        highlights = []
        if prefilter:
            sentences_list = self.prefiltering_sentences(sentences_list)

        if abstract == None:
            rank_scores = self.sr.get_scores(sentences_list, batch_size=self.batch_size)
        else:
            rank_scores = self.sr.get_scores(sentences_list, abstract, batch_size=self.batch_size)

        indices = []
        max_value = max(rank_scores)
        index_max = rank_scores.index(max_value)
        indices.append(index_max)
        rank_scores[index_max] = -1
        highlights.append(sentences_list[index_max])

        if NH == None:
            NH = len(sentences_list)

        while len(highlights) < NH:
            max_value = max(rank_scores)
            index_max = rank_scores.index(max_value)
            indices.append(index_max)
            rank_scores[index_max] = -1
            highlights.append(sentences_list[index_max])

        if rerank:
            zipped_lists = zip(indices, highlights)
            sorted_zipped_lists = sorted(zipped_lists)
            highlights = [h for _, h in sorted_zipped_lists]

        return highlights

    def get_highlights_oracle(self, sentences_list, real_highlights, NH=3):
        rank_scores = list(np.zeros(len(sentences_list)))
        highlights = []
        r_computer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], limit_length=False, max_n=2, alpha=0.5, stemming=False)

        for i, s in enumerate(sentences_list):
            m_score = 0.0
            for rh in real_highlights:
                score = r_computer.get_scores(s, rh)
                r2f = score["rouge-2"]["f"]
                if r2f > m_score:
                    m_score = r2f

            rank_scores[i] = m_score

        max_value = max(rank_scores)
        index_max = rank_scores.index(max_value)
        rank_scores[index_max] = -1
        highlights.append(sentences_list[index_max])

        while len(highlights) < NH:
            max_value = max(rank_scores)
            index_max = rank_scores.index(max_value)
            rank_scores[index_max] = -1
            highlights.append(sentences_list[index_max])


        return highlights

    def prefiltering_sentences(self, sentences_list):
        filtered_sentences = []
        for s in sentences_list:
            if self.valid_sentence(s):
                filtered_sentences.append(s)
        return filtered_sentences

    def valid_sentence(self, sent):
        # return True if a good sentence, False if it should be skipped

        doc = self.nlp(sent)
        valid_len=0
        for token in doc:
            if not (token.is_stop):
                valid_len +=1

        if valid_len< self.sent_min_length or valid_len > self.sent_max_length:
            return False

        return True










