import enchant
import itertools
import numpy as np
import os
import pandas as pd
from pke.unsupervised import TextRank, TopicRank
import re
from sacremoses import MosesDetokenizer
from scipy.special import softmax, expit as sigmoid
import spacy
import spacy_spanish_lemmatizer  # noqa
import spacy_stanza
import stanza
from stanza.pipeline.core import DownloadMethod
import torch
from tqdm import tqdm
import transformers

from typing import Literal, Optional

from . import strings


class NLProcessor(object):

    def __init__(self, name=None, stop_words=None, use_stanza=False, verbose=0, path=None):
        self.spacy_pipelines = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm'
        }
        self.spacy_attrs = [
            'lemma', 'norm', 'pos', 'tag', 'dep', 'ent_type', 'ent_iob', 'shape',
            'is_alpha', 'is_ascii', 'is_digit', 'is_space', 'is_punct', 'is_stop'
        ]

        self.verbose = verbose  # Print progress
        self.path = path or os.getcwd()  # Path to store model files

        self.name = None
        self.model = None
        self.lang = None

        self.spacy = None
        self.stanza = None
        self.spellchecker = None
        self.tokenizer = None

        if name is not None:
            self.load_model(name, stop_words=stop_words, use_stanza=use_stanza)

    def load_model(self, name, stop_words=None, use_stanza=False):
        self.name = self.spacy_pipelines.get(name, name)

        if self.verbose > 0:
            print('Load Spacy [{}]...'.format(self.name))

        self.spacy = self._load_spacy()

        if not self.spacy:
            if self.verbose > 0:
                print('Spacy [{}] could not be loaded...'.format(self.name))

            return self

        self.model = self.spacy

        self.tokenizer = self.model.tokenizer
        self.lang = self.model.lang

        if stop_words:
            self.model.Defaults.stop_words = set(stop_words)

        if use_stanza:
            if self.verbose > 0:
                print('Load Stanza [{}]...'.format(self.lang))

            self.stanza = self._load_stanza()
            self.tokenizer = spacy_stanza.tokenizer.StanzaTokenizer(
                self.stanza,
                self.model.vocab
            )
            self.model.tokenizer = self.tokenizer
        elif self.lang == 'es':
            self.model.replace_pipe('lemmatizer', 'spanish_lemmatizer')

        if self.verbose > 0:
            print('Load Enchant [{}]...'.format(self.lang))

        if enchant.dict_exists(self.lang):
            self.spellchecker = enchant.DictWithPWL(self.lang, '{}/new-words.txt'.format(self.path))

        return self

    def _load_spacy(self):
        path = '{}/spacy/{}'.format(self.path, self.name)
        name = path if os.path.exists(path) else self.name

        try:
            model = spacy.load(name)
        except OSError:
            if re.match(r'^[a-z]{2}_(?:core|dep|ent|sent)_(?:web|news|wiki|ud)_(?:sm|md|lg|trf)$', name):
                os.system('python -m spacy download {}'.format(name))  # nosec
                model = spacy.load(name)
            elif re.match(r'^[a-z]{2}$', name):
                model = spacy.blank(name)
            else:
                return

        if name != path:
            if self.verbose > 0:
                print('Save Spacy [{}]...'.format(self.name))

            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.to_disk(path)

        return model

    def _load_stanza(self):
        model = stanza.Pipeline(
            lang=self.lang,
            model_dir=self.path,
            download_method=DownloadMethod.REUSE_RESOURCES,
            verbose=True if self.verbose > 0 else False
        )

        return model

    def add_stop_words(self, words):
        if isinstance(words, str):
            words = [words]

        self.model.Defaults.stop_words |= set(words)

        return self

    def tokenize(self, text):
        return [tok.text for tok in self.tokenizer(text)]

    def detokenize(self, tokens):
        detokenizer = MosesDetokenizer(lang=self.lang)

        chunks = [list(group) for k, group in itertools.groupby(tokens, lambda x: x.isspace())]

        return ''.join([detokenizer.detokenize(chunk) if len(chunk) > 1 else chunk[0] for chunk in chunks])

    def to_array(self, text, attrs=None):
        if attrs is None:
            attrs = self.spacy_attrs

        d = [
            {
                attr: getattr(tok, attr + '_') if hasattr(tok, attr + '_') else getattr(tok, attr)
                for attr in ['text'] + attrs
            }
            for tok in self.model(text)
        ]

        return d

    def to_pandas(self, text, attrs=None):
        return pd.DataFrame(
            self.to_array(text, attrs=attrs)
        )

    def detect_language(self, text, name=None, return_probas=False):
        return Language(
            verbose=self.verbose,
            path='{}/{}'.format(self.path, 'language')
        ).load_model(name).predict(text, return_probas=return_probas)

    def detect_sentiment(self, text, name=None, return_probas=False):
        return Sentiment(
            verbose=self.verbose,
            path='{}/{}'.format(self.path, 'sentiment')
        ).load_model(name).predict(text, return_probas=return_probas)

    def detect_key_phrases(self, text, n=10, pos=None, new_words=None, clean_pos=False, **kwargs):
        _textrank_params = dict(
            window=2, top_percent=None, normalized=False
        )
        _topicrank_params = dict(
            threshold=0.74, method='average', heuristic=None
        )

        candidates = []

        tokens = self.tokenize(text)
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        for word in new_words or []:
            if word in tokens:
                candidates.append(word)

        kps_agg = pd.DataFrame(columns=['textrank', 'topicrank'])

        textrank_params = {k: kwargs.get(k, v) for k, v in _textrank_params.items()}
        kps_textrank = self.textrank(text, n=n, pos=None, **textrank_params)
        for k, v in kps_textrank.items():
            kps_agg.loc[k, 'textrank'] = v

        topicrank_params = {k: kwargs.get(k, v) for k, v in _topicrank_params.items()}
        kps_topicrank = self.topicrank(text, n=n, pos=None, **topicrank_params)
        for k, v in kps_topicrank.items():
            kps_agg.loc[k, 'topicrank'] = v

        kps_agg['score'] = kps_agg.fillna(0).sum(axis=1)
        kps_agg = kps_agg.sort_values('score', ascending=False)

        if clean_pos:
            pos_tokens = [
                tok['text'] for tok in self.to_array(text)
                if tok['pos'] in pos and tok['text'].isalnum()
            ]

            kps = []
            for s in list(kps_agg.index):
                s = [w for w in s.split() if w.isalpha() and w not in self.model.Defaults.stop_words]
                if any(np.in1d(s, pos_tokens)):
                    kps.append(' '.join(s))
        else:
            kps = list(kps_agg.index)

        candidates.extend(kps)

        return [i for i in candidates if len(i) > 1 and not any(i in c for c in candidates if c != i)][:n]

    def textrank(self, text, n=10, pos=None, window=2, top_percent=None, normalized=False):
        model = TextRank()
        model.load_document(
            input=self.model(text),
            language=self.lang,
            stoplist=self.model.Defaults.stop_words
        )
        model.candidate_selection(pos=pos)
        model.candidate_weighting(window=window, pos=pos, top_percent=top_percent, normalized=normalized)

        return dict(model.get_n_best(n))

    def topicrank(self, text, n=10, pos=None, threshold=0.74, method='average', heuristic=None):
        model = TopicRank()
        model.load_document(
            input=self.model(text),
            language=self.lang,
            stoplist=self.model.Defaults.stop_words
        )
        model.candidate_selection(pos=pos)
        model.candidate_weighting(threshold=threshold, method=method, heuristic=heuristic)

        return dict(model.get_n_best(n))

    def spellcheck(self, text):
        check = {}
        text = text.lower()
        tokens = self.tokenize(text)

        for tok in tokens:
            if tok.isalpha() and tok not in check and not self.spellchecker.check(tok):
                sug = self.spellchecker.suggest(tok)
                if len(sug) > 0 and sug[0].lower() != tok.lower():
                    check[tok] = sug[0]

        return check if len(check) > 0 else True

    def text_process(
        self,
        text: str,
        split: Literal['lines', 'words', 'tokens'] = 'lines',
        tasks: Optional[list[str]] = None
    ):
        if text is None:
            return text

        text = str(text)

        if tasks is None:
            tasks = [
                'normalize_unicode', 'to_lower', 'remove_hashtags', 'remove_mentions',
                'limit_charseq', 'remove_urls', 'remove_emails', 'remove_html',
                'remove_emojis', 'remove_emoticons', 'remove_laughters', 'fix_contractions', 'fix_punctuation'
            ]

        if split == 'tokens':
            chunks = self.tokenize(text)
        else:
            chunks = strings.split(text, method=split)

        for i, chunk in enumerate(chunks):
            for task in tasks:
                params = {}
                if isinstance(task, (list, tuple)):
                    task, params = task

                if callable(task):
                    fn = task
                elif isinstance(task, str):
                    if hasattr(self, 'text_process_{}'.format(task)):
                        fn = getattr(self, 'text_process_{}'.format(task))
                    elif hasattr(strings, task):
                        fn = getattr(strings, task)
                else:
                    continue

                chunk = fn(chunk, **params)

            chunks[i] = chunk

        if split == 'tokens':
            text = self.detokenize(chunks)
        else:
            text = strings.join(chunks, method=split)

        return text

    def text_process_spellcheck(self, text):
        check = self.spellcheck(text)

        if isinstance(check, dict):
            for err, alt in check.items():
                if len(alt) > 0:
                    text = text.replace(err, alt[0])

        return text


class Classifier(object):

    models_map = {}
    labels_map = {}

    def __init__(self, name=None, verbose=0, path=None):
        self.name = name  # Name of the transformer model (or models_map key)
        self.verbose = verbose  # Print progress
        self.path = path or os.getcwd()  # Path to store model files

        self.tokenizer = None
        self.model = None

        transformers.logging.set_verbosity_info()

        if self.name:
            self.load_model()

    def load_model(self, name=None):
        if name:
            self.name = name

        if not self.name and 'default' in self.models_map:
            self.name = 'default'

        if self.name in self.models_map:
            self.name = self.models_map[self.name]

        if not self.name:
            if self.verbose > 0:
                print('Unknown model name...')

            return self

        if self.verbose > 0:
            print('Load {}...'.format(self.name))

        path = '{}/{}'.format(self.path, self.name.split('/')[1])
        name = path if os.path.exists(path) else self.name

        if self.verbose > 0:
            print('Load tokenizer...')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

        if self.verbose > 0:
            print('Load model...')

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(name)

        if name != path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if self.verbose > 0:
                print('Save tokenizer...')

            self.tokenizer.save_pretrained(path)

            if self.verbose > 0:
                print('Save model...')

            self.model.save_pretrained(path)

        return self
    
    def predict(self, corpus, return_probas=False, batch_size = 1):
        if not self.model or not self.tokenizer:
            return

        if self.verbose > 0:
            print('Setup...')

        is_single_input = isinstance(corpus, str)
        corpus = [corpus] if is_single_input else corpus

        # Limit the number of tokens in each document to the maximum established by the model.
        # Take into account the tokens that the model adds at the beginning and end of the document.
        max_position_embeddings = int(self.model.config.max_position_embeddings)
        bos_token_id = int(self.model.config.bos_token_id or 0)
        eos_token_id = int(self.model.config.eos_token_id or 0)
        max_length = max_position_embeddings - (bos_token_id + eos_token_id)

        is_single_label = self.model.config.problem_type == 'single_label_classification'
        
        labels = self.model.config.id2label.items()
        logits = np.empty((len(corpus), len(labels)))

        if self.verbose > 0:
            print('Predict...')
        
        output = self.tokenizer.batch_encode_plus(
            corpus,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        drange = np.arange(0, len(corpus), batch_size)
        drange = tqdm(drange) if self.verbose > 0 else drange
        for i in drange:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=output['input_ids'][i:(i + batch_size)],
                    attention_mask=output['attention_mask'][i:(i + batch_size)]
                )
                logits[i:(i + batch_size), :] = outputs.logits.detach().numpy()

        if is_single_label:
            scores = softmax(logits, axis=1)
        else:
            scores = sigmoid(logits)

        probas = [
            {
                self.labels_map.get(v.lower(), v.lower()): np.round(score[k], 4) for k, v in labels
            } for score in scores
        ]

        if self.verbose > 0:
            print('Format...')

        result = []
        for proba in probas:
            keymax = list(proba)[np.argmax(list(proba.values()))]
            if return_probas:
                result.append({
                    'label': keymax,
                    'score': proba[keymax],
                    'probas': proba
                })
            else:
                result.append(keymax)

        if is_single_input:
            return result[0]

        return result


class Language(Classifier):

    models_map = {
        'default': 'papluca/xlm-roberta-base-language-detection'
    }


class Sentiment(Classifier):

    models_map = {
        'default': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'es': 'pysentimiento/robertuito-sentiment-analysis',
        'en': 'finiteautomata/bertweet-base-sentiment-analysis',
        'stars': 'nlptown/bert-base-multilingual-uncased-sentiment'
    }
    labels_map = {
        'neg': 'negative',
        'neu': 'neutral',
        'pos': 'positive',
        '1 star': '1',
        '2 stars': '2',
        '3 stars': '3',
        '4 stars': '4',
        '5 stars': '5'
    }


class Emotion(Classifier):

    models_map = {
        'default': 'SamLowe/roberta-base-go_emotions'
    }
