import json
import logging
from typing import Dict, List, Tuple

from overrides import overrides
from tqdm import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("hoppy")
class HoppyReader(DatasetReader):
    """DatasetReader for WikiHop and MedHop datasets.
    
    Parameters
    ----------
    tokenizer
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 convert_to_lowercase: bool = True) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._convert_to_lowercase = convert_to_lowercase

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            #dataset_json = json.load(dataset_file)
            #dataset = dataset_json['data']
        logger.info("Reading the dataset")
        instances = []
        i = 0
        for qa_item in tqdm(dataset[:10]):
            i += 1
            print('\n\nQuestion', i)
            answer = qa_item['answer']
            paragraphs = ""
            tokenized_paragraphs = []
            span_start = 0
            answer_found = False
            for paragraph in qa_item['supports']:
                if self._convert_to_lowercase:
                    paragraph = paragraph.lower()
                paragraph += " EOD "
                #if not answer_found:
                #    if paragraph.find(answer) != -1:
                #        span_start += paragraph.find(answer)
                #        answer_found = True
                #    else:
                #        span_start += len(paragraph)

                tokenized_paragraphs.extend(self._tokenizer.tokenize(paragraph))
                paragraphs += paragraph
                #print('len:', len(tokenized_paragraphs))

            #print(tokenized_paragraphs, type(tokenized_paragraphs), type(tokenized_paragraphs[1]))

            # What if answer isn't found verbatim?
            #span_end = span_start + len(answer)
            #print('\n', span_start, ' -- ', span_end, 'of', len(paragraphs),)
            #if span_end > len(paragraphs):
            #    print(paragraphs, answer)
            instance = self.text_to_instance(qa_item['query'],
                                             qa_item['supports'],
                                             #(span_start, span_end),
                                             answer,
                                             qa_item['candidates'],
                                             tokenized_paragraphs)
            instances.append(instance)
            break

                #for question_answer in paragraph_json['qas']:
                #    question_text = question_answer["question"].strip().replace("\n", "")
                #    answer_texts = [answer['text'] for answer in question_answer['answers']]
                #    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                #    span_ends = [start + len(answer) for start, answer in zip(span_starts,
                # answer_texts)]
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         query_text: str,
                         passage_text: str,
                         #char_span: Tuple[int, int],
                         answer_text: str,
                         candidates: List[str],
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        #print(passage_text[:100])
        #print(passage_tokens[:10])
        #char_span = char_span or ()

        '''# We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        #print(passage_tokens)
        passage_offsets = []
        last_char = 0
        for token in passage_tokens:
            passage_offsets.append((last_char, last_char + len(token)))
            last_char += len(token)
        #[(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        print(passage_offsets)
        #print(passage_offsets, passage_tokens)
        char_span_start, char_span_end = char_span
        print(answer_text)
        (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                     (char_span_start, char_span_end))
        '''
        token_spans: List[Tuple[int, int]] = []
        answer_tokens = self._tokenizer.tokenize(answer_text)
        span_start = -1
        span_end = -1
        #print(answer_tokens)
        print(passage_tokens[:100])
        for i in range(len(passage_tokens) - len(answer_tokens) + 1):
            j = 0
            #print(i+j, len(passage_tokens))
            #print(j, len(passage_tokens))
            while passage_tokens[i+j].text == answer_tokens[j].text:
                #print(str(passage_tokens[i+j]), str(answer_tokens[j]),
                #      str(passage_tokens[i+j]) == str(answer_tokens[j]))
                print(i+j, j, passage_tokens[i+j], answer_tokens[j], 'match')
                print(passage_tokens[i:i+len(answer_tokens)], answer_tokens)
                j += 1
                if j == len(answer_tokens):
                    span_start = i
                    span_end = i + len(answer_tokens) - 1  # inclusive range
                    print('its a match!:', passage_tokens[i:j], answer_tokens)
                    break
            if span_start != -1:
                break
        #print(span_start, span_end)
        if span_start == -1 or span_end == -1:
            print('not found:', answer_tokens)
        #    print(passage_tokens[:50])
        #    print(query_text)
        print(span_start, span_end)
        '''
        if error:
            logger.debug("Passage: %s", passage_text)
            logger.debug("Passage tokens: %s", passage_tokens)
            logger.debug("Question text: %s", query_text)
            logger.debug("Tokens in answer: %s", self._tokenizer.tokenize(answer_text))
            logger.debug("Answer: %s", answer_text)
        '''
        token_spans.append((span_start, span_end))
        if ([t.text for t in passage_tokens[span_start:span_end+1]]
                != [a.text for a in answer_tokens]):
            print(passage_tokens[span_start:span_end+1], 'doesnt match', answer_tokens)
        else:
            print(passage_tokens[span_start:span_end+1], 'matches', answer_tokens)

        return util.make_reading_comprehension_instance(self._tokenizer.tokenize(query_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_text,
                                                        additional_metadata=(
                                                            {"candidates": candidates}))

    @classmethod
    def from_params(cls, params: Params) -> 'HoppyReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
