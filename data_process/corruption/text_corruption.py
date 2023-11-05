# -*- coding: utf-8 -*-

"""Text Corruption Methods."""

import logging
import random

import nltk
from styleformer import Styleformer
from textflint.generation.transformation.UT.ocr import Ocr
from textflint.generation.transformation.UT.punctuation import Punctuation
from textflint.generation.transformation.UT.keyboard import Keyboard
from textflint.generation.transformation.UT.spelling_error import SpellingError
from textflint.generation.transformation.UT.typos import Typos
from textflint.generation.transformation.UT.tense import Tense
from textflint.generation.transformation.UT.append_irr import AppendIrr
from textflint.generation.transformation.UT.back_trans import BackTrans
from textflint.generation.transformation.UT.insert_adv import InsertAdv
from textflint.generation.transformation.UT.mlm_suggestion import MLMSuggestion
from textflint.generation.transformation.UT.swap_syn_word_embedding import SwapSynWordEmbedding
from textflint.generation.transformation.UT.swap_syn_wordnet import SwapSynWordNet

from textflint.generation.transformation.POS.prefix_swap import SwapPrefix
from textflint.generation.transformation.POS.multi_pos_swap import SwapMultiPOS
from textflint.generation.transformation.SA.double_denial import DoubleDenial
from textflint.input.component.sample.ut_sample import UTSample
from textflint.input.component.sample.pos_sample import POSSample
from textflint.input.component.sample.sa_sample import SASample
from textflint.input.component.sample.mrc_sample import MRCSample
import textflint.generation.transformation.UT as universal
import textflint.generation.transformation.POS as pos
import textflint.generation.transformation.SA as sentiment
import textflint.generation.transformation.MRC as machine
from typing import Dict
import re
import time
from nltk.tokenize import sent_tokenize


import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


TEXTFLINT_MODULES = {'universal': universal, 'pos': pos, 'machine': machine, 'sentiment': sentiment}
TEXTFLINT_SAMPLES = {'universal': UTSample, 'pos': POSSample, 'machine': MRCSample, 'sentiment': SASample}
ALL_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

# initialize
form_to_cas = Styleformer(style=1)
cas_to_form = Styleformer(style=0)
pass_to_act = Styleformer(style=3)
act_to_pass = Styleformer(style=2)
back_trans = BackTrans(
        from_model_name="facebook/wmt19-en-de",
        to_model_name="facebook/wmt19-de-en",
        device=None,
    )

def nonsense_corruption(text, severity=5):
    return " "


def get_textfint_output(func, text, corr_type):
    """

    :param func:
    :type func:
    :param text:
    :type text:
    :return:
    :rtype:
    """
    if corr_type == "pos":
        sentence = nltk.sent_tokenize(text)
        x = list()
        y = list()
        for sent in sentence:
            tags = nltk.pos_tag(nltk.word_tokenize(sent))
        x = [x[0] for x in tags]
        y = [y[1] for y in tags]
        data = {'x': x, 'y': y}
    elif corr_type == "sentiment":
        data = {'x': text, 'y': 'pos'}
    else:
        data = {'x': text}
    sample = TEXTFLINT_SAMPLES[corr_type](data)
    try:
        out = func.transform(sample)[0]
        out = out.dump()['x']
        # logger.info(f"Original text: {text}")
        # logger.info(f"ocr.transform(sample) = {ocr.transform(sample)}")
    except Exception as e:
        logger.error(f"Error in {func} corruption: {e}； Return original text.")
        logger.error(f"Original text: {text}")
        logger.error(f"{func}.transform(sample) = {func.transform(sample)}")
        return text
    return out


def ocr_corruption(text, corr_type="universal", severity=1):
    """

    :param text:
    :type text:
    :param severity:
    :type severity:
    :param corr_type:
    :type corr_type:
    :return:
    :rtype:
    """
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    ocr = Ocr(
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
    )
    out = get_textfint_output(ocr, text, corr_type)
    return out


def punctuation_corruption(text, corr_type="universal", **kwargs):
    punctuation = Punctuation()
    out = get_textfint_output(punctuation, text, corr_type)
    return out


def keyboard_corruption(text, corr_type="universal", severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    keyboard = Keyboard(
        min_char=1,
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
        include_special_char=True,
        include_numeric=True,
        include_upper_case=True,
        lang="en",
    )
    out = get_textfint_output(keyboard, text, corr_type)
    return out


def spell_error_corruption(text, corr_type="universal", severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    spell_error = SpellingError(
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
    )
    out = get_textfint_output(spell_error, text, corr_type)
    return out


def typos_corruption(text, corr_type="universal", severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    typos = Typos(
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
        stop_words=None,
        mode="random",
        skip_first_char=False,
        skip_last_char=False,
    )
    out = get_textfint_output(typos, text, corr_type)
    return out


def tense_corruption(text, corr_type="universal", **kwargs):
    tense = Tense()
    out = get_textfint_output(tense, text, corr_type)
    return out


def append_irr_corruption(text, corr_type="universal", **kwargs):
    append_irr = AppendIrr()
    out = get_textfint_output(append_irr, text, corr_type)
    return out


def back_trans_corruption(text, corr_type="universal", **kwargs):
    out = get_textfint_output(back_trans, text, corr_type)
    return out


def insert_adv_corruption(text, corr_type="universal", **kwargs):
    insert_adv = InsertAdv()
    out = get_textfint_output(insert_adv, text, corr_type)
    return out


def mlm_suggestion_corruption(text, corr_type="universal", severity=1):
    assert 0 < severity < 6
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    mlm_suggestion = MLMSuggestion(
        masked_model="bert-base-uncased",
        accrue_threshold=1,
        max_sent_size=100,
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
    )
    out = get_textfint_output(mlm_suggestion, text, corr_type)
    return out


def swap_syn_word_emb_corruption(text, corr_type="universal", severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    swap_syn_word_emb = SwapSynWordEmbedding(
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
    )
    out = get_textfint_output(swap_syn_word_emb, text, corr_type)
    return out


def swap_syn_word_net_corruption(text, corr_type="universal", severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    swap_syn_word_net = SwapSynWordNet(
        trans_min=1,
        trans_max=len(text),
        trans_p=error_rate,
    )
    out = get_textfint_output(swap_syn_word_net, text, corr_type)
    return out


def style_corruption(
        text,
        perturbs=['active_to_passive', 'passive_to_active', 'formal_to_casual', 'casual_to_formal'],
        **kwargs
):
    out_dict = {}
    sentences = sent_tokenize(text)
    res = ''
    
    for sentence in sentences:
        if 'formal_to_casual' in perturbs:
            sentence = form_to_cas.transfer(sentence)
        elif 'casual_to_formal' in perturbs:
            sentence = cas_to_form.transfer(sentence)
        elif 'passive_to_active' in perturbs:
            sentence = pass_to_act.transfer(sentence)
        else:
            sentence = act_to_pass.transfer(sentence)
        if len(sentence)>1 and sentence[-1].isalpha():
            sentence = sentence+'.'
        res = res+f'{sentence} '
    
    if 'formal_to_casual' in perturbs:
        out_dict['formal_to_casual'] = res
    if 'casual_to_formal' in perturbs:
        out_dict['casual_to_formal'] = res
    if 'passive_to_active' in perturbs:
        out_dict['passive_to_active'] = res
    if 'active_to_passive' in perturbs:
        out_dict['active_to_passive'] = res
        
    return out_dict


def to_passive_corruption(text, **kwargs):
    return style_corruption(text, perturbs=['active_to_passive'])["active_to_passive"]


def to_active_corruption(text, **kwargs):
    return style_corruption(text, perturbs=['passive_to_active'])["passive_to_active"]


def to_formal_corruption(text, **kwargs):
    return style_corruption(text, perturbs=['casual_to_formal'])["casual_to_formal"]


def to_casual_corruption(text, **kwargs):
    return style_corruption(text, perturbs=['formal_to_casual'])["formal_to_casual"]


def drop_first_corruption(text, **kwargs):
    words = text.split()
    new_row_drop_first = ' '.join(['[UNK]'] + words[1:])
    return new_row_drop_first


def drop_last_corruption(text, **kwargs):
    words = text.split()
    new_row_drop_last = ' '.join(words[:-1] + ['[UNK]'])
    return new_row_drop_last


def drop_first_and_last_corruption(text, **kwargs):
    words = text.split()
    new_drop_first_and_last = ' '.join(['[UNK]'] + words[1:-1] + ['[UNK]'])
    return new_drop_first_and_last


def shuffle_order_corruption(text, **kwargs):
    words = text.split()
    random.shuffle(words)
    new_row_shuffle_order = ' '.join(words)
    return new_row_shuffle_order


def random_word_delete_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    num_words_to_remove = int(len(text.split()) * error_rate)
    words = text.split()
    for _ in range(num_words_to_remove):
        words.remove(random.choice(words))
    new_random_delete = ' '.join(words)
    return new_random_delete


def random_word_insert_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    num_words_to_insert = int(len(text.split()) * error_rate)
    words = text.split()
    for _ in range(num_words_to_insert):
        words.insert(random.randint(0, len(words)), '[UNK]')
    new_random_insert = ' '.join(words)
    return new_random_insert


def random_word_swap_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    num_words_to_swap = int(len(text.split()) * error_rate)
    words = text.split()
    for _ in range(num_words_to_swap):
        word1 = random.choice(words)
        word2 = random.choice(words)
        words[words.index(word1)] = word2
        words[words.index(word2)] = word1
    new_random_swap = ' '.join(words)
    return new_random_swap


def random_char_insert_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    new_text = ""
    for i in range(len(text)):
        new_text += text[i]
        if text[i] == " ":  # don't insert after spaces
            continue
        if random.random() < error_rate:
            new_text += random.choice(ALL_CHARACTERS)
    return new_text


def random_char_replace_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    new_text = ""
    for i in range(len(text)):
        if random.random() < error_rate:
            new_text += random.choice(ALL_CHARACTERS)
        else:
            new_text += text[i]
    return new_text


def random_char_delete_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    new_text = ""
    for i in range(len(text)):
        if random.random() < error_rate:
            continue
        else:
            new_text += text[i]
    return new_text


def random_char_swap_corruption(text, severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    text = list(text)
    for i in range(len(text)):
        if random.random() < error_rate:
            swap_idx = random.choice(list(range(len(text))))
            text[i], text[swap_idx] = text[swap_idx], text[i]
    return "".join(text)


def _get_pos_tags(text):
    sentence = nltk.sent_tokenize(text)
    for sent in sentence:
        tags = nltk.pos_tag(nltk.word_tokenize(sent))
    return tags


def drop_nn_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    no_nns = list()
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('NN'):
            no_nns.append('[UNK]')
        else:
            no_nns.append(word)
    return ' '.join(no_nns)


def drop_rand_one_nn_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    rand_noun = []
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('NN'):
            rand_noun.append(word)
    if len(rand_noun) > 0:
        drop_nn = random.sample(rand_noun, k=1)[0]
        drop_one_nn = [x if x != drop_nn else '[UNK]' for x in text.split()]
    else:
        drop_one_nn = text.split()
    return ' '.join(drop_one_nn)


def drop_rand_one_vb_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    rand_noun = []
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('VB'):
            rand_noun.append(word)
    if len(rand_noun) > 0:
        drop_nn = random.sample(rand_noun, k=1)[0]
        drop_one_nn = [x if x != drop_nn else '[UNK]' for x in text.split()]
    else:
        drop_one_nn = text.split()
    return ' '.join(drop_one_nn)


def drop_vb_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    no_nns = list()
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('VB'):
            no_nns.append('[UNK]')
        else:
            no_nns.append(word)
    return ' '.join(no_nns)


def drop_vb_nn_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    no_nns = list()
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('NN') or tag.startswith('VB'):
            no_nns.append('[UNK]')
        else:
            no_nns.append(word)
    return ' '.join(no_nns)


def only_nn_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    no_nns = list()
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('NN'):
            no_nns.append(word)
        else:
            no_nns.append('[UNK]')
    return ' '.join(no_nns)


def only_vb_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    no_nns = list()
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('VB'):
            no_nns.append(word)
        else:
            no_nns.append('[UNK]')
    return ' '.join(no_nns)


def only_vb_nn_corruption(text, **kwargs):
    tags = _get_pos_tags(text)
    no_nns = list()
    for idx, (word, tag) in enumerate(tags):
        if tag.startswith('NN') or tag.startswith('VB'):
            no_nns.append(word)
        else:
            no_nns.append('[UNK]')
    return ' '.join(no_nns)


# POS type TBD
def swap_prefix_corruption(text, corr_type="pos", severity=1):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    swap_prefix = SwapPrefix(
        trans_max=len(text),
        trans_p=error_rate,
    )
    out = get_textfint_output(swap_prefix, text, corr_type)
    # return " ".join(out)
    return out

def double_denial_corruption(text, corr_type="sentiment", **kwargs):
    double_denial = DoubleDenial(
    )
    out = get_textfint_output(double_denial, text, corr_type)
    return out


def swap_multi_pos_nn_corruption(text, severity=1, corr_type="pos"):
    return swap_multi_pos_corruption(text, severity, corr_type, swap_type="NN")


def swap_multi_pos_jj_corruption(text, severity=1, corr_type="pos"):
    return swap_multi_pos_corruption(text, severity, corr_type, swap_type="JJ")


def swap_multi_pos_corruption(text, severity=1, corr_type="pos", swap_type="NN"):
    error_rate = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    swap_multi_pos = SwapMultiPOS(
        treebank_tag=swap_type,
        trans_max=len(text),
        trans_p=error_rate,
    )

    out = get_textfint_output(swap_multi_pos, text, corr_type)
    # return " ".join(out)
    return out
OPTION=['A','B','C','D','E','F','G','H']
def get_options(choices, option_content):
    option_list = []
    for idx, answer in enumerate(choices):
        optionstr = OPTION[idx]
        if option_content:
            option_list.append(f'({optionstr}) {answer}')
        else:
            option_list.append(f'({optionstr}')
    return option_list

def clean_question(question, generative = False): # delete context
    qlist = question.split('Options:')
    q = qlist[0].split('Context:')
    if not generative:
        res = 'Question: ' + q[0] + 'Options:' + qlist[1] + "\n"
    else:
        res = 'Question: ' + q[0] + "\n"
    
    return res


def circular_option(text, gt_choices, gt_choice, context=None, que_pr='', opt_pr=''):
    n = len(gt_choices)
    shift = random.randint(0,1)
    if shift == 0:
        new_choices = gt_choices[1:] + gt_choices[:1]
        new_choice = (gt_choice-1+n)%n
    else:
        new_choices = gt_choices[-1:] + gt_choices[:-1]
        new_choice = (gt_choice+1)%n
    choices_text = ''
    for idx, choice in enumerate(new_choices):
        optionstr = OPTION[idx]
        choices_text += f' ({optionstr}) {choice}'
    if context!=None:
        res = f'{que_pr} {text} \nContext:{context}{opt_pr}{choices_text}\n'
    else:
        res = f'{que_pr} {text} \n{opt_pr}{choices_text}\n'
    return res, new_choices, new_choice


def reverse_option(text, gt_choices, gt_choice, context=None, que_pr='', opt_pr=''):
    n = len(gt_choices)
    x,y = 0,n-1
    new_choices = gt_choices.copy()
    chioce_map = {}
    while(x<=y):
        tmp = new_choices[x]
        new_choices[x] = new_choices[y]
        new_choices[y] = tmp
        chioce_map[x] = y
        chioce_map[y] = x
        x+=1
        y-=1
    new_choice = chioce_map[gt_choice]
    choices_text = ''
    for idx, choice in enumerate(new_choices):
        optionstr = OPTION[idx]
        choices_text += f' ({optionstr}) {choice}'
    if context!=None:
        res = f'{que_pr} {text} \nContext:{context}{opt_pr}{choices_text}\n'
    else:
        res = f'{que_pr} {text} \n{opt_pr}{choices_text}\n'
    return res, new_choices, new_choice

def lowercase(text):
    return text.lower()



def match_case(source_word: str, target_word: str) -> str:
    """Return a version of the target_word where the case matches the source_word."""
    # Check for all lower case source_word
    if all(letter.islower() for letter in source_word):
        return target_word.lower()
    # Check for all caps source_word
    if all(letter.isupper() for letter in source_word):
        return target_word.upper()
    # Check for capital source_word
    if source_word and source_word[0].isupper():
        return target_word.capitalize()
    return target_word

CONTRACTION_MAP: Dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}
contraction_map: Dict[str, str] = CONTRACTION_MAP
reverse_contraction_map: Dict[str, str] = {value: key for key, value in contraction_map.items()}
reverse_contraction_pattern = re.compile(
            r"\b({})\b ".format("|".join(reverse_contraction_map.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )
def constraction(text):
    def cont(possible):
            match = possible.group(1)
            expanded_contraction = reverse_contraction_map.get(
                match, reverse_contraction_map.get(match.lower())
            )
            return match_case(match, expanded_contraction) + " "

    return reverse_contraction_pattern.sub(cont, text)

contraction_pattern = re.compile(
            r"\b({})\b".format("|".join(contraction_map.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )
def expansion(text):
    def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = contraction_map.get(match, contraction_map.get(match.lower()))
            return match_case(match, expanded_contraction)

    return contraction_pattern.sub(expand_match, text)

d_basic = {
    'lowercase':lowercase,
    'constraction':constraction,
    'expansion':expansion,

}
d_sentence = {'active_to_passive': to_passive_corruption,
              'passive_to_active': to_active_corruption,
              'formal_to_casual': to_casual_corruption,
              'casual_to_formal':to_formal_corruption,
              'back_trans': back_trans_corruption,
              }
d_word = {'swap_syn':swap_syn_word_net_corruption,
          'insert_adv': insert_adv_corruption,
          'append_irr_corruption': append_irr_corruption,
         }
d_character = {'char_ocr': ocr_corruption,
               'char_typos': typos_corruption,
               'char_spell_error': spell_error_corruption,
               'char_keyboard_corruption': keyboard_corruption,
               }
d_option = {'circular_option':circular_option,
            'reverse_option':reverse_option}


if __name__ == "__main__":
    # test_text = "What is the color of the sky?"
    test_text = "It is worth noting that the combination of all possible perturbations is not included in the table due to the vast number of combinations. However, the table provides a comprehensive overview of the performance of the models across a wide range of perturbations. Additionally, the table shows the original performance of the models on each dataset as well as the performance after each perturbation. The combination of CLIP-BART and Single Prompt seems to perform well overall, with relatively minor decreases in performance after most perturbations."
    time_start = time.time()
    # corr_text = swap_prefix_corruption(test_text, corr_type="pos", severity=5)
    corr_text = mlm_suggestion_corruption(test_text)
    # corr_text = swap_multi_pos_jj_corruption(test_text, severity=5, corr_type="pos")
    print(corr_text)
    print(time.time() - time_start)
    print(ocr_corruption("hello world", severity=1))
    print(ocr_corruption("What time of day?", severity=1))
    #print(punctuation_corruption("hello world", severity=1))
    print(keyboard_corruption("hello world", severity=5))
    print(back_trans_corruption("It is worth noting that the combination of all possible perturbations is not included in the table due to the vast number of combinations. However, the table provides a comprehensive overview of the performance of the models across a wide range of perturbations. Additionally, the table shows the original performance of the models on each dataset as well as the performance after each perturbation. The combination of CLIP-BART and Single Prompt seems to perform well overall, with relatively minor decreases in performance after most perturbations."))
    print(append_irr_corruption("hello world"))
    #print(swap_syn_word_emb_corruption("It is worth noting that the combination of all possible perturbations is not included in the table", severity=5))
    print(swap_syn_word_net_corruption("It is worth noting that the combination of all possible perturbations is not included in the table", severity=5))
    print(insert_adv_corruption("It is worth noting that the combination of all possible perturbations is not included in the table"))
    print(style_corruption("It is worth noting that the combination of all possible perturbations is not included in the table", severity=1, perturbs=['active_to_passive', 'passive_to_active', 'formal_to_casual', 'casual_to_formal']))
    #print(drop_last_corruption(test_text))
    #print(drop_first_corruption(test_text))
    #print(drop_first_and_last_corruption(test_text))
    #print(shuffle_order_corruption(test_text))
    #for severity in range(1, 5 + 1):
    #    print(random_word_delete_corruption(test_text, severity=severity))
    #    print(random_word_swap_corruption(test_text, severity=severity))
    #    print(random_char_insert_corruption(test_text, severity=severity))
    #    print(random_char_replace_corruption(test_text, severity=severity))
    #    print(random_char_delete_corruption(test_text, severity=severity))
    #    print(random_char_swap_corruption(test_text, severity=severity))
    #    print(mlm_suggestion_corruption(test_text, severity=severity))
    #    print("=====================================")
    #print(drop_nn_corruption(test_text))
    #print(drop_rand_one_nn_corruption(test_text))
    #print(swap_prefix_corruption(test_text, severity=5, corr_type="pos"))
    #print(double_denial_corruption(test_text))
    #print(swap_multi_pos_nn_corruption(test_text, severity=5, corr_type="pos"))
    #print(swap_multi_pos_jj_corruption(test_text, severity=5, corr_type="pos")) 