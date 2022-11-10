import pandas as pd
from datasets import load_dataset
from pprint import pprint
from transformers import DistilBertTokenizerFast
from transformers import GPT2Tokenizer, RobertaTokenizer, FlaubertTokenizer, XLMTokenizer, CTRLTokenizer, OpenAIGPTTokenizer, PreTrainedTokenizer


def split_text(text):
    text = text.split('\n ')
    return text
#write a function to clean the text
def clean_text(text):
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('\n', '')
    text = text.replace('\"', '')
    text = text.replace('  ', ' ')
    
    if(text[0] == '\''):
        text = text[1:]
    if(text[-1] == '\''):
        text = text[:-1]
    return text

#this function clean a split the text into sentences
def prepare_text(text):
    if(type(text) == str):
        splited_text = split_text(text)
        cleaned_text = [clean_text(sentence) for sentence in splited_text]
        return cleaned_text
    if(type(text) == list):
        arraytext = []
        for sentence in text:
            splited_text = split_text(sentence)
            cleaned_text = [clean_text(sentence) for sentence in splited_text]
            arraytext.append(cleaned_text)
        return arraytext

#defining the tokenizer
def create_tokenizer(name = 'distilbert-base-uncased-distilled-squad', vocab_size = 10000):
  tokenizer = DistilBertTokenizerFast.from_pretrained(name, vocab_size = vocab_size)

  return tokenizer

#tokenizing the splited text
def tokenize_function(text, tokenizer):
  tokenized_inputs = tokenizer(text, truncation = True, padding = 'max_length', max_length = 1024,)
  return tokenized_inputs

def tokenize_function_result(text, tokenizer):
  tokenized_inputs = tokenizer(text, truncation = True, padding = 'max_length', max_length = 128,)
  return tokenized_inputs

def tokenize_plus(text, tokenizer):
  try:
    tokenized_inputs = tokenizer(text)
  except:
    for word in text:
      tokenizer.add_tokens(word)
    tokenize_plus(text,tokenizer)
  return tokenized_inputs

#coverting from text or list of text to tokenized inputs
def text_to_token(text, tokenizer):
    if (type(text) == str):
        pure_text = prepare_text(text)
        tokenized_text = tokenize_function(pure_text, tokenizer)
        return tokenized_text
    if (type(text) == list):
        tokenized_text = []
        for sentence in text:
            pure_text = prepare_text(sentence)
            tokenized_sentence = tokenize_function(pure_text, tokenizer)
            tokenized_text.append(tokenized_sentence)
        return tokenized_text
    
def text_to_token_plus(text, tokenizer):
    #this function recieve a list of sentences or a list of lists of 
    #sentences and return a dict with the input ids or a list of a dicts with the inputs ids
    if (type(text) == str):
        pure_text = prepare_text(text)
        tokenized_text = tokenize_plus(pure_text, tokenizer)
        return tokenized_text
    if (type(text) == list):
        tokenized_text = []
        for sentence in text:
            pure_text = prepare_text(sentence)
            tokenized_sentence = tokenize_plus(pure_text, tokenizer)
            tokenized_text.append(tokenized_sentence)
        return tokenized_text

def token_to_text(tokens, tokenizer):
    if (type(tokens)== list):
        text = []
        for example in tokens:
          text_ex = tokenizer.batch_decode(example)
          text.append(text_ex)
        return text
    if (type(tokens) != list):
        text = tokenizer.batch_decode(tokens)
        return text


"""dialogues = data_train.iloc[0:10]['dialog'].values.tolist()
tokenized_text = text_to_token(dialogues)
tokenized_text[0]"""
