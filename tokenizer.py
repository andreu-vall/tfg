from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import argparse
import os


def bert_tokenize(data_path, override=False):
    
    token_dir = f"{data_path}/tokenized-text"
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenize is a function that takes a string and returns a list of strings (the tokens)
    # untokenize is a function that takes a list of strings (the tokens) and returns a string
    tokenize = bert_tokenizer.tokenize
    untokenize = bert_tokenizer.convert_tokens_to_string


    bert_path = f"{token_dir}/bert.pkl"
    if os.path.exists(bert_path) and not override:
        print("Loading already bert tokenized text from", bert_path)
        bert_tokenized_text = pd.read_pickle(bert_path)
        return bert_tokenized_text, tokenize, untokenize

    if override:
        print("Overriding existing bert tokenized text")
    else:
        print("First time, tokenizng the text")

    reviews_path = f"{data_path}/reviews.csv"
    df = pd.read_csv(reviews_path)
    df["text"] = df["text"].fillna('nan') # some text might be literally "nan" as string

    tqdm.pandas()
    bert_tokenized_text = df['text'].progress_apply(bert_tokenizer.tokenize).rename('tokenized_text_bert')

    print("Saving the tokenized text to", bert_path)
    bert_tokenized_text.to_pickle(bert_path) # pickle so that array of strings is kept, csv was a pain to read it

    return bert_tokenized_text, tokenize, untokenize



def load(data_path, tokenizer, override=False):
    assert tokenizer == 'bert-base-uncased', "Only bert tokenizer is supported"
    return bert_tokenize(data_path, override)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a new split for the data.')
    parser.add_argument('data_path', type=str, help='Path to the data')
    parser.add_argument('tokenizer', choices=['bert-base-uncased'], help='Tokenizer to use')
    parser.add_argument('--override', action='store_true', help='Override existing tokenized files')
    return parser.parse_args()


# File name tokenize.py was problematic, as some tokenizer library use it. tokenizer.py is fine
# Can I or should I use the GPU for tokenizing? It's only done 1 time and it doesn't matter really much for my TFG
if __name__ == "__main__":
    args = parse_arguments()
    load(args.data_path, args.tokenizer, args.override)
