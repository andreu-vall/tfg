from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import argparse
import os



def tokenize_text(data_path, name, tokenize_fn, override):

    token_dir = f"{data_path}/tokenized-text"
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)

    token_path = f"{token_dir}/{name}.pkl"
    if os.path.exists(token_path) and not override:
        print(f"Loading already {name} tokenized text from", token_path)
        tokenized_text = pd.read_pickle(token_path)
        return tokenized_text

    if override:
        print(f"Overriding existing {name} tokenized text")
    else:
        print(f"First time, tokenizng the text with {name}")

    reviews_path = f"{data_path}/reviews.csv"
    df = pd.read_csv(reviews_path)
    df["text"] = df["text"].fillna('nan') # some text might be literally "nan" as string

    tqdm.pandas()
    tokenized_text = df['text'].progress_apply(tokenize_fn).rename(f'{name}_tokenized_text')

    print("Saving the tokenized text to", token_path)
    tokenized_text.to_pickle(token_path) # pickle so that array of strings is kept, csv was a pain to read it

    return tokenized_text





def bert_tokenize(data_path, override):

    # Any existing tokenizer could be used. I could even train my own tokenizer but there would be no point in doing it

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenize is a function that takes a string and returns a list of strings (the tokens)
    # untokenize is a function that takes a list of strings (the tokens) and returns a string
    tokenize = bert_tokenizer.tokenize
    untokenize = bert_tokenizer.convert_tokens_to_string

    tokenized_text = tokenize_text(data_path, 'bert', tokenize, override)

    return tokenized_text, tokenize, untokenize



def space_tokenize(data_path, override):

    tokenize = lambda text: text.split(' ')
    untokenize = lambda tokens: ' '.join(tokens)
    
    tokenized_text = tokenize_text(data_path, 'space', tokenize, override)

    return tokenized_text, tokenize, untokenize



def load(data_path, tokenizer, override):

    assert tokenizer in ['tokenizer-bert-base-uncased', 'tokenizer-space']
    
    token_dir = f"{data_path}/tokenized-text"
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)

    if tokenizer == 'tokenizer-bert-base-uncased':
        return bert_tokenize(data_path, override)
    else:
        return space_tokenize(data_path, override)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a new split for the data.')
    parser.add_argument('data_path', type=str, help='Path to the data')
    parser.add_argument('tokenizer', choices=['tokenizer-bert-base-uncased', 'tokenizer-space'], help='Tokenizer to use')
    parser.add_argument('--override', action='store_true', help='Override existing tokenized files')
    return parser.parse_args()


# File name tokenize.py was problematic, as some tokenizer library use it. tokenizer.py is fine
# Can I or should I use the GPU for tokenizing? It's only done 1 time and it doesn't matter really much for my TFG
if __name__ == "__main__":
    args = parse_arguments()
    load(args.data_path, args.tokenizer, args.override)
