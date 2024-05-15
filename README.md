# PETER-Andreu


## Prerquisites
```
python >= 3
torch
```

## Usage
1. **create.ipynb [data]**: create a dataset with columns user, item, rating, text and save it to [data]
2. **split.py [data] [split_id] [seed]**: make a train/valid/test split for [data] and save it to [split_id]
3. **tokenizer.py [data] [tokenizer]***: tokenize the text column with a certain tokenizer (bert-base-uncased, others?)
4. **train.py [data] [split_id] [train_id] [--args]**: train a model with certain args and save it to out/[id]
5. **test.py [train_id]**: evaluate the fast metrics of a certain training on the test split
6. **generate.py [train_id] [strategy] [result_id] [--args].py**: decode test from text with a certain decoding strategy
7. **analysis.ipynb [train_id] [result_id]**: human analysis of the results genereated by a model using a strategy

\* Step 3 will be automatically done by train.py if not done beforehand. It's only done once, because tokenizing is kinda slow and might take around 5 minutes

## Sources
- https://github.com/lileipisces/PETER: repo cloned
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html: more readable train/test torch codes

