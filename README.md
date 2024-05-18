# PETER-Andreu

Train a transformers model from scratch to learn to generate personalized justifications on why you might like or dislike some items.

## Prerquisites
```
python >= 3
torch
```

## Usage
1. **create.ipynb [data_path]**: create a dataset with columns user, item, rating, text and save it to [data_path]
2. **split.py [data_path] [split_id] [seed]**: make a train/valid/test split for [data_path] and save it to [split_id]
3. **tokenizer.py [data_path] [tokenizer]***: tokenize the text column with a certain tokenizer (tokenizer-bert-base-uncased, others?)
4. **train.py [data_path] [tokenizer] [context_window] [split_id] [train_id] [--args]**: train a model with certain args and save it to out/[train_id]
5. **test.py [train_id]**: evaluate the fast metrics of a certain training on the test split
6. **generate.py [train_id] [strategy] [result_id] [--args]**: decode test from text with a certain decoding strategy
7. **analysis.ipynb [train_id] [result_id]**: human analysis of the results genereated by a model using a strategy

\* Step 3 will be automatically done by train.py if not done beforehand. It's only done once, because tokenizing is kinda slow and might take around 5 minutes


## Example usage
```
1. create_mcauley.ipynb (2014 Amazon Beauty Reviews)
2. python split.py data/amz-beauty-review split_id_1 1
3. python tokenizer.py data/amz-beauty-review tokenizer-bert-base-uncased (5 min)
4. python train.py data/amz-beauty-review tokenizer-bert-base-uncased 15 split_id_1 train_id_1 --epochs 3 (each epoch takes around 2m 30s amb window=15)
4. python train.py data/amz-beauty-review tokenizer-bert-base-uncased 5 split_id_1 train_id_14
5. python test.py train_id_1
6. python generate.py train_id_1 greedy result_id_1 (1m 40s)
```

data/amz-beauty-summary
tokenizer.py tarda només 20 seg
train.py context_window=20


## Sources
- https://github.com/lileipisces/PETER: repo cloned, used it as a backbone but modifying things to make it more easy to understand and more pythonic
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html: more readable train/test torch codes


## Ampliations
- To force the model to better alineate the predicted rating with the review content, an additional loss component could be added that penalized the inconsistency between the predicted_rating and the predicted_text
- To better exploit the knowledge in the datasest, we could add a item_context that gave the most probable words that all users of the database have given to a certain item for example (but not the reviews from the valid or test set)
- This could also be done in a user item, where it's feeded for example the kind of topics that the reviews has made till now

But right now I'll have to focus on what I have, explain it and formalize the maths involved in all my pipeline
