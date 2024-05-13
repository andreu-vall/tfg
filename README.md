# PETER-Andreu


## Prerquisites
```
python >= 3
torch
```


## Usage

1. **process.ipynb**: create a dataset with [users, items, ratings, text] from any source and save it in /data
2. **split.ipynb**: create or load a certain train/test/valid split
3. **transform.ipynb**: transform all the text into ID's (will be automatically done the first time training it)
4. **train.py [data_path] [index_dir] [id]**: train a model
5. **test.py [id]**: test the model


## Sources
- https://github.com/lileipisces/PETER: repo cloned
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html: more readable train/test torch codes
- https://github.com/alarca94/sequer-recsys23: some file structure and ideas, also the logging
