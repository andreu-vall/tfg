# PETER-Andreu


## Prerquisites
```
python >= 3
torch
```

## Steps
1. Process data, from raw_data (any format) to data (in data-name/reviews.csv)
2. Generate/load train/valid/test splits
3. Train a model with a data_path and index_dir, giving it an id
4. Test this model

## Usage
```
python train.py [data_path] [index_dir] [id]
python test.py [id]
```

## Sources
- https://github.com/lileipisces/PETER: repo cloned
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html: more readable train/test torch codes
- https://github.com/alarca94/sequer-recsys23: some file structure and ideas, also the logging
