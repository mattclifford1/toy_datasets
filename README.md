# toy_datasets
python package to get synthetic and toy datasets to use with ML analysis

# install
for now no PyPi upload so just install directly from github

todo: make a release version and install by adding to pyproject (get from python-utils how to) 

# examples
see the examples.py for examples of how to use

# dev
install the dev group from pdm/pyproject. then install this dir as editable
```
python -m pip install -e .
```

# todo
 - fix the splitting for normal dist loader
 - finish any other synthetic loaders
 - MIMIC loaders
 - have main with the new class method
 - add to abstract (from get_dataset)
   - do dim reduction
   - do scaling 
   - other things for feature parity?
 - other thigns to add to abstract
   - have the dim reduction plotting the same for train and test
