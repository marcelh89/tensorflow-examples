## Image classification to 0 or 1 (two classes output) 
The setup is based on https://github.com/nicknochnack/ImageClassification
- load images from filesystem
- prepare images (convert, scale)
- split data to train (70%), validation(20%), test(10%)
- create model (and save) / or load existing model
- use the model to classify images to "good" or "bad"

#### How to set up

1. create virtual environment and use it

2. install dependencies
```pip install -r requirements.txt```


Known Issues:
MatplotlibDeprecationWarning: Support for FigureCanvases without a required_interactive_framework attribute was deprecated in Matplotlib 3.6 and will be removed two minor releases later.
  fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

- workaround 1 - stick with version 3.5

- workaround 2 - use a specific backend for matplotlib https://matplotlib.org/stable/users/explain/backends.html
```
import matplotlib
matplotlib.use('macosx')
```
This opens an external window to show plots and blocks the index.py script from further execution until external window is closed.

TODO
- find out what mode PyCharm is using to show plots within the IDE in version 3.5 and not block further execution