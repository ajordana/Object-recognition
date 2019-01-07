# Object_recognition

Final project -- Object recognition

The aim of this project is to investigate a joint representation for image and text in order to automatize the text description of an image. In particular, we will study the Deep Sementic Matching (Deep SM) and the canonical correlation analysis (CCA) in order to create a multimodal retrieval : Image-to-Image search, Tag-to-Image search, and Image-to-Tag search. We will experiment those methods on the Pascal Sentence Dataset http://vision.cs.uiuc.edu/pascal-sentences/.


### Create conda environment :
conda env create -f environment.yml

### Load data :
git clone https://github.com/rupy/PascalSentenceDataset.git

python pascal_sentence_dataset.py

### train/test split
python traintestsplit.py

### Extract visual features
python cnn_features.py

### Extract text Lda
python text_LDA.py

### Extract text features
python text_features.py

### run both pipelines
python text2images.py
