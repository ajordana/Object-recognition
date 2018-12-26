# Object_recognition

Final project -- Object recognition


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
