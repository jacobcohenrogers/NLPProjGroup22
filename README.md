# NLPProjGroup22
## Authorship Attribution of Poems using NLP techniques
Our project explores using a variety of NLP techniques to attribute authorship of a set of poems. We used poems from 5 authors - Emily Dickinson, Robert Frost, Robert Burns, Shakespeare, and Walt Whitman. These authors were chosen because of their different writing styles. 

## Approaches -
- Stylometric features + Random Forest/Logistic Regression
- LSTM Model with GloVe Embeddings
- Fine-tuned BERT transformer

**NOTE** - We did most of the first approach in files in this github. The LSTM and Transformer were created and run on a google colab file that can be found here - 
- https://colab.research.google.com/drive/1gOfOBbvk9ZL3EyNfIBV1_uQvyekaprvj?usp=sharing
This links to a colab sheet where we ran and analyzed each of our models. We have included a pdf of the colab sheet at the time of the due date as a pdf and ipynb file in this repo. (**NEED COLAB PRO** to have enough RAM for the transformer)

## Installation - 
** No installation is necessary, the Colab file linked above works with the github directly to import the data when the relevant cells are run**

## Team members - 
Robert McDonald, Avalyn Mullikin, Jacob Rogers, V Verity

## Dataset
Poems were sourced from a variety of online libraries. Data was manually cleaned and split by us. 
- Poems by Robert Burns - https://www.gutenberg.org/ebooks/1279
- Poems by Shakespeare - https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
- Poems by Walt Whitman - https://www.gutenberg.org/ebooks/8388
- Poems by Emily Dickinson - https://www.gutenberg.org/ebooks/12242
- Poems by Robert Frost - https://www.gutenberg.org/ebooks/59824

## File overview 
- NLP_Group22_Project.ipynb - the ipynb file of our execution of the inital approaches, the LSTM, and the Transformer
- NLP_Group22_Project.ipynb - Colab.pdf - The pdf of the ipynb file of the due date to show that it was completed on time
- README.md - Readme
- character_prompts.sty - old starter data, irrelevant
- **data**
  - **raw_poems** - folder of the raw txt files of compilations of poems by each author
  - **raw_text** - folder of the split poems after processing
    - testing - folder of poem data for testing
      - Emilydickinson, Frost, Robertburns, Shakespeare, Waltwhitman - *subfolders containing the split poems by their respective author*
    - training - folder of poem data for training
      - Emilydickinson, Frost, Robertburns, Shakespeare, Waltwhitman - *subfolders containing the split poems by their respective author*
- **old** - **irrelevant** folder containing old test data use to set up the project
- **src**
  - **poem_splitters** - folder of the python code we used to split the poem compilations into their own txt files. Each author has their own custom splitting method, due to differences in the txt files of their individual poem compilations
  - convert_to_text.py - **irrelevant** old split method
  - create_mainfest.py - creates a csv that maps every poem files to its author and whether its in the train or test data. This structures the data for the learning models
  - feature_extraction.py - pulls usefule feautures from the poems to feedd to models like random forest or logistic regression. It grabs writing style stats, character patterns, parts of speech, and sentence embeddings
  - inspection.py - Loads saved models, prints out classification reports. Used for Random forest and Logistic Regression
  - modeling.py - trains logistic regression and random forest models on train set, tests them on tests set
  - preprocessing.py - handles cleaning up the raw poem text before feature extraction and modeling.

## Limitations

- Small dataset size limits generalization, especially for deep learning models.
- Poetry formatting required custom preprocessing for each author.
- Some models struggled with overfitting due to the limited number of training samples.

## Acknowledgements

- HuggingFace Transformers for BERT modeling
- NLTK and scikit-learn for preprocessing and classical models
- Project Gutenberg and MIT OCW for public domain poetry datasets
