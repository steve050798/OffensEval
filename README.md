# NIT_Agartala_NLP_Team at SemEval-2019 Task 6 
System Submission for SemEval Task 6: OffensEval 2019 (https://competitions.codalab.org/competitions/20011)

Abstract:
Developed an Ensemble Approach (Vote based) Classifier for Offensive Language detection trained on the OLID dataset (https://scholar.harvard.edu/malmasi/olid). Also includes a simple LSTM network to compare performance with DLL methods

Files:
  1. proto.py - Ensemble model approach
  2. LSTM.ipynb - Deep Learning Approach (Rudimentary Model)
  
Resources Required:
  1. CMU POS Tagger (http://www.cs.cmu.edu/~ark/TweetNLP/)
  2. OLID Training Data (https://scholar.harvard.edu/malmasi/olid)
  3. GLoVe Embeddings (Current Version uses the Twitter, 2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vector variant)for LSTM (https://nlp.stanford.edu/projects/glove/)

Getting the code Ready:
1. Set the global variables filename and test_filename to your dataset paths
2. Download ark tweet nlp and extract into the code location (https://bit.ly/33x2WJT) also download the python wrapper (https://github.com/ianozsvald/ark-tweet-nlp-python/blob/master/CMUTweetTagger.py) (Used as library) 

Changing Subtasks:
Perform the following Changes to run different subtasks
  1. Change X,y= DataPreprocessing(data_q) replace 'q' with the subtask required
  2. Change the test_filename if performing submission prediction

Detatiled System Description:
https://www.aclweb.org/anthology/papers/S/S19/S19-2124/
