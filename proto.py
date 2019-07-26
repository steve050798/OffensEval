#GENERAL LIBRARIES
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd

#TWEET POS TAGGER
import CMUTweetTagger

#DENSE FEATURES
from textblob import TextBlob
from scipy import sparse
from pyphen import Pyphen


#FOR ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize


#VECTORIZERS
count_vect=None
tfidf_transformer=None

#DATA SET FILE
filename = "Datasets/offenseval-training-v1.tsv"
test_filename = 'Datasets/testset-taska.tsv'
header = ['id','tweet','subtask_a','subtask_b','subtask_c']
docCount=0
backup=None
data_a=None
data_b=None
data_c=None

#OUTPUT FILE
output_file=open("Output","w")


#DICTIONARY OF MODELS BEING USED
models = {"Linear Support Vector Classifier": LinearSVC(random_state=0,class_weight='balanced'),
	"Logistic Regression L1": LogisticRegression(penalty='l1',random_state=0,solver='saga',class_weight='balanced'),
	"Logistic Regression L2": LogisticRegression(penalty='l2',random_state=0,solver='lbfgs',max_iter=250,class_weight='balanced'),
	"PA Classifier": PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3,class_weight='balanced'),
	"SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3,class_weight='balanced'),
	"Ridge Classifier": RidgeClassifier(class_weight='balanced')
	}

def readData():
	global filename,header,data_a,data_b,data_c,backup

	#REMOVING NOISE
	#open("training.tsv","w").write(open(filename,"r").read().replace("URL","").replace("@USER",""))

	#CREATING A PANDAS DATAFRAME
	data = pd.read_table(filename, names=header)
	backup=data
	data_a = data[['tweet','subtask_a']]
	data_b = data[['tweet','subtask_b']]
	data_c = data[['tweet','subtask_c']]
	data_b = data_b[data.subtask_b != 'NULL']
	data_c = data_c[data.subtask_c != 'NULL']
	data_b = data_b.dropna()
	data_c = data_c.dropna()
	data_b = data_b.reset_index(drop=True)
	data_c = data_c.reset_index(drop=True)
	data_a['subtask'] = data_a['subtask_a']
	data_a = data_a[['tweet','subtask']]
	data_b['subtask'] = data_b['subtask_b']
	data_b = data_b[['tweet','subtask']]
	data_c['subtask'] = data_c['subtask_c']
	data_c = data_c[['tweet','subtask']]


def DataPreprocessing(data, train=1):
	
	global docCount

	#EXTRACTING DENSE FEATURES
	sentiment=np.array([])
	word_count=np.array([])
	char_count=np.array([])
	sent_count=np.array([])
	syl_count=np.array([])
	mention_count=np.array([])
	url_count=np.array([])
	special_count=np.array([])
	cat_count=np.array([])
	dic=Pyphen(lang='en')
	for text in data["tweet"]:
		blob=TextBlob(text)

		#OPTIONAL SPELLING CORRECTION
		#data.loc[docCount,"tweet"]=str(blob.correct())
		#print(data.loc[docCount,"tweet"],type(data.loc[docCount,"tweet"]))
		
		url_count=np.append(url_count,blob.words.count("URL"))
		mention_count=np.append(mention_count,blob.words.count("USER"))
		cat_count=np.append(cat_count,sum(c=='#' for c in text))
		special_count=np.append(special_count,sum(not c.isalnum() and c!=' ' and c!='@' and c!='#' for c in text))
		syl_count=np.append(syl_count,len(TextBlob(dic.inserted(text).replace('-',' ')).words))
		char_count=np.append(char_count,len(text))
		word_count=np.append(word_count,len(blob.words))
		sent_count=np.append(sent_count,len(blob.sentences))
		sentiment=np.append(sentiment,blob.sentiment.polarity)
		docCount+=1
	
	#INITIALIZING STEMMER AND STOP WORD CORPUS	
	stop_words=set(stopwords.words('english'))
	porter_stemmer = PorterStemmer()

	#POS TAGGING
	POS=CMUTweetTagger.runtagger_parse(data["tweet"])
	POSDictionary={"N":"nn","O":"pro","S":"np","^":"nnps","Z":"nnpz","L":"vl","M":"nv","V":"md","A":"adj",
				"R":"adv","!":"int","D":"det","P":"ppt","&":"cc","T":"rp","X":"ex","Y":"exv","#":"cat",
				"@":"tar","~":"dsc",",":"punc","$":"num","U":"url","E":"emo","G":"abr"}
	
	#PREPROCESSING (REMOVE STOP WORDS AND STEMMING)
	docCount=0
	for doc in POS:
		filtered_sentence=[]
		for word in doc:
			if word[0] not in stop_words:
				filtered_sentence.append(porter_stemmer.stem(word[0]))#+'_'+POSDictionary[word[1]])
		data.loc[docCount,"tweet"]=filtered_sentence
		data.loc[docCount,"tweet"]=" ".join(data.loc[docCount,"tweet"])
		docCount+=1

	#REPLACING LABEL (subtask) WITH INTEGER
	if(train==1):
		data['label'] = data['subtask'].factorize()[0]
	data['sentiment'] = sentiment + 1
	data['sent_count'] = sent_count
	data['word_count'] = word_count
	data['syl_count'] = syl_count
	data['url_count'] = url_count
	data['mention_count'] = mention_count
	data['cat_count'] = cat_count
	data['special_count'] = special_count
	
	#SEPERATING FEATURES AND LABELS
	X=data[['tweet','sentiment','sent_count','word_count','syl_count','url_count','mention_count','special_count','cat_count']]
	if train==1:
		y=data['label']
	else:
		y=None
	return X,y

def WriteMetrics(model,accuracy,conf_mat):
	global output_file
	output_file.write("\n-------------------------------------------------------------------------\n")
	output_file.write(model+"\n")
	output_file.write("Accuracy= "+str(accuracy)+"\n")
	output_file.write("Confusion Matrix= "+"\n")
	output_file.write(str(conf_mat)+"\n")
	TP=conf_mat[0][0]
	FN=conf_mat[0][1]
	FP=conf_mat[1][0]
	TN=conf_mat[1][1]
	recall_off=TP/(TP+FN)
	recall_not=TN/(TN+FP)
	precision_off=TP/(TP+FP)
	precision_not=TN/(TN+FN)
	Fscore_off=2*recall_off*precision_off/(recall_off+precision_off)
	Fscore_not=2*recall_not*precision_not/(recall_not+precision_not)
	Fscore_avg=(Fscore_off*(TP+FN)+Fscore_not*(FP+TN))/(FP+TP+TN+FN)
	metric_dict={"Precision":[precision_off,precision_not],
				"Recall":[recall_off,recall_not,],
				"Fscore":[Fscore_off,Fscore_not],"Label":["OFF","NOT"]}
	metric_dataframe=pd.DataFrame(metric_dict)
	metric_dataframe.set_index("Label",inplace=True)
	output_file.write(str(metric_dataframe))
	output_file.write("\nWeighted Fscore: "+str(Fscore_avg))
	#output_file(metrics.classification_report(y_test, y_pred, 
	#                                   target_names=data['subtask'].unique()))
	output_file.write("\n-------------------------------------------------------------------------\n")


def DisplayMetrics(model,accuracy,conf_mat):
	print("-------------------------------------------------------------------------")
	print(model)
	print("Accuracy=",accuracy)
	print("Confusion Matrix: ")
	print(conf_mat,"\n")
	TP=conf_mat[0][0]
	FN=conf_mat[0][1]
	FP=conf_mat[1][0]
	TN=conf_mat[1][1]
	recall_off=TP/(TP+FN)
	recall_not=TN/(TN+FP)
	precision_off=TP/(TP+FP)
	precision_not=TN/(TN+FN)
	Fscore_off=2*recall_off*precision_off/(recall_off+precision_off)
	Fscore_not=2*recall_not*precision_not/(recall_not+precision_not)
	Fscore_avg=(Fscore_off*(TP+FN)+Fscore_not*(FP+TN))/(FP+TP+TN+FN)
	metric_dict={"Precision":[precision_off,precision_not],
				"Recall":[recall_off,recall_not,],
				"Fscore":[Fscore_off,Fscore_not],"Label":["0","1"]}
	metric_dataframe=pd.DataFrame(metric_dict)
	metric_dataframe.set_index("Label",inplace=True)
	print(metric_dataframe)
	print("\nWeighted Fscore: "+str(Fscore_avg))
	#print(metrics.classification_report(y_test, y_pred, 
	#                                    target_names=data['subtask'].unique()))
	print("-------------------------------------------------------------------------\n")

def FeatureSelection(X,type,range):
	#VECTORIZING THE DATA
	global count_vect, tfidf_transformer
	count_vect = CountVectorizer(analyzer=type,stop_words="english",ngram_range=range,lowercase=True,max_features=None)
	X_counts = count_vect.fit_transform(X)
	open("Vocab","w").write("\n".join(count_vect.get_feature_names()))
	tfidf_transformer = TfidfTransformer()
	X_tfidf = tfidf_transformer.fit_transform(X_counts)
	return X_tfidf


def TrainML(X,y):

	global models,backup, test_filename

	#CROSS VALIDATION MODEL
	#HOLD OUT METHOD
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

	#K-FOLD CROSS VALIDATION
	noOfFolds=10
	kfold=StratifiedKFold(noOfFolds, True, 1)
	
	#FEATURE SELECTION
	dense_features = np.hstack((np.array(X['sentiment']).reshape((docCount,1)),
								np.array(X['sent_count']).reshape((docCount,1)), 
								np.array(X['word_count']).reshape((docCount,1)),
								np.array(X['url_count']).reshape((docCount,1)),
								np.array(X['mention_count']).reshape((docCount,1)),
								np.array(X['special_count']).reshape((docCount,1)),
								np.array(X['cat_count']).reshape((docCount,1)),
								np.array(X['syl_count']).reshape((docCount,1))
								))
	dense_features=normalize(dense_features)
	X=X['tweet']
	dense_features=sparse.csr_matrix(dense_features)
	X_tfidf=FeatureSelection(X,'word',(1,3))
	features=sparse.hstack([X_tfidf,dense_features])
	features=features.tocsr()
	
	#TRAINING ALL MODELS
	for model in models.keys():
		accuracy=0
		conf_mat=np.array([[0,0],[0,0]])
		#TRAINING
		for train_index,test_index in kfold.split(features,y):
			X_train, X_test = features[train_index], features[test_index]
			y_train, y_test = y[train_index], y[test_index]


			hyp = models[model].fit(X_train, y_train)

			#PREDICTING unseen
			y_pred=hyp.predict(X_test)
			
			'''
			#SAVING PREDICTIONS
			count=0
			true_positive=open("True_Positive","w")
			true_negative=open("True_Negative","w")
			false_positive=open("False_Positive","w")
			false_negative=open("False_Negative","w")
			for i in test_index:
				if y_pred[count]==0 and y_test[i]==1:
					false_positive.write(backup['tweet'][i])
					false_positive.write(" "+str(i))
					false_positive.write("\n\n")
				elif y_pred[count]==1 and y_test[i]==0:
					false_negative.write(backup['tweet'][i])
					false_negative.write(" "+str(i))
					false_negative.write("\n\n")
				elif y_pred[count]==0 and y_test[i]==0:
					true_positive.write(backup['tweet'][i])
					true_positive.write(" "+str(i))
					true_positive.write("\n\n")
				else:
					true_negative.write(backup['tweet'][i])
					true_negative.write(" "+str(i))
					true_negative.write("\n\n")
				count+=1
			'''
			
			#CALCULATING ACCURACY AND CONFUSION MATRIX
			accuracy+=metrics.accuracy_score(y_test,y_pred)
			conf_mat+=metrics.confusion_matrix(y_test, y_pred)
		
		accuracy/=noOfFolds
		conf_mat=conf_mat/noOfFolds
		DisplayMetrics(model,accuracy,conf_mat)
		#WriteMetrics(model,accuracy,conf_mat)
	

	y_pred=np.array([])
	index=0
	for i in range(len(y_test)):
		votes = np.array([0,0])
		for model in models.keys():
			votes[models[model].predict(X_test[index])]+=1
		y_pred=np.append(y_pred,np.argmax(votes))
		index+=1
	accuracy = metrics.accuracy_score(y_test,y_pred)
	conf_mat = metrics.confusion_matrix(y_test, y_pred)
	DisplayMetrics("Ensemble model",accuracy,conf_mat)
	WriteMetrics("Ensemble model",accuracy,conf_mat)

	# #WRITING DATA FOR SUBMISSION
	# dataTest= pd.read_table(test_filename)
	# backup=dataTest
	# dataTest=dataTest[["tweet"]]
	# print(type(dataTest))
	# X,y=DataPreprocessing(dataTest,0)
	# dense_features = np.hstack((np.array(X['sentiment']).reshape((docCount,1)),
	# 							np.array(X['sent_count']).reshape((docCount,1)), 
	# 							np.array(X['word_count']).reshape((docCount,1)),
	# 							np.array(X['url_count']).reshape((docCount,1)),
	# 							np.array(X['mention_count']).reshape((docCount,1)),
	# 							np.array(X['special_count']).reshape((docCount,1)),
	# 							np.array(X['cat_count']).reshape((docCount,1)),
	# 							np.array(X['syl_count']).reshape((docCount,1))
	# 							))
	# dense_features=normalize(dense_features)
	# X=X['tweet']
	# dense_features=sparse.csr_matrix(dense_features)
	# X_counts=count_vect.transform(X)
	# X_tfidf=tfidf_transformer.transform(X_counts)
	# features=sparse.hstack([X_tfidf,dense_features])
	# features=features.tocsr()
	# y_pred=np.array([])
	# print(type(X_test))
	# index=0
	# for i in range(860):
	# 	votes = np.array([0,0])
	# 	for model in models.keys():
	# 		votes[models[model].predict(features[index])]+=1
	# 	y_pred=np.append(y_pred,np.argmax(votes))
	# 	index+=1
	# y_pred=list(y_pred) 
	# #PREDICT ALL TWEETS
	# #CONVERT PREDICTION TO ORIGINAL LABELS
	# labelDict={0:'OFF',1:'NOT'}
	# y_pred=np.array([labelDict[i] for i in y_pred]).reshape((docCount,1))
	# submission=pd.DataFrame(np.hstack((np.array(backup["id"]).reshape((docCount,1)),y_pred)))	#CREATE A PANDAS DATAFRAME
	# submission.to_csv("submission.csv", encoding='ascii',index=False,header=False)	#STORE CSV WITHOUT HEADER AND INDEX



if __name__ == '__main__':
	global docCount, data_a, data_b, data_c
	readData()
	X,y=DataPreprocessing(data_a)
	TrainML(X,y)
	print(data_a.head())
	#print(data_b.head())
	#print(data_c.head())

