
#Author : Somaiah Thimmaiah Balekuttira (4-oct-2017)


import numpy as np
from nltk.corpus import stopwords
import re






#feature_extraction is used to extract the features. Using a unigram model with bag of words
def feature_extraction():
    
    #shuffle_data is used to shuffle the data after splitting the data into test and train 
    
    
    
    #parse is used to parse the sentences into individual words
    def parse(sentence):
        return re.compile('\w+').findall(sentence)
   
    train_sentences=[]
    labels=[]
    bagofwords=[]
    sentences=[]
    
    #add the file name of the training data below
    train_sen=open('sentences.txt','r')
    
    #add the file name of the training labels data below

    train_label=open('labels.txt','r')	    

    #add the file name of the testing data below
    test_sen=open('test_sentences.txt','r')

    
    #add the file name of the testing labels below
    lab=open('test_labels.txt','r')
    
    for sentence in test_sen:
        sentences.append(parse(sentence))
    
    for sentence in train_sen:
        train_sentences.append(parse(sentence))
     
    for label in lab:
        labels.append(label.strip())
        
    labels=np.matrix(labels).astype(int)
    labels=np.matrix.transpose(labels)
    

    #making bag of words from the training data (sentences.txt)
    for i in range(len(train_sentences)):
        for word in train_sentences[i]:
            if word not in stopwords.words("english"):
                bagofwords.append(word)
    bagofwords=set(bagofwords)

    word2int={}
    for i,word in enumerate(bagofwords):
           word2int[word] = i

   #making feature vector matrix     
    w, h = len(bagofwords), len(sentences)

    Matrix = [[0 for x in range(w)] for y in range(h)] 

    for i in range(len(sentences)): 
        for word in bagofwords: 
            if word in sentences[i]: 
                    Matrix[i][word2int[word]]=sentences[i].count(word)
    
    test_features=np.array(Matrix)
    test_labels=np.array(labels) 
    
    return test_features,test_labels
    
    
    






#This function shuffles and splits data into testing and training 
def split_data():
    def shuffle_data(matrix, labels):
        assert len(matrix) == len(labels)
        q = np.random.permutation(len(matrix))
        return matrix[q], labels[q]
    
    features,labels=shuffle_data(feat,labels)
    
    #split 70% train 30% test
    index=int(len(features)*0.7)
    training_features, test_features = features[:index,:], features[index:,:]
    training_labels, test_labels=labels[:index,:], labels[index:,:]





#evalution function 
def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)





#This function trains the model and returns the input and output layer weights

def perceptron_weight_train(features,labels):
    def predict(vector,weights):
        
         activation=np.matmul(vector,np.matrix.transpose(weights))
        
         if (activation>0):
               return 1
         else:
               return -1
    w=np.zeros(shape=(features.shape[1],1))
    weights=np.transpose(w)
    labels[labels==0]=-1  
    epoch=0 
     
    while(epoch!=150):
        for i in range(features.shape[0]):
            predicts=predict(features[i],weights)
            if(predicts!=labels[i]):
                    temp=features[i]
                    temp=np.reshape(temp,(1,weights.shape[1]))
                    weights+=np.matmul(labels[i],temp)
                
         
        epoch+=1
        
    #saving the trained model weights in percep_weight file      
    np.save("percep_weight.npy",weights)    
    return weights




#predict_perceptron predicts the labels of unseen testing data
def predict_perceptron(test_features,weights):
    
    def predict(vector,weights):
        
        activation=np.dot(vector,weights.T)
        
        if (activation>0):
               return 1
        else:
               return -1
            
    predicted =[]

    for i in range(test_features.shape[0]):
            predicts=predict(test_features[i],weights)
            if(predicts==1):
	    	predicted.append(predicts)
	    elif(predicts==-1):
		temp=0
		predicted.append(temp)	 
    #returns the predicted labels        
    return predicted





# Entry POINT MAIN CODE starts from here

#feature extration from unseen testing data
test_features,test_labels=feature_extraction()


#loading the weights for perceptron from the file percep.weight.npy which was found after training the model on the training data on sentences.txt
perceptron_weights = np.load('percep_weight.npy')

#percep_predicted has the final predicted labels for the unseen testing data
percep_predicted=predict_perceptron(test_features,perceptron_weights)
precision_percep , recall_percep , f1_percep = evaluate(percep_predicted, test_labels)
print "Perceptron results", precision_percep, recall_percep, f1_percep
