from bestbayes import Bayes_Classifier
import glob, os
bobj = Bayes_Classifier()

bobj.train()

# st=bobj.loadFile("movies_reviews/movies-1-.txt")
# bobj.classify("it was not crap")
# stopwords=bobj.tokenize(bobj.loadFile("stopwords.txt"))
# print(stopwords)
path="test/"
# fname=os.listdir(path)
lFileList = []
# for fl in glob.glob(os.path.join(path,'*.txt')):
#     lFileList.append(fl)
# fnames=[]
# ftokens=[]
actual_pos=0
actual_neg=0
actual_neu=0

predicted_false_pos=0
predicted_false_neg=0
predicted_false_neu=0
predicted_true_pos=0
predicted_true_neg=0
predicted_true_neu=0
counter=0
for fl in glob.glob(os.path.join(path,'*.txt')):
    print(counter)
    fname=fl.split("-")
    if(fname[1] == "1") or (fname[1]=="2"):
        actual_neg+=1
    elif(fname[1] == "4") or (fname[1]=="5"):
        actual_pos+=1
    else:
        actual_neu+=1
    
    result=bobj.classify(bobj.loadFile(fl))
    if result=="positive" and (fname[1] == "4") or (fname[1]=="5"):
        predicted_true_pos+=1
    elif (result=="positive" and (not (fname[1] == "4") or (fname[1]=="5"))):
        predicted_false_pos+=1
    elif result=="negative" and (fname[1] == "1") or (fname[1]=="2"):
        predicted_true_neg+=1
    elif result=="negative" and ( not (fname[1] == "1") or (fname[1]=="2")):
        predicted_false_neg+=1
    elif(result=="neutral" and (fname[1]=="3")):
        predicted_true_neu+=1
    elif(result=="neutral" and (not fname[1]=="3")):
        predicted_false_neu+=1
    counter+=1


# print("Actual positive: ",actual_pos,"\t Predicted tue_positive",predicted_true_pos,"\t Predicted false positive",predicted_false_pos)
# print("Actual negative: ",actual_neg,"\t Predicted tuue negative",predicted_true_neg,"\t Predicted false negative",predicted_false_neg)
# print("Actual neutral: ",actual_neu,"\t Predicted true neutral",predicted_true_neu,"\t Predicted false neutral",predicted_false_neu)
true_predictions= predicted_true_neg+predicted_true_neu+predicted_true_pos
false_predictions=predicted_false_neg+predicted_false_neu+predicted_false_pos
# print("True predictions: ",true_predictions)
# print("False predictions:",false_predictions)
total_predictions= actual_neg+actual_neu+actual_pos
precision=float(true_predictions/(true_predictions+predicted_false_pos))*100
recall=float(true_predictions/(true_predictions+predicted_false_neg))*100
fmeasure=float(2*(precision*recall)/(precision+recall))
accuracy=float(true_predictions/total_predictions)
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-measure: ",fmeasure)