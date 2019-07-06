from bayes_train import Bayes_Classifier
import glob,os
bobj = Bayes_Classifier()
bobj.train()
def evaluate():
    path="testing"
    fileList=[]
    true_pos = 0
    true_neg = 0
    true_neu = 0
    pred_true_pos = 0
    pred_true_neg = 0
    pred_true_neu = 0
    pred_false_pos = 0
    pred_false_neg = 0
    pred_false_neu = 0
    total_predictions = 0
    for fl in glob.glob(os.path.join(path,"*.txt")):
        # fileList.append(file)
        total_predictions += 1
        fname = fl.split("-")
        if fname[1] =="1" or fname[1] == "2":
            true_neg+=1
        elif fname[1] == "4" or fname == "5":
            true_pos+=1
        else:
            true_neu+=1

        result = bobj.classify(bobj.readFile(fl))
        if result == "Positive" and (fname[1] == "4" or fname == "5"):
            pred_true_pos+=1
        elif result == "Positive" and not (fname[1] == "4" or fname == "5"): 
            pred_false_pos+=1
        elif result == "Negative" and (fname[1] == "1" or fname == "2"):
            pred_true_pos+=1
        elif result == "Negative" and not (fname[1] == "1" or fname == "2"):
            pred_false_pos+=1
        elif result == "Neutral" and (fname[1] == "3"):
            pred_true_neu+=1
        elif result == "Positive" and not (fname[1] == "3"):
            pred_false_neu+=1

    TP = pred_true_neg + pred_true_neu + pred_true_pos
    FP = pred_false_neg + pred_false_neu + pred_false_pos
    
    print("True classifications: ",TP)
    print("False classifications: ",FP)
    return

def test():
    vect = input("Enter text to classify: ")
    result = bobj.classify(vect)

    print("The input text is ", result)
    return

def main():
    while(True):
        try:
            choice = input("Enter 1 to check the efficiency of the algorithm\nEnter 2 to test a new string\nEnter 3 to exit\n")
            if choice == "1":
                evaluate()
            elif choice == "2":
                test()
            elif choice == "3":
                break
            else:
                print("Invalid entry\nTry again ....\n")
        except (KeyboardInterrupt,SystemExit):
            print("\nExiting ...\n")
            break
if __name__ == "__main__":
    main()