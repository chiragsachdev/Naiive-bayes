import math, os, pickle, glob, re
# from pathlib import Path

class Bayes_Classifier:
    def __init__(self, train_dir="training"):
        # initialize elements for constructor for object variables
        self.features={}
        self.features["pos"]={}
        self.features["neg"]={}
        self.features["neu"]={}
        self.likelihood={}
        self.likelihood["pos"]={}
        self.likelihood["neg"]={}
        self.likelihood["neu"]={}
        self.posCount=0
        self.negCount=0
        self.neuCount=0
        self.total=0

    def train(self):
        # training for classifier
        if os.path.isfile("features.bin"):
            self.features=self.loadFile("features.bin")
            self.likelihood=self.loadFile("likelihood.bin")
            count=self.loadFile("count.bin")
            self.posCount=count[0]
            self.negCount=count[1]
            self.neuCount=count[2]
        else:
            path="training"
            filelst=[]
            print("Training the Algorithm...")
            counter=0
            for fl in glob.glob(os.path.join(path,"*.txt")):
                # print("2")
                # print(fl)
                filelst.append(fl)
                fileText=self.readFile(fl)
                fname=fl.split("-")
                ftokens=self.tokenize(fileText)
                if (fname[1]=="1" or fname[1]=="2"):
                    self.negCount+=1
                    for tok in ftokens:
                        if tok in self.features["neg"]:
                            self.features["neg"][tok]+=1
                        else:
                            self.features["neg"][tok]=1
                elif (fname[1]=="4" or fname[1]=="5"):
                    self.posCount+=1
                    for tok in ftokens:
                        if tok in self.features["pos"]:
                            self.features["pos"][tok]+=1
                        else:
                            self.features["pos"][tok]=1
                elif fname[1]=="3":
                    self.neuCount+=1
                    for tok in self.tokenize(fileText):
                        if tok in self.features["neu"]:
                            self.features["neu"][tok]+=1
                        else:
                            self.features["neu"][tok]=1
            
            print("Training Done!\n")
            self.likelihood["pos"]=dict([val,math.log((self.features["pos"][val]/self.posCount))] for val in self.features["pos"])
            self.likelihood["neg"]=dict([val,math.log((self.features["neg"][val]/self.negCount))] for val in self.features["neg"])
            self.likelihood["neu"]=dict([val,math.log((self.features["neu"][val]/self.neuCount))] for val in self.features["neu"])
            self.saveFile([self.posCount,self.negCount,self.neuCount],"count.bin")
            self.saveFile(self.features,"features.bin")
            self.saveFile(self.likelihood,"likelihood.bin")
        return
        


    def classify(self, input_vector):
        # deciding the outcome of the given input
        iv_tok=self.tokenize(input_vector)
        totalCount = self.negCount+self.posCount+self.neuCount
        posLikelihood = math.log(float(self.posCount/totalCount))
        negLikelihood = math.log(float(self.negCount/totalCount))
        result=""
        for tok in iv_tok:
            if tok in self.likelihood["pos"]:
                posLikelihood+=self.likelihood["pos"][tok]
            elif tok not in self.likelihood["pos"]:
                posLikelihood+=math.log(0.000001)
            elif tok in self.likelihood["neg"]:
                negLikelihood+=self.likelihood["neg"][tok]
            elif tok not in self.likelihood["neg"]:
                negLikelihood+=math.log(0.000001)
        if posLikelihood > negLikelihood:
            result = "Positive"
        elif negLikelihood > posLikelihood:
            result = "Negative"
        elif posLikelihood == negLikelihood:
            result = "Neutral"

        return result

    def readFile(self,fileName):
        # load the string from a file
        fp = open(fileName,"r")
        text=fp.read()
        fp.close
        return text

    def saveFile(self, data, fileName):
        # write file onto disk
        fp = open(fileName, "wb")
        p = pickle.Pickler(fp)
        p.dump(data)
        fp.close()
        return

    def loadFile(self, fileName):
        # load file from disk
        fp = open(fileName,"rb")
        u = pickle.Unpickler(fp)
        data = u.load()
        fp.close()
        return data

    def tokenize(self, text):
        # return list of tokens from given string
        lTokens = []
        sToken = ""
        for c in text:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == "-":
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))
                
        if sToken != "":
            lTokens.append(sToken)

        return lTokens