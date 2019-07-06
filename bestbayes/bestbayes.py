import math, os, pickle, re, glob, math

class Bayes_Classifier:

   def __init__(self, trainDirectory = "movie_reviews/"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text.'''
      self.features={}
      self.features["pos"]={}
      self.features["neg"]={}
      self.features["neu"]={}
      self.likelihood={}
      self.likelihood["pos"]={}
      self.likelihood["neg"]={}
      self.likelihood["neu"]={}
      self.p=0
      self.n=0
      self.neu=0
      self.stopwords=self.tokenize(self.loadFile("stopwords.txt"))

   def train(self):   
      '''Trains the Naive Bayes Sentiment Classifier.'''

      if(os.path.isfile("features.txt")):
            self.features=self.load("features.txt")
            count=self.load("count.txt")
            self.likelihood=self.load("likelihood.txt")
            self.p=int(count[0])
            self.n=int(count[1])
        
      else:
         # stopwords=self.tokenize(self.loadFile("stopwords.txt"))
         path="movies_reviews/"
         # fname=os.listdir(path)
         lFileList = []
         for fl in glob.glob(os.path.join(path,'*.txt')):
            lFileList.append(fl)
         # fnames=[]
         # ftokens=[]
         for fl in lFileList:
            s=self.loadFile(fl)
            fname=fl.split("-")
            ftokens=self.tokenize(s)
            if(len(ftokens)<5):
               continue
            if(fname[1] == "1") or (fname[1]=="2"):
               self.n+=1
               for word in ftokens:
                  if word in self.stopwords:
                      continue
                  if word not in self.features["neg"]:
                     self.features["neg"][word]=1
                  else:
                     self.features["neg"][word]+=1
            elif(fname[1] == "4") or (fname[1]=="5"):
               self.p+=1
               for word in ftokens:
                  if word in self.stopwords:
                    continue
                  if word not in self.features["pos"]:
                     self.features["pos"][word]=1
                  else:
                     self.features["pos"][word]+=1
            else:
               self.neu+=1
               for word in ftokens:
                  if word in self.stopwords:
                    continue
                  if word not in self.features["neu"]:
                     self.features["neu"][word]=1
                  else:
                     self.features["neu"][word]+=1
                   
         # for word in self.features["pos"]:
         #    if self.features["pos"][word]<5:
         #       temp=self.features["pos"].pop(word)
               
         # for word in self.features["neg"]:
         #    if self.features["neg"][word]<5:
         #       temp=self.features["neg"].pop(word)
         
         # for word in self.features["neu"]:
         #    if self.features["neu"][word]<5:
         #       temp=self.features["neu"].pop(word)
         self.likelihood["pos"]=dict([val,math.log(self.features["pos"][val]/len(self.features["pos"]))] for val in self.features["pos"])
         self.likelihood["neg"]=dict([val,math.log(self.features["neg"][val]/len(self.features["neg"]))] for val in self.features["neg"])
         self.likelihood["neu"]=dict([val,math.log(self.features["neu"][val]/len(self.features["neu"]))] for val in self.features["neu"])
         self.save([self.p,self.n],"count.txt")
         self.save(self.features,"features.txt")
         self.save(self.likelihood,"likelihood.txt")
         # print(self.likelihood["pos"])



   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
      total=self.p+self.n
      prob_pos=math.log(self.p/total)
      prob_neg=math.log(self.n/total)
      # prob_neu=math.log(self.neu/total)
      prob_classify_neg=prob_neg
      prob_classify_pos=prob_pos
      # prob_classify_neu=prob_neu
      flist=self.tokenize(sText)
      
      #positive probability
      for fl in flist:
         if fl in self.stopwords:
            continue
         if fl not in self.likelihood["pos"]:
            prob_classify_pos+=0
            # continue
         else:
            if self.features["pos"][fl]<5:
               continue
            prob_classify_pos+=self.likelihood["pos"][fl]

      #negative probability
      for fl in flist:
         if fl in self.stopwords:
            continue
         if fl not in self.likelihood["neg"]:
            prob_classify_neg+=0
            # continue
         else:
            if self.features["neg"][fl]<5:
               continue
            prob_classify_neg+=self.likelihood["neg"][fl]

      #neutral probability
      # for fl in flist:
      #    if fl not in self.likelihood["neu"]:
      #       prob_classify_neu+=0
      #    else:
      #       prob_classify_neu+=self.likelihood["neu"][fl]
      # print("positive: ",prob_neg)
      # print("negative",prob_pos)
      # print ("prob pos",prob_classify_pos)
      # print("prob neg",prob_classify_neg)
      prob_classify_neg-=prob_neg
      prob_classify_pos-=prob_pos

      result=""
      if prob_classify_neg<prob_classify_pos:
         result="positive"
      elif prob_classify_neg>prob_classify_pos:
         result="negative"
      else:
         result="neutral"
      # print("...")

      return result


   def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "wb")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "rb")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      '''Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order).'''

      lTokens = []
      sToken = ""
      for c in sText:
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
