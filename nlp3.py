# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:25:02 2019

@author: Barış
"""
import dynet_config
dynet_config.set_gpu()
import dynet as dy
import random

def defineWordVector(tempWord,wordVector):
    curVector = []

    for i in range(0,len(wordVector)):
        if wordVector[i] == tempWord:
            curVector.append(1)
        else:
            curVector.append(0)

    return curVector

f = open('trumpspeeches.txt','r+')
trumph = f.read()
f.close()
wordList = trumph.split(' ')
wordList = [temp for temp in wordList if temp != '']
wordList = [temp for temp in wordList if temp != "SPEECH"]
wordList = [temp for temp in wordList if not temp.isdigit()]
wordVector = list(set(wordList))
    
vectorSize = len(wordVector)

hiddenNeuronNumber = int(vectorSize/100)
    
m = dy.ParameterCollection()
W = m.add_parameters((hiddenNeuronNumber,vectorSize*2))
V = m.add_parameters((vectorSize,hiddenNeuronNumber))
b = m.add_parameters((hiddenNeuronNumber))

dy.renew_cg()

x = dy.vecInput(vectorSize*2)
output = dy.logistic(V*(dy.tanh((W*x)+b)))

y = dy.vecInput(vectorSize)
loss = dy.squared_distance(output,y)

trainer = dy.SimpleSGDTrainer(m)

epoch = 0
while epoch < 10:
    epochLoss = 0
    for i in range(0,len(wordList)-2):
        x.set(defineWordVector(wordList[0],wordVector) + defineWordVector(wordList[1],wordVector))
        y.set(defineWordVector(wordList[2],wordVector))
        
        loss.backward()
        trainer.update()
        
        epochLoss += loss.value(recalculate=True)
        print("Epoch Number: " + str(epoch) + " Training Percentage: %" + str(100*i/len(wordList)))
    print("Epoch number: " + str(epoch) + " Epoch Loss value: " + str(epochLoss/(len(wordList)-2)))
    epoch  += 1

startingWords = []
endingWords = []
for i in range(0,len(wordList)-1):
    curWord = wordList[i]
    if curWord[-1] == '.':
        endingWords.append(wordList[i])
        startingWords.append(wordList[i+1])

for i in range(0,5):      
    
    index1 = random.randint(0,len(startingWords)+1)
    index2 = index1 + 1
    tempStr = wordList[index1] + " " + wordList[index2]
    
    while(wordList[index2] not in endingWords):
        
        tempInput1 = defineWordVector(wordList[index1],wordVector)
        tempInput2 = defineWordVector(wordList[index2],wordVector)
        x.set(tempInput1+tempInput2)
    
        tempOut = output.value(recalculate = True)
        index1 = index2
        index2 = tempOut.index(max(tempOut))
        tempStr = tempStr + " " + wordList[index2]
        
    print(tempStr)

        
        