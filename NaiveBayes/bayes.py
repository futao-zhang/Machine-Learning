#! /usr/bin/python
# -*-coding:UTF-8-*-

from numpy import *

def loadDateSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], \
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], \
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], \
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'], \
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], \
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 代表侮辱性文字，0代表正常言论
    return postingList,classVec

# 获取所有单词的集合
def createVocabList(dataSet):
    vocabSet = set([]) # 创建一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 两个合集的并集 
    return list(vocabSet)

# 遍历查看该单词是否出现，出现该单词则将该单词置1
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList) # 创建一个其中所含元素都为0的向量
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    return returnVec

# 朴素贝叶斯分类器训练函数
# param trainMatrix: 文件单词矩阵 [[1,0,1,1,1....],[],[]...]
# param trainCategory: 文件对应的类别[0,1,1,0....]，列表长度等于单词矩阵数，
# 其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 文档中属于侮辱性文档的概率P(1)
    # 构造单词出现次数列表 [0,0,0,.....]
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 整个数据集单词出现总数
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 是否是侮辱性文件
        if trainCategory[i] == 1:
            # 如果是侮辱性文件，对侮辱性文件的向量进行加和
            p1Num += trainMatrix[i]
            # 对向量中的所有元素进行求和，也就是计算所有侮辱性文件中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个单词出现的概率
    # p1Vect = p1Num/p1Denom
    # 类别0，即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    # 即 在0类别下，每个单词出现的概率
    # p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

# vec2Classify: 代表待测数据（即要分类的向量）
# p0Vec：类别0对应的单词的概率列表
# p1Vec：类别1对应的单词的概率列表
# pClass：类别1侮辱性文件的出现概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    # 将乘法转换为加法
        # 乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        # 加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    p1 = sum(vec2Classify * p1Vec) + log(pClass)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass)
    if p1 > p0:
        return 1
    else:
        return 0

# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocablist, inputSet):
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] += 1
    return returnVec


def testingNB():
    listOPosts, listClasses = loadDateSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,"classified as %d" %(classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,"classified as %d" %(classifyNB(thisDoc, p0V, p1V, pAb)))


# 文本解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split('\\W*',bigString)
    return [tok.lower() for tok in bigString if len(tok) > 2]

def spamTest():
    docList = [] # 文本字符串列表
    classList = []
    fullText = []
    # 1. 加载数据集（导入并解析文本数据）
    for i in range(1,26):
        # 读取文本中的字符串并转换为小写的不加空字符串的列表的形式
        # 添加分类为1的文本
        # 切分，解析数据，并归类为 1 类别
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 添加分类为0的文本
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 2. 创建单词集合
    vocabList = createVocabList(docList)
    # 随机构建训练集
    traingingSet = range(50)
    testSet = []
    # 随机取 10 个邮件用来测试
    # 选择出的数字所对应的文档被添加到测试集，同时也将其从训练集删除
    # 这种随机选择一部分作为训练集，而剩余部分作为测试集的过程称为留存交叉验证
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(traingingSet[randIndex])
        del(traingingSet[index])
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    trainClasses = []
    for docIndex in traingingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 4. 训练数据
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 5. 测试数据并计算误差
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is %0.2f' %float(errorCount/len(testSet)))
