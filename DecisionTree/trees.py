from math import log

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannoEnt -= prob * log(prob,2)
    return shannoEnt


def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 按照给定特征划分数据集
# 三个输入参数：待划分的数据集，划分数据集的特征，需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # axis列为value的数据集【该数据集需要排除axis列】
    # 判断axis列的值是否为value
    for featVec in dataSet:
        # chop out axis used for splitting
        # [:axis]表示前axis行，即若 axis 为2，就是取 featVec 的前 2 行
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis] # 把axis之前的列保存下来
            # list.append(object) 向列表中添加一个对象object
            # list.extend(sequence) 把一个序列seq的内容添加到列表中
            # 1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
            # 2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
            reduceFeatVec.extend(featVec[axis+1:])# [axis+1:]表示从跳过 axis 的 axis+1行，取接下来的数据
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)# 数据集的原始信息熵
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]# 获取对应的feature下的所有数据
        uniqueVals = set(featList)# 获取剔重后的集合，使用set对list数据进行去重
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵 
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy #信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature 

# 如果只有一列，采用多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key():
            classCount[vote] = 0
            classCount[vote] +=1
    sortedClassCount = sorted(classCount.iteritems(),\
                             key = operator.iteritems(1), reverse = True)
    return sortedClassCount[0][0] 

# 创建树的函数代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}} # 初始化MyTree
    # 注：labels列表是可变对象，在Python函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的分支做分类
    featValues = [example[bestFeat] for example in dataSet] # 最优列的特征取值
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]  # 求出剩余的标签label
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 测试算法：使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0] # 获取tree的根节点对于的key值
    secondDict = inputTree[firstStr] # 除去根节点后的树的分支(通过key得到根节点对应的value)
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr) # 获得特征的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict': # 如果值仍为字典，继续分类
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key] #否则，输出对应键的值
    return classLabel


