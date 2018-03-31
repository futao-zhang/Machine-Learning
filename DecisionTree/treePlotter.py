import matplotlib.pyplot as plt

#使用文本注解绘制树节点 
#包含了边框的类型，边框线的粗细等 
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细 ,pad指的是外边框锯齿形（圆形等）的大小  
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8", pad = 1)
# 定义决策树的叶子结点的描述属性 round4表示圆形 
leafNode = dict(boxstyle = "round4", fc = "0.8", pad = 1)
arrow_args = dict(arrowstyle = "<-")#定义箭头属性


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # annotate是关于一个数据点的文本    
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点    
    # annotate的作用是添加注释，nodetxt是注释的内容，  
    # nodetype指的是输入的节点（边框）的形状  
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',\
                           xytext = centerPt, textcoords = 'axes fraction', \
                           va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)

# 获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0] # 获取tree的根节点对于的key值
    # 遇到的问题是mytree.keys()获得的类型是dict_keys，而dict_keys不支持索引，
    # 我的解决办法是把获得的dict_keys强制转化为list即可  
    secondDict = myTree[firstStr] # 根据键值得到对应的值，即根据第一个特征分类的结果
    for key in secondDict.keys(): # 获取第二个小字典中的key  
        if type(secondDict[key]).__name__ == 'dict': # type()函数判断子节点是否为字典类型
            numLeafs += getNumLeafs(secondDict[key]) # 包含的话进行递归从而继续循环获得新的分支所包含的叶节点的数量  
        else:
            numLeafs += 1 # 不包含的话就停止迭代并把现在的小字典加一表示这边有一个分支  
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0] # 获取tree的根节点对于的key值
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth #间隔 间隔得问题一定要多考虑
    return maxDepth

def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},\
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]

    return listOfTrees[i]


# 作用是计算tree的中间位置    
# cntrpt起始位置,parentpt终止位置,txtstring：文本标签信息  
def plotMidText(cntrPt, parentPt, txtString):
    # cntrPt 起点坐标 子节点坐标   parentPt 结束坐标 父节点坐标  
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    #找到x和y的中间位置  
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
    
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0] # 获取tree的根节点对于的key值
    # 计算子节点的坐标   
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) # 绘制线上的文字
    plotNode(firstStr, cntrPt, parentPt, decisionNode) # 绘制节点
    secondDict = myTree[firstStr]
    #每绘制一次图，将y的坐标减少1.0/plottree.totald，间接保证y坐标上深度的  
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    
    
    
# 类似于Matlab的figure，定义一个画布，背景为白色
def createPlot(inTree): 
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图   
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops) # frameon表示是否绘制坐标轴矩形
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

