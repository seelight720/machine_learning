import numpy as np
import pandas as pd

def loaddata(filename):
    with open(filename, "r", encoding="utf-8") as f:
        i = 0
        head=[]
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            lineback = line
            arrlineback = lineback.split('\t')

            if i==0:
                head=arrlineback
            else:
                for j in range(len(arrlineback)):
                    if j==0:
                        rowid=arrlineback[j]
                    else:
                        with open(filename+"_od", "a+", encoding="utf-8") as f:
                            if arrlineback[j]=="":
                                f.writelines(rowid+","+head[j]+",0"+ "\n")
                            else:
                                f.writelines(rowid + "," + head[j] + ","+arrlineback[j] + "\n")
            if i % 1000 == 0:
                print(i)
            i = i + 1

if __name__ == '__main__':
    loaddata("D:\PycharmProjects\宁德数据分析\OD-总2022.txt")