
def loadodata(filename):
    TAZ_level1={}
    TAZ_level2 = {}
    TAZ_level3 = {}

    with open(filename, "r") as source_file:
        for line in source_file:
            try:
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                arrline = line.split(',')
                if arrline[3] in TAZ_level1.keys():
                    TAZ_level1[arrline[3]]=TAZ_level1[arrline[3]]+arrline[0]+"|"
                else:
                    TAZ_level1[arrline[3]]=  arrline[0] + "|"


                if arrline[3] in TAZ_level2.keys():
                    TAZ_level2[arrline[3]] = TAZ_level2[arrline[3]] + arrline[1] + "|"
                else:
                    TAZ_level2[arrline[3]] = arrline[1] + "|"


                if arrline[3] in TAZ_level3.keys():
                    TAZ_level3[arrline[3]]=TAZ_level3[arrline[3]]+arrline[2]+"|"
                else:
                    TAZ_level3[arrline[3]]=  arrline[2] + "|"
            except:
                pass
    return TAZ_level1,TAZ_level2,TAZ_level3

def savedata(dict,savefilename):
    for tempkey in dict.keys():
        with open(savefilename, "a+",encoding="UTF-8") as files:
            files.writelines(tempkey+","+dict[tempkey][0:-1]+"\n")

if __name__ == '__main__':
    TAZ_level1,TAZ_level2,TAZ_level3=loadodata("pois.txt")
    savedata(TAZ_level1,"TAZ_level1.txt")
    savedata(TAZ_level2, "TAZ_level2.txt")
    savedata(TAZ_level3, "TAZ_level3.txt")
