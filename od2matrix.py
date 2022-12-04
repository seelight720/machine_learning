import numpy as np
import pandas as pd

def load_od_data(input_filename,filter1,row_id,col_id,value_id,save_filename):
    row_dict={}
    col_dict={}
    matrix={}

    if len(filter1)>0:
        valid_data = []
        with open(input_filename, "r", encoding="ANSI") as f:
            i = 0
            for line in f.readlines():
                if i % 100000 == 0:
                    print("Read data {0}".format(i))
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                lineback = line
                arrlineback = lineback.split(',')

                if arrlineback[0] in filter1:
                    valid_data.append(arrlineback)
                i=i+1

        #获取行、列的唯一值
        print("Get row, col value")
        for j in range(len(valid_data)):
            row_dict[valid_data[j][row_id]]=0
            col_dict[valid_data[j][col_id]]=0

        # 计算矩阵求和值
        print("Cal matrix value")
        for j in range(len(valid_data)):
            matrix_key=valid_data[j][row_id]+","+valid_data[j][col_id]
            if matrix_key in matrix.keys():
                matrix[matrix_key]=matrix[matrix_key]+float(valid_data[j][value_id])
            else:
                matrix[matrix_key]=float(valid_data[j][value_id])

        # 保存矩阵值
        print("Save matrix")
        with open(save_filename , "a+", encoding="utf-8") as f:
            m = 0
            for key_temp_row_dict in row_dict.keys():
                if m==0:
                    #添加head
                    m = m + 1
                    f.write("id")
                    for key_temp_col_dict in col_dict.keys():
                        f.write(",{0}".format(key_temp_col_dict))
                    f.write("\n")
                    # 添加head

                    # 添加首行数据
                    f.write("{0}".format(key_temp_row_dict))
                    for key_temp_col_dict in col_dict.keys():
                        # f.write(",{0}".format(matrix[key_temp_row_dict+","+key_temp_col_dict]))
                        if key_temp_row_dict + "," + key_temp_col_dict in matrix.keys():
                            f.write(",{0}".format(matrix[key_temp_row_dict + "," + key_temp_col_dict]))
                        else:
                            f.write(",0")
                    f.write("\n")
                    # 添加首行数据
                else:
                    f.write("{0}".format(key_temp_row_dict))
                    for key_temp_col_dict in col_dict.keys():
                        # f.write(",{0}".format(matrix[key_temp_row_dict+","+key_temp_col_dict]))
                        if key_temp_row_dict + "," + key_temp_col_dict in matrix.keys():
                            f.write(",{0}".format(matrix[key_temp_row_dict + "," + key_temp_col_dict]))
                        else:
                            f.write(",0")
                    f.write("\n")
    else:
        #不用filter
        valid_data = []
        with open(input_filename, "r", encoding="ANSI") as f:
            i = 0
            for line in f.readlines():
                if i % 100000 == 0:
                    print("Read data {0}".format(i))
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                lineback = line
                arrlineback = lineback.split(',')


                valid_data.append(arrlineback)
                i = i + 1

        # 获取行、列的唯一值
        print("Get row, col value")
        for j in range(len(valid_data)):
            row_dict[valid_data[j][row_id]] = 0
            col_dict[valid_data[j][col_id]] = 0

        # 计算矩阵求和值
        print("Cal matrix value")
        for j in range(len(valid_data)):
            matrix_key = valid_data[j][row_id] + "," + valid_data[j][col_id]
            if matrix_key in matrix.keys():
                matrix[matrix_key] = matrix[matrix_key] + float(valid_data[j][value_id])
            else:
                matrix[matrix_key] = float(valid_data[j][value_id])

        # 保存矩阵值
        print("Save matrix")
        with open(save_filename, "a+", encoding="utf-8") as f:
            m = 0
            for key_temp_row_dict in row_dict.keys():
                if m == 0:
                    # 添加head
                    m = m + 1
                    f.write("id")
                    for key_temp_col_dict in col_dict.keys():
                        f.write(",{0}".format(key_temp_col_dict))
                    f.write("\n")
                    # 添加head

                    # 添加首行数据
                    f.write("{0}".format(key_temp_row_dict))
                    for key_temp_col_dict in col_dict.keys():
                        # f.write(",{0}".format(matrix[key_temp_row_dict+","+key_temp_col_dict]))
                        if key_temp_row_dict + "," + key_temp_col_dict in matrix.keys():
                            f.write(",{0}".format(matrix[key_temp_row_dict + "," + key_temp_col_dict]))
                        else:
                            f.write(",0")
                    f.write("\n")
                    # 添加首行数据
                else:
                    f.write("{0}".format(key_temp_row_dict))
                    for key_temp_col_dict in col_dict.keys():
                        # f.write(",{0}".format(matrix[key_temp_row_dict+","+key_temp_col_dict]))
                        if key_temp_row_dict + "," + key_temp_col_dict in matrix.keys():
                            f.write(",{0}".format(matrix[key_temp_row_dict + "," + key_temp_col_dict]))
                        else:
                            f.write(",0")
                    f.write("\n")



if __name__ == '__main__':
    filter1=['20220912', '20220913', '20220914', '20220915', '20220916']
    filter2 = ['0', '1', '2', '3', '4']
    filter3=[]
    load_od_data("links_flows出行.txt",filter1,4,1,5,'matrix出行.txt')