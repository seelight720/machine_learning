import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

def loaddata(txt_path, delimiter):
    # ---
    # 功能：读取只包含数字的txt文件，并转化为array形式
    # txt_path：txt的路径；delimiter：数据之间的分隔符
    # ---
    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = line.split(delimiter)
        temp_data = list(map(float, data_split[0:]))
        data_list.append(temp_data)


    data_array = np.array(data_list)
    return data_array


# def get_data():
#     """生成聚类数据"""
#     from sklearn.datasets import make_blobs
#
#     x_value, y_value = make_blobs(n_samples=1000, n_features=40, centers=3, )
#     return x_value, y_value
# def plot_xy(x_values, label, title):
#     """绘图"""
#     df = pd.DataFrame(x_values, columns=['x', 'y'])
#     df['label'] = label
#     sns.scatterplot(x="x", y="y", hue="label", data=df)
#     plt.title(title)
#     plt.show()
# def main():
#     x_value, y_value = get_data()
#
#     # PCA 降维
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     x_pca = pca.fit_transform(x_value)
#     plot_xy(x_pca, y_value, "PCA")
#
#     # t-sne 降维
#     from sklearn.manifold import TSNE
#     tsne = TSNE(n_components=2)
#     x_tsne = tsne.fit_transform(x_value)
#     print()
#     plot_xy(x_tsne, y_value, "t-sne")

def t_sne(x_value):
    # t-sne 降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3)
    x_tsne= tsne.fit_transform(x_value)
    x_tsne=pd.DataFrame(x_tsne)
    x_tsne.to_csv('时间指标3wei4.csv')
    print()
    # plot_xy(x_tsne, y_value, "t-sne")


if __name__ == '__main__':
    data_array = loaddata("时间指标8维.txt", ",")
    t_sne(data_array)
    #main()

