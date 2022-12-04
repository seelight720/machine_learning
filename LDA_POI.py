import tomotopy as tp
import matplotlib.pyplot as plt
import re
from time import sleep
import numpy as np
import pyLDAvis

dir = {}
for n, line in enumerate(open('POI面积加权.txt', encoding='utf-8')):
    line = line.strip('\n')
    line = re.split('[,]', line)
    # print(line)
    # sleep(1)
    # print(dir)
    # sleep(1)
    if line[0] not in dir:
        dir[line[0]] = []
        dir[line[0]].append(line[1:])
    else:
        dir[line[0]].append(line[1:])


with open('dir.txt', 'w', encoding='utf-8') as f:
    for i in dir.keys():
        str = ""
        for j in dir[i]:
            # print(j)
            str += ' '.join([j[0]] * int(int(j[1])/1000))+" "
        f.write(str+'\n')
        # sleep(1)


corpus = tp.utils.Corpus()
for n, line in enumerate(open('dir.txt', encoding='utf-8')):
    corpus.add_doc(line.strip().split())

per_list = []
for k in range(6, 7):

    mdl = tp.LDAModel(min_cf=0, min_df=0, rm_top=0, k=k,
                      alpha=100, eta=1000, corpus=corpus)
    mdl.train(0)
    mdl.burn_in = 400
    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
    ))
    print('Removed Top words: ', *mdl.removed_top_words)

    # Let's train the model
    for i in range(0, 800, 20):
        print('Iteration: {:04}, LL per word: {:.4}'.format(
            i, mdl.ll_per_word))
        mdl.train(20)
    print('Iteration: {:04}, LL per word: {:.4}'.format(1000, mdl.ll_per_word))
    print()

    per_list.append(mdl.perplexity)
    # corpus.add_doc(str.strip().split())

# np.savetxt("square.csv", per_list, delimiter=',')
# plt.plot(np.arange(2, 11), per_list)
# plt.show()

topic_term_dists = np.stack(
    [mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
np.savetxt("./topic2term_dist_%d.csv" %
           (k), topic_term_dists, delimiter=',')
np.savetxt("./doc2topic_dist_%d.csv" %
           (k), doc_topic_dists, delimiter=',')


doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
    # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    sort_topics=False
)
pyLDAvis.save_html(prepared_data, 'level2_with_square.html')

ID_list = []
for n, line in enumerate(open("POI面积加权.txt", encoding='utf-8')):

    if line.__contains__(',,'):
        continue
    line = line.strip('\n')
    ID = int(re.split('[,|]', line)[0])
    if ID not in ID_list:
        ID_list.append(ID)
print(ID_list)
np.savetxt("as.csv", ID_list)
print(vocab)
