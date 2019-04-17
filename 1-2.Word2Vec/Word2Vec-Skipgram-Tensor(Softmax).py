'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()

# 3 Words Sentence
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split() # 拼接成一个大句子后空格分词得到词序列
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}

# Word2Vec Parameter
batch_size = 20
embedding_size = 2 # To show 2 dim embedding graph
voc_size = len(word_list)

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False) # 从data中非放回采样size

    # 使用对角矩阵作为voc_size个单词的onehot特征表示
    # data由(target, context)的词对组成，在构建skip_grams时生成，context为target前、后一个词
    # 随机选择不重复的size个词对，将target和context分别作为input、label
    
    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # target
        random_labels.append(np.eye(voc_size)[data[i][1]])  # context word

    return random_inputs, random_labels

# Make skip gram of one size window
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]] # 选第i个词作为target
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]] # 其前后两个词作为context

    # skip_grams数组中存储了所有的(target,context)对，类似2-gram但区别是skip_grams中等于还多存储了反向的2-gram
    for w in context:
        skip_grams.append([target, w]) 
        
# Model 从target预测context
inputs = tf.placeholder(tf.float32, shape=[None, voc_size]) # N * voc_size
labels = tf.placeholder(tf.float32, shape=[None, voc_size])

# W and WT is not Traspose relationship
W = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0)) #                              voc_size * embedding_size
WT = tf.Variable(tf.random_uniform([embedding_size, voc_size], -1.0, 1.0)) # 

# 
hidden_layer = tf.matmul(inputs, W) # [batch_size, embedding_size] # M_n*voc_size) * M_voc_size*embedding_size
output_layer = tf.matmul(hidden_layer, WT) # [batch_size, voc_size] # M_n*embedding_size * M_embedding_size*voc_size

# 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 5000个epoch，每一轮随机选择构建batch_size个skip_grams的样本
    for epoch in range(5000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, labels: batch_labels})

        if (epoch + 1)%1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        trained_embeddings = W.eval()

for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
