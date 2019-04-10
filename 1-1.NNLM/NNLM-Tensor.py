# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk"]

# 唯一词表
word_list = " ".join(sentences).split()
word_list = list(set(word_list))

# 为每个word分配index，及生成相应的word、index的映射dict
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # number of steps ['i like', 'i love', 'i hate']
n_hidden = 2 # number of hidden units

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        # 取每个词
        word = sen.split()
        
        # 用除末字母外的子串的index作为x
        input = [word_dict[n] for n in word[:-1]]
        
        # 每个单词的末字母的index作为y
        target = word_dict[word[-1]]
        
        # 生成对角矩阵，每一行作为index行的onehot特征
        # 将index对应的行取出来，从而得到x和y的特征
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Model
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

# 输入为steps * dim
input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]

# 第一层，随机初始化hidden个神经元，每个的输入维度为steps * n_class
# H存储weight, d存储bias
# 本层的输出维度最终为n_hidden
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))

# 第二层，随机初始化n_class个神经元，每个的维度为n_hidden，对应上一层的输出维度
# H存储weight, d存储bias
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# tanh(input * H + d) -> l0
# l0 * U + b -> model
tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b # [batch_size, n_class]

# 定义loss为所有样本平均的cross_entropy(logits, label)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction = tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
input = [sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])
