import random
import numpy as np
import tensorflow as tf
class MLP:
    def __init__(self,vocab_size,hidden_size):
        self._vocab_size=vocab_size
        self._hidden_size=hidden_size
    def build_graph(self):
        tf.compat.v1.disable_eager_execution()
        self._X= tf.compat.v1.placeholder(tf.float32,shape=[None,self._vocab_size])
        self._Y= tf.compat.v1.placeholder(tf.int32,shape=[None,])
        NUM_CLASSES=20
        w1=tf.compat.v1.get_variable(name='weights_input_hidden',shape=[self._vocab_size,self._hidden_size],initializer=tf.random_normal_initializer(seed=100))
        b1=tf.compat.v1.get_variable(name='biases_input_hidden',shape=[self._hidden_size],initializer=tf.random_normal_initializer(seed=100))
        w2 = tf.compat.v1.get_variable(name='weights_hidden_output', shape=[self._hidden_size,NUM_CLASSES],
                             initializer=tf.random_normal_initializer(seed=100))
        b2 = tf.compat.v1.get_variable(name='biases_hidden_output', shape=[NUM_CLASSES],
                             initializer=tf.random_normal_initializer(seed=100))
        hidden=tf.matmul(self._X,w1)+b1
        hidden=tf.sigmoid(hidden)
        output=tf.matmul(hidden,w2)+ b2
        label_onehotencode=tf.one_hot(indices=self._Y,depth=NUM_CLASSES,dtype=float)
        loss=tf.nn.softmax_cross_entropy_with_logits(labels=label_onehotencode,logits=output)
        loss=tf.reduce_mean(loss)
        prob=tf.nn.softmax(output)
        predicted_labels=tf.argmax(prob,axis=1)
        predicted_labels=tf.squeeze(predicted_labels)
        return predicted_labels,loss
    def trainer(self,loss,learning_rate):
        train=tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train
class DataReader:
    def __init__(self,datapath,batch_size,vocab_size):
        self._batch_size=batch_size
        with open(datapath) as f:
            d_lines=f.read().splitlines()
        self._data=[]
        self._label=[]
        for data_id,line in enumerate(d_lines):
            vector=[0.0 for _ in range(vocab_size)]
            feature=line.split("<fff>")
            label=int(feature[0])
            doc_id=int(feature[1])
            rep=feature[2].split()
            for f in rep:
                content=f.split(":")
                index,value=int(content[0]),float(content[1])
                vector[index]=value
            self._data.append(vector)
            self._label.append(label)
        self._data=np.array(self._data)
        self._label=np.array(self._label)
        self._num_epoch=0
        self._batch_id=0
    def next_batch(self):
        start=self._batch_id*self._batch_size
        end=start+self._batch_size
        self._batch_id +=1
        if end +self._batch_size > len(self._data):
            end=len(self._data)
            self._num_epoch +=1
            self._batch_id=0
            indice=range(len(self._data))
            random.seed(100)
            random.shuffle(indice)
            self._data, self._label = self._data[indice],self._label[indice]
            return self._data[start:end],self._labels[start:end]
def save_parameters(name,value,epoch):
        filename=name.replace(':','-colon-') + '-epoch-{}.txt'.format(epoch)
        if len(value.shape)==1:
            string=','.join(str(number) for number in value)
        else:
            string='\n'.join(','.join([str(number) for number in value[row]]) for row in range(value.shape[0]))
        with open('D:/20news-bydate/saved_para/'+filename,'w') as f:
            f.write(string)
def restore_parameters(name,epoch):
        filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
        with open('D:/20news-bydate/saved_para/' + filename, 'r') as f:
            lines=f.read().splitlines()
        if len(lines)==1:
            value=[float(number) for number in lines[0].split(',')]
        else:
            value=[float(number) for number in [lines[row].split(',') for row in range(len(lines))]]

with open("D:/20news-bydate/word_idfs.txt") as f:
    vocab_size=len(f.read().splitlines())
train_data_reader = DataReader(datapath='D:/20news-bydate/tf_idf.txt',batch_size=50,vocab_size=vocab_size)
test_data_reader = DataReader(datapath='D:/20news-bydate/tf_idf_test.txt', batch_size=50, vocab_size=vocab_size)
mlp=MLP(vocab_size=vocab_size,hidden_size=50)
predicted_labels, loss= mlp.build_graph()
train=mlp.trainer(loss=loss,learning_rate=0.01)
with tf.compat.v1.Session() as sess:
    epoch=10
    trainable_variabe=tf.compat.v1.trainable_variables()
    for variable in trainable_variabe:
        save_parameters(
            name=variable.name,
            value=variable.eval(),
            epoch=train_data_reader._num_epoch
        )
    for variable in trainable_variabe:
        saved_value=restore_parameters(variable.name,epoch)
        assign_op=variable.assign(saved_value)
        sess.run(assign_op)
    num_true_preds=0
    while True:
        test_data,test_label=test_data_reader.next_batch()
        test_label_hat=sess.run(
            predicted_labels,feed_dict={
                mlp._X: test_data,
                mlp._Y : test_label
            }
        )
        matches=np.equal(test_label_hat,test_label)
        num_true_preds += np.sum(matches.astype(float))
        if test_data_reader._batch_id==0:
            break
    print("Epoch:",epoch)
    print("Accuracy",num_true_preds/len(test_data_reader._data))


