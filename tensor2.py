import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_data(datapath,vocab_size):
    data=[]
    labels=[]
    doc_ids=[]
    with open(datapath) as f:
        d_lines=f.read().splitlines()
    for d in d_lines:
        content=d.split('<fff>')
        label,doc_id= int(content[0]),int(content[1])
        labels.append(label)
        doc_ids.append(doc_id)
        rep=content[2].split()
        vector = [0.0 for _ in range(vocab_size)]
        for feature in rep:
            index,value=int(feature.split(':')[0]),float(feature.split(':')[1])
            vector[index]=value
            data.append(vector)
    return labels,doc_ids,data
def save_data(value):
    if len(value.shape) == 1:
        string = ','.join(str(word) for word in value)
    else:
        a = []
        for row in len(value.shape[0]):
            a.append(','.join(str(word) for word in value[row]))
        string = '\n'.join(a)
    return string
with open('D:/20news-bydate/word_idfs.txt') as f:
    vocab_size=len(f.read().splitlines())
train_label,train_doc_id,train_data=load_data(datapath='D:/20news-bydate/tf_idf.txt',vocab_size=vocab_size)
test_label,test_doc_id,test_data=load_data(datapath='D:/20news-bydate/tf_idf_test.txt',vocab_size=vocab_size)
num_classes=np.argmax(train_label)+1
hidden_size=50
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='sigmoid'),
    tf.keras.layers.Dense(num_classes)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics = ['accuracy']
)
model.fit(train_data,train_label,epochs=2)
filename="D:/20news-bydate/weights_and_biases"
with open(filename,"w") as f:
    for layer in model.layers:
        weights,biases=layer.get_weights()[0],layer.get_weights[1]
        f.write(save_data(weights)+"\n"+save_data(biases))



