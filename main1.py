import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
import pickle
from models import hccr_cnnnet
from tensorflow.lite.python import lite

char_dict = "dictionary"
charlabels={}
#负责讲ont_hot映射成中文
with open(char_dict,'rb') as f:
    charlabels = pickle.load(f)

x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='image_batch')

logits=hccr_cnnnet(x,train=False,regularizer=None,channels=1)
prob_batch = tf.nn.softmax(logits)
saver=tf.train.Saver()
sess = tf.Session()

saver.restore(sess, "./checkpoint/trainnum_200000_-48001")

def convolutional(input):
    s=sess.run(prob_batch, feed_dict={x: input})
    printa = []
    for charlist in s:
        maxlist=[]
        A = np.argpartition(charlist, -10)[-10:]
        B = charlist[A]
        for i,j in zip(A,B):
            maxlist.append([i,j])
        maxlist=sorted(maxlist,key=(lambda x:x[1]),reverse=True)
        for element in maxlist:
            chari = charlabels[element[0]]
            printa.append(chari)
    return printa


# webapp
app = Flask(__name__)


@app.route('/api/zhongwen', methods=['POST'])
def mnist():
    input = (np.array(request.json, dtype=np.uint8)).reshape(1, 4096)
    output1 = convolutional((input).reshape(1,64,64,1))
    print(output1)
    return jsonify(results=output1)

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=80)
