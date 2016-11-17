from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import os
import shutil
from six.moves import urllib
import sys
from tensorflow.contrib import layers
import numpy as np
import tensorflow as tf
import gzip

####### download data ############

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
datadir = os.path.abspath('./Data')

if not os.path.exists(datadir):
    os.mkdir(datadir)
    
if not os.path.exists(os.path.join(datadir,TRAIN_IMAGES)):
    print('start downloading the data')
    def processbar(filename):
        def _process(count,block_size,total_size):
            per = float(count*block_size)*100/total_size
            if per > 100:
                sys.stdout.write('\r>>downloading {0} 100%\n--downloaed {0}\n'.format(filename))
            else:
                sys.stdout.write('\r>>downloading %s %.1f%%'%(filename,per))          
            sys.stdout.flush()
        return _process
    fname = TRAIN_IMAGES
    surl = SOURCE_URL+fname
    urllib.request.urlretrieve(surl, os.path.join(datadir,fname),processbar(fname))
    
    fname = TRAIN_LABELS
    surl = SOURCE_URL+fname
    urllib.request.urlretrieve(surl, os.path.join(datadir,fname),processbar(fname))
    
    fname = TEST_IMAGES
    surl = SOURCE_URL+fname
    urllib.request.urlretrieve(surl, os.path.join(datadir,fname),processbar(fname))

    fname = TEST_LABELS
    surl = SOURCE_URL+fname
    urllib.request.urlretrieve(surl, os.path.join(datadir,fname),processbar(fname))

####### unzip data ############

with gzip.GzipFile(datadir+'/'+TRAIN_IMAGES) as f:
    buf = f.read()   
train_magic1,nimg,nrow, ncol= np.frombuffer(buf,np.dtype('>i4'),4)
train_image = np.frombuffer(buf,np.dtype('u1'),offset=16)

with gzip.GzipFile(datadir+'/'+TRAIN_LABELS) as f:
    buf = f.read()   
train_magic2,nlbl = np.frombuffer(buf,np.dtype('>i4'),2)
train_label = np.frombuffer(buf,np.dtype('u1'),offset=8)
assert(train_magic1==2051)
assert(train_magic2==2049)
images = [] # use python list, avoid reallocation when append elements
for i in range(nimg):
    img = train_image[i*nrow*ncol:(i+1)*nrow*ncol].reshape(nrow,ncol)
    images.append(img)

images = np.array(images) # change to np.array

data = images.reshape([-1,28,28,1])/255.0
label = train_label.reshape([-1,1])
datatype = tf.float32
print('data loaded')

#####  Helper class #######
class DSet(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __repr__(self):
        return 'DSet: x:%s, y:%s'%(self.x.shape,self.y.shape)
    
    def __len__(self):
        return len(self.x)
    
class Dataset(object):
    def __init__(self,data,label,per_for_test):
        self._per_for_test = per_for_test
        self._data = data
        self._label = label
        n_test = int(len(self)*self._per_for_test)
        n_train = len(self)-n_test
        idx_train = np.zeros(len(self),dtype=bool)
        idx_train[np.random.choice(np.arange(len(self)),n_train,replace=False)]=True
        
        
        self.train = DSet(self._data[idx_train,:],self._label[idx_train,:])
        self.test = DSet(self._data[~idx_train,:],self._label[~idx_train,:])
        
        self._train_perm = np.arange(len(self.train.x))
        
        self._index_in_epoch = 0
        self._epochs_completed = 0
        
    
    def __len__(self):
        return len(self._data)
    
    # only select batch in training data
    def next_batch(self, batch_size):
        assert batch_size <= len(self.train)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch>len(self.train.x):
            self._epochs_completed += 1
            perm = np.arange(len(self.train.x))
            self.train.x = self.train.x[perm]
            self.train.y = self.train.y[perm]
            start = 0
            self._index_in_epoch=batch_size
            assert batch_size <= len(self.train)
        end = self._index_in_epoch
        return DSet(self.train.x[start:end], self.train.y[start:end])
    
    def __repr__(self):
        return 'Dataset(data=, label=, per_for_test=%f): <train:(%s-%s);  test:(%s-%s)>' % (
            self._per_for_test,
            str(self.train.x.shape),
            str(self.train.y.shape),
            str(self.test.x.shape),
            str(self.test.y.shape),
        )
    
mnist = Dataset(data,label,0.20)

##### define model ########
def flatten(feat):
    _shape = feat.get_shape()
    nfeat = np.prod(_shape[1:]).value 
    with tf.variable_scope('flatten'):
        feat = tf.reshape(feat,shape=[-1,nfeat],name='flattened')
    return feat

def one_hot(target,nlabel):
    with tf.variable_scope('onehot'):
        one_hot_target = tf.one_hot(tf.cast(tf.reshape(target,[-1]),tf.int32),nlabel)
    return one_hot_target

def cross_entropy_loss(logit,onehot):
    with tf.variable_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit,onehot))
    return loss

def logit_accuracy(logit,one_hot):
    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(one_hot,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    return accuracy

def conv(in_tensor,in_channel,out_channel,name='conv',datatype = tf.float32):
    with tf.variable_scope(name):
        h_conv1 = layers.conv2d(in_tensor,out_channel,[5,5],activation_fn=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
        
#         w = tf.Variable(tf.zeros(shape=[5,5,in_channel,out_channel],dtype=datatype))
#         b = tf.Variable(tf.constant(0,shape=[out_channel],dtype=datatype))
#         conv = tf.nn.conv2d(in_tensor,w,strides=[1,1,1,1],padding='SAME')
#         z = tf.nn.bias_add(conv,b)
#         x = tf.nn.relu(z)
    return h_pool1

def max_pool_2x2(tensor_in,name='maxpool'):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(tensor_in,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return out

def fc(in_tensor,n_out,name='fc',datatype = tf.float32,act=None):
    n_in = in_tensor.get_shape()[1].value
    with tf.variable_scope(name):
        w = tf.Variable(tf.zeros(shape=[n_in,n_out],dtype=datatype))
        b = tf.Variable(tf.constant(0,shape=[n_out],dtype=datatype))
        z = tf.matmul(in_tensor,w)+b
        z = act(z) if act else z
    return z,w,b

def conv_net(feat):
    conv1 = conv(feat,1,32,name='conv1')
    conv2 = conv(conv1,32,64,name='conv2')
    flattened = flatten(conv2)
    fc1,w1,b1 = fc(flattened,1024,name='fc1',act=tf.nn.relu)
    logit,w2,b2 = fc(fc1,10,name='fc1',act=None)

    return logit,w1,b1
    
    
    
feat = tf.placeholder(datatype,shape=(None, 28, 28, 1))
logits,w,b = conv_net(feat)

target = tf.placeholder(datatype,shape=(None,1))
onehot_tgt = one_hot(target,10)

accuracy = logit_accuracy(logits,onehot_tgt)
loss = cross_entropy_loss(logits,onehot_tgt)

optimizer = layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),
                               optimizer='SGD',learning_rate=0.01)
#regularizer = tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
#loss=loss+5e-4*regularizer
#step = tf.Variable(0,dtype=datatype)
#learning_rate = tf.train.exponential_decay(0.01,step*batchsize,N,0.95,staircase=True)
#optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=step)

testdict = {feat: mnist.test.x[0:1000,:], target: mnist.test.y[0:1000,:]}

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in xrange(3000):
        batch = mnist.next_batch(100)
        traindict = {feat: batch.x, target: batch.y}
        sess.run(optimizer,feed_dict=traindict)
        if i%10==0:
            print(str(i),
                  loss.eval(feed_dict=traindict),
                  accuracy.eval(feed_dict=testdict),
                 )
