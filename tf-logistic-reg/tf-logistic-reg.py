
# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
import numpy as np

run_id = str(np.random.randint(100))

print("Training with ID: " + run_id)

with tf.name_scope("Inputs"):
    x = tf.placeholder(tf.float32, shape=[None, 784])
with tf.name_scope("Outputs"):
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Create variables for weights and bias as tensors full of zeros

with tf.name_scope("W"):
    W = tf.Variable(tf.zeros([784,10]))
with tf.name_scope("b"):
    b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Forward_Prop"):
    y = tf.matmul(x,W) + b

with tf.name_scope("Cost"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
tf.summary.scalar('cost_scalar', cross_entropy)


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter('out.graph/run_' + run_id ,sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        batch = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch[0],y_: batch[1]})
        
        cost, summary = sess.run([cross_entropy, merge], feed_dict={x: batch[0],y_: batch[1]})
        writer.add_summary(summary,i)
        
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    final_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) 

    num_summary = tf.Summary()
    num_summary.value.add(tag="accuracy", simple_value=final_accuracy)
    writer.add_summary(num_summary)

    print(final_accuracy)

