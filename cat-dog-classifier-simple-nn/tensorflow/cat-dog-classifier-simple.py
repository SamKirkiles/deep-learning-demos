
import tensorflow as tf
import numpy as np
import os
import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with tf.name_scope("data"):
	train_x_orig, train_y, test_x_orig, test_y, classes, original_shape = load_data.load_data()


# Feature Normilization
train_x_orig = (train_x_orig-np.mean(train_x_orig))/(np.max(train_x_orig)-np.min(train_x_orig))
test_x_orig = (test_x_orig-np.mean(test_x_orig))/(np.max(test_x_orig)-np.min(test_x_orig))

# Number of training examples and number of features
m = train_x_orig.shape[1]
n = train_x_orig.shape[0]

# Dimensons
dims = [n, 4, 4, 1]

with tf.name_scope("inputs"):
	x = tf.placeholder(tf.float32,shape=[n,None],name="x")

with tf.name_scope("outputs"):
	y_ = tf.placeholder(tf.float32,shape=[1,None],name="y")

with tf.name_scope("hidden_layer_1"):
	W1 = tf.Variable(tf.random_normal(shape=(7,n), stddev=0.1), name="W1", dtype=tf.float32)
	b1 = tf.Variable(tf.random_normal(shape=(7,1), stddev=0.1), name="b1", dtype=tf.float32)

	foward_prop_1 =  tf.add(tf.matmul(W1,x),b1)

	layer_1 = tf.nn.relu(foward_prop_1)

with tf.name_scope("hidden_layer_2"):
	W2 = tf.Variable(tf.random_normal(shape=(7,7), stddev=0.1),name="W2", dtype=tf.float32)
	b2 = tf.Variable(tf.random_normal(shape=(7,1), stddev=0.1), name="b2", dtype=tf.float32)

	foward_prop_2 = tf.add(tf.matmul(W2,layer_1),b2)

	layer_2 = tf.nn.relu(foward_prop_2)

with tf.name_scope("output_layer"):
	W3 = tf.Variable(tf.random_normal(shape=(1,7), stddev=0.1),name="W3", dtype=tf.float32)
	b3 = tf.Variable(tf.random_normal(shape=(1,1), stddev=0.1), name="b3", dtype=tf.float32)

	foward_prop_3 = tf.add(tf.matmul(W3,layer_2),b3)

	layer_output = tf.nn.sigmoid(foward_prop_3)



with tf.name_scope("cross_entropy_cost"):
	cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_,tf.float32),logits=tf.cast(layer_output,tf.float32)))
tf.summary.scalar("cross_entropy_",cross_entropy)

#Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# Visualize image
example_image = tf.reshape(tf.transpose(x), original_shape)
index = np.random.randint(m)
tf.summary.image("Example", example_image[index:index+1,:,:,:])

with tf.Session() as sess:
	# Intiailize all variables and merge summaries
	tf.global_variables_initializer().run()
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('out.graph/train_' + str(np.random.randint(100)), sess.graph)

	# Example Image
	image, image_summary = sess.run([example_image, merged], feed_dict={x:train_x_orig, y_:train_y})
	train_writer.add_summary(image_summary)

	#Find intial cost
	cost = sess.run(cross_entropy,feed_dict={x:train_x_orig, y_:train_y})
	print("Initial cost: " + str(cost))

	# Add cost to tensorboard
	summary = sess.run(merged, feed_dict={x:train_x_orig, y_:train_y})
	train_writer.add_summary(summary)


	# Training loop

	iterations = 1000

	for i in range(iterations):

		step, summary = sess.run([train_step,merged],feed_dict={x:train_x_orig, y_:train_y})
		train_writer.add_summary(summary,i)
	
		if i%(iterations/10) == 0: 
			print(sess.run(cross_entropy,feed_dict={x:train_x_orig, y_:train_y}))

	#Find final cost
	cost = sess.run(cross_entropy,feed_dict={x:train_x_orig, y_:train_y})
	print("Final cost: " + str(cost))


	with tf.name_scope("accuracy"):

		output_accuracy = tf.equal(tf.greater_equal(layer_output, 0.5), tf.greater_equal(y_, 0.5))
		accuracy = tf.reduce_mean(tf.cast(output_accuracy,tf.float32)) * 100

	print("Final Accuracy on Training Set: " + str(accuracy.eval(feed_dict={x:train_x_orig, y_:train_y})))


	print("Final Accuracy on Test Set: " + str(accuracy.eval(feed_dict={x:test_x_orig, y_:test_y})))
	







	

