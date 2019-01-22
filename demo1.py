import pandas as pd  # A beautiful library to help us work with data as tables
import tensorflow as tf  # Fire from the gods

if __name__ == "__main__":

    # Let's have Pandas load our dataset as a dataframe
    dataframe = pd.read_csv("datasetcsv.csv")
    # remove columns we don't care about
    dataframe = dataframe.drop(["unknown.17", "unknown.16", "unknown.15", "unknown.14", "unknown.13", "unknown.12",
                                "unknown.11", "unknown.10", "unknown.9", "unknown.8", "unknown.7", "unknown.6",
                                "unknown.5"
                                   , "unknown.4", "unknown.3", "unknown.2", "unknown.1", "unknown"],
                               axis=1)
    # We'll only use the first 10 rows of the dataset in this example
    # dataframe = dataframe[0:30]
    # Let's have the notebook show us how the dataframe looks now

    inputX = dataframe.loc[:,
             ['radius ', 'texture', 'perimeter', 'area', 'smoothness ', 'compactness ', 'concavity', 'concave  '
                 , 'points', 'symmetry', 'fractal', ' dimension']].values

    inputY = dataframe.loc[:, ["Label"]].values

    for x in range(inputY.size):
        if inputY[x] == "M":
            inputY[x] = 1
        else:
            inputY[x] = 0

    # Let's prepare some parameters for the training process

    # Parameters
    n_input = 12  # features
    n_hidden = 4  # hidden nodes
    n_output = 1  # lables
    learning_rate = 0.001
    training_epochs = 100000  # simply iterations
    display_step = 10000  # to split the display
    n_samples = inputY.size  # number of the instances

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # weights from input -> hidden -> output
    W1 = tf.Variable(tf.zeros([n_input, n_hidden]))
    W2 = tf.Variable(tf.zeros([n_hidden, n_output]))

    # Bias input -> hidden -> output
    b1 = tf.Variable(tf.zeros([n_hidden]))
    b2 = tf.Variable(tf.zeros([n_output]))

    # calculation of the layers
    L2 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    hy = tf.nn.sigmoid(tf.matmul(L2, W2) + b2)

    # calculate the coast of our calculations and then optimaze it

    cost = tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    """cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n_samples)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
     """
    # Initialize variabls and tensorflow session
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        # lets Do  Our real traing
        for i in range(training_epochs):
            sess.run(optimizer, feed_dict={X: inputX, Y: inputY})
            # Take a gradient descent step using our inputs and  labels

            # That's all! The rest of the cell just outputs debug messages.
            # Display logs per epoch step

            if (i) % display_step == 0:
                cc = sess.run(cost, feed_dict={X: inputX, Y: inputY})
                print("Training step:", '%04d' % (i), "cost=", "{:.35f}".format(cc))
                # print("\n  W1=", sess.run(W1), " \n W1=", sess.run(W2),
                # "\n b1=", sess.run(b1), "b2=", sess.run(b2) )

        print("\n ------------------------------------Optimization "
              "Finished!------------------------------------------\n")

        training_cost = sess.run(cost, feed_dict={X: inputX, Y: inputY})
        print("Training cost=", training_cost,
              "\n W1 = \n", sess.run(W1), "\n W2= \n", sess.run(W2),
              "\n b1=", sess.run(b1), '\n', "\n b2=", sess.run(b2), '\n')

        answer = tf.equal(tf.floor(hy + 0.1), Y)
        accuracy = tf.reduce_mean(tf.cast(answer, "float32"))

        print(sess.run([hy], feed_dict={X: inputX, Y: inputY}))
        print("Accuracy: ", accuracy.eval({X: inputX, Y: inputY}) * 100, "%")
        print("final Coast = ", training_cost)
        print("Parameters  :", "\n learning rate  = ", learning_rate, "\n epoches = ", training_epochs,
              " \n hidden layers  = ", n_hidden, "\n coast function \n optimazer gradient decent ")

        sess.close()
