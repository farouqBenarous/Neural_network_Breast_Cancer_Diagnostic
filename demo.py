import pandas as pd              # A beautiful library to help us work with data as tables
import tensorflow as tf          # Fire from the gods

if __name__ == "__main__":

    # Let's have Pandas load our dataset as a dataframe
    dataframe = pd.read_csv("datasetcsv.csv")
    # remove columns we don't care about
    dataframe = dataframe.drop(["unknown.17","unknown.16","unknown.15","unknown.14","unknown.13","unknown.12",
                                "unknown.11","unknown.10","unknown.9","unknown.8","unknown.7","unknown.6","unknown.5"
                                   ,"unknown.4","unknown.3","unknown.2","unknown.1","unknown"],
                                axis=1)
    # We'll only use the first 10 rows of the dataset in this example
    dataframe = dataframe[0:30]
    #Let's have the notebook show us how the dataframe looks now


    inputX = dataframe.loc[:, ['radius ', 'texture','perimeter','area','smoothness ','compactness ','concavity','concave  '
                                  ,'points','symmetry','fractal',' dimension']].values

    inputY = dataframe.loc[:,["Label"]].values


    for x  in  range(inputY.size):
        if( inputY[x] == "M") :
            inputY[x] = 1
        else:
            inputY[x] =0




    #Let's prepare some parameters for the training process

    # Parameters
    learning_rate = 0.000001
    training_epochs = 2000 #simply iterations
    display_step = 50
    n_samples = inputY.size  # number of the instances

    #now lets build our NN Model

    x = tf.placeholder(tf.float32, [None, 12])  # Okay TensorFlow, we'll feed you an array of examples. Each example

    # be an array of 12  real values (radius ,texture,perimeter,area,smoothness ,compactness ,concavity,concave
    # ,points,symmetry,fractal, dimension).
    # "None" means we can feed you any number of examples
    # Notice we haven't fed it the values yet

    W = tf.Variable(tf.zeros([12, 1]))
    # Maintain a 12 x 12 real matrix for the weights that we'll keep updating
    # through the training process (make them all zero to begin with)

    b = tf.Variable(tf.zeros([2]))  # Also maintain two bias values

    y_values = tf.add(tf.matmul(x, W), b)
    # The first step in calculating the prediction would be to multiply
    # the inputs matrix by the weights matrix then add the biases

    y = tf.nn.softmax(y_values)
    # Then we use softmax as an "activation function" that translates the
    # numbers outputted by the previous layer into probability form

    y_ = tf.placeholder(tf.float32, [None,1])
    # For training purposes, we'll also feed you a matrix of labels
    # Cost function: Mean squared error
    cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n_samples)
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize variabls and tensorflow session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # lets Do  Our real traing
    for i in range(training_epochs):
        sess.run(optimizer,feed_dict={x: inputX, y_: inputY})
        # Take a gradient descent step using our inputs and  labels

        # That's all! The rest of the cell just outputs debug messages.
        # Display logs per epoch step
        if (i) % display_step == 0:
            cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
            print("Training step:", '%04d' % (i), "cost=", "{:.35f}".format(cc) )
            # , \"W=", sess.run(W), "b=", sess.run(b))

    print("\n ------------------------------------Optimization Finished!------------------------------------------\n")

    training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
    print("Training cost=", training_cost, "\n W= \n", sess.run(W), "\n b=", sess.run(b), '\n')

    sess.run(y, feed_dict={x: inputX})

    sess.run(tf.nn.softmax([1., 2.]))

    sess.close()


