import tensorflow as tf
from tensorflow.python.framework import ops
from Preprocessor import Preprocessor
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy import misc


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_h, n_w, n_c, n_y):

    X = tf.placeholder(tf.float32, shape=[None, n_h, n_w, n_c], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, n_y], name='Y')
    print(str(X))
    print(str(Y))

    return X, Y


def initialize_parameters():


    W1 = tf.get_variable("W1", [5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    # to add to regularizer
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W1)
    b1 = tf.get_variable("b1", [32], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [3, 3, 32, 48], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [48, 1], initializer=tf.zeros_initializer())

    W2 = tf.get_variable("W2", [3, 3, 48, 64], initializer=tf.contrib.layers.xavier_initializer())
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W2)
    b2 = tf.get_variable("b2", [64], initializer=tf.zeros_initializer())

    Wf = tf.get_variable("Wf", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W2)
    bf = tf.get_variable("bf", [128], initializer=tf.zeros_initializer())

    # Wf = tf.get_variable("Wf", [10, 30], initializer=tf.contrib.layers.xavier_initializer())
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, Wf)
    # bf = tf.get_variable("bf", [10, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "Wf": Wf,
                  "bf": bf
                   }

    return parameters


def forward_propagation(X, parameters, keep_prob, training):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Wf = parameters['Wf']
    bf = parameters['bf']

    # input = 32x32x3 o/p = 30x30x32
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    # A1 = tf.layers.batch_normalization(A1, training=training)
    # input = 30x30x32 o/p = 28x28x48
    Z1 = tf.nn.conv2d(A1, W3, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)

    # input = 28x28x48 o/p = 14x14x48
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # A1 = tf.nn.dropout(A1, keep_prob=keep_prob)

    #input = 14x14x48 o/p = 12x12x64
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding ='SAME')
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    # input = 12x12x64 o/p = 10x10x128
    Z2 = tf.nn.conv2d(A2, Wf, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)

    # A2 = tf.layers.batch_normalization(A2, training=training)
    #input = 10x10x128 o/p = 5x5x128
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    P2 = tf.contrib.layers.flatten(P2)
    Zf = tf.contrib.layers.fully_connected(P2, num_outputs=800, activation_fn=None)
    Zf = tf.contrib.layers.dropout(Zf, keep_prob=keep_prob, is_training=training)
    Zf = tf.contrib.layers.fully_connected(Zf, num_outputs=10, activation_fn=None)
    return Z1, Zf


def compute_cost(Zf, Y, beta=0.1):

    #regularizer
    # regularizer = tf.contrib.layers.l2_regularizer(beta)
    # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

    logits = Zf
    labels = Y
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    # Loss function using L2 Regularization
    loss = cost

    return loss


def model(X_train, Y_train, X_test, Y_test, op, file=None, learning_rate=0.001,
          num_epochs=11, minibatch_size=32, print_cost=True):

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (m, n_h, n_w, n_c) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[1]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_h, n_w, n_c, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z1, Zf = forward_propagation(X, parameters, keep_prob=keep_prob, training=training)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Zf, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    #Initialize saver
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        if op == 'train':
            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.  # Defines a cost related to an epoch
                train_acc = 0.
                validation_cost = 0
                num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
                minibatch_X = None
                minibatch_Y = None
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob:0.8, training:True})

                    epoch_cost += minibatch_cost / num_minibatches
                minibatch_validation_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test, keep_prob: 0.8, training:True})
                validation_cost += minibatch_validation_cost
                # Print the cost every epoch
                train_summary(epoch+1, epoch_cost, validation_cost, Zf, X, Y, minibatch_X, minibatch_Y, X_test, Y_test, keep_prob, training)
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
            saver.save(sess, './model/model')
            # plot the cost
            # plt.show()

        else:
            label_dic = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                         7: 'horse', 8: 'ship', 9: 'truck'}

            #Load model
            img = misc.imread(file)
            if not img.shape == (32, 32, 3):
                print("shape_issue")
            else:
                img = np.expand_dims(img, axis=0)
                saver.restore(sess, "./model/model")

                print(label_dic[sess.run(tf.argmax(Zf), feed_dict={X: img, keep_prob:1, training:False})[0]])
                layer_op = sess.run(Z1, feed_dict={X: img, keep_prob: 1, training: False})
                vv2 = layer_op[0, :, :, :]  # in case of bunch out - slice first img
                def vis_conv(v, ix, iy, ch, cy, cx, p=0):
                    v = np.reshape(v, (iy, ix, ch))
                    ix += 2
                    iy += 2
                    npad = ((1, 1), (1, 1), (0, 0))
                    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
                    v = v[:,:,:32]
                    v = np.reshape(v, (iy, ix, cy, cx))
                    v = np.transpose(v, (2, 0, 3, 1))  # cy,iy,cx,ix
                    v = np.reshape(v, (cy * iy, cx * ix))
                    return v

                # W_conv1 - weights
                ix = 32  # data size
                iy = 32
                ch = 48
                cy = 4  # grid from channels:  32 = 4x8
                cx = 8

                #  h_conv1 - processed image
                v = vis_conv(vv2, ix, iy, ch, cy, cx)
                plt.figure(figsize=(8, 8))
                plt.imshow(v, cmap="Greys_r", interpolation='nearest')
                plt.show()


# def train_accuracy(Zf, X, Y, X_test, Y_test, keep_prob, training):
#     # Calculate the correct predictions
#     correct_prediction = tf.equal(tf.argmax(Zf, 1), tf.argmax(Y, 1))
#     # Calculate accuracy on the test set
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#     # train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1, training:False})
#     test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1, training: False})
#     return test_accuracy

def train_summary(epoch, epoch_cost, validation_cost, Zf, X, Y, X_train, Y_train, X_test, Y_test, keep_prob, training):

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Zf, 1), tf.argmax(Y, 1))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1, training:False})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1, training:False})

    if epoch -1 == 0:
        print("Loop" + '\t' + "Train Loss" + '\t' + "Train Acc" + '\t' + "Test Loss" + '\t' + "Test Acc")
    print("%i \t %f \t %f \t %f \t %f" % (epoch, epoch_cost, train_accuracy*100, validation_cost, test_accuracy*100))



def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')

    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)

    sess.close

    return one_hot


def trainer(op, file):

    p = Preprocessor()
    X_train, labels_train, X_test, labels_test = p.load_data()
    Y_train = np.squeeze(one_hot_matrix(labels=labels_train, C=10), -1).T
    Y_test = np.squeeze(one_hot_matrix(labels=labels_test, C=10), -1).T
    X_train = X_train / 255
    X_test = X_test / 255
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    model(X_train, Y_train, X_test, Y_test, op, file)


def main(args):
    op = args[0]
    file = None
    if op == 'test':
        file = args[1]
    trainer(op, file)


if __name__ == "__main__":
    main(sys.argv[1:])



