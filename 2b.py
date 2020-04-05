# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

colors = ['k', 'y', 'c', 'm', 'g', 'r', 'b']
lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]

# Load data
def load_data(filename):
    all_data = np.loadtxt(filename, delimiter=',')
    data = dict()
    data['x'] = all_data[:, :1]
    data['y'] = all_data[:, 1:]

    # Random permutation
    n_samples = data['x'].shape[0]
    indices = np.random.permutation(n_samples)
    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

    # Normalization
    data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
    data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])
    
    return data['x'], data['y'], n_samples

# Feature matrix (degree = 3)
def create_feature_matrix(x, n_features=3):
    tmp_features = []
    for deg in range(1, n_features+1):
        tmp_features.append(np.power(x, deg))
    
    return np.column_stack(tmp_features)

# Regression
def polynomial_regression(data_x, data_y, n_samples, lmbd, n_features=3):
    
    tf.reset_default_graph()
    
    # Model
    X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(n_features))
    bias = tf.Variable(0.0)
    
    w_col = tf.reshape(w, (n_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)
    
    # Loss function
    Y_col = tf.reshape(Y, (-1, 1))
    
    # L2 regularizator
    l2_reg = lmbd * tf.reduce_mean(tf.square(w))
    # l2_reg = lmbd * tf.nn.l2_reg(w_col)
    
    mse = tf.reduce_mean(tf.square(hyp - Y_col))
    loss = tf.add(mse, l2_reg)
    
    # Optimizer
    opt_op = tf.train.AdamOptimizer().minimize(loss)
    
    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        epochs = 100
        for epoch in range(epochs):
            
            epoch_loss = 0
            for sample in range(n_samples):
                
                feed = {X: data_x[sample].reshape((1, n_features)), Y: data_y[sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss
            
            epoch_loss /= n_samples
            if (epoch + 1) % 10 == 0:
                print('Lambda: {} | Epoch: {}/{} | avg loss: {:.5f}'.format(lmbd, epoch+1, epochs, epoch_loss)) 

        # Final values
        w_val = sess.run(w)
        bias_val = sess.run(bias)
        print('w = ', w_val, 'bias = ', bias_val)
        xs = create_feature_matrix(np.linspace(-2, 4, 100))
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        final_loss = sess.run(loss, feed_dict={X: data_x, Y: data_y})
        print('final loss = ', final_loss)
        
        return xs, hyp_val, final_loss
    
    
def main():
    np.set_printoptions(suppress=True, precision=5)
    
    data_x, data_y, n_samples = load_data('./data/corona.csv')
    
    plt.scatter(data_x[:, 0], data_y, c='lightblue')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    data_x = create_feature_matrix(data_x)
    
    final_cost_f = []
    for i in range(len(lambdas)):
        xs, hyp_val, final_loss = polynomial_regression(data_x, data_y, n_samples, lambdas[i])
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=colors[i], label='lmbd={}'.format(lambdas[i]))
        final_cost_f.append(final_loss)
        
    # Graph 1
    plt.xlim([-2, 4])
    plt.ylim([-3, 2])
    plt.legend()
    plt.show()
    
    # Graph 2
    plt.scatter(lambdas, final_cost_f, c='r')
    plt.xlabel('lambda')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    main()
    
# %%

