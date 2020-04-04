# %%
import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt

MAX_FEATURES = 6
colors = ['k', 'y', 'c', 'm', 'g', 'r']

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

# Feature matrix
def create_feature_matrix(x, n_features):
    tmp_features = []
    for deg in range(1, n_features+1):
        tmp_features.append(np.power(x, deg))
    
    return np.column_stack(tmp_features)

def polynomial_regression(data_x, data_y, n_samples, n_features):
    
    tf.reset_default_graph()
    data_x = create_feature_matrix(data_x, n_features)
    
    # Model
    X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(n_features))
    bias = tf.Variable(0.0)
    
    w_col = tf.reshape(w, (n_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)
    
    # Loss function
    Y_col = tf.reshape(Y, (-1, 1))
    mse = tf.reduce_mean(tf.square(hyp - Y_col))
    
    # AdamOptimizer
    opt_op = tf.train.AdamOptimizer().minimize(mse)
    
    # Training
    n_epochs = 100
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for sample in range(n_samples):
                feed = {X: data_x[sample].reshape((1, n_features)), Y: data_y[sample]}
                _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
                epoch_loss += curr_loss
                
            epoch_loss /= n_samples
            if (epoch + 1) % 10 == 0:
                print('Degree: {} | Epoch: {}/{} | avg loss: {:.5f}'.format(n_features, epoch+1, n_epochs, epoch_loss))
        
        w_val = sess.run(w)
        bias_val = sess.run(bias)
        print('w = ', w_val, 'bias = ', bias_val)
        xs = create_feature_matrix(np.linspace(-2, 4, 100), n_features)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        final_loss = sess.run(mse, feed_dict={X: data_x, Y: data_y})
        print('Final loss = ', final_loss)
        
        return xs, hyp_val, final_loss

def main():
    np.set_printoptions(suppress=True, precision=5)
    
    data_x, data_y, n_samples = load_data('./data/corona.csv')
    
    plt.scatter(data_x[:, 0], data_y, c='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    final_cost_f = []
    n_features = 1
    
    for i in range(MAX_FEATURES):    
        xs, hyp_val, final_loss = polynomial_regression(data_x, data_y, n_samples, n_features)
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=colors[i], label='deg(p)={}'.format(n_features))
        final_cost_f.append(final_loss)
        n_features += 1
        
    # graph 1
    plt.xlim([-2, 4])
    plt.ylim([-3, 2])
    plt.legend()
    plt.show()
    
    # graph 2
    plt.scatter([1, 2, 3, 4, 5, 6], final_cost_f, c='r')
    plt.xlabel('degree')
    plt.ylabel('loss')
    plt.show()
 
if __name__ == "__main__":
    main()
    
# %%
