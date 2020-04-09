# %%

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def config():
    filename = './data/Prostate_Cancer.csv'
    k = 3
    n_classes = 2
    n_features = 4
    
    return filename, k, n_classes, n_features

class KNN:
    
    def __init__(self, n_features, n_classes, data, k):
        self.n_features = n_features
        self.n_classes = n_classes
        self.data = data
        self.k = k
        # self.weighted = weighted
        
        # Model 
        # X - matrica podataka, Q - vektor upita 
        self.X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
        self.Q = tf.placeholder(shape=(n_features), dtype=tf.float32)

        # Racunamo kvadratnu euklidstku udaljenost i uzimamo minimalnih k
        dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), axis=1))
        _, idxs = tf.nn.top_k(-dists, self.k)
        
        self.classes = tf.gather(self.Y, idxs)
        self.dists = tf.gather(dists, idxs)
        
        # if weighted:
        #     self.w = 1 / self.dists
        # else:
        #     self.w = tf.fill([k], 1/k)
        
        # Mnozimo svaki red svojim glasom i sabiramo glasove po kolonama
        # w_col = tf.reshape(self.w, (k, 1))
        self.classes_one_hot = tf.one_hot(self.classes, n_classes)
        self.scores = tf.reduce_sum(self.classes_one_hot, axis=0)
        # self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)
        
        # Klasa sa najvise glasova je hipoteza
        self.hyp = tf.argmax(self.scores)
        
    # Ako imamo odgovore za upit racunamo accuracy
    def predict(self, test_data):
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            n_queries = test_data['x'].shape[0]
            matches = 0
            
            for i in range(n_queries):
                
                feed = {
                    self.X: self.data['x'],
                    self.Y: self.data['y'],
                    self.Q: test_data['x'][i]
                }
                
                hyp_val = sess.run(self.hyp, feed_dict=feed)
                
                if test_data['y'] is not None:
                    actual = test_data['y'][i]
                    match = (int(hyp_val) == int(actual))
                    if match:
                        matches += 1
                    # if i % 2 == 0:
                    #     print('Test example: {}/{} | Predicted: {} | Actual: {} | Match: {}'.format(i+1, n_queries, hyp_val, actual, match))
                        
            acurracy = matches / n_queries
            print('{} matches out of {} examples'.format(matches, n_queries))
            
            return acurracy
        
# Get data from file and store it in data dict
def get_data(filename):
    data = dict()
    data['x'] = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(2, 3, 4, 5))
    data['y'] = np.loadtxt(filename, dtype=str, delimiter=',', skiprows=1, usecols=1)
    
    return data

# Random permutation
def shuffle(data, n_samples):
    indices = np.random.permutation(n_samples)
    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

# Normalization
def normalize(data):
    data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
    
# Plot first graph
def show_graph(data, idxs_0, idxs_1):
    plt.scatter(data['x'][idxs_0, 0], data['x'][idxs_0, 1], c='b', edgecolors='k', label='benigni')
    plt.scatter(data['x'][idxs_1, 0], data['x'][idxs_1, 1], c='r', edgecolors='k', label='manigni')
    plt.legend()
    plt.show()
    
def main():
    
    filename, k, n_classes, n_features = config()
    
    data = get_data(filename)
    
    predictions = len(data['y'])
    
    for i in range(0, predictions):
        if data['y'][i] == 'B':
            data['y'][i] = 0
        else:
            data['y'][i] = 1
    
    n_samples = data['x'].shape[0]
    
    shuffle(data, n_samples)
    normalize(data)
    
    train_ratio = 0.8
    n_train = int(train_ratio * n_samples)
    
    train_data = dict()
    train_data['x'] = data['x'][:n_train]
    train_data['y'] = data['y'][:n_train]
    
    test_data = dict()
    test_data['x'] = data['x'][n_train:]
    test_data['y'] = data['y'][n_train:]
    
    knn = KNN(n_features, n_classes, train_data, k)
    accuracy = knn.predict(test_data)
    
    print('Accuracy: ', accuracy)
 
    idxs_0 = []
    idxs_1 = []
    
    for i in range(0, len(train_data['y'])):
        
        if int(train_data['y'][i]) == 0:
            idxs_0.append(i)
        else:
            idxs_1.append(i)
        
    # Graph 1
    show_graph(train_data, idxs_0, idxs_1)
    
if __name__ == '__main__':
    main()

# %%
