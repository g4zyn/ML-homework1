# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def config():
    filename = './data/Prostate_Cancer.csv'
    n_classes = 2
    n_features = 4
    k_range = [1, 15]
    
    return filename, n_classes, n_features, k_range

class KNN():
    def __init__(self, n_features, n_classes, data, k):
        self.n_features = n_features
        self.n_classes = n_classes
        self.data = data
        self.k = k
        
        self.X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
        self.Q = tf.placeholder(shape=(n_features), dtype=tf.float32)
        
        dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), axis=1))
        _, idxs = tf.nn.top_k(-dists, self.k)
        
        self.classes = tf.gather(self.Y, idxs)
        self.dists = tf.gather(dists, idxs)
        
        self.classes_one_hot = tf.one_hot(self.classes, n_classes)
        self.scores = tf.reduce_sum(self.classes_one_hot, axis=0)
        
        self.hyp = tf.argmax(self.scores) 

    def predict(self, test_data):
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            n_queries = test_data['x'].shape[0]
            matches = 0
            
            for i in range(n_queries):
                feed = {
                    self.X: self.data['x'],
                    self.Y: self.data['y'],
                    self.Q: self.data['x'][i]
                }
                hyp_val = sess.run(self.hyp, feed_dict=feed)
    
                if test_data['y'] is not None:
                    actual = test_data['y'][i]
                    match = (int(hyp_val) == int(actual))
                    if match:
                        matches += 1
                    
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
    
# Plot dependance graph
def show_graph(k_differ, accuracies):
    _, drawing = plt.subplots()
    drawing.plot(k_differ, accuracies, c='g')
    drawing.set(xlabel='k', ylabel='Accuracy', title='Dependence Accuracy of value k in k-NN')

    drawing.grid()
    plt.show()

def main():
    
    filename, n_classes, n_features, k_range = config()
    
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
    
    accuracies = []
    k_differ = []
    avg_accuracy = 0
    
    for i in range(k_range[0], k_range[1]+1):
        knn = KNN(n_features, n_classes, train_data, i)
        accuracy = knn.predict(test_data)
        avg_accuracy += accuracy
        accuracies.append(accuracy)
        k_differ.append(i)
        print('Accuracy = {:.2f} | k = {}'.format(accuracy, i))    
        
    avg_accuracy /= k_range[1]
    print('Average accuracy: ', avg_accuracy)

    show_graph(k_differ, accuracies)
    
if __name__ == '__main__':
    main()
    
# %%