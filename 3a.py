import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class KNN:
    
    def __init__(self, n_features, n_classes, data, k, weighted=False):
        self.n_features = n_features
        self.n_classes = n_classes
        self.data = data
        self.k = k
        self.weighted = weighted
        
        # Model 
        # X - matrica podataka, Q - vektor upita 
        self.X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
        self.Q = tf.placeholder(shape=(n_features), dtype=tf.float32)

        # Racunamo kvadratnu euklidstku udaljenost i uzimamo minimalnih k
        dists = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.X, self.Q)), axis=1))
        _, idxs = tf.nn.top_k(-dists, self.k)
        
        self.classes = tf.gather(self.Y, idxs)
        self.dists = tf.gather(dists, idxs)
        
        if weighted:
            self.w = 1 / self.dists
        else:
            self.w = tf.fill([k], 1/k)
        
        # Mnozimo svaki red svojim glasom i sabiramo glasove po kolonama
        w_col = tf.reshape(self.w, (k, 1))
        self.classes_one_hot = tf.one_hot(self.classes, n_classes)
        self.scores = tf.reduce_mean(w_col * self.classes_one_hot, axis=0)
        
        # Klasa sa najvise glasova je hipoteza
        self.hyp = tf.arg_max(self.scores)
        
    # Ako imamo odgovore za upit racunamo accuracy
    def predict(self, query_data):
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            n_queries = query_data['x'].shape[0]
            matches = 0
            
            for i in range(n_queries):
                
                feed = {
                    self.X: self.data['x'],
                    self.Y: self.data['y'],
                    self.Q: self.data['x'][i]
                }
                
                hyp_val = sess.run(self.hyp, feed_dict=feed)
                
                if query_data['y'] is not None:
                    actual = query_data['y'][i]
                    match = (hyp_val == actual)
                    if match:
                        matches += 1
                    if i % 10 == 0:
                        print('Test example: {}/{} | Predicted: {} | Actual: {} | Match: {}'.format(i+1, n_queries, hyp_val, actual, match))
                        
            acurracy = matches / n_queries
            print('{} matches out of {} examples'.format(matches, n_queries))
            
            return acurracy
