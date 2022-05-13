import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from node2vec import Node2Vec
import networkx as nx

def read_data(path):
    citef = path+".cites"
    contentf = path+".content"
    edges = []
    nodes = []
    with open(citef) as f:
        for line in f:
            edges.append(line.strip().split())
    with open(contentf) as f:
        for line in f:
            nodes.append(line.strip().split())
    edges_np = np.array(edges)
    nodes_np = np.array(nodes)
    return edges_np, nodes_np


# nodes features
def extract_features(graph, graph_nodes_list):
    degree_c = nx.degree_centrality(graph)
    # katz_c = nx.katz_centrality(graph)
    close_c = nx.closeness_centrality(graph)
    cluster_coef = nx.clustering(graph)
    graph_np = nx.to_numpy_array(graph, nodelist=graph_nodes_list)
    eigen_values, eigen_vectors = np.linalg.eigh(graph_np)
    sorted_index = np.argsort(eigen_values)[::-1]
    eigen_vectors_sorted = eigen_vectors[:, sorted_index]
    features = [ [degree_c[node], close_c[node], cluster_coef[node]] for node in graph_nodes_list]
    return eigen_vectors_sorted, np.array(features)


def lr_fit(X_train, y_train, X_test, y_test, C=0.5, max_iter=100):
    lr = LogisticRegression(C=C, max_iter=max_iter)
    lr.fit(X_train, y_train)
    score_train = lr.score(X_train, y_train)
    score_test = lr.score(X_test, y_test)
    print(f"score_train: {score_train: .4f}, score_test: {score_test: .4f}")
    return lr, score_train, score_test


def cross_valid(features, labels, C_list=[0.1, 0.5, 1, 1.5, 2, 2.5, 5], iter_max=200):
    print("cross validation")
    c_best = 1
    s_best = 0
    # lr_best = None
    for C in C_list:
        clf = LogisticRegression(C=C, max_iter=iter_max)
        scores = cross_val_score(clf, features, labels, cv=5)
        score_ave = np.mean(scores)
        if score_ave > s_best:
            c_best = C
            s_best = score_ave
            # lr_best = clf
        print(f"C: {C:2.2f}, score_ave: {score_ave :.4f}")
    print(f"c_best: {c_best:2.1f}, s_best: {s_best :.4f}")
    return c_best


def learn(features, labels, C_list=[0.1, 0.5, 1, 2, 5], iter_max=200):
    index = np.arange(len(features))
    index_train, index_test = train_test_split(index, test_size=1000, random_state=2)
    print(f"index_train: {index_train.shape}, index_test: {index_test.shape}")
    
    c_best = cross_valid(features, labels, C_list, iter_max)
    print("train and test")
    lr, score_train, score_test = lr_fit(features[index_train], labels[index_train], features[index_test], labels[index_test], C=c_best)
    

def main(args):
    path = os.path.join(args.data_home, args.dataset, args.dataset)
    edges, nodes = read_data(path)
    print(f"edges.shape: {edges.shape}, nodes.shape: {nodes.shape}")
    nodes_list = nodes[:, 0]
    nodes_class = list(set(nodes[:, -1]))
    print(f"len(nodes_list): {len(nodes_list)}, len(nodes_class): {len(nodes_class)}")
    class2label = {c:i for i,c in enumerate(nodes_class)}
    print("class2label: ", class2label)
    labels = np.array([class2label[c] for c in nodes[:, -1]])
    
    if args.count:
        onehots = nodes[:, 1: -1].astype('float64')
        print(f"onehots.shape: {onehots.shape}, labels.shape: {labels.shape}")
        np.save(os.path.join(args.data_home, args.dataset, 'onehots.npy'), onehots)
    if args.words:
        features = np.load(os.path.join(args.data_home, args.dataset, 'onehots.npy'))
        print(f"features.shape: {features.shape}, labels.shape: {labels.shape}")
        learn(features, labels, C_list=args.c_list, iter_max=args.iter_max)
    if args.extract:
        graph = nx.from_edgelist(edges)
        eigen_vectors, features = extract_features(graph, nodes_list)
        print(f"eigen_vectors.shape: {eigen_vectors.shape}, features.shape: {features.shape}")
        np.save(os.path.join(args.data_home, args.dataset, 'eigenvec.npy'), eigen_vectors)
        np.save(os.path.join(args.data_home, args.dataset, 'features.npy'), features)
    if args.feats:
        eigen_vectors = np.load(os.path.join(args.data_home, args.dataset, 'eigenvec.npy'))
        features = np.load(os.path.join(args.data_home, args.dataset, 'features.npy'))
        print(f"eigen_vectors.shape: {eigen_vectors.shape}, features.shape: {features.shape}")
        feats_hstack = np.hstack((eigen_vectors[:, :args.truncate], features))
        print(f"feats_hstack.shape: {feats_hstack.shape}, labels.shape: {labels.shape}")
        learn(feats_hstack, labels, C_list=args.c_list, iter_max=args.iter_max)
    if args.node2vec:
        graph = nx.from_edgelist(edges)
        nodes2vecs = Node2Vec(graph, dimensions=args.dimvec, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers)
        model = nodes2vecs.fit(window=args.windows, min_count=args.min_count, batch_words=args.batch_words)
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}.vec')
        model.wv.save(EMBEDDING_FILENAME)
        EMBEDDING_MODEL_FILENAME = os.path.join(args.data_home, args.dataset, f'node2vec_{args.dimvec}.model')
        model.save(EMBEDDING_MODEL_FILENAME)
    if args.vecs:
        from gensim.models import KeyedVectors
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}.vec')
        wv = KeyedVectors.load(EMBEDDING_FILENAME)
        nodes2vecs = np.array([wv[node] for node in nodes_list])
        print(f"nodes2vecs.shape: {nodes2vecs.shape}, labels.shape: {labels.shape}")
        learn(nodes2vecs, labels, C_list=args.c_list, iter_max=args.iter_max)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="node_classification")
    parser.add_argument("--data_home", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--feats", action="store_true")
    parser.add_argument("--vecs", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--node2vec", action="store_true")
    parser.add_argument("--truncate", type=int, default=512)
    parser.add_argument("--iter_max", type=int, default=500)
    parser.add_argument("--dimvec", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=80)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--batch_words", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument('-cl','--c_list', nargs='+', help='<Required> Set flag', type=float, default=[0.1, 1, 5, 10, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path_save", type=str, default='models')
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)



