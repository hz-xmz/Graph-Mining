import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from node2vec import Node2Vec
import networkx as nx

def read_data(path='data/facebook/facebook_combined.txt'):
    edges = []
    with open(path) as f:
        for line in f:
            edges.append(line.strip().split())
    edges_np = np.array(edges)
    nodes_set = set(edges_np.flatten())
    nodes_np = np.array([node for node in nodes_set])
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
    
    
def link_pred(node2feat, edges_p, edges_n):
    size = len(edges_p)
    scores_p = []
    scores_n = []
    for i in range(size):
        try:
            x, y = edges_p[i]
            xv, yv = node2feat[x], node2feat[y]
            score = np.dot(xv, yv)/(np.linalg.norm(xv)*np.linalg.norm(yv))
            scores_p.append(score)
        except Exception as e:
            pass
            # print('exception: ', e)
    for edge in edges_n:
        try:
            x, y = edge
            xv, yv = node2feat[x], node2feat[y]
            score = np.dot(xv, yv)/(np.linalg.norm(xv)*np.linalg.norm(yv))
            scores_n.append(score)
        except Exception as e:
            pass
            # print('exception: ', e)
    return scores_p, scores_n
    

def main(args):
    # path = os.path.join(args.data_home, args.dataset, args.dataset)
    if args.scratch:
        edges, nodes = read_data()
        print(f"edges.shape: {edges.shape}, nodes.shape: {nodes.shape}")
        nodes_list = nodes
        print(f"len(nodes_list): {len(nodes_list)}")
        edges_train, edges_test = train_test_split(edges, test_size=0.5, random_state=2)
        print(f"edges_train.shape: {edges_train.shape}, edges_test.shape: {edges_test.shape}")
        nodes_set_train = set(edges_train.flatten())
        nodes_list_train = np.array([node for node in nodes_set_train])
        nodes_set_test = set(edges_test.flatten())
        nodes_list_test = np.array([node for node in nodes_set_test])
        path = os.path.join(args.data_home, args.dataset,)
        np.save(os.path.join(args.data_home, args.dataset, "edges.npy"), edges)
        np.save(os.path.join(args.data_home, args.dataset, "nodes.npy"), nodes)
        np.save(os.path.join(args.data_home, args.dataset, "edges_train.npy"), edges_train)
        np.save(os.path.join(args.data_home, args.dataset, "edges_test.npy"), edges_test)
        np.save(os.path.join(args.data_home, args.dataset, "nodes_list_train.npy"), nodes_list_train)
        np.save(os.path.join(args.data_home, args.dataset, "nodes_list_test.npy"), nodes_list_test)
    else:
        edges = np.load(os.path.join(args.data_home, args.dataset, "edges.npy"))
        nodes = np.load(os.path.join(args.data_home, args.dataset, "nodes.npy"))
        edges_train = np.load(os.path.join(args.data_home, args.dataset, "edges_train.npy"))
        edges_test = np.load(os.path.join(args.data_home, args.dataset, "edges_test.npy"))
        nodes_list_train = np.load(os.path.join(args.data_home, args.dataset, "nodes_list_train.npy"))
        nodes_list_test = np.load(os.path.join(args.data_home, args.dataset, "nodes_list_test.npy"))
        nodes_list = nodes
    edges_nega = np.load(os.path.join(args.data_home, args.dataset, "edges_nega.npy"))
    print(f"edges.shape: {edges.shape}, nodes.shape: {nodes.shape}")
    print(f"len(nodes_list): {len(nodes_list)}")
    print(f"edges_train.shape: {edges_train.shape}, edges_test.shape: {edges_test.shape}")
    print(f"nodes_list_train.shape: {nodes_list_train.shape}, nodes_list_test.shape: {nodes_list_test.shape}")
    print(f"edges_nega.shape: {edges_nega.shape}")
    
    if args.extract:
        graph = nx.from_edgelist(edges_train)
        eigen_vectors, features = extract_features(graph, nodes_list_train)
        print(f"eigen_vectors.shape: {eigen_vectors.shape}, features.shape: {features.shape}")
        np.save(os.path.join(args.data_home, args.dataset, 'eigenvec.npy'), eigen_vectors)
        np.save(os.path.join(args.data_home, args.dataset, 'features.npy'), features)
    if args.feats:
        eigen_vectors = np.load(os.path.join(args.data_home, args.dataset, 'eigenvec.npy'))
        features = np.load(os.path.join(args.data_home, args.dataset, 'features.npy'))
        print(f"eigen_vectors.shape: {eigen_vectors.shape}, features.shape: {features.shape}")
        feats_hstack = np.hstack((eigen_vectors[:, :args.truncate], features))
        print(f"feats_hstack.shape: {feats_hstack.shape}")
        node2feats = {}
        for i, feat_h in enumerate(feats_hstack):
            node2feats[nodes_list_train[i]] = feat_h
        scores_p, scores_n = link_pred(node2feats, edges_test, edges_nega)
        print(f"len(scores_p): {len(scores_p)}, len(scores_n): {len(scores_n)}")
        label_true = [1 for _ in range(len(scores_p))]
        label_false = [0 for _ in range(len(scores_n))]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(label_true+label_false, scores_p+scores_n)
        print(f"auc: {auc: .4f}")
    if args.node2vec:
        graph = nx.from_edgelist(edges_train)
        nodes2vecs = Node2Vec(graph, dimensions=args.dimvec, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers)
        model = nodes2vecs.fit(window=5, min_count=1, batch_words=4)
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}.vec')
        model.wv.save(EMBEDDING_FILENAME)
        EMBEDDING_MODEL_FILENAME = os.path.join(args.data_home, args.dataset, f'node2vec_{args.dimvec}.model')
        model.save(EMBEDDING_MODEL_FILENAME)
    if args.vecs:
        from gensim.models import KeyedVectors
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}.vec')
        wv = KeyedVectors.load(EMBEDDING_FILENAME)
        scores_p, scores_n = link_pred(wv, edges_test, edges_nega)
        print(f"len(scores_p): {len(scores_p)}, len(scores_n): {len(scores_n)}")
        label_true = [1 for _ in range(len(scores_p))]
        label_false = [0 for _ in range(len(scores_n))]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(label_true+label_false, scores_p+scores_n)
        print(f"auc: {auc: .4f}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="link_prediction")
    parser.add_argument("--data_home", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="facebook")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--feats", action="store_true")
    parser.add_argument("--vecs", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--node2vec", action="store_true")
    parser.add_argument("--truncate", type=int, default=1024)
    parser.add_argument("--iter_max", type=int, default=500)
    parser.add_argument("--dimvec", type=int, default=200)
    parser.add_argument("--walk_length", type=int, default=30)
    parser.add_argument("--num_walks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('-cl','--c_list', nargs='+', help='<Required> Set flag', type=float, default=[0.1, 1, 5, 10, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path_save", type=str, default='models')
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)



