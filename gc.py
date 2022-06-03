import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from node2vec import Node2Vec
import networkx as nx

def read_data(path='data/email'):
    file_edges = os.path.join(path, 'email-Eu-core.txt')
    file_labels = os.path.join(path, 'email-Eu-core-department-labels.txt')
    labels = []
    edges  = []
    with open(file_edges) as f:
        for line in f:
            edges.append(line.strip().split())
    with open(file_labels) as f:
        for line in f:
            labels.append(line.strip().split())
    edges_np = np.array(edges)
    nodes_np = np.array(labels)
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


def extract_spectral(L):
    eigen_values, eigen_vectors = np.linalg.eigh(L)
    sorted_index = np.argsort(eigen_values)[::-1]
    eigen_vectors_sorted = eigen_vectors[:, sorted_index]
    return eigen_vectors_sorted
    
    
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


def cluster(feats, labels, n_cluster = [5, 10, 15, 20, 25, 30, 35, 40]):
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    n2scores = {}
    for n in n_cluster:
        kmeans = KMeans(n_clusters=n, random_state=2)
        preds = kmeans.fit_predict(feats)
        score_nmi = nmi(preds, labels)
        n2scores[n] = score_nmi
    return n2scores

def spectral_cluster(spectral, labels, n_cluster = [5, 10, 15, 20, 25, 30, 35, 40]):
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    from sklearn.preprocessing import normalize
    n2scores = {}
    for n in n_cluster:
        feats = normalize(spectral[:, :50])
        print(f"feats.shape: {feats.shape}")
        kmeans = KMeans(n_clusters=n, random_state=2)
        preds = kmeans.fit_predict(feats)
        score_nmi = nmi(preds, labels)
        n2scores[n] = score_nmi
    return n2scores

    
def main(args):
    # path = os.path.join(args.data_home, args.dataset, args.dataset)
    if args.scratch:
        edges, nodes = read_data()
        path = os.path.join(args.data_home, args.dataset,)
        np.save(os.path.join(args.data_home, args.dataset, "edges.npy"), edges)
        np.save(os.path.join(args.data_home, args.dataset, "nodes.npy"), nodes)
    else:
        edges = np.load(os.path.join(args.data_home, args.dataset, "edges.npy"))
        nodes = np.load(os.path.join(args.data_home, args.dataset, "nodes.npy"))
        
    nodes_list = nodes[:, 0]
    labels_list = [int(label) for label in nodes[:, 1]]
    node2label = {r[0]: int(r[1]) for r in nodes}
    print(f"edges.shape: {edges.shape}, nodes.shape: {nodes.shape}")
    print(f"nodes_list.shape: {nodes_list.shape}, len(labels_list): {len(labels_list)}")
    print('len(node2label): ', len(node2label))
    
    if args.spectral:
        graph = nx.from_edgelist(edges)
        laplacian = nx.normalized_laplacian_matrix(graph, nodes_list)
        print(type(laplacian), laplacian.shape)
        spectral = extract_spectral(laplacian.todense())
        print(f"spectral.shape: {spectral.shape}")
        np.save(os.path.join(args.data_home, args.dataset, 'spectral.npy'), spectral)
    
    if args.extract:
        graph = nx.from_edgelist(edges)
        eigen_vectors, features = extract_features(graph, nodes_list)
        print(f"eigen_vectors.shape: {eigen_vectors.shape}, features.shape: {features.shape}")
        np.save(os.path.join(args.data_home, args.dataset, 'eigenvec.npy'), eigen_vectors)
        np.save(os.path.join(args.data_home, args.dataset, 'features.npy'), features)
        
    if args.feats:
        from sklearn.preprocessing import normalize
        spectral = np.load(os.path.join(args.data_home, args.dataset, 'spectral.npy'))
        scores = spectral_cluster(spectral, labels_list)
        print(f"cluster scores: ")
        score_list = []
        for n in scores:
            print(f"n={n}, nmi: {scores[n]: .4f}")
            score_list.append(scores[n])
        print(f"average of scores: {np.mean(score_list): .4f}")
            
    if args.node2vec:
        graph = nx.from_edgelist(edges)
        nx.set_node_attributes(graph, node2label, 'label')
        nodes2vecs = Node2Vec(graph, dimensions=args.dimvec, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers)
        model = nodes2vecs.fit(window=args.windows, min_count=args.min_count, batch_words=args.batch_words)
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}_{args.dmax}.vec')
        model.wv.save(EMBEDDING_FILENAME)
        EMBEDDING_MODEL_FILENAME = os.path.join(args.data_home, args.dataset, f'node2vec_{args.dimvec}_{args.dmax}.model')
        model.save(EMBEDDING_MODEL_FILENAME)
        
    if args.node2wvec:
        graph = nx.from_edgelist(edges)
        centrality = nx.eigenvector_centrality(graph)
        weighted_edges_list = []
        for e in graph.edges:
            # d = min()
            # weighted_edges_list.append([e[1], e[0], 1+2*centrality[e[0]]])
            # weighted_edges_list.append([e[0], e[1], 1+2*centrality[e[1]]])
            weighted_edges_list.append([e[1], e[0], min(graph.degree[e[0]], args.dmax)])
            weighted_edges_list.append([e[0], e[1], min(graph.degree[e[1]], args.dmax)])
        print(f'len(weighted_edges_list): {len(weighted_edges_list)}')
        print('weighted_edges_list: ', weighted_edges_list[:10])
        DG = nx.DiGraph()
        DG.add_weighted_edges_from(weighted_edges_list)
        print('DG weights 0: ', DG['0'])
        print('DG weights 1: ', DG['1'])
        nodes2vecs = Node2Vec(DG, dimensions=args.dimvec, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers)
        model = nodes2vecs.fit(window=args.windows, min_count=args.min_count, batch_words=args.batch_words)
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}_w{args.dmax}.vec')
        model.wv.save(EMBEDDING_FILENAME)
        EMBEDDING_MODEL_FILENAME = os.path.join(args.data_home, args.dataset, f'node2vec_{args.dimvec}_w{args.dmax}.model')
        model.save(EMBEDDING_MODEL_FILENAME)
        
    if args.node2cvec:
        graph = nx.from_edgelist(edges)
        centrality = nx.eigenvector_centrality(graph)
        jc = nx.jaccard_coefficient(graph, edges)
        for u, v, p in jc:
            graph[u][v]['weight'] = 1 + p*args.dmax
        # weighted_edges_list = []
        # for e in graph.edges:
        #     weighted_edges_list.append([e[1], e[0], 1+2*centrality[e[0]]])
        #     weighted_edges_list.append([e[0], e[1], 1+2*centrality[e[1]]])
        #     # weighted_edges_list.append([e[1], e[0], min(graph.degree[e[0]], args.dmax)])
        #     # weighted_edges_list.append([e[0], e[1], min(graph.degree[e[1]], args.dmax)])
        # print(f'len(weighted_edges_list): {len(weighted_edges_list)}')
        # print('weighted_edges_list: ', weighted_edges_list[:10])
        # DG = nx.DiGraph()
        # DG.add_weighted_edges_from(weighted_edges_list)
        # print('DG weights 0: ', DG['0'])
        # print('DG weights 1: ', DG['1'])
        # nodes2vecs = Node2Vec(DG, dimensions=args.dimvec, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers)
        nodes2vecs = Node2Vec(graph, dimensions=args.dimvec, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers)
        model = nodes2vecs.fit(window=args.windows, min_count=args.min_count, batch_words=args.batch_words)
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}_c{args.dmax}.vec')
        model.wv.save(EMBEDDING_FILENAME)
        EMBEDDING_MODEL_FILENAME = os.path.join(args.data_home, args.dataset, f'node2vec_{args.dimvec}_c{args.dmax}.model')
        model.save(EMBEDDING_MODEL_FILENAME)
        
    if args.vecs:
        from gensim.models import KeyedVectors
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}_{args.dmax}.vec')
        wv = KeyedVectors.load(EMBEDDING_FILENAME)
        feats = [wv[node] for node in nodes_list]
        scores = cluster(feats, labels_list, n_cluster=args.c_list)
        print(f"cluster scores: ")
        score_list = []
        for n in scores:
            print(f"n={n}, nmi: {scores[n]: .4f}")
            score_list.append(scores[n])
        print(f"average of scores: {np.mean(score_list): .4f}")
            
    if args.wvecs:
        from gensim.models import KeyedVectors
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}_w{args.dmax}.vec')
        wv = KeyedVectors.load(EMBEDDING_FILENAME)
        feats = [wv[node] for node in nodes_list]
        scores = cluster(feats, labels_list)
        print(f"cluster scores: ")
        score_list = []
        for n in scores:
            print(f"n={n}, nmi: {scores[n]: .4f}")
            score_list.append(scores[n])
        print(f"average of scores: {np.mean(score_list): .4f}")
        
    if args.cvecs:
        from gensim.models import KeyedVectors
        EMBEDDING_FILENAME = os.path.join(args.data_home, args.dataset, f'nodes_{args.dimvec}_c{args.dmax}.vec')
        print(EMBEDDING_FILENAME)
        wv = KeyedVectors.load(EMBEDDING_FILENAME)
        feats = [wv[node] for node in nodes_list]
        scores = cluster(feats, labels_list)
        print(f"cluster scores: ")
        score_list = []
        for n in scores:
            print(f"n={n}, nmi: {scores[n]: .4f}")
            score_list.append(scores[n])
        print(f"average of scores: {np.mean(score_list): .4f}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="graph_clustering")
    parser.add_argument("--data_home", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="email")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--feats", action="store_true")
    parser.add_argument("--vecs", action="store_true")
    parser.add_argument("--wvecs", action="store_true")
    parser.add_argument("--cvecs", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--node2vec", action="store_true")
    parser.add_argument("--node2wvec", action="store_true")
    parser.add_argument("--node2cvec", action="store_true")
    parser.add_argument("--truncate", type=int, default=50)
    parser.add_argument("--iter_max", type=int, default=500)
    parser.add_argument("--dmax", type=int, default=0)
    parser.add_argument("--dimvec", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=80)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--batch_words", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument('-cl','--c_list', nargs='+', help='<Required> Set flag', type=int, default=[0.1, 1, 5, 10, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path_save", type=str, default='models')
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)