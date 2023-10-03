import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
def import_facebook_data(path):
    """
    Import facebook data from the given path and return the list of edges.

    Args:
        path (str): Path to the facebook_combined.txt file.

    Returns:
        numpy.ndarray: Nx2 array of edges.
    """
    df = pd.read_csv(path, sep=' ', header=None)
    val=df.values
    ## add reverse edges to make undirected graph
    reverse_edges = np.flip(val, axis=1)
    val = np.concatenate((val, reverse_edges), axis=0)
    ## no duplicates in graph, so we proceed
    return val

def import_bitcoin_data(path):
    """
    Import bitcoin data from the given path and return the list of edges.

    Args:
        path (str): Path to the soc-sign-bitcoinotc.csv file.

    Returns:
        numpy.ndarray: Nx2 array of edges.
    """

    # Taking average weight for directed edges on both sides

    df = pd.read_csv(path, sep=',', header=None)
    df_new=df
    df_new = df_new.iloc[:, 0:3]

    col_list = list(df_new.columns)
    x, y = col_list.index(0), col_list.index(1)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    flip = df_new[col_list]


    flip.columns=[0,1,2]
    frame=[df_new,flip]

    df_new=pd.concat(frame)
    df_new.reset_index(drop=True,inplace=True)

    df_new=df_new.groupby([0,1]).mean().astype(int).reset_index()
    val1 = df_new.values

    df = df.iloc[:, [0, 1]]
    df1=df.values
    reverse_edges = np.flip(df1, axis=1)
    df1 = np.concatenate((df1, reverse_edges), axis=0)
    #remove duplicate rows from val
    value= np.unique(df1, axis=0)
    unique = np.unique(value[:,0])
    
    unique_dict = {i: val for i, val in enumerate(unique.tolist())}
    reversed_dict = {val: key for key, val in unique_dict.items()}

    data= [[reversed_dict[val1], reversed_dict[val2]]for val1, val2 in value]
    #change data to numpy array
    data=np.array(data)
    return data

def modularity_change(node, cluster_val, new_cluster_val, adjacency_list, node_value_from_cluster, number_of_edges):
    """
    Calculate the change in modularity after moving a node from one cluster to another.

    Returns:
        float: Change in modularity.
    """
    nodes_x = set()
    nodes_y = set()
    nodes_x.update(node_value_from_cluster[cluster_val])
    nodes_y.update(node_value_from_cluster[new_cluster_val])
    
    sigma_x = 0
    for node1 in node_value_from_cluster[cluster_val]:
        sigma_x += len(adjacency_list[node1])
    
    sigma_y = 0
    for node1 in node_value_from_cluster[new_cluster_val]:
        sigma_y += len(adjacency_list[node1])

    deg_node = len(adjacency_list[node])
    edges_in_cluster = 0

    edges_in_new_cluster = 0
    for node1 in adjacency_list[node]:
        if node1 in nodes_x:
            edges_in_cluster += 1
        elif node1 in nodes_y:
            edges_in_new_cluster += 1

    change_after_rem_node = -2 * edges_in_cluster / (2 * number_of_edges)
    change_after_add_node = 2 * edges_in_new_cluster / (2 * number_of_edges)
    
    change_after_rem_node = change_after_rem_node + (sigma_x ** 2 - (sigma_x - deg_node) ** 2) / (4 * number_of_edges * number_of_edges)
    change_after_add_node = change_after_add_node + (sigma_y ** 2 - (sigma_y + deg_node) ** 2) / (4 * number_of_edges * number_of_edges)

    return change_after_rem_node + change_after_add_node

def spectralDecomp_OneIter(nodes_connectivity_list,plot=False):
    """
    Perform spectral decomposition of the graph for one iteration and return the results.

    Args:
        nodes_connectivity_list (numpy.ndarray): Nx2 array of edges.
        plot (bool): If True, plot the results.

    Returns:
        numpy.ndarray: N-length array of fielder vector.
        numpy.ndarray: NxN adjacency matrix of the graph.
        numpy.ndarray: Nx2 array of graph partition.
    """

    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list)
    
    # Create the adjacency matrix of the graph
    adj_mat = nx.adjacency_matrix(G).todense()

    # Create the degree matrix of the graph
    deg_mat = np.diag(np.sum(adj_mat, axis=1))

    # Create the laplacian matrix of the graph
    lap_mat = deg_mat - adj_mat

    # Get the eigenvalues and eigenvectors of the laplacian matrix
    eig_vals, eig_vecs = np.linalg.eigh( lap_mat)

    # Sort the eigenvectors in the increasing order of eigenvalues
    idx = eig_vals.argsort()
    sorted_eig_vecs = eig_vecs[:, idx]
    
    # Get the eigenvector corresponding to the second smallest eigenvalue
    fielder_vec = sorted_eig_vecs[:, 1]

    idx=np.argsort(fielder_vec)
    sorted_fielder_vec = fielder_vec[idx]
    # Get the number of nodes in the graph
    n = eig_vals.shape[0]

    # Create the sorted adjacency matrix
    sorted_adj_mat = adj_mat[idx][:, idx]
    # Create the graph partition
    graph_partition = np.zeros((n, 2), dtype=int)


    smallest_first = np.where(fielder_vec >= 0)[0][0]
    
    smallest_second = np.where(fielder_vec < 0)[0][0]
   
    for i in range(n):
        if fielder_vec[i] > 0:
            graph_partition[i, 0] = i
            graph_partition[i, 1] = smallest_first
        else:
            graph_partition[i, 0] = i
            graph_partition[i, 1] = smallest_second
    unique_id=np.unique(graph_partition[:,1])
    
    partition = [[i for i, x in graph_partition if x == j] for j in unique_id]

    #modularity = nx.algorithms.community.modularity(G, partition)
    #print("Modularity in one iteration is: ", modularity)
    if(plot):
        #Plot the original graph
        nx.draw_spring(G, with_labels=False, node_size=10)
        plt.savefig("graph.png")

        # Plot the sorted Fiedler vector
        plt.figure()
        plt.scatter(np.arange(sorted_fielder_vec.size), sorted_fielder_vec,s=2)
        plt.title("Sorted Fiedler vector",fontsize=20)
        plt.xlabel("Nodes corresponding to sorted Fiedler vector value") 
        plt.ylabel("Fiedler vector value")
        plt.savefig("sorted_fiedler_vector.png",dpi=500)

        #Plot adjacency matrix before sorting(sorted y axis in increasing order)
        plt.figure()
        plt.imshow(adj_mat,origin='lower')
        plt.title("Adjacency matrix")
        plt.xlabel("Node ID")
        plt.ylabel("Node ID")
        plt.savefig("adjacency_matrix.png")

        #Plot the sorted adjacency matrix
        plt.figure()
        plt.imshow(sorted_adj_mat,origin='lower')
        plt.title("Sorted adjacency matrix")
        plt.xlabel("Nodes corresponding to sorted Fiedler vector value") 
        plt.ylabel("Nodes corresponding to sorted Fiedler vector value")
        plt.savefig("sorted_adjacency_matrix.png")

        #Plot the graph partition
        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, edgelist=nodes_connectivity_list, edge_color='black', width=1)
        nodelist = graph_partition[graph_partition[:, 1] == smallest_first, 0].tolist()
        nx.draw_networkx_nodes(G, pos, nodelist, node_color='r', node_size=10)
        nodelist = graph_partition[graph_partition[:, 1] == smallest_second, 0].tolist()
        nx.draw_networkx_nodes(G, pos, nodelist, node_color='b', node_size=10)
        plt.title("Graph with partitions")
        plt.savefig("graph_partition.png")

    return fielder_vec, adj_mat, graph_partition

def spectralDecomposition(nodes_connectivity_list_fb):

    start = time.time()

    fielder_vec, adj_mat, graph_partition=spectralDecomp_OneIter(nodes_connectivity_list_fb,plot=False)
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_fb)
    
    graph_partition_new=graph_partition

    ## Using brute force for number of iterations
    for iter in range(10):
        partition_parent_list=[[],[]]
        unique, counts = np.unique(graph_partition_new[:,1], return_counts=True)
        for i in range(len(unique)):
            partition_parent_list[0].append(unique[i])
            partition_parent_list[1].append(counts[i])
        
        max_id=np.argmax(counts)
        parent = unique[max_id]
        nodes_connectivity_list = []
        for i in nodes_connectivity_list_fb:
            if graph_partition_new[i[0],1] == parent and graph_partition_new[i[1],1] == parent:
                nodes_connectivity_list.append([i[0],i[1]])
        nodes_connectivity_list = np.array(nodes_connectivity_list)
        unique = np.unique(nodes_connectivity_list[:,0])
       
        unique_dict = {i: val for i, val in enumerate(unique)}
        reversed_dict = {val: key for key, val in unique_dict.items()}

        nodes_connectivity_list_new = [[reversed_dict[val1], reversed_dict[val2]]for val1, val2 in nodes_connectivity_list]

        fielder_vec, adj_mat, graph_partition1=spectralDecomp_OneIter(nodes_connectivity_list_new,plot=False)
        graph_partition_new=[[unique_dict[val1], unique_dict[val2]]for val1, val2 in graph_partition1]
       
        val=[row[0] for row in graph_partition_new]
      
        for i in range(len(graph_partition)):
            if(graph_partition[i,0] not in val):
                graph_partition_new.append(graph_partition[i])
      
        graph_partition_new=np.array(graph_partition_new)
        graph_partition_new = graph_partition_new[np.argsort(graph_partition_new[:, 0])] 
        graph_partition=graph_partition_new
        unique_id=np.unique(graph_partition[:,1])
        partition = [[i for i, x in enumerate(graph_partition[:,1]) if x == j] for j in unique_id]
        
        modularity = nx.algorithms.community.modularity(G, partition)
    print("Spectral Modularity after iteration ",iter+1," is ",modularity)

    node_colors = {}
    for i in range(graph_partition.shape[0]):
        node_colors[i] = 'C{}'.format(graph_partition[i, 1])

    # plot the graph with different colors for each community
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=list(node_colors.values()), node_size=10)
    nx.draw_networkx_edges(G, pos)
    plt.figure()
    plt.title("Community Graph",fontsize=20)
    plt.savefig("Community Graph.png")

    end = time.time()
    print("Time taken for Spectral decomposition is: ",end-start)
    return graph_partition

def createSortedAdjMat(graph_partition, nodes_connectivity_list):
    """
    Create the sorted adjacency matrix of the entire graph.

    Args:
        graph_partition (numpy.ndarray): Nx2 array of graph partition.
        nodes_connectivity_list (numpy.ndarray): Nx2 array of edges.

    Returns:
        numpy.ndarray: NxN adjacency matrix of the graph.
    """
    # Get the number of nodes in the graph
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list)
    
    # Create the adjacency matrix of the graph
    adj_mat = nx.adjacency_matrix(G).todense()
    idx = np.argsort(graph_partition[:, 1])

    sorted_adj_mat = adj_mat[idx][:, idx]

    # Plot the sorted adjacency matrix
    plt.figure()
    plt.imshow(sorted_adj_mat,origin='lower')
    plt.title("Sorted adjacency matrix")
    plt.xlabel("Nodes corresponding to sorted Fiedler vector value")
    plt.ylabel("Nodes corresponding to sorted Fiedler vector value")
    plt.savefig("Final_sorted_adjacency_matrix.png")
    plt.close()
    return sorted_adj_mat

def louvain_one_iter(nodes_connectivity_list):
    """
    Run one iteration of louvain algorithm and return the resulting graph partition.

    Args:
        nodes_connectivity_list (numpy.ndarray): Nx2 array of edges.

    Returns:
        numpy.ndarray: Nx2 array of graph partition.
    """
    start= time.time()
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list)
    number_of_edges = len(nodes_connectivity_list)/2
    # adjacency list
    adjacency_list = {}

    for edge in nodes_connectivity_list:
        if edge[0] not in adjacency_list:
            adjacency_list[edge[0]] = []
        adjacency_list[edge[0]].append(edge[1])
    
    #Initial cluster value of each node to itself
    cluster_value_from_node = {}
    for node in adjacency_list:
        cluster_value_from_node[node] = node
    
    modularity = 0
    for node in adjacency_list:
        modularity += len(adjacency_list[node]) ** 2
        
    modularity = -modularity /( 4 *number_of_edges *number_of_edges)
    
    nodes = list(adjacency_list.keys())

    for _ in range(10):
        count = 0
        for node1 in nodes:
            node_value_from_cluster = {}
            for node in cluster_value_from_node:
                if node_value_from_cluster.get(cluster_value_from_node[node]) == None:
                    node_value_from_cluster[cluster_value_from_node[node]] = []
                node_value_from_cluster[cluster_value_from_node[node]].append(node)
            
            node2 = 0
            new_cluster = 0
            delta = -10000
            
            for node in adjacency_list[node1]:
                c = cluster_value_from_node[node]
                if cluster_value_from_node[node1] == c:
                    continue
                    
                delQ = modularity_change(node1, cluster_value_from_node[node1], c, adjacency_list, node_value_from_cluster, number_of_edges)
                if delQ > delta:
                    delta = delQ
                    node2 = node1
                    new_cluster = c
            

            if modularity + delta <= modularity:
                count += 1
                continue
            modularity = modularity + delta
            cluster_value_from_node[node2] = new_cluster
        if count == len(nodes):
            break
    
    print("Louvain Modularity is:",modularity)

    node_value_from_cluster = {}
    for key in cluster_value_from_node:
        if node_value_from_cluster.get(cluster_value_from_node[key]) == None:
            node_value_from_cluster[cluster_value_from_node[key]] = key
        else:
            node_value_from_cluster[cluster_value_from_node[key]] = min(key, node_value_from_cluster[cluster_value_from_node[key]])
    
    number_of_nodes = len(nodes)
    partition = np.zeros((number_of_nodes, 2),dtype=int)
    for i in range(number_of_nodes):
        partition[i][0] = nodes[i]
        partition[i][1] = node_value_from_cluster[cluster_value_from_node[nodes[i]]]
        
    partition=np.array(partition)

    # node_colors = {}
    # for i in range(partition.shape[0]):
    #     node_colors[i] = 'C{}'.format(partition[i, 1])

    # # plot the graph with different colors for each community
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_color=list(node_colors.values()), node_size=10)
    # nx.draw_networkx_edges(G, pos)
    # plt.title("Community Graph Louvain",fontsize=20)

    # plt.savefig("Community Graph Louvain.png")
    end=time.time()
    print("Time taken for Louvain Algorithm is: ",end-start)
    return partition
    
if __name__ == "__main__":

    start=time.time()
    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    
    #fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)

    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    
    end=time.time()
    print("Total time taken is: ",end-start)