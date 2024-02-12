import torch
import numpy as np
import logging
from tqdm.auto import tqdm


class BertCluster:
    """
    Structure to hold the cluster
    """
    def __init__(self, embeddings: list[torch.Tensor], start_id: int = None, label: int = -1):
        """
        :param embeddings: given word context embeddings
        :param start_id: is or list od ids
        :param label: label or list of labels
        """
        self.embeddings = embeddings

        self.ids = []
        if start_id is None:
            logging.warning("start id for the cluster is not specified")
        elif isinstance(start_id, int):
            self.ids.append(start_id)
        elif isinstance(start_id, list):
            self.ids = start_id

        if isinstance(label, int):
            self.labels = [label]
        elif isinstance(label, list):
            self.labels = label

    def __len__(self):
        return len(self.ids)

    def __eq__(self, other):
        return set(self.ids) == set(other.ids)

    def extend_cluster(self, cluster):
        """
        Extends the cluster with elements of another cluster
        :param cluster: another cluster (BertCluster)
        """
        self.embeddings.extend(cluster.embeddings)
        self.ids.extend(cluster.ids)
        self.labels.extend(cluster.labels)


class BertHierarchicalClustering:
    """
    structure to perform hierarchical clustering
    """
    def __init__(self, clusters: list[BertCluster] = None):
        self.clusters = dict()
        if clusters is not None:
            self.clusters = {
                clusters[i].ids[0]: clusters[i]
                for i in range(len(clusters))
            }
        self.dist_mat = self.create_dist_mat()

    def create_dist_mat(self) -> dict[dict[int: float]]:
        """
        :return: create the matrix (actually dict of dicts) of distances between clusters
        """
        dist_mat = dict()
        for cluster_id_1 in self.clusters.keys():
            dist_mat[cluster_id_1] = dict()
            for cluster_id_2 in self.clusters.keys():
                if cluster_id_2 in dist_mat and cluster_id_1 in dist_mat[cluster_id_2]:
                    dist_mat[cluster_id_1][cluster_id_2] = dist_mat[cluster_id_2][cluster_id_1]
                else:
                    dist_mat[cluster_id_1][cluster_id_2] = self.cluster_distance(self.clusters[cluster_id_1],
                                                                                 self.clusters[cluster_id_2])
        return dist_mat

    def __len__(self):
        return len(self.clusters)

    def add_cluster(self, cluster: BertCluster):
        """
        adds cluster to the structure
        :param cluster: cluster to add
        """
        if cluster.ids[0] in self.clusters.keys():
            raise KeyError("The clusters has the same id")
        new_id = cluster.ids[0]
        self.clusters[new_id] = cluster
        self.dist_mat[new_id] = {
            cluster_id: self.cluster_distance(cluster, self.clusters[cluster_id])
            for cluster_id in self.clusters.keys()
        }
        for cluster_id in self.clusters.keys():
            if cluster_id == new_id:
                self.dist_mat[cluster_id][new_id] = 0.0
            else:
                self.dist_mat[cluster_id][new_id] = BertHierarchicalClustering.cluster_distance(
                    self.clusters[cluster_id], cluster)

    def delete_cluster(self, cluster_id: int):
        """
        :param cluster_id: cluster id to delete the cluster
        """
        if not cluster_id in self.clusters.keys():
            raise KeyError("There is no such cluster id")
        self.clusters.pop(cluster_id)
        self.dist_mat.pop(cluster_id)
        for cluster_id_dop in self.clusters.keys():
            self.dist_mat[cluster_id_dop].pop(cluster_id)

    @staticmethod
    def cluster_distance(cluster1: BertCluster, cluster2: BertCluster) -> float:
        """
        :return: the distance between two clusters
            at present it is minima of distances between embeddings
        """
        if cluster1 == cluster2:
            return 0.0
        distances = [
            BertHierarchicalClustering.vector_cluster_distance(vector, cluster2)
            for vector in cluster1.embeddings
        ]
        return np.min(distances)

    @staticmethod
    def vector_cluster_distance(vector: torch.Tensor, cluster: BertCluster):
        """
        :return: the distance between cluster and embedding
            at present it is minima of distances between embeddings
        """
        distances = []
        for cluster_vector in cluster.embeddings:
            distance = torch.sum((vector - cluster_vector) ** 2).cpu().numpy()
            distances.append(distance)
        return np.min(distances)

    def find_nearest_clusters(self) -> tuple[int, int]:
        """
        find two nearest clusters by checking all distances in self.dist_mat
        :return: cluster_id_1, cluster_id_2 : tuple of two cluster ids
        """
        ans = None, None
        min_dist = None
        for cluster_id_1 in self.clusters.keys():
            for cluster_id_2 in self.clusters.keys():
                if cluster_id_1 == cluster_id_2:
                    continue
                check_dist = self.dist_mat[cluster_id_1][cluster_id_2]
                if min_dist is None or check_dist < min_dist:
                    min_dist = check_dist
                    ans = cluster_id_1, cluster_id_2
        return ans

    def reduce_one_cluster(self):
        """
        Makes one step in hierarchical algorithm
        """
        cluster_id_1, cluster_id_2 = self.find_nearest_clusters()
        new_cluster = self.clusters[cluster_id_1]
        new_cluster.extend_cluster(self.clusters[cluster_id_2])
        self.delete_cluster(cluster_id_1)
        self.delete_cluster(cluster_id_2)
        self.add_cluster(new_cluster)

    def make_clusters(self, num):
        """
        perform hierarchical algorithm
        :param num: desired number of clusters
        """
        for _ in tqdm(list(range(len(self) - num))):
            self.reduce_one_cluster()