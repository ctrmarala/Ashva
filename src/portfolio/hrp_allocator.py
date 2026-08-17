"""
Ashva Hierarchical Risk Parity (HRP) Portfolio Allocator
Implements Marcos López de Prado's HRP algorithm using hierarchical tree clustering and recursive bisection
to allocate capital across uncorrelated strategies and assets without Markowitz matrix inversion instability.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


class HierarchicalRiskParityAllocator:
    """
    Allocates portfolio weights across multiple strategies or assets using Hierarchical Risk Parity (HRP).
    """

    def __init__(self, linkage_method: str = "single"):
        self.linkage_method = linkage_method

    @staticmethod
    def _correl_distance(corr_matrix: np.ndarray) -> np.ndarray:
        """
        Computes angular distance matrix: d(i, j) = sqrt(0.5 * (1 - rho_i,j))
        """
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def _get_quasi_diag(linkage_matrix: np.ndarray) -> List[int]:
        """
        Sorts clustered items by tree hierarchy (Quasi-Diagonalization).
        """
        link = linkage_matrix.astype(int)
        num_items = len(link) + 1
        curr_order = [link[-1, 0], link[-1, 1]]
        
        while max(curr_order) >= num_items:
            new_order = []
            for item in curr_order:
                if item < num_items:
                    new_order.append(item)
                else:
                    left_child = link[item - num_items, 0]
                    right_child = link[item - num_items, 1]
                    new_order.extend([left_child, right_child])
            curr_order = new_order

        return curr_order

    @staticmethod
    def _get_cluster_variance(cov_matrix: np.ndarray, cluster_indices: List[int]) -> float:
        """
        Computes inverse-variance portfolio variance for a cluster.
        """
        sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
        inv_diag = 1.0 / np.diag(sub_cov)
        weights = inv_diag / np.sum(inv_diag)
        cluster_var = float(np.dot(np.dot(weights, sub_cov), weights))
        return cluster_var

    def _recursive_bisection(self, cov_matrix: np.ndarray, sorted_indices: List[int]) -> np.ndarray:
        """
        Recursively splits clusters and allocates weights using Inverse Cluster Variance.
        """
        weights = pd.Series(1.0, index=sorted_indices)
        clusters = [sorted_indices]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # Bisect cluster into two sub-clusters
                    split = len(cluster) // 2
                    c1 = cluster[:split]
                    c2 = cluster[split:]
                    
                    var1 = self._get_cluster_variance(cov_matrix, c1)
                    var2 = self._get_cluster_variance(cov_matrix, c2)
                    
                    # Inverse variance allocation factor
                    alpha = 1.0 - var1 / (var1 + var2)
                    
                    weights.loc[c1] *= alpha
                    weights.loc[c2] *= (1.0 - alpha)

                    if len(c1) > 1:
                        new_clusters.append(c1)
                    if len(c2) > 1:
                        new_clusters.append(c2)
            clusters = new_clusters

        return weights.sort_index().values

    def allocate(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes HRP portfolio weights given a DataFrame of strategy/asset returns.
        
        :param returns_df: DataFrame where each column is an asset/strategy return series
        :return: Dictionary mapping asset/strategy name to allocation weight (sum = 1.0)
        """
        clean_df = returns_df.dropna()
        if clean_df.shape[1] == 1:
            return {clean_df.columns[0]: 1.0}

        corr = clean_df.corr().values
        cov = clean_df.cov().values

        # 1. Tree Clustering
        dist = self._correl_distance(corr)
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method=self.linkage_method)

        # 2. Quasi-Diagonalization
        sorted_indices = self._get_quasi_diag(link)

        # 3. Recursive Bisection
        hrp_weights = self._recursive_bisection(cov, sorted_indices)

        # Normalize and map to column names
        total_w = np.sum(hrp_weights)
        norm_weights = hrp_weights / total_w if total_w > 0 else np.ones(len(hrp_weights)) / len(hrp_weights)

        return {col: float(norm_weights[i]) for i, col in enumerate(clean_df.columns)}
