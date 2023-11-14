import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture


class Clustering():
    def __init__(self, model_name, params):
        super(Clustering, self).__init__()
        self.model_name = model_name
        self.model = self.init_model(model_name, params)
        self.model_names = ['AffinityPropagation', 'DBSCAN', 'KMeans', 'MiniBatchKMeans',
                            'SpectralClustering', 'GaussianMixture']
        self.init_model(model_name, params)

    def init_model(self, name, params):
        if name == 'AffinityPropagation':
            model = AffinityPropagation(**params)
        elif name == 'DBSCAN':
            model = DBSCAN(**params)
        elif name == 'KMeans':
            model = KMeans(**params)
        elif name == 'MiniBatchKMeans':
            model = MiniBatchKMeans(**params)
        elif name == 'SpectralClustering':
            model = SpectralClustering(**params)
        elif name == 'GaussianMixture':
            model = GaussianMixture(**params)
        else:
            raise RuntimeError(f'No such pre-defined ML model: {name}')
        self.model = model

    def fit_predict(self, X):
        if self.model != 'DBSCAN' and self.model_name in self.model_names:
            fit_y = self.model.fit(X).predict(X)
        else:
            fit_y = self.model.fit_predict(X)

        self.num_fit_data, self.fit_x, self.fit_y = X.shape[0], X, fit_y
        self.cluster_ids = np.unique(self.fit_y)
        self.num_cluster = len(self.cluster_ids)
        self.cluster_members = {cluster_id: np.where(self.fit_y == cluster_id)[0] for cluster_id in self.cluster_ids}
        self.cluster_nums = {id: len(self.cluster_members[id]) for id in self.cluster_ids}
        self.min_cluster_num, self.max_cluster_num = min(self.cluster_nums.values()), max(self.cluster_nums.values())
        return self.fit_y

    def compute_score(self, link_label_array, threshold):
        self.threshold = threshold
        self.cluster_labels = {id: [link_label_array[member_id] for member_id in members] for id, members in
                               self.cluster_members.items()}
        self.cluster_label_dists = {id: sum(labels) / len(labels) for id, labels in self.cluster_labels.items()}
        self.score = sum([self.cluster_nums[id] for id, ratio in self.cluster_label_dists.items() if
                          threshold <= ratio <= 1 - threshold]) / self.num_fit_data
        return self.score

    def draw_cluster_dist(self, normalization='min_max'):
        # 根据正样本的比例升序排列
        self.sorted_cluster_ids, self.sorted_label_dist = (list(t) for t in
                                                           zip(*sorted(self.cluster_label_dists.items(),
                                                                       key=lambda kv: (kv[1], kv[0]))))
        if normalization == 'min_max':
            self.normalized_cluster_nums = [
                (self.cluster_nums[id] - self.min_cluster_num) / (self.max_cluster_num - self.min_cluster_num)
                for id in self.sorted_cluster_ids]
        elif normalization == 'percentage':
            self.normalized_cluster_nums = [self.cluster_nums[id] / self.num_fit_data for id in self.sorted_cluster_ids]

        out_of_threshold_ids = [self.sorted_cluster_ids[i] for i, v in enumerate(self.sorted_label_dist) if
                                self.threshold < v < 1 - self.threshold]
        self.verified_score = sum([self.cluster_nums[id] for id in out_of_threshold_ids]) / self.num_fit_data

        x = [i for i in range(self.num_cluster)]
        plt.figure()
        plt.bar(x, self.sorted_label_dist, color='blue')
        plt.plot(x, self.normalized_cluster_nums, c='orange')
        plt.plot(x, [self.threshold for i in range(self.num_cluster)], c='red')
        plt.plot(x, [1 - self.threshold for i in range(self.num_cluster)], c='red')
        plt.show()
        return self.verified_score

    def predict(self, X):
        self.pred_x, self.pred_y = X, self.model.predict(X)
        self.cluster_label_assignment = {id: 1 if sum(labels) / len(labels) > 0.5 else 0 for id, labels in
                                         self.cluster_labels.items()}
        self.pred_y_label = np.array([self.cluster_label_assignment[yhat] for yhat in self.pred_y])
        return self.pred_y, self.pred_y_label
