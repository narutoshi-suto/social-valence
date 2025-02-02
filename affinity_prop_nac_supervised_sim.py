import numpy as np
import pandas as pd
import os
import sys
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.io import loadmat, savemat
from scipy.stats import cumfreq
import matplotlib.pyplot as plt

# Define cell types, mice, and sessions
vCelltype = ['D2']
#D2 '317-2','323-1','415-1_GroupO','27-2','307-2','340-2','340-2_01'
#D1 '335-2_01','289-2','287-2','270-2'
#D2BLAInh '39-2_reframe', '332-1_01_re', '347-1_GroupI', '342-3'
#Rspo2 '242-2_GroupD', '246-1_GroupE','310-2_GroupS_re','285-1_GroupL_s2'
vMouse = [[ '317-2','323-1','415-1_GroupO','27-2','307-2_re']]
vSession = ['session1','session2','session3']
dir1 = 'C:/Users/pine5/OneDrive - The University of Tokyo/code/Data'
window_size = 10  # Number of rows to aggregate
step_size = 5  # Number of rows to slide the window each time


def downsample_matrix(matrix, window_size, step_size):
    num_windows = (matrix.shape[0] - window_size) // step_size + 1
    downsampled = np.empty((num_windows, matrix.shape[1]))
    frame_indices = np.zeros((num_windows, 2), dtype=int)
    for i in range(num_windows):
        start_row = i * step_size
        end_row = start_row + window_size
        downsampled[i, :] = matrix[start_row:end_row, :].mean(axis=0)
        #downsampled[i, :] = matrix[start_row:end_row, :].sum(axis=0)
        
        frame_indices[i, :] = [start_row, end_row - 1]
    return downsampled, frame_indices


def process_single_combination(cell_type, mouse_id, session_id):
    mat_file = os.path.join(dir1, cell_type, f'{cell_type}-{mouse_id}', session_id, 'ClusterData.mat')
    mat = loadmat(mat_file)
    D_ = mat['D']
    #X_ = mat['DF'] #df/f
    Y = mat['SP']  # spike count
    Y = np.where(np.isnan(Y) | (Y == 0), Y, 1)
    X_ = Y
    X_ = pd.DataFrame(Y).rolling(window=10, center=True,
                      axis=0, min_periods=1).sum().to_numpy()

    # Create a StandardScaler object
    #scaler = StandardScaler()
    # Fit the scaler to the matrix and transform it (Z-scoring)
    #X_ = scaler.fit_transform(X_)

    Xds, frame_ids_X = downsample_matrix(X_, window_size, step_size)
    Dds, frame_ids_D = downsample_matrix(D_, window_size, step_size)
    conditions = ['agressor', 'non_agressor', 'both', 'all']
    cluster_types = ['activity', 'neuron']

    # for multiplier in np.arange(6.0, 8.5, 0.5):
    multiplier1 = 4.0
    multiplier2 = 1.0
    mat_content = {}  # for saving
    for i, condition in enumerate(conditions):
        for j, cluster_type in enumerate(cluster_types):
            rad = 10 / 2 * 2
            #rad = 10 / 2 * (1.5)
            if i < 2:
                pick = (Dds[:, i] < rad)
            elif condition == 'both':
                pick = (Dds < rad).any(axis=1)
            else:
                pick = np.ones(Xds.shape[0], dtype=bool)

            picked_frame_ids = frame_ids_D[pick, :]

            X = Xds[pick, :]

            # Remove columns that are all zeros along the column axis
            non_zero_columns = ~np.all(X == 0, axis=0)
            X = X[:, non_zero_columns]


            if cluster_type == 'activity':
               corr_matrix = np.corrcoef(X, rowvar=True)
            elif cluster_type == 'neuron':
               corr_matrix = np.corrcoef(X, rowvar=False)

            distance_matrix = 1 - corr_matrix
            similarity_matrix = -distance_matrix
            # (n_samples_X, n_features) = (variable, observation)
            similarity_distance1 = - pairwise_distances(X, metric='euclidean') ** 2
            similarity_distance2 = - pairwise_distances(X.T, metric='euclidean') ** 2

            if cluster_type == 'activity':
                preference1 = np.min(similarity_distance1) * multiplier1
                # preference1 = -1
                af = AffinityPropagation(
                    preference=preference1, random_state=0, damping=0.9, max_iter=1000).fit(X)
            else:
                # af = AffinityPropagation(affinity='precomputed', random_state=0, damping=0.9, max_iter=1000).fit(similarity_matrix)
                preference2 = np.min(similarity_matrix) * multiplier2
                # preference2 = -0.6
                af = AffinityPropagation(affinity='precomputed', preference=preference2,
                                         random_state=0, damping=0.9, max_iter=1000).fit(similarity_matrix)  # -1.5

            A_flattened = af.affinity_matrix_.flatten()
            a, low_lim, binsize, extrapoints = cumfreq(
                A_flattened, numbins=100)
            a = a / a[-1]
            x_values = np.linspace(low_lim, low_lim + binsize * a.size, a.size)

            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_
            n_clusters = len(cluster_centers_indices)

            if cluster_type == 'activity':
                preference = preference1
            elif cluster_type == 'neuron':
                preference = preference2

            data_to_save = {
                'cluster_centers_indices': cluster_centers_indices,
                'labels': labels,
                'n_clusters': n_clusters,
                'X': X,
                'pick': pick,
                'preference': preference,
                'picked_frame_ids': picked_frame_ids,
                'multiplier1': multiplier1,
                'multiplier2': multiplier2
            }
            field_name = f'{condition}_activity_clustered_by_{cluster_type}'
            mat_content[field_name] = data_to_save

    save_file = os.path.join(dir1, cell_type, f'{cell_type}-{mouse_id}', session_id, f'cluster_4mul_1mul_sp_nosum.mat')
    savemat(save_file, mat_content)

for ci, cell_type in enumerate(vCelltype):
    for mouse_id in vMouse[ci]:
        for session_id in vSession:
            process_single_combination(cell_type, mouse_id, session_id)

sys.exit(0)
