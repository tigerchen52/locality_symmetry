import numpy as np
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()


def matrix_locality(M) -> float:
    """
    :param M: input attention weight matrix
    :return: locality value [0, 1]
    """
    length = len(M)
    sum_value = list()
    for index in range(length):
        factor = [1 / (2 ** (abs(i-index))) for i in range(length)]
        vector = M[index]
        scaled = [vector[i] * factor[i] for i in range(length)]
        sum_value.append(sum(scaled))
    locality = sum(sum_value) / len(sum_value)
    return locality


def cal_matrix_symmetry(M) -> float:
    """
    :param M: input attention weight matrix
    :return: symmetry value [0, 1]
    """
    seq_length = len(M)
    sum_symmetry = list()
    for index in range(seq_length):
        left, right = M[index][0:index], M[index][index + 1:seq_length]
        l_len, r_len = len(left), len(right)
        min_seg_len = min(l_len, r_len)
        if min_seg_len == 0: continue
        left_seq, right_seq = left[-min_seg_len:], list(reversed(right[:min_seg_len]))
        vec_sym = np.array([abs(left_seq[j] - right_seq[j]) for j in range(len(left_seq))]).reshape(-1, 1)
        normalized_vec_sym = min_max_scaler.fit_transform(vec_sym)
        sum_symmetry.extend([v[0] for v in normalized_vec_sym])
    symmetry = 1 - sum(sum_symmetry) / len(sum_symmetry)

    return symmetry