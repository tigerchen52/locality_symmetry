# The Locality and Symmetry of Positional Encodings (EMNLP Findings 2023)
This repository provides a PyTorch implementation of the locality and symmetry metrics for positional encodings.

+ [arXiv Paper](https://arxiv.org/pdf/2310.12864.pdf)

## Documentation

To calculate the locality and symmetry of positional encodings, you need the code in `locality_symmetry/calculate.py`:
```py
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
```

## Experiments

We provide the following [experiments](experiments) to run BERT experiments.

## Citation
Lihu Chen, Gaël Varoquaux, Fabian M. Suchanek. "The Locality and Symmetry of Positional Encodings". In EMNLP Findings 2023.
```
@article{chen2023locality,
  title={The Locality and Symmetry of Positional Encodings},
  author={Chen, Lihu and Varoquaux, Ga{\"e}l and Suchanek, Fabian M},
  journal={arXiv preprint arXiv:2310.12864},
  year={2023}
}
```
