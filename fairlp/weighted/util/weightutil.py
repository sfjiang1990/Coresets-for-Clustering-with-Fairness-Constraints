def weighted_size(weight):
    return sum(weight)

def weighted_size_by_idx(weight, idx_list):
    return sum([weight[i] for i in idx_list])