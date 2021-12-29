import torch
import torch.nn.functional as F
# Taken from gist: https://gist.githubusercontent.com/vadimkantorov/bd1616a3a9eea89658ea3efb1f9a1d5d

def adjusted_rand_index(true_mask, pred_mask):
    """
    Provides an implementation of the Adjusted Rand Index. Ignores points with no cluster label
    in `true_mask` (ie; those points where `true_mask` is all zero). This means it provides a
    "foreground" segmentation metric
    Args:
        true_mask: tensor or np.array of size (batch, max_objects, time, channel, height, width)
        pred_mask: predicted tensor of same size
    """
    """
    B, max_num_entities, T, C, H, W = true_mask.shape
    desired_shape = (B, T*C*H*W, max_num_entities)
    true_mask = true_mask.reshape(desired_shape)
    pred_mask = pred_mask.reshape(desired_shape)
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (n_points <= n_true_groups and n_points <= n_pred_groups), ("adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32)  # One hot encoding
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32)

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    _all_equal = lambda values: torch.all(torch.eq(values, values[..., :1]), dim=-1)
    both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)
