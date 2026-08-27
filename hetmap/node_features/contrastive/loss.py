import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss.

    z1, z2: L2-normalised embeddings (B, H).
    Row i in z1 is the positive pair for row i in z2.
    All other off-diagonal cross-row combinations are in-batch negatives.
    """
    sim = z1 @ z2.T / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def info_nce_masked(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temp: float,
    connected_mask: torch.Tensor,
) -> torch.Tensor:
    """NT-Xent with false-negative masking.

    Same as info_nce but positions (i, j) where anchor i and target j share a
    dependency edge are excluded from the negative gradient (set to -inf before
    softmax). This prevents connected files that were not sampled as a positive
    pair from being pushed apart.

    connected_mask: (B, B) bool tensor.
        True  = anchor i and target j are connected — exclude from negatives.
        False = treat as a negative (normal NT-Xent behaviour).
        Diagonal must always be False (those are the true positives).
    """
    sim = z1 @ z2.T / temp
    sim = sim.masked_fill(connected_mask, float('-inf'))
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
