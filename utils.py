################################################################################
#                                                                               #
# This code utilizes PyTorch to create a synthetic linear dataset. The function #
# 'get_linear' generates samples of input-output pairs for a linear regression  #
# problem.                                                                      #
#                                                                               #
################################################################################


from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor, linspace, normal
from typing import Optional, Tuple
from config import dataset_size


def get_linear(samples: int, batch: Optional[int]) -> Tuple[Tensor, Tensor, DataLoader]:
    """
        Generates a synthetic linear dataset with input-output pairs.

    Args:
        samples (int): Number of samples in the dataset.
        batch (Optional[int]): Batch size for DataLoader.

    Returns:
        tuple: Tuple containing input data, output data, and DataLoader.
    """
    X = linspace(-10, 10, dataset_size).view(-1, 1)
    y = 2 * X + 1
    y += normal(0, 2, y.shape)

    dataset = TensorDataset(X, y.view(-1, 1))
    dataloader = DataLoader(
        dataset, batch_size=batch, shuffle=True
    )  # shuffle should not matter

    return (X, y, dataloader)
