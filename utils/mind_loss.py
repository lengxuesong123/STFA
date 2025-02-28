import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConstantPad2d, ReplicationPad2d

from builder import LOSSES


def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise squared euclidean distance of input coordinates.

    Args:
        x: input coordinates, input shape should be (1, dim, #input points)
    Returns:
        dist: pairwise distance matrix, (#input points, #input points)
    """
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


@LOSSES.register_module('mind')
class MINDSSCLoss(nn.Module):
    """
    Modality-Independent Neighbourhood Descriptor Dissimilarity Loss for Image Registration
    References: https://link.springer.com/chapter/10.1007/978-3-642-40811-3_24

    Args:
        radius (int): radius of self-similarity context.
        dilation (int): the dilation of neighbourhood patches.
        penalty (str): the penalty mode of mind dissimilarity loss.
    """
    def __init__(
        self,
        radius: int = 2,
        dilation: int = 2,
        penalty: str = 'l2',
    ) -> None:
        super().__init__()
        self.kernel_size = radius * 2 + 1
        self.dilation = dilation
        self.radius = radius
        self.penalty = penalty
        self.mshift1, self.mshift2, self.rpad1, self.rpad2 = self.build_kernels(
        )

    def build_kernels(self):
        six_neighbourhood = torch.Tensor([[0, 1], [1, 0], [1, 1], [0, -1], [-1, 0], [-1, -1]]).long()

        # squared distances
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask, square distance equals 2
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 2)[mask, :]
        mshift1 = torch.zeros(idx_shift1.shape[0], 4, 3, 3)
        mshift1.view(-1)[torch.arange(idx_shift1.shape[0]) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
        mshift1.requires_grad = False

        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :]
        mshift2 = torch.zeros(idx_shift2.shape[0], 4, 3, 3)
        mshift2.view(-1)[torch.arange(idx_shift2.shape[0]) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1
        mshift2.requires_grad = False

        # maintain the output size
        rpad1 = ReplicationPad2d(self.dilation)
        rpad2 = ReplicationPad2d(self.radius)
        return mshift1, mshift2, rpad1, rpad2

    def mind(self, img: torch.Tensor) -> torch.Tensor:
        mshift1 = self.mshift1.to(img)
        mshift2 = self.mshift2.to(img)
        # compute patch-ssd
        ssd = F.avg_pool2d(self.rpad2(
            (F.conv2d(self.rpad1(img), mshift1, dilation=self.dilation) -
             F.conv2d(self.rpad1(img), mshift2, dilation=self.dilation))**2),
                           self.kernel_size,
                           stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var,
                               mind_var.mean() * 0.001,
                               mind_var.mean() * 1000)
        mind = torch.div(mind, mind_var)
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([2, 1, 0]), :, :]

        return mind

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute the MIND-SSC loss.

        Args:
            source: source image, tensor of shape [BNHWD].
            target: target image, tensor fo shape [BNHWD].
        """
        assert source.shape == target.shape, 'input and target must have the same shape.'
        if self.penalty == 'l1':
            mind_loss = torch.abs(self.mind(source) - self.mind(target))
        elif self.penalty == 'l2':
            mind_loss = torch.square(self.mind(source) - self.mind(target))
        else:
            raise ValueError(
                f'Unsupported penalty mode: {self.penalty}, available modes are l1 and l2.'
            )

        return torch.mean(mind_loss)  # the batch and channel average

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(radius={self.radius},'
                     f'dilation={self.dilation},'
                     f'penalty=\'{self.penalty}\')')
        return repr_str


if __name__ =="__main__" :
    a = torch.rand(24,4,256,256,1)
    b = torch.rand(24,4,256,256,1)
    mind = MINDSSCLoss()
    loss = mind(a,b)
    print(loss)