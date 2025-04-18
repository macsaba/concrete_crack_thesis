# This implementation is from the deep learning practice
from torch import nn

class DiceLoss(nn.Module):
    # NOTE: this only works for binary classification problems

    def __init__(self):
        super().__init__()

    def forward(self, mask, prediction, *args, **kwargs):

        mask = mask.view(*prediction.shape) # make sure prediction and mask are the same shape

        # calculate true positives, false positives, false negatives
        TPs = mask * prediction # elementwise product
        FPs = (1 - mask) * prediction
        FNs = mask * (1 - prediction)

        # calculate the number of TPs, FPs, and FNs
        TP = TPs.sum()
        FP = FPs.sum()
        FN = FNs.sum()

        dice_score = (2 * TP + 1) / (2 * TP + FP + FN + 1) # epsilon = 1

        return 1 - dice_score