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
    
import torch
from torch.utils.data import DataLoader

def dice_score(pred, target, epsilon=1e-6):
    """
    Computes the Dice score for binary segmentation.
    """
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + epsilon) / (union + epsilon)


def evaluate_dice_index(model, dataset, batch_size=4, device=None):
    """
    Evaluate the Dice score of a PyTorch model on a given dataset.
    
    Parameters:
        model: Trained PyTorch model
        dataset: Dataset returning (image, mask) pairs
        batch_size: Batch size for DataLoader
        device: torch.device('cuda') or torch.device('cpu') or None
    
    Returns:
        Average Dice score over the dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dice_total = 0.0
    n_batches = 0
    scores = []
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # dataloader returns dict with batched values
            batch = {input_type: input.to(device) for input_type, input in batch.items()}
            preds = model(batch['image'])
            masks = batch['mask'].to(device)
            for pred, mask in zip(preds, masks):
                score = dice_score(pred, mask)
                dice_total += score.item()
                scores.append(score.item())
                predictions.append(pred)
                n_batches += 1

    return [dice_total / n_batches if n_batches > 0 else 0.0, scores, predictions]