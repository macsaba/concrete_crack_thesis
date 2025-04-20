# This implementation is from the deep learning practice
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt

def train(model, loss_fn, optim, train_ds, val_ds, num_epochs = 1, accum_scale = 1, dice_idcs = [], epoch_dice_idcs = [], val_dice_idcs = [], train_loss = [], val_loss = [],  best_model_wts = {}):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # initialise metric calculations
    conf_mtx = torch.zeros((2, 2)) # initial count of confusion matrix entries is 0
    conf_mtx_calc = ConfusionMatrix(task = 'binary')
    best_val_score = 0
    for epoch_idx in range(num_epochs):
        model.train()
        train_loss_batch = 0
        val_loss_batch = 0
        for i, batch in enumerate(train_ds):
            # dataloader returns dict with batched values
            batch = {input_type: input.to(device) for input_type, input in batch.items()}
            pred = model(batch['image'])

            mask = batch['mask']
            loss = loss_fn(mask, pred)
            train_loss_batch = train_loss_batch + loss
            # gradients will be summed, so loss should be scaled to maintain learning rate
            (loss / accum_scale).backward()

            conf_mtx = conf_mtx + conf_mtx_calc(pred.to('cpu'), mask.to('cpu')) # update confusion matrix

            # update metrics and step with optimizer at the end of each real batch
            if (i + 1) % accum_scale == 0 or i + 1 == len(train_ds):
                tn, fp, fn, tp = conf_mtx.flatten()
                dice_idx = (2 * tp + 1) / (2 * tp + fp + fn + 1)
                dice_idcs.append(dice_idx.item())
                conf_mtx = torch.zeros((2, 2))
                optim.step()
                optim.zero_grad()
        # compute epoch scores
        tn, fp, fn, tp = conf_mtx_calc.compute().flatten()
        dice_idx = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        epoch_dice_idcs.append(dice_idx.item())

        # compute train loss:
        train_loss.append(train_loss_batch.to('cpu').item()/len(train_ds))

        print('Train loss: ', train_loss[-1])
        model.eval()
        for batch in val_ds:
            batch = {input_type: input.to(device) for input_type, input in batch.items()}
            with torch.no_grad():
                pred = model(batch['image'])

            mask = batch['mask']
            val_loss_batch = val_loss_batch + loss_fn(mask, pred)
            
            _ = conf_mtx_calc(pred.to('cpu'), batch['mask'].to('cpu'))
        tn, fp, fn, tp = conf_mtx_calc.compute().flatten()
        dice_idx = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        val_dice_idcs.append(dice_idx.item())

        # compute val loss:
        val_loss.append(val_loss_batch.to('cpu').item()/len(val_ds))

        print('Epoch ',epoch_idx + 1, '. finished.' )
        print('Validation loss: ', val_loss[-1])

        # save best parameters
        if val_dice_idcs[-1] > best_val_score:
            best_model_wts.clear()  # clear previous weights
            best_model_wts.update(model.state_dict())  # copy new best weights
            best_val_score = val_dice_idcs[-1]



def validate(model, loss_fn, optim, train_ds, val_ds, num_epochs = 1, accum_scale = 1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    conf_mtx_calc = ConfusionMatrix(task = 'binary')
    dice_idcs, epoch_dice_idcs, val_dice_idcs = [], [], []

    model.eval()
    for batch in val_ds:
        batch = {input_type: input.to(device) for input_type, input in batch.items()}
        with torch.no_grad():
            pred = model(batch['image'])
            _ = conf_mtx_calc(pred.to('cpu'), batch['mask'].to('cpu'))
    tn, fp, fn, tp = conf_mtx_calc.compute().flatten()
    dice_idx = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    val_dice_idcs.append(dice_idx.item())

    return dice_idcs, epoch_dice_idcs, val_dice_idcs

def test2(x):
    x[1] = 3