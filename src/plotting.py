import matplotlib.pyplot as plt
import numpy as np

def plot_image_mask_grid(pairs, columns=3, scale = 6, path = None):
    """
    Plots image-mask pairs in a grid.

    Args:
        pairs: List of (image, mask) tuples, where image and mask are numpy arrays in HWC format.
        columns: Number of image-mask columns per row.
        figsize: Size of each column in inches (width, height).
    """
    rows = int(np.ceil(len(pairs) / columns))

    # calculate image size
    im_H = pairs[0][0].shape[0]
    im_W = pairs[0][0].shape[1]
    padding = im_W/10
    fig_W = columns * 2 * im_W + (columns * 2 +1)*padding
    fig_H = rows*im_H + (rows+1)*padding
    # norm and scale
    fig_W = fig_W/fig_H*scale
    fig_H = scale
    fig, axs = plt.subplots(rows, columns * 2, figsize=(fig_W, fig_H))
    #fig, axs = plt.subplots(rows, columns * 2)

    if rows == 1:
        axs = np.expand_dims(axs, 0)  # handle single-row case

    for i in range(rows * columns):
        row = i // columns
        col = i % columns
        ax_img = axs[row, col * 2]
        ax_mask = axs[row, col * 2 + 1]

        if i < len(pairs):
            image, mask = pairs[i]
            ax_img.imshow(image)
            ax_img.axis('off')

            ax_mask.imshow(mask.squeeze(), cmap='gray')
            ax_mask.axis('off')
        else:
            # Empty subplot if not enough data
            ax_img.axis('off')
            ax_mask.axis('off')

        # Write 'image' and 'mask' labels below empty subplots
        if i < columns:
            ax_img.set_title('Image', fontsize=20*scale/10)
            ax_mask.set_title('Ground truth', fontsize=20*scale/10)
    #plt.subplots_adjust(hspace=0.1, wspace=0.1)  # finer manual control

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_image_mask_prediction_grid(triplets, columns=3, scale=6, path=None):
    """
    Plots image-mask-prediction triplets in a grid.

    Args:
        triplets: List of (image, mask, prediction) tuples, where each element is a numpy array in HWC or HW format.
        columns: Number of image sets per row.
        scale: Scaling factor for figure size.
        path: Optional path to save the figure.
    """
    rows = int(np.ceil(len(triplets) / columns))

    # Image dimensions
    im_H = triplets[0][0].shape[0]
    im_W = triplets[0][0].shape[1]
    padding = im_W / 10
    fig_W = columns * 3 * im_W + (columns * 3 + 1) * padding
    fig_H = rows * im_H + (rows + 1) * padding

    # Normalize and scale figure size
    fig_W = fig_W / fig_H * scale
    fig_H = scale

    fig, axs = plt.subplots(rows, columns * 3, figsize=(fig_W, fig_H))

    if rows == 1:
        axs = np.expand_dims(axs, 0)  # handle single-row case

    for i in range(rows * columns):
        row = i // columns
        col = i % columns
        ax_img = axs[row, col * 3]
        ax_mask = axs[row, col * 3 + 1]
        ax_pred = axs[row, col * 3 + 2]

        if i < len(triplets):
            image, mask, pred = triplets[i]

            ax_img.imshow(image)
            ax_img.axis('off')

            ax_mask.imshow(mask.squeeze(), cmap='gray')
            ax_mask.axis('off')

            ax_pred.imshow(pred.squeeze(), cmap='gray')
            ax_pred.axis('off')
        else:
            ax_img.axis('off')
            ax_mask.axis('off')
            ax_pred.axis('off')

        # Add titles only for the first row
        if i < columns:
            ax_img.set_title('Image', fontsize=20 * fig_W / 10)
            ax_mask.set_title('Ground Truth', fontsize=20 * fig_W / 10)
            ax_pred.set_title('Prediction', fontsize=20 * fig_W / 10)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_image_mask_prediction_grid_with_dice(triplets, columns=3, scale=6, path=None):
    """
    Plots image-mask-prediction triplets with Dice score in a grid.

    Args:
        triplets: List of (image, mask, prediction, dice_score) tuples.
        columns: Number of image sets per row.
        scale: Scaling factor for figure size.
        path: Optional path to save the figure.
    """
    rows = int(np.ceil(len(triplets) / columns))

    im_H = triplets[0][0].shape[0]
    im_W = triplets[0][0].shape[1]
    padding = im_W / 10
    fig_W = columns * 3 * im_W + (columns * 3 + 1) * padding
    fig_H = rows * im_H + (rows + 1) * padding

    fig_W = fig_W / fig_H * scale
    fig_H = scale

    fig, axs = plt.subplots(rows, columns * 3, figsize=(fig_W, fig_H))

    if rows == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(rows * columns):
        row = i // columns
        col = i % columns
        ax_img = axs[row, col * 3]
        ax_mask = axs[row, col * 3 + 1]
        ax_pred = axs[row, col * 3 + 2]

        if i < len(triplets):
            image, mask, pred, dice = triplets[i]

            ax_img.imshow(image)
            ax_img.axis('off')

            ax_mask.imshow(mask.squeeze(), cmap='gray')
            ax_mask.axis('off')

            ax_pred.imshow(pred.squeeze(), cmap='gray')
            ax_pred.axis('off')

            # Display provided Dice score
            ax_pred.text(0.5, -0.15, f"Dice: {dice:.3f}", transform=ax_pred.transAxes,
                         ha='center', va='top', fontsize=10, color='black')

        else:
            ax_img.axis('off')
            ax_mask.axis('off')
            ax_pred.axis('off')

        if i < columns:
            ax_img.set_title('Image', fontsize=20 * fig_W / 10)
            ax_mask.set_title('Ground Truth', fontsize=20 * fig_W / 10)
            ax_pred.set_title('Prediction', fontsize=20 * fig_W / 10)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()