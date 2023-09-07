################################################################################
#                                                                               #
# This script is designed to visualize the deep learning training process. It   #
# employs the Matplotlib library to animate and display the training iterations #
# and model updates. It defines a function 'init' that initializes training     #
# parameters. An animation is generated using 'FuncAnimation' to visualize the  #
# model's training progress, displaying data this script facilitates a visual   #
# representation of deep learning training with dynamic updates.                #                                                            #
#                                                                               #
################################################################################


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import config
import Model
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from utils import get_linear


def init(
    batch_size: int = config.BATCH_SIZE,
    LR: float = config.LEARNING_RATE,
    model: Module = config.model,
    sample_size=100,
) -> None:
    """
    Initializes the dataset using RNG and visualizes the training process of a deep learning model.
    with an animation, drawing the predictions of the model on every Iteration

    Args:
        batch_size (int, optional):  Batch size for training. Defaults to config.BATCH_SIZE (10).
        LR (float, optional): Learning rate for the optimizer. Defaults to config.LEARNING_RATE (1e-3).
        model (nn.Module, optional): The neural network model for training. Defaults to LinearRegression.
        sample_size (int, optional): Size of the synthetic dataset. Defaults to 100.

    Returns:
        None
    """
    # TODO: Check if CUDA is avalible and move model and data to GPU
    X, y, dataloader = get_linear(sample_size, batch_size)  # Get the synthetic samples
    fig = plt.figure(figsize=(8, 5))  # init figure

    # Define the optimizer and loss function
    # print(model.parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    epochs = config.epochs

    # Define th function to be called on each frame of animation
    def animate(i: int) -> None:
        """
        Animate and visualize the training progress of a deep learning model. this
        function is called once every frame (Every 0.1 Seconds) until the model has
        finished training

        Args:
            i (int): Current animation frame index.

        Returns:
            None
        """
        plt.clf()

        for batch_X, batch_y in dataloader:
            # Perform forward pass and calculate loss
            output = model(batch_X)
            loss = loss_fn(batch_y, output)

            # Clear gradients and perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l = loss.detach().numpy()  # Get current loss

        # Plot the data and model predictions
        plt.title(f"Visual Deep Learning")
        plt.scatter(X.numpy(), y.numpy(), label="Data")
        plt.plot(X, model(X).detach().numpy(), c="crimson")

        # Display loss value
        plt.figtext(0.7, 0.005, f"Loss: {l:.2f}", fontsize=12, color="black")

        # Set plot style and show grid
        plt.style.use("seaborn-v0_8-pastel")  # seaborn pastel is a lovely theme
        plt.grid()

        # Update model parameters
        optimizer.step()  # called twice but too scared it will break everything

    # Easier to see tried 60 everything happend too quickly
    frame_rate = 10

    # Finally we draw the create and draw the animation
    ani = FuncAnimation(
        fig, animate, frames=epochs, interval=1 / frame_rate, repeat=False
    )
    plt.show()


# Here for testing reasons, to allow for quick testing without going through the GUI
if __name__ == "__main__":
    init()
