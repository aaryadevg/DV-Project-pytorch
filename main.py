################################################################################
#                                                                               #
#   This script utilizes the Tkinter library to establish a graphical           #
#   interface, enabling users to configure and initiate the training of deep    #
#   learning models. The interface offers choices for specifying training mode, #
#   activation function, model type, learning rate, and dataset size.           #
#   Additionally, it provides a visualization of the model training process by  #
#   displaying the model's output.                                              #
#                                                                               #
################################################################################


from tkinter import *
from tkinter import ttk
from config import (
    TrainingMode,
    Activation,
    Models,
    activation_fn_sigmoid,
    activation_fn_relu,
)
import Model
import linearReg
from torch.nn import ReLU, Tanh


def draw_training() -> None:
    """
    Retrives the values of the hyperparameter input elements
    and set up the parameters to be passed to the animation plotting
    code (LinReg.init) to set up the figure for animation and draw the
    animation
    """
    # Retrive the values
    selected_training_mode = training_mode_var.get()
    selected_activation = activation_var.get()
    selected_model = models_var.get()
    dataset_size = int(dataset_entry.get())

    # If we are doing sample gradient descent the batch size should be 1
    bs = 1
    if selected_training_mode == "Mini_Batch":
        bs = 10

    # Initialize the activation function to be used based on user Selection
    act_fn = activation_fn_sigmoid
    if selected_activation == "ReLU":
        act_fn = activation_fn_relu
    elif selected_activation == "Tanh":
        act_fn = Tanh()

    # Initialize the model to be used based on user Selection
    model = None
    if selected_model == "ANN":
        model = Model.ANNModel(
            activation=act_fn
        )  # Only the ANN model needs an activation function
    elif selected_model == "Linear":
        model = Model.LinearRegression()

    lr = float(learning_rate_entry.get())

    # All parameters set up ready to go on an initialize a figure to plot
    linearReg.init(batch_size=bs, LR=lr, model=model, sample_size=dataset_size)


# Create the main window
root = Tk()
root.title("Visual Deep Learning")
root.geometry("300x320")  # Set the window size (width 300px, height 320px)

# Hyperparameter input section
frame = Frame(root)
frame.pack(padx=10, pady=10)

# Create dropdowns for each enum.
# Each combo box is dynamically populated with all values in the enum
# Thus if we want to add another activation then we can update the enum
# and this will still function
training_mode_var = StringVar(root)
training_mode_dropdown = ttk.Combobox(
    frame, textvariable=training_mode_var, values=[mode.name for mode in TrainingMode]
)
training_mode_dropdown.set("Select Training Mode")
training_mode_dropdown.pack(pady=10)

activation_var = StringVar(root)
activation_dropdown = ttk.Combobox(
    frame, textvariable=activation_var, values=[act.name for act in Activation]
)
activation_dropdown.set("Select Activation")
activation_dropdown.pack(pady=10)

models_var = StringVar(root)
models_dropdown = ttk.Combobox(
    frame, textvariable=models_var, values=[model.name for model in Models]
)
models_dropdown.set("Select Model")
models_dropdown.pack(pady=10)

# Entry for Learning Rate
# NOTE: We are not doing any checks on the learning rate, done for teaching purpose
Label(frame, text="Learning Rate:").pack(pady=5)
learning_rate_entry = Entry(frame)
learning_rate_entry.insert(0, "0.01")  # Default value
learning_rate_entry.pack(pady=5)

# NOTE: We are not doing any checks on the Sample size and a large sample size will crash the application
#       cannot check if x number of samples will fit in memory ahead of time.
Label(frame, text="Sample Size:").pack(pady=5)
dataset_entry = Entry(frame)
dataset_entry.insert(0, "100")  # Default value
dataset_entry.pack(pady=5)

# Train button
train_button = ttk.Button(frame, text="Start Training", command=draw_training)
train_button.pack(pady=10)

root.mainloop()
