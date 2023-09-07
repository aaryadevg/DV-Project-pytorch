# Machine Learning Hyperparameter Visualization Tool

## Overview

This interactive learning tool is designed to help users gain a deeper understanding of the impact of hyperparameters on machine learning models. It allows users to experiment with different hyperparameters, observe model behavior during training, and visualize the training progress.

## Features

- **User-Friendly Frontend:** A Tkinter-based graphical user interface (GUI) allows users to select a machine learning model, set hyperparameters, and initiate the training process.

- **PyTorch-Based Trainer:** The trainer component handles model training, including the generation of synthetic data and default settings for missing hyperparameters.

- **Matplotlib-Powered Visualization:** The plotting section provides dynamic visualizations of the training process, displaying model predictions and training loss in real-time.

## Getting Started

Follow these steps to get started with the tool:

1. **Clone the Repository:**
```bash
git clone https://github.com/aaryadevg/DV-Project-pytorch
cd DV-Project-pytorch
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Application:**
   - On most systems, you can run the application using:
     ```bash
     python main.py
     ```
   - If you encounter issues or if your system uses Python 3.x by default, you can try:
     ```bash
     python3 main.py
     ```


4. **Use the GUI:** Launch the Tkinter GUI, select a model, set hyperparameters, and start the training process.

## Usage

- Select a machine learning model from the dropdown menu.
- Adjust hyperparameters using the input fields and sliders.
- Click the "Start Training" button to initiate training.
- Observe the dynamic visualizations in the Matplotlib-based plotting window.

## Contributing

Contributions to this project are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Special thanks to the contributors and users of this tool for their valuable feedback and support.

## Contact

For questions or inquiries about the project, please contact [Aaryadev Ghosalkar](mailto:aaryadevg@gmail.com).

