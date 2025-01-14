# Real vs AI-Generated Image Classification

This project focuses on building a machine learning model capable of distinguishing between real and AI-generated images using a ResNet18 architecture, implemented in PyTorch. The model is trained and validated on labeled datasets of images, achieving high accuracy in both training and testing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

With the rapid advancement of AI, distinguishing between real and AI-generated images has become an increasingly important task. This project addresses this challenge by using transfer learning on ResNet18, a pre-trained Convolutional Neural Network (CNN), to classify images into two categories: **Real** and **AI-Generated**.

### Key Highlights:
1. **Transfer Learning**: Utilizes a pre-trained ResNet18 model to leverage existing image recognition capabilities.
2. **High Accuracy**: Achieved a test accuracy of up to 97.5%.
3. **Streamlined Workflow**: Includes functions for training, validation, and testing for efficient experimentation.

---

## Features

- End-to-end pipeline for image classification.
- Use of PyTorch for model development.
- GPU-accelerated training using CUDA (if available).
- Customizable hyperparameters for training.
- Detailed performance metrics for evaluation.

---

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Libraries**:
  - torchvision
  - torch
  - matplotlib
  - numpy
  - tqdm

---

## Dataset Structure

The dataset is organized into two main directories:
- **Train Data**: Contains labeled subfolders for real and AI-generated images.
- **Test Data**: Images without prior classification.

### Example Structure:
```
/dataset
  /train
    /real
      image1.jpg
      image2.jpg
    /ai_generated
      image1.jpg
      image2.jpg
  /test
    image1.jpg
    image2.jpg
```
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url/image-classification.git
   cd image-classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate # Linux/Mac
   env\Scripts\activate # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare the Dataset**:
   - Ensure the dataset follows the [Dataset Structure](#dataset-structure).

2. **Train the Model**:
   ```bash
   python train.py
   ```

3. **Validate the Model**:
   - Use the validation function in `train.py` to monitor performance after each epoch.

4. **Test the Model**:
   - Place test images in the `test` directory and run the testing script:
     ```bash
     python test.py
     ```

---

## Project Workflow

1. **Data Preprocessing**:
   - Resizes all images to 224x224.
   - Normalizes pixel values using:
     ```python
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ```

2. **Model Architecture**:
   - ResNet18 model with the final fully connected layer modified:
     ```python
     model.fc = nn.Linear(num_features, 2)
     ```

3. **Training**:
   - Uses Cross-Entropy Loss:
     ```python
     criterion = nn.CrossEntropyLoss()
     ```
   - Optimized with Adam:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
     ```

4. **Validation**:
   - Evaluates model performance on unseen validation data.

5. **Testing**:
   - Predicts labels for images in the test set and evaluates overall accuracy.

---

## Model Performance

| Metric       | Value        |
|--------------|--------------|
| Train Loss   | 0.19 (Epoch 10)|
| Train Accuracy | 98.9%       |
| Test Loss    | 0.0786       |
| Test Accuracy | 97.5%       |

---

## Contributing

Contributions are welcome! Please fork this repository, create a new branch, and submit a pull request. Ensure your code follows the existing style and is thoroughly tested.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

If you have any questions or feedback, feel free to reach out:
- **Email**: akshat25chauhan.205@gmail.com
- **GitHub**: [AkZcH](https://github.com/AkZcH)

