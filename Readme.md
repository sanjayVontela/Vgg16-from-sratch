<h1>VGG16 Model for Image Classification</h1>

This project implements a VGG16 model for image classification from scratch using PyTorch. The model is built and trained from scratch, and transfer learning is applied by modifying the model's final layers to adapt to a custom dataset.

<h1>Dependencies</h1>
To run this project, you'll need the following libraries:
<ul>
<li>Python 3.x</li>
<li>NumPy</li>
<li>OpenCV</li>
<li>Matplotlib</li>
<li>Seaborn</li>
<li>Pandas</li>
<li>Tqdm</li>
<li>PyTorch</li>
<li>scikit-learn</li>
</ol>


<h1>Dataset Preparation</h1>
<ol>
<li>Training and Testing Data: Organize your dataset into TRAIN and TEST folders, with each subfolder representing a different class.</li>

<li>Data Loading: Images are loaded using OpenCV, resized to 128x128, and normalized. The data is stored in NumPy arrays (x_train, x_test) with corresponding labels (y_train, y_test).</li>

<li>One-Hot Encoding: Labels are one-hot encoded using OneHotEncoder from scikit-learn.</li>

<li>Data Splitting: The training data is split into training and validation sets.</li>
</ol>

<h1>Model Architecture</h1>
The VGG16 model is implemented from scratch in PyTorch, featuring:
<ul>
<li>Multiple convolutional layers with ReLU activation functions.</li>
<li>Max pooling layers to reduce spatial dimensions.</li>
<li>Adaptive average pooling to handle different input sizes.</li>
<li>Fully connected layers with ReLU and dropout layers for classification.</li>
<li>For transfer learning, the output layer is modified to match the number of classes in the custom dataset.</li>
</ul>

<h1>Training the Model</h1>
The training process involves:
<ol>
<li>Loss Function: Cross-Entropy Loss is used for multi-class classification.</li>
<li>Optimizer: Adam Optimizer with a learning rate of 0.0001.</li>
<li>Epochs: The model is trained for 50 epochs.</li>
<li>Training Loop: Includes forward pass, loss computation, backpropagation, and optimizer steps.</li>
<li>Validation Loop: Evaluates the model's performance on the validation set after each epoch.</li>
<li>Model Saving: The best model is saved based on the highest validation accuracy achieved during training.</li>
</ol>
