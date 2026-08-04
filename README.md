🍽️ NutriScan India

NutriScan India is an AI-powered web application that classifies Indian food images using Deep Learning. The system is built using TensorFlow and deployed through a Flask-based web interface. Users can upload an image of a food item and receive instant predictions along with a confidence score.

This project demonstrates the practical implementation of Computer Vision and Deep Learning integrated with Web Development to create a real-world AI application.


🚀 Features

Upload food images through a web interface

AI-based food classification using CNN

Instant prediction results

Confidence score display

Clean and responsive UI

Organized project structure


🛠️ Technologies Used

Python

Flask

TensorFlow / Keras

NumPy

Pillow

HTML5

CSS3


📂 Project Structure
NutriScanIndia/
│
├── dataset/                 # Training dataset
├── static/
│   └── style.css            # Styling file
├── templates/
│   └── index.html           # Frontend UI
├── uploads/                 # Uploaded images folder
├── model.h5                 # Trained CNN model
├── train_model.py           # Model training script
├── food_data.json           # Food category details
├── app.py                   # Main Flask application
└── README.md


⚙️ Installation and Setup
Step 1: Clone the Repository
git clone https://github.com/your-username/NutriScanIndia.git
cd NutriScanIndia
Step 2: Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate
Step 3: Install Required Dependencies
pip install flask tensorflow pillow numpy
Step 4: Run the Application
python app.py
Step 5: Open in Browser
http://127.0.0.1:5000


🧠 Model Details

Model Type: Convolutional Neural Network (CNN)

Input Image Size: 128x128 pixels

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Output: Multi-class food classification


🖥️ How the Application Works

The user uploads a food image.

The image is resized and preprocessed.

The trained CNN model analyzes the image.

The predicted food category and confidence score are displayed on the screen.


🎯 Project Objective

The goal of NutriScan India is to demonstrate how Artificial Intelligence can be applied to food recognition systems. It combines deep learning and web technologies to create an interactive and practical AI solution.


🔮 Future Improvements

Add detailed nutritional information for each food item

Deploy the application on cloud platforms (AWS, Render, Heroku)

Improve accuracy using transfer learning models

Add mobile-friendly UI

Integrate live camera capture functionality


👩‍💻 Author

Riya Chaubey
B.Tech Student | AI & ML Enthusiast
