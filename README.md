> **⚠️ IMPORTANT NOTICE:** Wait 50 seconds after opening the site because it's on a free tier, so it takes time to load.

# Smart Health AI - Disease Prediction System

Welcome to the Smart Health AI project. This application uses machine learning models to help predict diseases based on clinical data and medical factors.

> **🚧 WORK IN PROGRESS:** Please note that this project is currently under development. **Only the Diabetes prediction section is fully working** at the moment. The Liver Disease and Breast Cancer modules are still being developed.

## 🌟 What This Project Has
- **Flask Web Backend:** A robust Python backend handling web requests, APIs, and model inference.
- **Machine Learning Integration:**
  - **Diabetes Prediction:** AutoML model (Fully Working)
  - **Liver Disease Prediction:** AutoML model (In Progress)
  - **Breast Cancer Detection:** CNN model for image scanning (In Progress)
- **SQLite Database:** A `users.db` database integrated to store and track historical predictions securely.
- **Interactive UI Dashboard:** A frontend for users to input their medical details, get predictions, and view their prediction history.

## 🚀 How to Run Locally

Follow these steps to run the application on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ketan-Gehlot/SMART-DISEASE-PREDCTION.git
   cd SMART-DISEASE-PREDCTION
   ```

2. **Run the setup requirements:**
   Make sure you have Python installed. You may optionally create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

4. **Start the server:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:5000/`.
