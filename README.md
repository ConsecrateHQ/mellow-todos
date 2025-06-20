# mellow-todos

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Install Python 3.12.9

Make sure you have Python 3.12.9 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### 2. Set up a virtual environment (Recommended)

It's highly recommended to use a virtual environment to avoid conflicts with other Python projects.

#### Option A: Using venv (Built-in Python)

```bash
# Create a virtual environment
python -m venv mellow-todos-env

# Activate the virtual environment
# On macOS/Linux:
source mellow-todos-env/bin/activate
# On Windows:
# mellow-todos-env\Scripts\activate

# You should see (mellow-todos-env) in your terminal prompt
```

#### Option B: Using Anaconda/Miniconda

```bash
# Create a new conda environment with Python 3.12.9
conda create -n mellow-todos python=3.12.9

# Activate the environment
conda activate mellow-todos
```

### 3. Install requirements.txt

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

**Note:** Make sure your virtual environment is activated before running this command.

### 4. Create a .env file with the necessary variables

Create a `.env` file in the root directory of the project and add the following variables:

```env
GOOGLE_API_KEY=your_google_api_key_here
FIREBASE_CREDENTIALS_PATH=path/to/your/firebase-config.json
```

Replace `your_google_api_key_here` with your actual Google API key and `path/to/your/firebase-config.json` with the path to your Firestore credentials file.

### 5. Download your Firestore config file

Download your Firestore configuration file and place it at the root folder, on the same level as the `.env` file.

### 6. Run inference

To run the YOLO detection with webcam input, execute:

```bash
python webcam_yolo_detect.py
```

## Project Structure

### Core Files

- **`webcam_yolo_detect.py`** - Main application script that runs real-time YOLO object detection using your webcam feed. This script captures video from your camera, processes each frame through the YOLO model, and displays the results with bounding boxes around detected objects.

- **`process_JSON.py`** - Utility script for processing and handling JSON data. This likely contains functions for parsing, validating, and manipulating JSON files related to the object detection results or configuration data.

- **`my_model.pt`** - Pre-trained YOLO (You Only Look Once) model file in PyTorch format. This contains the neural network weights and architecture used for object detection. The model has been trained to recognize and classify various objects in images.

### Configuration Files

- **`requirements.txt`** - Contains all Python package dependencies needed to run the project
- **`.env`** - Environment variables file (you need to create this) containing API keys and configuration paths
- **Firestore config file** - Firebase/Firestore credentials JSON file (download from Firebase console and place in root directory)
