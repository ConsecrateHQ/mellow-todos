# mellow-todos

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Install Python 3.12.9

Make sure you have Python 3.12.9 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### 2. Set up a virtual environment

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

### 4. Obtain Required Credentials

Before creating your `.env` file, you'll need to obtain the necessary credentials:

#### Google AI API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google Account
3. Click "Create API Key"
4. Copy the generated API key - you'll need this for your `.env` file

#### Firestore Configuration File

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Select your project (or create a new one if you don't have one)
3. Click on the gear icon (Project Settings) in the left sidebar
4. Navigate to the "Service accounts" tab
5. Click "Generate new private key"
6. Download the JSON file - this is your Firestore configuration file
7. Save this file in the root directory of your project (same level as README.md)

### 5. Create a .env file with the necessary variables

Create a `.env` file in the root directory of the project and add the following variables:

```env
GOOGLE_API_KEY=your_google_api_key_here
FIREBASE_CREDENTIALS_PATH=path/to/your/firebase-config.json
```

Replace `your_google_api_key_here` with your actual Google API key (obtained from Google AI Studio) and `path/to/your/firebase-config.json` with the path to your Firestore credentials file.

### 6. Run inference

To run the YOLO detection with webcam input, execute:

```bash
python webcam_yolo_detect.py
```

## How It Works

This project combines computer vision and AI to automatically digitize handwritten TODO lists:

1. **YOLO Object Detection** - The webcam captures your handwritten TODO list and YOLO detects various symbols (checkboxes, status indicators) and classifies their states (NOT_STARTED, IN_PROGRESS, COMPLETED, MEETING).

2. **AI-Powered OCR** - When a stable page view is detected, the `ai_playground` module sends the image to Google Gemini AI, which:

   - Performs OCR on handwritten text
   - Maps detected symbols to task statuses
   - Analyzes task hierarchy (main tasks vs subtasks)
   - Automatically assigns tasks to relevant projects
   - Converts everything into structured JSON format

3. **Firebase Integration** - The processed data is stored in Firestore, creating a digital representation of your handwritten TODO list with timestamps, project associations, and hierarchical task structure.

4. **Automatic Processing** - The system runs in "fully automatic mode" by default, detecting when you show a TODO list to the camera and processing it without manual intervention.

## Project Structure

### Core Files

- **`webcam_yolo_detect.py`** - Main application script that runs real-time YOLO object detection using your webcam feed. This script captures video from your camera, processes each frame through the YOLO model, and displays the results with bounding boxes around detected objects.

- **`ai_playground.py`** - AI-powered OCR (Optical Character Recognition) module that uses Google Gemini AI to extract and analyze handwritten TODO lists from images. This module:

  - Takes images captured by the webcam and performs intelligent OCR
  - Analyzes handwritten text and maps YOLO-detected symbols to task statuses
  - Converts handwritten TODO lists into structured JSON format
  - Intelligently detects task hierarchies (main tasks vs subtasks) based on indentation
  - Automatically assigns tasks to projects based on content analysis
  - Handles multi-line tasks and preserves original text formatting
  - Returns structured data with status classifications (NOT_STARTED, IN_PROGRESS, MEETING, COMPLETED)

- **`process_JSON.py`** - Utility script for processing and handling JSON data. This likely contains functions for parsing, validating, and manipulating JSON files related to the object detection results or configuration data.

- **`my_model.pt`** - Pre-trained YOLO (You Only Look Once) model file in PyTorch format. This contains the neural network weights and architecture used for object detection. The model has been trained to recognize and classify various objects in images.

### Configuration Files

- **`requirements.txt`** - Contains all Python package dependencies needed to run the project
- **`.env`** - Environment variables file (you need to create this) containing API keys and configuration paths
- **Firestore config file** - Firebase/Firestore credentials JSON file (download from Firebase console and place in root directory)
