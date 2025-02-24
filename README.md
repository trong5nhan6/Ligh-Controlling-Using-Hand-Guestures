# Light-Controlling-Using-Hand-Gestures

## Project Overview

This project is designed to control lights using hand gestures. It utilizes deep learning models to detect specific hand gestures and map them to control commands for lighting systems. The project includes scripts for data generation, model training, and gesture detection.

## Project Structure

-   **generate_data/**: Contains scripts and data for generating training data.
    
    -   `generate_landmark_data.py`: Script to generate landmark data from hand gestures.
        
    -   `data2/`: Directory for storing generated data.
        
    -   `sign_imgs2/`: Directory for storing sign images.
    -   **requirements.txt**: List of dependencies required for the project.

        
-   **model/**: Contains machine learning models and related files.
    
    -   `MLP.py`: Multi-Layer Perceptron model implementation.
        
    -   `model_24-02 22`: Pre-trained model file.


        
-   **controller.py** : Script to control lights based on detected gestures.
    
-   **detect_simulation.py**: Main script to control lights based on detected gestures.
    
-   **hand_geture.yaml**: Configuration file for gesture settings.
    
    
-   **.gitignore**: Specifies files to be ignored by Git.
    
-   **README.md**: This file, providing an overview and instructions for the project.
    

## How to Run the Project

1.  **Install Dependencies**: Ensure you have Python installed. Install the required packages by running:
    

>     conda create -n hand_gesture_env python=3.10
>     conda activate hand_gesture_env
>     cd generate_data
>     pip install -r requirements.txt

    
1.  **Generate Data**:
           
    -   Run the  `generate_landmark_data.py`  script:
        
>         python generate_landmark_data.py

        
    -   Follow the on-screen instructions to capture hand gestures. Note the following:
        
        -   **Thumb up**: Turns on 3 lights.
            
        -   **Index finger**: Turns on light 1.
            
        -   **Index and middle fingers**: Turns on light 2.
            
        -   **Index, middle, and ring fingers**: Turns on light 3.
            
        -   **All five fingers**: Turns off all 3 lights.
            
        -   Press  `a`  to start capturing gestures for each finger. After capturing the first five gestures, press  `b` to continue.
            
        -   Capture data for three files, pressing  `q`  to end one file and start a new one.
            
3.  **Train the Model**:
    
    -   Use the generated data to train the model by running the appropriate training script in the  `model`  directory.
        
4.  **Run the Controller**:
    
    -   Execute the  `generate_landmark_data.py`  script to start the light control system:
        

>         python generate_landmark_data.py

        
   
 -   Perform the gestures in front of the camera to control the lights as per the defined mappings.
        

## Notes

-   Ensure proper lighting and camera setup for accurate gesture detection.
    
-   Regularly update the training data to improve model accuracy.
    
-   Refer to  `hand_geture.yaml`  for customizing gesture mappings and settings.