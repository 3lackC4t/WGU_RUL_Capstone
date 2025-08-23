# WGU_RUL_Capstone
## Description
Unexpected bearing failures can result in millions of losses per year in lost production in manufacturing as well as potential interruption of service in critical industries. These failures are most often a result of fatigue failure of rotating bearings subject axial and radial thrust as a result of supporting the load of a piece of rotating machinery. Over time the bearings degrade and become damaged until they reach a material failure point, resulting in the failure of the machine as well. The ability to predict these failures in time to perform preventive maintenance would result in millions of annual revenue protected as well as substantial reductions in service interruptions in critical industries such as power, water treatment, or oil and gas.

This model can be used to predict early failure of rotational bearings and can be used to assist in predictive maintenance of such machines.

## Quick Start Guide

1. Download the following dependencies
    a. Python 3.13.5 (or later if applicable)
    b. Most recent version of pip python package installer
2. Extract the “RUL_WGU” archive to the directory in which the model is intended to be run.
3. Execute the following command in the root directory of the project in order to activate the virtual environment
    a. For Windows
        1. $”.venv/scripts/activate.ps1”
    b. For Linux/Mac
        1. source .venv/bin/activate
4. Execute the following command to download python packages from inside the root directory
    a. pip install -r requirements.txt
5. Execute the following to start the server
    a. python main.py
6. Open a web browser and navigate to “http://localhost:5000” to access the frontend of the application
7. Inside of the root project directory there is a folder titled test_data. This contains sampled vibrational data to test the performance of the application.
8. In the ‘analysis’ page use the GUI to input the desired bearing file
9. Fill in the L10 value
10. If desired, open up the “L10 Advanced” section to calculate a custom L10 value
11. Enter the RPM of the machine the bearings were captured on. For the test data the RPM is 1000RPM
12. Hit “Analyze”
13. If desired, explore the about page linked in the top left.

## Model Training
1. If desired the model can be trained again for demonstration
The model will check for the presence of the autoencoder, reference_data and preprocessor files in the model_data directory when it runs, if it finds them it will load them from save data. 
2. In order to trigger training, simply delete those files and rerun the program as above. 
3. The current proportion setting on the model is 1.0 for full model training. This will take roughly 30 minutes to fully train. If desired, reduce proportion for faster training.
