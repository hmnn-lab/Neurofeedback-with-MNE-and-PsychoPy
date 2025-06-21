
# Suceesfully working codes!

The latest codes working and tested with pilot for Neurofeedback training.

The Welcome Page collects user name and ID to initialize a session; the Baseline Window records baseline EEG data; the Paradigm Choice Window allows selection of processing paradigms like PSD or coherence; the Preprocessing Dialog handles artifact removal and data preparation; and the Real-Time Feedback Window visualizes live EEG results. These windows, managed via a QStackedWidget, provide a streamlined, modular workflow for real-time EEG processing.

IMPORTANT FILES: 
+ new_main.py : contain the main architecture nad control of the app

+ final_baseline_window: contains the code for baseline recording or loading pre-recorded baseline file 

+ final_preproc_window: contains code for dialog box that appears for Preprocessing and calculqate baseline value for selected modality

+ paradigm_choice_window: conatins codes to allow user to select modality, select channels and frequency bands

+ feedback_window: contains codes for the final window of the app to start and stop streaming LSL streams and launch the game; also contains tting of real time feedback values. 


