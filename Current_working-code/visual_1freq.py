import psychopy
from psychopy.hardware import keyboard
from psychopy import visual, core
import numpy as np

def visualize_eeg_feedback(power_change, monitor_name="TestMonitor", freq_band="Alpha (8-12 Hz)"):
    """
    Visualize EEG feedback using PsychoPy with a dynamic circle radius based on power change values.
    
    Parameters:
    -----------
    pow_change_values : list or array
        List of power change values to update the circle's radius.
    monitor_name : str, optional
        Name of the monitor (default: "TestMonitor").
    freq_band : str, optional
        Name of the frequency band to display (default: "Alpha (8-12 Hz)").
        
    Returns:
    --------
    None
    """
    # Create a window
    mywin = visual.Window([1920, 1080], monitor=monitor_name, color=[0, 0, 0], 
                         fullscr=True, units="deg")
    
    # Create fixation cross
    fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), 
                               lineWidth=2, closeShapes=False, lineColor="white")
    
    # Create square and circle
    sqr = visual.Rect(win=mywin, size=[20, 20], fillColor='black', pos=(0.0, 0.0))
    mycir = visual.Circle(win=mywin, radius=5.0, edges=128, lineWidth=1.5, 
                         lineColor='white', fillColor='red', pos=(0, 0))
    
    # Create text stimulus for frequency band
    freq_band_name = visual.TextStim(win=mywin, text=freq_band, pos=(-30, 0), 
                                    color=(1, 1, 1), opacity=0.75, 
                                    anchorHoriz='center', anchorVert='bottom', 
                                    height=10, ori=0.0)
    
    # Parameters for smooth radius change
    max_radius = 10
    min_radius = 2
    
    # Create keyboard component
    kb = keyboard.Keyboard()
    
    # Main visualization loop
    while True:
        # Check for keyboard input to exit
        keys = kb.getKeys(['escape', 'q'])
        if 'escape' in keys or 'q' in keys:
            break
        
        # Update circle radius based on latest power change value
        if len(power_change) > 0:
            current_power_change = power_change[-1]
            new_radius = np.interp(np.mean(current_power_change), [0, 100], 
                                 [min_radius, max_radius])
            mycir.radius = new_radius
        
        # Draw all stimuli
        sqr.draw()
        mycir.draw()
        fixation.draw()
        freq_band_name.draw()
        
        # Update the window
        mywin.flip()
    
    # Clean up and close the window
    mywin.close()
    core.quit()