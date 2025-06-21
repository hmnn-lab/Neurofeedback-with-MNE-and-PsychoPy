import numpy as np
from psychopy import visual, core
from psychopy.hardware import keyboard

def setup_and_run_visuals(power_changes=None, update_interval=0.5):
    """
    Set up and optionally run a PsychoPy visualization for EEG neurofeedback with a moving dot.
    
    Parameters:
    -----------
    power_changes : list or array, optional
        List of [theta_power, alpha_power] values (0-100) to update dot position.
        If provided, the function runs an update loop. If None, returns components for manual updates.
    update_interval : float, optional
        Time interval (seconds) between updates (default: 0.5).
        
    Returns:
    --------
    dict : Contains the following visualization components:
        - mywin: PsychoPy window
        - line1: X-axis Line stimulus
        - line2: Y-axis Line stimulus
        - freq_band_1_name: TextStim for theta band
        - freq_band_2_name: TextStim for alpha band
        - dot: Circle stimulus (moving dot)
        - kb: Keyboard component
        - clock: Core clock
        - update_timer: Countdown timer
        - window_width: Window width in pixels
        - window_height: Window height in pixels
    """
    # Create a window
    mywin = visual.Window(
        [500, 500],
        monitor="TestMonitor",
        color=[-1, -1, -1],
        fullscr=False,
        units="pix"
    )
    window_width, window_height = mywin.size
    
    # Create quadrant lines
    line1 = visual.Line(
        win=mywin,
        start=(-window_width / 2, 0),
        end=(window_width / 2, 0),
        units='pix',
        lineWidth=2.0,
        pos=(0, 0),
        color=(-1, -1, -1),
        name='X-axis'
    )
    line2 = visual.Line(
        win=mywin,
        start=(0, -window_height / 2),
        end=(0, window_height / 2),
        units='pix',
        lineWidth=2.0,
        pos=(0, 0),
        color=(-1, -1, -1),
        name='Y-axis'
    )
    
    # Create text stimuli for frequency band names
    freq_band_1_name = visual.TextStim(
        win=mywin,
        text="Theta (4-7 Hz)",
        pos=(0, -30),
        color=(1, 1, 1),
        opacity=0.75,
        anchorHoriz='left',
        anchorVert='center',
        height=10,
        ori=0.0
    )
    freq_band_2_name = visual.TextStim(
        win=mywin,
        text="Alpha (8-12 Hz)",
        pos=(-30, 0),
        color=(1, 1, 1),
        opacity=0.75,
        anchorHoriz='center',
        anchorVert='bottom',
        height=10,
        ori=90.0
    )
    
    # Create moving red dot
    dot = visual.Circle(
        win=mywin,
        radius=20,
        edges=128,
        fillColor='red',
        lineColor='white',
        pos=(0, 0)
    )
    
    # Create keyboard component
    kb = keyboard.Keyboard()
    
    # Create clock and timer
    clock = core.Clock()
    update_timer = core.CountdownTimer(update_interval)
    
    # If power_changes is provided, run an update loop
    if power_changes is not None:
        epoch_count = 0
        while True:
            # Check for escape key to exit
            if kb.getKeys(['escape']):
                break
            
            # Update visualization when timer expires
            if update_timer.getTime() <= 0:
                # Map power changes to window size
                x_pos = np.interp(power_changes[0], [0, 100], [-window_width / 2, window_width / 2])
                y_pos = np.interp(power_changes[1], [0, 100], [-window_height / 2, window_height / 2])
                dot.setPos((x_pos, y_pos))
                
                # Reset update timer and increment epoch count
                update_timer.reset(update_interval)
                epoch_count += 1
                
                # Draw stimuli
                dot.draw()
                freq_band_1_name.draw()
                freq_band_2_name.draw()
                line1.draw()
                line2.draw()
                mywin.flip()
    
    # Draw initial stimuli (if no power_changes or after loop)
    dot.draw()
    freq_band_1_name.draw()
    freq_band_2_name.draw()
    line1.draw()
    line2.draw()
    mywin.flip()
    
    # Return components for further use
    return {
        'mywin': mywin,
        'line1': line1,
        'line2': line2,
        'freq_band_1_name': freq_band_1_name,
        'freq_band_2_name': freq_band_2_name,
        'dot': dot,
        'kb': kb,
        'clock': clock,
        'update_timer': update_timer,
        'window_width': window_width,
        'window_height': window_height
    }

setup_and_run_visuals(power_changes=None, update_interval=0.5)
