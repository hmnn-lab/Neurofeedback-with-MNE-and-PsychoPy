import psychopy
import PyQt6
from psychopy import visual, core, event
from PyQt6.QtWidgets import QApplication
import logging

def run_fixation_display(duration, start_event=None):
    """
    Display a fixation cross using PsychoPy for the specified duration.

    Args:
        duration (float): Duration to display the fixation cross in seconds.
        start_event (multiprocessing.Event, optional): Event to synchronize start with other processes.
    """
    logging.debug("Starting run_fixation_display")
    try:
        # Create a PsychoPy window
        mywin = visual.Window([500, 500], monitor="TestMonitor", color=[-1, -1, -1], fullscr=True, units="deg")
        fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), 
                                   lineWidth=2, closeShape=False, lineColor="white")
        
        # Set the start event immediately to unblock recording
        if start_event is not None:
            start_event.set()
            logging.debug("Start event set")

        # Display fixation
        start_time = core.getTime()
        while core.getTime() - start_time < duration:
            fixation.draw()
            mywin.flip()
            # Allow PyQt6 event loop to process events
            QApplication.processEvents()
            # Check for escape key to stop early
            if 'escape' in event.getKeys(keyList=['escape']):
                logging.info("Fixation display stopped by user")
                break
        
        # Clean up
        mywin.close()
        logging.debug("Fixation window closed")
    except Exception as e:
        logging.error(f"run_fixation_display failed: {e}")
        raise
