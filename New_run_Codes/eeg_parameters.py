def get_eeg_parameters():
    """
    Prompt the user to input EEG processing parameters and return them as individual variables.

    Returns:
    --------
    tuple
        Contains the EEG parameters as individual variables:
        - step (float): Time step in seconds
        - time_window (float): Time window in seconds
        - n_channels (int): Number of channels
        - feed_ch_names (list): List of channel names
        - low_freq (float): Lower frequency bound (e.g., for alpha band)
        - high_freq (float): Upper frequency bound (e.g., for alpha band)
        - band_name (str): Name of the frequency band (e.g., 'Alpha')
    """
    try:
        step = float(input("Enter the time step (in seconds, e.g., 0.01): "))
        if step <= 0:
            raise ValueError("Time step must be positive.")

        time_window = float(input("Enter the time window (in seconds, e.g., 5): "))
        if time_window <= 0:
            raise ValueError("Time window must be positive.")

        n_channels = int(input("Enter the number of channels (e.g., 2): "))
        if n_channels <= 0:
            raise ValueError("Number of channels must be positive.")

        feed_ch_names = input("Enter channel names (comma-separated, e.g., O1,Pz): ")
        feed_ch_names = [name.strip() for name in feed_ch_names.split(",") if name.strip()]
        if not feed_ch_names:
            raise ValueError("At least one channel name must be provided.")

        low_freq = float(input("Enter the lower frequency bound (e.g., 8): "))
        if low_freq < 0:
            raise ValueError("Lower frequency must be non-negative.")

        high_freq = float(input("Enter the upper frequency bound (e.g., 12): "))
        if high_freq <= low_freq:
            raise ValueError("Upper frequency must be greater than lower frequency.")

        band_name = input("Enter the name of the frequency band (e.g., Alpha): ").strip()
        if not band_name:
            raise ValueError("Frequency band name cannot be empty.")

        return step, time_window, n_channels, feed_ch_names, low_freq, high_freq, band_name

    except ValueError as e:
        print(f"Error: {e}")
        return None