from pylsl import StreamInlet, resolve_stream

def get_lsl_channel_names(stream_name):
    try:
        # Resolve the LSL stream by name
        print(f"Looking for LSL stream with name '{stream_name}'...")
        streams = resolve_stream('name', stream_name)
        
        if not streams:
            print(f"No stream named '{stream_name}' found.")
            return
        
        # Create an inlet to read from the stream
        inlet = StreamInlet(streams[0])
        
        # Get stream info
        info = inlet.info()
        
        # Get channel count
        channel_count = info.channel_count()
        
        if channel_count == 0:
            print("No channels found in the stream.")
            return
        
        # Get channel names from the stream's description
        channel_names = []
        ch = info.desc().child("channel")
        for i in range(channel_count):
            label = ch.child_value("label")
            channel_names.append(label)
        
        # Display channel names
        print(f"Channel names for stream '{stream_name}':")
        for i, name in enumerate(channel_names, 1):
            print(f"Channel {i}: {name}")
            
    except Exception as e:
        print(f"Error accessing LSL stream: {e}")

if __name__ == "__main__":
    stream_name = "obci_eeg1"
    get_lsl_channel_names(stream_name)