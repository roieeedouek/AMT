import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm  # Import LogNorm from colors module
import matplotlib.animation as animation
from scipy import signal, optimize
import pyroomacoustics as pra
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

# Try to import numba for JIT compilation
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("Numba JIT compilation available for acceleration")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, running without JIT acceleration")
    
    # Create dummy decorators when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    # Dummy parallel range
    def prange(*args):
        return range(*args)

class AcousticTracker:
    """Acoustic multi-target tracking system using home theater setup with pyroomacoustics and Kalman filtering"""
    
    def __init__(self, room_dim=(5.0, 6.0, 2.4), speed_of_sound=343.0):
        """Initialize the acoustic tracker
        
        Args:
            room_dim (tuple): Room dimensions (width, length, height) in meters
            speed_of_sound (float): Speed of sound in m/s at room temperature
        """
        self.room_dim = room_dim
        self.c = speed_of_sound
        self.fs = 48000  # Increased sampling rate for better resolution
        
        # Set up the room simulation
        self.setup_room()
        
        # For storing targets and tracking data
        self.targets = []  # List of targets with positions, velocities, etc.
        self.tracking_data = {}  # Store tracking results
        
        # Store correlation history for visualization
        self.correlation_history = []
        
        # Store previous correlations for MTI filtering
        self.prev_correlations = {}
        
        # Dictionary to store Kalman filters for each target
        self.kalman_filters = {}
    
    def setup_room(self):
        """Set up the room, speakers, and microphones with optimized geometry"""
        width, length, height = self.room_dim
        
        # Create the room with material
        material = pra.Material(energy_absorption=0.2)
        
        self.room = pra.ShoeBox(
            self.room_dim, 
            fs=self.fs,
            materials=material,
            max_order=3  # Max reflection order
        )
        
        # Speaker positions - 5.1 setup
        self.speakers = [
            # Front speakers
            np.array([0.2, 0.5, 1.0]),         # Front Left
            np.array([width/2, 0.3, 0.7]),     # Center
            np.array([width - 0.2, 0.5, 1.0]), # Front Right
            # Surround speakers
            np.array([0.5, length - 0.5, 1.2]),         # Surround Left
            np.array([width - 0.5, length - 0.5, 1.2]), # Surround Right
        ]
        
        # Microphone positions - soundbar-like arrangement at the front
        self.mics = [
            np.array([width/2 - 0.5, 0.3, 1.0]),  # Left Mic
            np.array([width/2 + 0.5, 0.3, 1.0]),  # Right Mic
            np.array([width/2      , 0.3, 0.9]),  # Center Mic Up
            np.array([width/2      , 0.3, 1.1]),  # Center Mic Down
        ]
        
        # Create microphone array
        self.mic_array = np.array(self.mics).T  # pyroomacoustics expects shape (3, n_mics)
        self.room.add_microphone_array(self.mic_array)
        
        # Set up Zadoff-Chu sequences for each speaker
        self.zc_sequences = self.generate_zc_sequences(len(self.speakers))
        
        # Calculate direct path delays for each speaker-mic pair
        self.direct_delays = {}
        for s_idx, speaker_pos in enumerate(self.speakers):
            for m_idx, mic_pos in enumerate(self.mics):
                direct_dist = np.linalg.norm(mic_pos - speaker_pos)
                direct_delay_samples = int(direct_dist / self.c * self.fs)
                self.direct_delays[(s_idx, m_idx)] = direct_delay_samples
                
        # Enhancement 4: Calculate geometric quality for z-resolution
        # This helps us estimate how well our setup can resolve z-position
        try:
            z_resolution_quality = self._calculate_geometry_quality()
            print(f"Z-axis resolution quality: {z_resolution_quality:.2f} (higher is better)")
        except Exception as e:
            print(f"Warning: Could not calculate geometry quality: {str(e)}")
    
    def _calculate_geometry_quality(self):
        """Calculate the geometric quality for z-axis resolution
        
        Returns:
            float: A quality metric (higher is better)
        """
        # A simple metric that measures diversity in vertical angles
        vertical_angles = []
        
        # Calculate angles between all speaker-mic pairs
        for speaker_pos in self.speakers:
            for mic_pos in self.mics:
                # Vector from speaker to mic
                vec = mic_pos - speaker_pos
                
                # Calculate vertical angle
                horizontal_dist = np.sqrt(vec[0]**2 + vec[1]**2)
                vertical_angle = np.arctan2(vec[2], horizontal_dist)
                vertical_angles.append(vertical_angle)
        
        # Calculate standard deviation of angles
        # Higher std dev means more diverse angles, which is better for z-resolution
        vertical_angles = np.array(vertical_angles)
        return np.std(vertical_angles) * 10  # Scale up for readability
    
    def generate_zc_sequences(self, n_speakers, seq_length=127):
        """Generate unique Zadoff-Chu sequences for each speaker
        
        Args:
            n_speakers (int): Number of speakers
            seq_length (int): Length of each sequence - increased for better resolution
            
        Returns:
            list: List of ZC sequences in time domain
        """
        sequences = []
        
        for i in range(n_speakers):
            # Use different root indices for minimal cross-correlation
            u = 2*i + 1  # Odd numbers are coprime to powers of 2
            
            # Generate the sequence in frequency domain
            n = np.arange(seq_length)
            if seq_length % 2 == 0:  # Even length
                zc_freq = np.exp(-1j * np.pi * u * n * (n+1) / seq_length)
            else:  # Odd length
                zc_freq = np.exp(-1j * np.pi * u * n**2 / seq_length)
            
            # Convert to time domain using IFFT
            zc_time = np.fft.ifft(zc_freq)
            
            # Modulate to carrier frequency (21 kHz)
            fc = 21000
            t = np.arange(seq_length) / self.fs
            carrier = np.exp(1j * 2 * np.pi * fc * t)
            modulated = zc_time * carrier
            
            # Make it real for pyroomacoustics
            # Take the real part and normalize
            real_signal = np.real(modulated)
            real_signal = real_signal / np.max(np.abs(real_signal))
            
            sequences.append(real_signal)
        
        return sequences
    
    def init_kalman_filter(self, target_id, pos, dt=0.1):
        """Initialize an enhanced Kalman filter for a target with acceleration model
        
        Args:
            target_id (int): Target ID
            pos (np.array): Initial position
            dt (float): Time step in seconds
            
        Returns:
            KalmanFilter: Initialized Kalman filter
        """
        # Enhancement 3: Add acceleration to the state model
        # We'll track 3D position, velocity, and acceleration: [x, y, z, vx, vy, vz, ax, ay, az]
        dim_x = 9  # Expanded state dimension
        dim_z = 3  # Measurement dimension (x, y, z position)
        
        # Create Kalman filter
        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # State transition matrix for constant acceleration model
        # [1, 0, 0, dt, 0, 0, 0.5*dt^2, 0, 0]
        # [0, 1, 0, 0, dt, 0, 0, 0.5*dt^2, 0]
        # [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt^2]
        # [0, 0, 0, 1, 0, 0, dt, 0, 0]
        # [0, 0, 0, 0, 1, 0, 0, dt, 0]
        # [0, 0, 0, 0, 0, 1, 0, 0, dt]
        # [0, 0, 0, 0, 0, 0, 1, 0, 0]
        # [0, 0, 0, 0, 0, 0, 0, 1, 0]
        # [0, 0, 0, 0, 0, 0, 0, 0, 1]
        kf.F = np.eye(dim_x)
        # Position updated by velocity
        kf.F[:3, 3:6] = np.eye(3) * dt
        # Position updated by acceleration (0.5 * dt^2)
        kf.F[:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        # Velocity updated by acceleration
        kf.F[3:6, 6:9] = np.eye(3) * dt
        
        # Measurement function (we only measure position)
        kf.H = np.zeros((dim_z, dim_x))
        kf.H[:3, :3] = np.eye(3)
        
        # Process noise covariance - tuned for better precision
        # Lower process noise for more stable tracking
        q = 0.001  # Base process noise
        
        # Use less process noise for z-axis to make it more stable
        q_pos = np.array([q, q, q*0.5])       # Position process noise
        q_vel = np.array([q*5, q*5, q*2.5])   # Velocity process noise
        q_acc = np.array([q*10, q*10, q*5])   # Acceleration process noise
        
        # Create diagonal process noise matrix
        q_dim = block_diag(
            q_pos[0], q_pos[1], q_pos[2],
            q_vel[0], q_vel[1], q_vel[2],
            q_acc[0], q_acc[1], q_acc[2]
        )
        kf.Q = q_dim
        
        # Adjust measurement noise for better z-axis stability
        # More weight on x and y measurements compared to z
        r_x = 0.05  # Reduced for more precision when measurements are good
        r_y = 0.05
        r_z = 0.1   # Higher noise for z-measurements as they are less reliable
        kf.R = np.diag([r_x, r_y, r_z])
        
        # Initial state
        kf.x = np.zeros(dim_x)
        kf.x[:3] = pos  # Position
        
        # Set initial velocity if available
        if target_id < len(self.targets):
            kf.x[3:6] = self.targets[target_id]['velocity']  # Velocity
        
        # Initial acceleration is zero
        kf.x[6:9] = np.zeros(3)
        
        # Initial state covariance
        kf.P = np.eye(dim_x) * 0.1
        # Higher initial uncertainty for z-axis
        kf.P[2, 2] = 0.2    # More uncertainty in z position
        kf.P[5, 5] = 0.3    # More uncertainty in z velocity
        kf.P[8, 8] = 0.5    # More uncertainty in z acceleration
        
        return kf
    
    def add_target(self, position, velocity=(0, 0, 0), radius=0.2, name=None):
        """Add a target (person) to track
        
        Args:
            position (tuple): Initial position (x, y, z) in meters
            velocity (tuple): Initial velocity vector (vx, vy, vz) in m/s
            radius (float): Approximate radius of the target
            name (str): Name for the target
            
        Returns:
            dict: The added target
        """
        target_id = len(self.targets)
        position = np.array(position)
        velocity = np.array(velocity)
        
        target = {
            'id': target_id,
            'position': position,
            'velocity': velocity,
            'radius': radius,
            'name': name if name else f"Target_{target_id}",
            'history': [position.copy()],  # Track position history
            'estimated_positions': [],  # Raw estimated positions
            'filtered_positions': [position.copy()]  # Kalman filtered positions
        }
        
        self.targets.append(target)
        self.tracking_data[target_id] = {
            'true_positions': [position.copy()],
            'estimated_positions': [],
            'filtered_positions': [position.copy()]
        }
        
        # Initialize Kalman filter for this target
        self.kalman_filters[target_id] = self.init_kalman_filter(target_id, position)
        
        return target
    
    def update_targets(self, dt):
        """Update target positions based on their velocities
        
        Args:
            dt (float): Time step in seconds
        """
        for target in self.targets:
            # Update position
            new_pos = target['position'] + target['velocity'] * dt
            
            # Simple boundary reflection
            for i in range(3):
                dim_size = self.room_dim[i]
                if new_pos[i] < target['radius'] or new_pos[i] > dim_size - target['radius']:
                    target['velocity'][i] *= -0.8  # Bounce with some energy loss
                    new_pos = target['position'] + target['velocity'] * dt
            
            target['position'] = new_pos
            target['history'].append(new_pos.copy())
            
            # Store in tracking data
            self.tracking_data[target['id']]['true_positions'].append(new_pos.copy())
    
    def simulate_echoes(self, block_length=2048):
        """Simulate acoustic echoes using simplified model
        
        Args:
            block_length (int): Signal block length
            
        Returns:
            dict: Dictionary mapping (speaker_idx, mic_idx) to received signals
        """
        received_signals = {}
        
        # For each speaker-mic pair, create direct path signals and reflections
        for s_idx, speaker_pos in enumerate(self.speakers):
            for m_idx, mic_pos in enumerate(self.mics):
                # Direct path
                direct_delay = self.direct_delays[(s_idx, m_idx)]
                
                # Create a buffer for the received signal
                # For simplicity, we'll just use a buffer that can hold the sequence plus some delay
                signal = self.zc_sequences[s_idx]
                buffer_length = block_length + 1000  # Extra room for delays
                mic_signal = np.zeros(buffer_length)
                
                # First, add the direct path
                if direct_delay < buffer_length - len(signal):
                    mic_signal[direct_delay:direct_delay+len(signal)] += signal
                
                # Then add reflections from targets (simplified)
                for target in self.targets:
                    # Calculate path: speaker -> target -> mic
                    to_target = np.linalg.norm(target['position'] - speaker_pos)
                    from_target = np.linalg.norm(mic_pos - target['position'])
                    total_dist = to_target + from_target
                    delay_samples = int(total_dist / self.c * self.fs)
                    
                    # Calculate attenuation (simplified model)
                    # In reality, this would depend on many factors
                    attenuation = 0.7 / (to_target * from_target)
                    
                    # Add the reflection
                    if delay_samples < buffer_length - len(signal):
                        mic_signal[delay_samples:delay_samples+len(signal)] += attenuation * signal
                
                # Store the received signal
                received_signals[(s_idx, m_idx)] = mic_signal
        
        return received_signals
    
    def detect_echoes(self, received_signals):
        """Detect echoes in received signals using correlation and MTI filtering
        
        Args:
            received_signals (dict): Received signals for each speaker-mic pair
            
        Returns:
            dict: Detected echo data
        """
        echo_data = {}
        
        for (s_idx, m_idx), received_signal in received_signals.items():
            # Get the ZC sequence for this speaker
            sequence = self.zc_sequences[s_idx]
            
            # Cross-correlate the received signal with the sequence
            # This will give peaks at the delay times
            correlation = np.abs(signal.correlate(received_signal, sequence, mode='valid'))
            
            # Normalize
            correlation = correlation / np.max(correlation) if np.max(correlation) > 0 else correlation
            
            # Apply MTI filtering - compare with previous correlation
            pair_id = (s_idx, m_idx)
            if pair_id in self.prev_correlations:
                # Get previous correlation
                prev_corr = self.prev_correlations[pair_id]
                
                # Make sure the lengths match
                min_len = min(len(correlation), len(prev_corr))
                correlation = correlation[:min_len]
                prev_corr = prev_corr[:min_len]
                
                # Calculate difference to highlight moving objects
                diff = np.abs(correlation - prev_corr)
                
                # Normalize difference
                diff = diff / np.max(diff) if np.max(diff) > 0 else diff
            else:
                # First frame, no difference available
                diff = correlation
            
            # Store current correlation for next time
            self.prev_correlations[pair_id] = correlation.copy()
            
            # Find peaks in the difference - these are moving targets
            threshold = 0.3
            # scipy.signal.find_peaks is more robust than manual approach
            peaks, peak_info = signal.find_peaks(diff, height=threshold, distance=20)
            
            # Sort by amplitude
            if len(peaks) > 0:
                peak_heights = peak_info['peak_heights']
                sorted_indices = np.argsort(peak_heights)[::-1]  # Descending
                peaks = peaks[sorted_indices]
                peak_heights = peak_heights[sorted_indices]
            else:
                peak_heights = []
            
            # Get direct path delay
            direct_delay = self.direct_delays[pair_id]
            
            # Calculate elapsed time from direct path for each peak
            time_diffs = []
            for peak in peaks:
                # Time difference between echo and direct path
                time_diff = peak - direct_delay
                time_diffs.append(time_diff)
            
            # Store the peak data
            echo_data[pair_id] = {
                'peaks': peaks,
                'heights': peak_heights,
                'time_diffs': time_diffs,
                'correlation': correlation,
                'diff': diff
            }
        
        return echo_data
    
    def locate_targets(self, echo_data, dt=0.1, step=0):
        """Locate targets using multilateration from detected echoes with Kalman filtering
        
        Args:
            echo_data (dict): Echo detection data
            dt (float): Time step in seconds
            step (int): Current step number for debugging
            
        Returns:
            list: Estimated positions for each target
        """
        # Create ranging ellipses from the detected peaks
        ellipses = []
        
        for (s_idx, m_idx), data in echo_data.items():
            speaker_pos = self.speakers[s_idx]
            mic_pos = self.mics[m_idx]
            
            # For each peak (potentially a target echo)
            for i, (peak, height, time_diff) in enumerate(zip(data['peaks'], data['heights'], data['time_diffs'])):
                if i >= len(self.targets):
                    break  # We only have this many targets
                
                # Skip negative time differences (which would be before direct path)
                if time_diff <= 0:
                    continue
                
                # Convert time difference to distance
                additional_distance = time_diff * self.c / self.fs
                direct_distance = np.linalg.norm(mic_pos - speaker_pos)
                total_path_length = direct_distance + additional_distance
                
                # Create ellipse
                ellipse = {
                    'speaker_idx': s_idx,
                    'mic_idx': m_idx,
                    'speaker_pos': speaker_pos,
                    'mic_pos': mic_pos,
                    'path_length': total_path_length,
                    'height': height
                }
                ellipses.append(ellipse)
        
        # For each target, find the ellipses that match its predicted position
        target_ellipses = self.match_ellipses_to_targets(ellipses)
        
        # Locate each target using its group of ellipses and apply Kalman filtering
        estimated_positions = []
        filtered_positions = []
        
        for i, target in enumerate(self.targets):
            # Get Kalman filter for this target
            kf = self.kalman_filters[target['id']]
            
            # Predict next state (time update)
            kf.predict()
            
            # If we have ellipses for this target
            if i < len(target_ellipses) and len(target_ellipses[i]) >= 2:
                # Use two-stage multilateration to find the position with high precision
                # Save error maps every 1 steps for debugging
                save_error_map = (step % 1 == 0)
                position = self.multilateration_twostage(target_ellipses[i], save_error_map=save_error_map)
                
                if position is not None:
                    # Use position as measurement
                    kf.update(position)
                    
                    # Get filtered position from Kalman state
                    filtered_pos = kf.x[:3]
                    
                    # Store the estimated and filtered positions
                    estimated_positions.append(position)
                    filtered_positions.append(filtered_pos)
                    
                    target['estimated_positions'].append(position)
                    target['filtered_positions'].append(filtered_pos)
                    
                    self.tracking_data[target['id']]['estimated_positions'].append(position)
                    self.tracking_data[target['id']]['filtered_positions'].append(filtered_pos)
                else:
                    # No position found, use prediction
                    filtered_pos = kf.x[:3]
                    
                    estimated_positions.append(None)
                    filtered_positions.append(filtered_pos)
                    
                    target['estimated_positions'].append(None)
                    target['filtered_positions'].append(filtered_pos)
                    
                    self.tracking_data[target['id']]['estimated_positions'].append(None)
                    self.tracking_data[target['id']]['filtered_positions'].append(filtered_pos)
            else:
                # Not enough ellipses, use prediction
                filtered_pos = kf.x[:3]
                
                estimated_positions.append(None)
                filtered_positions.append(filtered_pos)
                
                target['estimated_positions'].append(None)
                target['filtered_positions'].append(filtered_pos)
                
                self.tracking_data[target['id']]['estimated_positions'].append(None)
                self.tracking_data[target['id']]['filtered_positions'].append(filtered_pos)
        
        return estimated_positions, filtered_positions
    
    def match_ellipses_to_targets(self, ellipses, max_dist=1.0):
        """Match ellipses to targets based on predicted positions
        
        Args:
            ellipses (list): List of ellipse dictionaries
            max_dist (float): Maximum distance for a valid association
            
        Returns:
            list: List of ellipse groups for each target
        """
        if not ellipses:
            return [[] for _ in range(len(self.targets))]
        
        # Initialize empty groups for each target
        target_ellipses = [[] for _ in range(len(self.targets))]
        
        # For each target, find compatible ellipses
        for target_idx, target in enumerate(self.targets):
            # Get Kalman filter for this target
            kf = self.kalman_filters[target['id']]
            
            # Get predicted position from Kalman state
            predicted_pos = kf.x[:3]
            
            # Evaluate each ellipse for compatibility with this target
            for ellipse in ellipses:
                # Evaluate how well this ellipse matches the target's predicted position
                speaker_pos = ellipse['speaker_pos']
                mic_pos = ellipse['mic_pos']
                path_length = ellipse['path_length']
                
                # Calculate the path length if the reflection was from the predicted position
                to_target = np.linalg.norm(predicted_pos - speaker_pos)
                from_target = np.linalg.norm(mic_pos - predicted_pos)
                predicted_path = to_target + from_target
                
                # Calculate error between predicted and actual path lengths
                path_error = abs(predicted_path - path_length)
                
                # If the error is small enough, the ellipse is compatible with this target
                if path_error < max_dist:
                    target_ellipses[target_idx].append(ellipse)
        
        # For any ellipses that could belong to multiple targets,
        # assign to the target with the smallest path error
        for ellipse in ellipses:
            candidates = []
            
            for target_idx, target in enumerate(self.targets):
                if ellipse in target_ellipses[target_idx]:
                    # Get Kalman filter for this target
                    kf = self.kalman_filters[target['id']]
                    
                    # Get predicted position from Kalman state
                    predicted_pos = kf.x[:3]
                    
                    # Calculate the path length if the reflection was from the predicted position
                    speaker_pos = ellipse['speaker_pos']
                    mic_pos = ellipse['mic_pos']
                    path_length = ellipse['path_length']
                    
                    to_target = np.linalg.norm(predicted_pos - speaker_pos)
                    from_target = np.linalg.norm(mic_pos - predicted_pos)
                    predicted_path = to_target + from_target
                    
                    # Calculate error between predicted and actual path lengths
                    path_error = abs(predicted_path - path_length)
                    
                    candidates.append((target_idx, path_error))
            
            # If this ellipse is assigned to multiple targets, keep only the best match
            if len(candidates) > 1:
                # Sort by path error (ascending)
                candidates.sort(key=lambda x: x[1])
                
                # Remove from all except the best match
                for target_idx, _ in candidates[1:]:
                    if ellipse in target_ellipses[target_idx]:
                        target_ellipses[target_idx].remove(ellipse)
        
        return target_ellipses
    
    # JIT-compiled error function for much faster computation
    @staticmethod
    @jit(nopython=True)
    def _compute_error_numba(point, speaker_positions, mic_positions, path_lengths, prev_point=None, z_weight=0.5):
        """Numba-accelerated error function
        
        Args:
            point (np.array): Position to evaluate
            speaker_positions (np.array): Array of speaker positions
            mic_positions (np.array): Array of mic positions
            path_lengths (np.array): Array of path lengths
            prev_point (np.array, optional): Previous position estimate
            z_weight (float): Weight for z-axis continuity
            
        Returns:
            float: Total error
        """
        # Basic error from ellipses
        total_error = 0.0
        
        for i in range(len(path_lengths)):
            speaker_to_point = np.sqrt(
                (point[0] - speaker_positions[i, 0])**2 + 
                (point[1] - speaker_positions[i, 1])**2 + 
                (point[2] - speaker_positions[i, 2])**2
            )
            
            point_to_mic = np.sqrt(
                (point[0] - mic_positions[i, 0])**2 + 
                (point[1] - mic_positions[i, 1])**2 + 
                (point[2] - mic_positions[i, 2])**2
            )
            
            computed_path = speaker_to_point + point_to_mic
            error = (computed_path - path_lengths[i])**2
            total_error += error
        
        # If we have a previous point, add continuity constraint
        if prev_point is not None:
            # Calculate distance from previous point, with higher weight on z-axis
            z_change = (point[2] - prev_point[2])**2
            # Penalize z-axis changes more heavily to reduce z-axis jumping
            total_error += z_weight * z_change
        
        return total_error

    def _compute_error(self, point, ellipses, prev_point=None):
        """Helper function to compute error for a point
        
        Args:
            point (np.array): Position to evaluate
            ellipses (list): List of ellipses
            prev_point (np.array, optional): Previous position estimate for continuity
            
        Returns:
            float: Total error
        """
        # Convert ellipses to arrays for numba if available
        if NUMBA_AVAILABLE and len(ellipses) > 0:
            # Convert to numba-friendly format
            speaker_positions = np.array([e['speaker_pos'] for e in ellipses])
            mic_positions = np.array([e['mic_pos'] for e in ellipses])
            path_lengths = np.array([e['path_length'] for e in ellipses])
            
            # Use the JIT-compiled function
            return self._compute_error_numba(
                point, speaker_positions, mic_positions, path_lengths, prev_point)
        
        # Fallback to standard implementation
        total_error = 0
        for ellipse in ellipses:
            speaker_to_point = np.linalg.norm(point - ellipse['speaker_pos'])
            point_to_mic = np.linalg.norm(ellipse['mic_pos'] - point)
            computed_path = speaker_to_point + point_to_mic
            error = (computed_path - ellipse['path_length'])**2
            total_error += error
        
        # If we have a previous point, add continuity constraint
        # especially for z-axis to avoid z-axis ambiguity/jumping
        if prev_point is not None:
            # Calculate distance from previous point, with higher weight on z-axis
            z_change = (point[2] - prev_point[2])**2
            # Penalize z-axis changes more heavily to reduce z-axis jumping
            z_weight = 0.5  # Weight for z-continuity constraint
            total_error += z_weight * z_change
        
        return total_error
        
    # Create an optimizable objective function for scipy.optimize
    def _create_objective_function(self, ellipses, prev_point=None):
        """Create an objective function for scipy.optimize
        
        Args:
            ellipses (list): List of ellipse dictionaries
            prev_point (np.array, optional): Previous position for continuity
            
        Returns:
            callable: Objective function for optimization
        """
        # Convert to arrays once for optimization
        if len(ellipses) > 0:
            speaker_positions = np.array([e['speaker_pos'] for e in ellipses])
            mic_positions = np.array([e['mic_pos'] for e in ellipses])
            path_lengths = np.array([e['path_length'] for e in ellipses])
            
            # Simple wrapper for _compute_error to work with scipy.optimize
            def objective(x):
                return self._compute_error_numba(
                    x, speaker_positions, mic_positions, path_lengths, prev_point) \
                    if NUMBA_AVAILABLE else \
                    self._compute_error(x, ellipses, prev_point)
                    
            return objective
        
        # Fallback if no ellipses
        return lambda x: float('inf')
    
    def multilateration_twostage(self, ellipses, coarse_res=10, fine_res=20, save_error_map=False):
        """Highly optimized multilateration using numerical optimization algorithms
        
        Args:
            ellipses (list): List of ellipse dictionaries
            coarse_res (int): Resolution for coarse search (for grid-based fallback)
            fine_res (int): Resolution for fine search (for grid-based fallback)
            save_error_map (bool): Whether to save error maps for debugging
            
        Returns:
            np.array: Estimated position
        """
        if not ellipses:
            return None
        
        # Enhancement 3: Historical Tracking Improvements - Get trajectory information
        prev_point = None
        trajectory_direction = None
        trajectory_points = []
        target_id = None
        
        try:
            # Check if we have a target ID for these ellipses
            for target_idx, target_ellipses in enumerate(self.match_ellipses_to_targets(ellipses)):
                if any(e in target_ellipses for e in ellipses):
                    # Found the target that these ellipses belong to
                    target_id = target_idx
                    
                    # Get the Kalman filter for this target
                    kf = self.kalman_filters.get(target_id)
                    if kf is not None:
                        # Use the Kalman state as previous point
                        prev_point = kf.x[:3].copy()
                        
                        # Get trajectory information if available
                        target = self.targets[target_id]
                        if len(target['filtered_positions']) >= 3:
                            # Get the last several points to create a trajectory
                            trajectory_points = target['filtered_positions'][-3:]
                            
                            # Calculate trajectory direction
                            if len(trajectory_points) >= 2:
                                trajectory_direction = trajectory_points[-1] - trajectory_points[-2]
                                # Normalize direction vector
                                trajectory_norm = np.linalg.norm(trajectory_direction)
                                if trajectory_norm > 0:
                                    trajectory_direction = trajectory_direction / trajectory_norm
                    break
        except Exception as e:
            print(f"Warning: Error getting trajectory information: {str(e)}")
                
        # Enhancement 2: Dynamic Resolution Adaptation
        # Increase z-resolution for better precision
        z_res_multiplier = 2
        
        # Stage 1: Coarse search
        x = np.linspace(0, self.room_dim[0], coarse_res)
        y = np.linspace(0, self.room_dim[1], coarse_res)
        z = np.linspace(0, self.room_dim[2] + 0.3, coarse_res * z_res_multiplier)
        
        best_point = None
        min_error = float('inf')
        
        # Create the objective function for optimization
        objective_fn = self._create_objective_function(ellipses, prev_point)
        
        # Get an initial guess based on historical information or room center
        initial_guess = np.array([
            self.room_dim[0] / 2, 
            self.room_dim[1] / 2, 
            self.room_dim[2] / 2
        ])
        
        # If we have trajectory information, use it for a better initial guess
        if prev_point is not None:
            # Start with previous point
            initial_guess = prev_point.copy()
            
            # If we have trajectory direction, predict next position
            if trajectory_direction is not None:
                step_size = 0.1  # Small step forward
                initial_guess = prev_point + trajectory_direction * step_size
            
            # Ensure within room bounds
            initial_guess[0] = np.clip(initial_guess[0], 0, self.room_dim[0])
            initial_guess[1] = np.clip(initial_guess[1], 0, self.room_dim[1])
            initial_guess[2] = np.clip(initial_guess[2], 0, self.room_dim[2])
        
        # Define bounds for optimization (room dimensions)
        bounds = [
            (0, self.room_dim[0]),            # x bounds
            (0, self.room_dim[1]),            # y bounds
            (0, self.room_dim[2] + 0.3)       # z bounds with slight extension
        ]
        
        # Best results so far from any method
        best_point = None
        min_error = float('inf')
        
        # Try multiple optimization methods and starting points for robustness
        
        # 1. Nelder-Mead (Simplex) method - fast and works well for non-linear problems
        try:
            result = optimize.minimize(
                objective_fn, 
                initial_guess, 
                method='Nelder-Mead',
                bounds=bounds, 
                options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3}
            )
            if result.success and objective_fn(result.x) < min_error:
                min_error = objective_fn(result.x)
                best_point = result.x
        except Exception:
            pass
            
        # 2. Powell method - generally reliable
        try:
            result = optimize.minimize(
                objective_fn, 
                initial_guess, 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 100, 'xtol': 1e-3, 'ftol': 1e-3}
            )
            if result.success and objective_fn(result.x) < min_error:
                min_error = objective_fn(result.x)
                best_point = result.x
        except Exception:
            pass
            
        # If either optimization method succeeded with a good error
        optimization_success = best_point is not None and min_error < 0.1
        
        # If optimization failed or we need to visualize error maps, use grid search as fallback
        if not optimization_success or save_error_map:
            # Skip error map creation unless explicitly requested
            if save_error_map:
                try:
                    # Choose a few z-slices to visualize
                    z_slices = [0.5, 1.0, 1.5, 2.0]
                    error_maps = {}
                    
                    # Reduced resolution maps for visualization
                    map_res = max(5, coarse_res // 2)
                    map_x = np.linspace(0, self.room_dim[0], map_res)
                    map_y = np.linspace(0, self.room_dim[1], map_res)
                    
                    for z_val in z_slices:
                        try:
                            # Create error map for this z-slice
                            error_map = np.zeros((len(map_y), len(map_x)))
                            
                            for i, yi in enumerate(map_y):
                                for j, xi in enumerate(map_x):
                                    point = np.array([xi, yi, z_val])
                                    error_map[i, j] = objective_fn(point)
                            
                            error_maps[z_val] = error_map
                        except Exception:
                            pass
                    
                    # Save the error maps
                    self._save_error_maps(error_maps, ellipses, "coarse")
                except Exception:
                    pass
            
            # Fallback grid search if optimization failed
            if not optimization_success:
                # Create a grid of points around the initial guess
                search_scale = 0.6  # Focused search area
                x_range = self.room_dim[0] * search_scale
                y_range = self.room_dim[1] * search_scale
                z_range = self.room_dim[2] * search_scale
                
                # Use initial guess as search center
                x_center = initial_guess[0]
                y_center = initial_guess[1]
                z_center = initial_guess[2]
                
                # Generate search grid
                x = np.linspace(max(0, x_center - x_range/2), 
                              min(self.room_dim[0], x_center + x_range/2), coarse_res)
                y = np.linspace(max(0, y_center - y_range/2),
                              min(self.room_dim[1], y_center + y_range/2), coarse_res)
                z = np.linspace(max(0, z_center - z_range/2),
                              min(self.room_dim[2] + 0.3, z_center + z_range/2), 
                              int(coarse_res * 1.5))  # Higher z-resolution
                
                # Create grid of points
                X, Y, Z = np.meshgrid(x, y, z)
                points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                
                # Use mini-batches for efficient processing
                batch_size = 500
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    for point in batch:
                        error = objective_fn(point)
                        
                        if error < min_error:
                            min_error = error
                            best_point = point.copy()
        
        # If we still don't have a solution, return None
        if best_point is None:
            return None
        
        # Final refinement using L-BFGS-B with best point as starting point
        # This is a more sophisticated algorithm that works well for fine-tuning
        try:
            result = optimize.minimize(
                objective_fn, 
                best_point,
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-5}
            )
            if result.success and objective_fn(result.x) < min_error:
                best_point = result.x
        except Exception:
            pass
        
        # Enhancement 3: Apply trajectory consistency check
        if trajectory_points and len(trajectory_points) >= 2:
            # Calculate direction from last points
            last_direction = trajectory_points[-1] - trajectory_points[-2]
            last_speed = np.linalg.norm(last_direction)
            
            if last_speed > 0:
                # Normalize to get direction vector
                last_direction = last_direction / last_speed
                
                # Calculate direction to new point
                new_direction = best_point - trajectory_points[-1]
                new_speed = np.linalg.norm(new_direction)
                
                if new_speed > 0:
                    new_direction = new_direction / new_speed
                    
                    # Calculate angle between directions
                    angle = np.arccos(np.clip(np.dot(last_direction, new_direction), -1.0, 1.0))
                    
                    # If angle is too large (sudden direction change) and speed is reasonable
                    if angle > np.pi/2 and new_speed > 0.5:
                        # This is likely an ambiguity error - adjust the point
                        adjusted_point = trajectory_points[-1] + last_direction * new_speed * 0.5
                        
                        # Verify that adjusted point still matches ellipses reasonably well
                        adjusted_error = self._compute_error(adjusted_point, ellipses, prev_point)
                        
                        # If error is acceptable, use the adjusted point
                        if adjusted_error < min_error * 1.5:  # Allow some increase in error
                            best_point = adjusted_point
        
        return best_point
        
    def _save_error_maps(self, error_maps, ellipses, stage, best_point=None):
        """Save error maps to visualize the error landscape
        
        Args:
            error_maps (dict): Dictionary mapping z-heights to error maps
            ellipses (list): List of ellipses used
            stage (str): 'coarse' or 'fine'
            best_point (np.array, optional): Best point found so far
        """
        try:
            import os
            output_dir = "amt_debug_images/error_maps"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a timestamp for this batch
            import time
            timestamp = int(time.time())
            
            # For each z-slice
            for z, error_map in error_maps.items():
                try:
                    # Create a figure
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot the error map using a logarithmic scale for better visualization
                    error_map_positive = error_map.copy()
                    error_map_positive[error_map_positive <= 0] = 1e-10  # Replace zeros with small value
                    
                    vmin = np.min(error_map_positive)  # Avoid log(0)
                    vmax = np.max(error_map_positive)
                    
                    # Use regular colormap without LogNorm if there are issues
                    try:
                        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
                        im = ax.imshow(error_map_positive, cmap='viridis', norm=norm, 
                                     extent=[0, self.room_dim[0], 0, self.room_dim[1]], 
                                     origin='lower', interpolation='bilinear')
                    except Exception as e:
                        print(f"Warning: Error with LogNorm: {str(e)}. Using regular colormap.")
                        im = ax.imshow(error_map_positive, cmap='viridis',
                                     extent=[0, self.room_dim[0], 0, self.room_dim[1]], 
                                     origin='lower', interpolation='bilinear')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Error')
                    
                    # Plot the best point if provided
                    if best_point is not None:
                        ax.plot(best_point[0], best_point[1], 'rx', markersize=12, label='Best point')
                    
                    # Plot the ellipse foci (speakers and microphones)
                    for ellipse in ellipses:
                        speaker_pos = ellipse['speaker_pos']
                        mic_pos = ellipse['mic_pos']
                        
                        # Only plot if they're close to this z-slice
                        if abs(speaker_pos[2] - z) < 0.5:
                            ax.plot(speaker_pos[0], speaker_pos[1], 'ro', markersize=8, label='Speaker')
                        
                        if abs(mic_pos[2] - z) < 0.5:
                            ax.plot(mic_pos[0], mic_pos[1], 'bo', markersize=8, label='Microphone')
                        
                        try:
                            # Draw the ellipse (simplified 2D projection)
                            path_length = ellipse['path_length']
                            foci_distance = np.linalg.norm(speaker_pos[:2] - mic_pos[:2])  # 2D distance
                            
                            # Only draw if the ellipse is physically possible
                            if path_length > foci_distance:
                                a = path_length / 2  # Semi-major axis
                                c = foci_distance / 2  # Half distance between foci
                                b = np.sqrt(a**2 - c**2)  # Semi-minor axis
                                
                                # Center of ellipse
                                center = (speaker_pos[:2] + mic_pos[:2]) / 2
                                
                                # Angle of ellipse
                                angle = np.arctan2(mic_pos[1] - speaker_pos[1], mic_pos[0] - speaker_pos[0])
                                angle_deg = np.degrees(angle)
                                
                                # Create ellipse
                                ellipse_patch = Ellipse(xy=center, width=2*a, height=2*b, angle=angle_deg, 
                                                      fill=False, edgecolor='gray', linestyle='--', alpha=0.7)
                                ax.add_patch(ellipse_patch)
                        except Exception as e:
                            print(f"Warning: Error drawing ellipse: {str(e)}")
                    
                    # Add labels and title
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_title(f'Error Map at Z={z:.2f}m - {stage.capitalize()} Search')
                    
                    # Add room boundary
                    ax.set_xlim(0, self.room_dim[0])
                    ax.set_ylim(0, self.room_dim[1])
                    ax.grid(True, alpha=0.3)
                    
                    # Handle duplicate labels in legend
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='best')
                    
                    # Save the figure
                    filename = f"{output_dir}/errormap_{stage}_{timestamp}_z{z:.2f}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"Saved error map to {filename}")
                except Exception as e:
                    print(f"Warning: Error processing error map at z={z}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error saving error maps: {str(e)}")
    
    def run_tracking(self, duration=5.0, steps=50):
        """Run the complete tracking simulation with performance optimizations
        
        Args:
            duration (float): Total simulation duration in seconds
            steps (int): Number of tracking steps
            
        Returns:
            dict: Tracking data
        """
        dt = duration / steps
        
        # Clear previous correlation history
        self.correlation_history = []
        
        # Store ellipses for debugging
        self.ellipses_history = []
        
        # Create output directory for saving images
        import os
        output_dir = "amt_debug_images/ellipses"
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance tracking
        import time
        total_start_time = time.time()
        step_times = []
        
        # Clear any cached computations
        if hasattr(self, '_location_cache'):
            self._location_cache = {}
        
        # Pre-allocate memory for history
        self.correlation_history = [None] * steps
        self.ellipses_history = [None] * steps
        
        for step in range(steps):
            step_start = time.time()
            print(f"Processing step {step+1}/{steps}...")
            
            # Update target positions
            self.update_targets(dt)
            
            # Simulate propagation and echoes
            received_signals = self.simulate_echoes()
            
            # Detect echoes
            echo_data = self.detect_echoes(received_signals)
            
            # Store echo data for visualization
            self.correlation_history[step] = echo_data
            
            # Create ranging ellipses for tracking
            ellipses = []
            for (s_idx, m_idx), data in echo_data.items():
                speaker_pos = self.speakers[s_idx]
                mic_pos = self.mics[m_idx]
                
                # For each peak (potentially a target echo)
                for i, (peak, height, time_diff) in enumerate(zip(data['peaks'], data['heights'], data['time_diffs'])):
                    if i >= len(self.targets):
                        break  # We only have this many targets
                    
                    # Skip negative time differences
                    if time_diff <= 0:
                        continue
                    
                    # Convert time difference to distance
                    additional_distance = time_diff * self.c / self.fs
                    direct_distance = np.linalg.norm(mic_pos - speaker_pos)
                    total_path_length = direct_distance + additional_distance
                    
                    # Create ellipse
                    ellipse = {
                        'speaker_idx': s_idx,
                        'mic_idx': m_idx,
                        'speaker_pos': speaker_pos,
                        'mic_pos': mic_pos,
                        'path_length': total_path_length,
                        'height': height
                    }
                    ellipses.append(ellipse)
            
            # Store ellipses for this step
            self.ellipses_history[step] = ellipses
            
            # Visualize ellipses for debugging (only periodically to save time)
            if step % 3 == 0 or step == steps - 1:
                self._visualize_ellipses(ellipses, step, output_dir)
            
            # Locate targets with Kalman filtering (pass step number for debugging)
            estimated_positions, filtered_positions = self.locate_targets(echo_data, dt, step=step)
            
            # Performance tracking
            step_end = time.time()
            step_time = step_end - step_start
            step_times.append(step_time)
            print(f"  Step {step+1} completed in {step_time:.3f} seconds")
        
        # Create animation of ellipses over time (only if needed)
        if steps > 5:
            self._create_ellipses_animation(output_dir)
        
        # Report performance statistics
        total_time = time.time() - total_start_time
        avg_step_time = sum(step_times) / len(step_times)
        print(f"\nTracking Performance Summary:")
        print(f"  Total tracking time: {total_time:.2f} seconds")
        print(f"  Average step time: {avg_step_time:.3f} seconds")
        print(f"  Steps per second: {1.0/avg_step_time:.2f}")
        
        return self.tracking_data
        
    def _visualize_ellipses(self, ellipses, step, output_dir):
        """Visualize the ellipses used for multilateration
        
        Args:
            ellipses (list): List of ellipse dictionaries
            step (int): Current step number
            output_dir (str): Directory to save images
        """
        try:
            # Create a figure with two subplots: top-down view and front view
            fig, (ax_top, ax_front) = plt.subplots(1, 2, figsize=(18, 8))
            
            # TOP-DOWN VIEW (X-Y plane)
            # Set limits
            ax_top.set_xlim(0, self.room_dim[0])
            ax_top.set_ylim(0, self.room_dim[1])
            
            # Draw room boundaries
            ax_top.plot([0, self.room_dim[0], self.room_dim[0], 0, 0], 
                       [0, 0, self.room_dim[1], self.room_dim[1], 0], 'k-', alpha=0.5)
            
            # Draw speakers
            for i, pos in enumerate(self.speakers):
                ax_top.plot(pos[0], pos[1], 'ro', markersize=8, label=f'Speaker {i+1}' if i == 0 else "")
            
            # Draw microphones
            for i, pos in enumerate(self.mics):
                ax_top.plot(pos[0], pos[1], 'bo', markersize=8, label=f'Mic {i+1}' if i == 0 else "")
            
            # Draw targets (ground truth)
            for i, target in enumerate(self.targets):
                pos = target['history'][step]
                ax_top.plot(pos[0], pos[1], 'gs', markersize=10, label=f'{target["name"]} (True)' if i == 0 else "")
            
            # Draw ellipses in top-down view
            for i, ellipse in enumerate(ellipses):
                speaker_pos = ellipse['speaker_pos']
                mic_pos = ellipse['mic_pos']
                path_length = ellipse['path_length']
                
                # Calculate ellipse properties (2D projection in X-Y plane)
                foci_distance = np.linalg.norm(speaker_pos[:2] - mic_pos[:2])
                
                # Only draw if the ellipse is physically possible
                if path_length > foci_distance:
                    try:
                        a = path_length / 2  # Semi-major axis
                        c = foci_distance / 2  # Half distance between foci
                        b = np.sqrt(a**2 - c**2)  # Semi-minor axis
                        
                        # Center of ellipse
                        center = (speaker_pos[:2] + mic_pos[:2]) / 2
                        
                        # Angle of ellipse
                        angle = np.arctan2(mic_pos[1] - speaker_pos[1], mic_pos[0] - speaker_pos[0])
                        angle_deg = np.degrees(angle)
                        
                        # Create ellipse
                        ellipse_patch = Ellipse(xy=center, width=2*a, height=2*b, angle=angle_deg, 
                                              fill=False, edgecolor=f'C{i%10}', linestyle='-', alpha=0.7,
                                              label=f'Ellipse S{ellipse["speaker_idx"]}→M{ellipse["mic_idx"]}' if i == 0 else "")
                        ax_top.add_patch(ellipse_patch)
                        
                        # Draw a line connecting the foci
                        ax_top.plot([speaker_pos[0], mic_pos[0]], [speaker_pos[1], mic_pos[1]], 
                                   color=f'C{i%10}', linestyle=':', alpha=0.5)
                    except Exception as e:
                        print(f"Error drawing top-view ellipse: {e}")
            
            # Grid and labels for top-down view
            ax_top.grid(True, alpha=0.3)
            ax_top.set_xlabel('X (m)')
            ax_top.set_ylabel('Y (m)')
            ax_top.set_title(f'Step {step}: Multilateration Ellipses (Top-down view)')
            
            # Add legend to top-down view
            handles, labels = ax_top.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_top.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # FRONT VIEW (X-Z plane)
            # Set limits
            ax_front.set_xlim(0, self.room_dim[0])
            ax_front.set_ylim(0, self.room_dim[2])
            
            # Draw room boundaries
            ax_front.plot([0, self.room_dim[0], self.room_dim[0], 0, 0], 
                         [0, 0, self.room_dim[2], self.room_dim[2], 0], 'k-', alpha=0.5)
            
            # Draw speakers in front view (X-Z plane)
            for i, pos in enumerate(self.speakers):
                ax_front.plot(pos[0], pos[2], 'ro', markersize=8, label=f'Speaker {i+1}' if i == 0 else "")
            
            # Draw microphones in front view
            for i, pos in enumerate(self.mics):
                ax_front.plot(pos[0], pos[2], 'bo', markersize=8, label=f'Mic {i+1}' if i == 0 else "")
            
            # Draw targets (ground truth) in front view
            for i, target in enumerate(self.targets):
                pos = target['history'][step]
                ax_front.plot(pos[0], pos[2], 'gs', markersize=10, label=f'{target["name"]} (True)' if i == 0 else "")
            
            # Draw ellipses in front view (X-Z plane)
            for i, ellipse in enumerate(ellipses):
                speaker_pos = ellipse['speaker_pos']
                mic_pos = ellipse['mic_pos']
                path_length = ellipse['path_length']
                
                # Calculate ellipse properties (2D projection in X-Z plane)
                # For X-Z plane projection, use first and third component (x, z)
                foci_distance_xz = np.sqrt((speaker_pos[0] - mic_pos[0])**2 + (speaker_pos[2] - mic_pos[2])**2)
                
                # Only draw if the ellipse is physically possible
                if path_length > foci_distance_xz:
                    try:
                        a = path_length / 2  # Semi-major axis
                        c = foci_distance_xz / 2  # Half distance between foci
                        b = np.sqrt(a**2 - c**2)  # Semi-minor axis
                        
                        # Center of ellipse (X-Z plane)
                        center_xz = ((speaker_pos[0] + mic_pos[0])/2, (speaker_pos[2] + mic_pos[2])/2)
                        
                        # Angle of ellipse in X-Z plane
                        angle_xz = np.arctan2(mic_pos[2] - speaker_pos[2], mic_pos[0] - speaker_pos[0])
                        angle_deg_xz = np.degrees(angle_xz)
                        
                        # Create ellipse
                        ellipse_patch_xz = Ellipse(xy=center_xz, width=2*a, height=2*b, angle=angle_deg_xz, 
                                                 fill=False, edgecolor=f'C{i%10}', linestyle='-', alpha=0.7)
                        ax_front.add_patch(ellipse_patch_xz)
                        
                        # Draw a line connecting the foci
                        ax_front.plot([speaker_pos[0], mic_pos[0]], [speaker_pos[2], mic_pos[2]], 
                                      color=f'C{i%10}', linestyle=':', alpha=0.5)
                    except Exception as e:
                        print(f"Error drawing front-view ellipse: {e}")
            
            # Grid and labels for front view
            ax_front.grid(True, alpha=0.3)
            ax_front.set_xlabel('X (m)')
            ax_front.set_ylabel('Z (m)')
            ax_front.set_title(f'Step {step}: Multilateration Ellipses (Front view)')
            
            # Add suptitle to the figure
            fig.suptitle(f'Step {step}: Multilateration Ellipses', fontsize=16)
            
            # Save figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            filename = f"{output_dir}/ellipses_step{step:03d}.png"
            fig.savefig(filename, dpi=300)
            plt.close(fig)
            
            # Create and save a side view (Y-Z plane) as well
            fig_side, ax_side = plt.subplots(figsize=(10, 8))
            
            # Set limits
            ax_side.set_xlim(0, self.room_dim[1])
            ax_side.set_ylim(0, self.room_dim[2])
            
            # Draw room boundaries
            ax_side.plot([0, self.room_dim[1], self.room_dim[1], 0, 0], 
                         [0, 0, self.room_dim[2], self.room_dim[2], 0], 'k-', alpha=0.5)
            
            # Draw speakers in side view (Y-Z plane)
            for i, pos in enumerate(self.speakers):
                ax_side.plot(pos[1], pos[2], 'ro', markersize=8, label=f'Speaker {i+1}' if i == 0 else "")
            
            # Draw microphones in side view
            for i, pos in enumerate(self.mics):
                ax_side.plot(pos[1], pos[2], 'bo', markersize=8, label=f'Mic {i+1}' if i == 0 else "")
            
            # Draw targets (ground truth) in side view
            for i, target in enumerate(self.targets):
                pos = target['history'][step]
                ax_side.plot(pos[1], pos[2], 'gs', markersize=10, label=f'{target["name"]} (True)' if i == 0 else "")
            
            # Draw ellipses in side view (Y-Z plane)
            for i, ellipse in enumerate(ellipses):
                speaker_pos = ellipse['speaker_pos']
                mic_pos = ellipse['mic_pos']
                path_length = ellipse['path_length']
                
                # Calculate ellipse properties (2D projection in Y-Z plane)
                # For Y-Z plane projection, use second and third component (y, z)
                foci_distance_yz = np.sqrt((speaker_pos[1] - mic_pos[1])**2 + (speaker_pos[2] - mic_pos[2])**2)
                
                # Only draw if the ellipse is physically possible
                if path_length > foci_distance_yz:
                    try:
                        a = path_length / 2  # Semi-major axis
                        c = foci_distance_yz / 2  # Half distance between foci
                        b = np.sqrt(a**2 - c**2)  # Semi-minor axis
                        
                        # Center of ellipse (Y-Z plane)
                        center_yz = ((speaker_pos[1] + mic_pos[1])/2, (speaker_pos[2] + mic_pos[2])/2)
                        
                        # Angle of ellipse in Y-Z plane
                        angle_yz = np.arctan2(mic_pos[2] - speaker_pos[2], mic_pos[1] - speaker_pos[1])
                        angle_deg_yz = np.degrees(angle_yz)
                        
                        # Create ellipse
                        ellipse_patch_yz = Ellipse(xy=center_yz, width=2*a, height=2*b, angle=angle_deg_yz, 
                                                 fill=False, edgecolor=f'C{i%10}', linestyle='-', alpha=0.7)
                        ax_side.add_patch(ellipse_patch_yz)
                        
                        # Draw a line connecting the foci
                        ax_side.plot([speaker_pos[1], mic_pos[1]], [speaker_pos[2], mic_pos[2]], 
                                      color=f'C{i%10}', linestyle=':', alpha=0.5)
                    except Exception as e:
                        print(f"Error drawing side-view ellipse: {e}")
            
            # Grid and labels for side view
            ax_side.grid(True, alpha=0.3)
            ax_side.set_xlabel('Y (m)')
            ax_side.set_ylabel('Z (m)')
            ax_side.set_title(f'Step {step}: Multilateration Ellipses (Side view)')
            
            # Add legend to side view
            handles, labels = ax_side.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_side.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Save side view figure
            plt.tight_layout()
            filename = f"{output_dir}/ellipses_side_step{step:03d}.png"
            fig_side.savefig(filename, dpi=300)
            plt.close(fig_side)
            
        except Exception as e:
            print(f"Error in _visualize_ellipses: {e}")
            # Fallback to simple top-down view if the multi-view approach fails
            try:
                # Create a figure for 2D view (top-down)
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Set limits
                ax.set_xlim(0, self.room_dim[0])
                ax.set_ylim(0, self.room_dim[1])
                
                # Draw room boundaries
                ax.plot([0, self.room_dim[0], self.room_dim[0], 0, 0], 
                      [0, 0, self.room_dim[1], self.room_dim[1], 0], 'k-', alpha=0.5)
                
                # Draw speakers
                for i, pos in enumerate(self.speakers):
                    ax.plot(pos[0], pos[1], 'ro', markersize=8, label=f'Speaker {i+1}' if i == 0 else "")
                
                # Draw microphones
                for i, pos in enumerate(self.mics):
                    ax.plot(pos[0], pos[1], 'bo', markersize=8, label=f'Mic {i+1}' if i == 0 else "")
                
                # Draw targets (ground truth)
                for i, target in enumerate(self.targets):
                    pos = target['history'][step]
                    ax.plot(pos[0], pos[1], 'gs', markersize=10, label=f'{target["name"]} (True)' if i == 0 else "")
                
                # Draw ellipses
                for i, ellipse in enumerate(ellipses):
                    speaker_pos = ellipse['speaker_pos']
                    mic_pos = ellipse['mic_pos']
                    path_length = ellipse['path_length']
                    
                    # Calculate ellipse properties (2D projection)
                    foci_distance = np.linalg.norm(speaker_pos[:2] - mic_pos[:2])
                    
                    # Only draw if the ellipse is physically possible
                    if path_length > foci_distance:
                        a = path_length / 2  # Semi-major axis
                        c = foci_distance / 2  # Half distance between foci
                        b = np.sqrt(a**2 - c**2)  # Semi-minor axis
                        
                        # Center of ellipse
                        center = (speaker_pos[:2] + mic_pos[:2]) / 2
                        
                        # Angle of ellipse
                        angle = np.arctan2(mic_pos[1] - speaker_pos[1], mic_pos[0] - speaker_pos[0])
                        angle_deg = np.degrees(angle)
                        
                        # Create ellipse
                        ellipse_patch = Ellipse(xy=center, width=2*a, height=2*b, angle=angle_deg, 
                                            fill=False, edgecolor=f'C{i%10}', linestyle='-', alpha=0.7,
                                            label=f'Ellipse S{ellipse["speaker_idx"]}→M{ellipse["mic_idx"]}' if i == 0 else "")
                        ax.add_patch(ellipse_patch)
                        
                        # Draw a line connecting the foci
                        ax.plot([speaker_pos[0], mic_pos[0]], [speaker_pos[1], mic_pos[1]], 
                              color=f'C{i%10}', linestyle=':', alpha=0.5)
                
                # Legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                
                # Grid and labels
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'Step {step}: Multilateration Ellipses (Top-down view)')
                
                # Save figure
                plt.tight_layout()
                filename = f"{output_dir}/ellipses_step{step:03d}.png"
                fig.savefig(filename, dpi=300)
                plt.close(fig)
            except Exception as e2:
                print(f"Error in fallback visualization: {e2}")
        
    def _create_ellipses_animation(self, output_dir):
        """Create an animation of ellipses over time
        
        Args:
            output_dir (str): Directory to save animation
        """
        # Skip if we don't have enough history
        if not hasattr(self, 'ellipses_history') or len(self.ellipses_history) < 2:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Function to update animation frame
        def update_frame(step):
            ax.clear()
            
            # Set limits
            ax.set_xlim(0, self.room_dim[0])
            ax.set_ylim(0, self.room_dim[1])
            
            # Draw room boundaries
            ax.plot([0, self.room_dim[0], self.room_dim[0], 0, 0], 
                   [0, 0, self.room_dim[1], self.room_dim[1], 0], 'k-', alpha=0.5)
            
            # Draw speakers
            for i, pos in enumerate(self.speakers):
                ax.plot(pos[0], pos[1], 'ro', markersize=8)
            
            # Draw microphones
            for i, pos in enumerate(self.mics):
                ax.plot(pos[0], pos[1], 'bo', markersize=8)
            
            # Get ellipses for this step
            if step < len(self.ellipses_history):
                ellipses = self.ellipses_history[step]
            else:
                ellipses = []
                
            # Draw targets (ground truth)
            for i, target in enumerate(self.targets):
                if step < len(target['history']):
                    pos = target['history'][step]
                    ax.plot(pos[0], pos[1], 'gs', markersize=10, label=target['name'])
                
                    # Draw trajectory up to this point
                    history = np.array(target['history'][:step+1])
                    ax.plot(history[:, 0], history[:, 1], 'g-', alpha=0.5)
                    
                    # Draw estimated position if available
                    if step < len(target['estimated_positions']) and target['estimated_positions'][step] is not None:
                        est_pos = target['estimated_positions'][step]
                        ax.plot(est_pos[0], est_pos[1], 'rx', markersize=8, label=f'{target["name"]} (Est)')
                        
                    # Draw filtered position
                    if step < len(target['filtered_positions']):
                        filt_pos = target['filtered_positions'][step]
                        ax.plot(filt_pos[0], filt_pos[1], 'yx', markersize=8, label=f'{target["name"]} (Filt)')
            
            # Draw ellipses
            for i, ellipse in enumerate(ellipses):
                speaker_pos = ellipse['speaker_pos']
                mic_pos = ellipse['mic_pos']
                path_length = ellipse['path_length']
                
                # Calculate ellipse properties (2D projection)
                foci_distance = np.linalg.norm(speaker_pos[:2] - mic_pos[:2])
                
                # Only draw if the ellipse is physically possible
                if path_length > foci_distance:
                    a = path_length / 2  # Semi-major axis
                    c = foci_distance / 2  # Half distance between foci
                    b = np.sqrt(a**2 - c**2)  # Semi-minor axis
                    
                    # Center of ellipse
                    center = (speaker_pos[:2] + mic_pos[:2]) / 2
                    
                    # Angle of ellipse
                    angle = np.arctan2(mic_pos[1] - speaker_pos[1], mic_pos[0] - speaker_pos[0])
                    angle_deg = np.degrees(angle)
                    
                    # Create ellipse
                    ellipse_patch = Ellipse(xy=center, width=2*a, height=2*b, angle=angle_deg, 
                                        fill=False, edgecolor=f'C{i%10}', linestyle='-', alpha=0.7)
                    ax.add_patch(ellipse_patch)
                    
                    # Draw a line connecting the foci
                    ax.plot([speaker_pos[0], mic_pos[0]], [speaker_pos[1], mic_pos[1]], 
                           color=f'C{i%10}', linestyle=':', alpha=0.5)
            
            # Grid and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Step {step}: Multilateration Ellipses (Top-down view)')
            
            # Show legend only for the first target
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(self.ellipses_history),
            interval=200, blit=True
        )
        
        # Save animation
        filename = f"{output_dir}/ellipses_animation.mp4"
        try:
            # Try to use FFMpegWriter if available
            writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='AMT+'), bitrate=5000)
            anim.save(filename, writer=writer)
            print(f"Saved ellipses animation to {filename}")
        except Exception as e:
            print(f"Error with FFMpegWriter: {str(e)}")
            try:
                # Fall back to PillowWriter if FFMpeg is not available
                print("Falling back to PillowWriter...")
                gif_filename = f"{output_dir}/ellipses_animation.gif"
                anim.save(gif_filename, writer='pillow', fps=5)
                print(f"Animation saved as GIF to {gif_filename}")
            except Exception as e2:
                print(f"Could not save animation: {str(e2)}")
        
        plt.close(fig)
    
    def visualize(self, use_filtered=True):
        """Visualize the room, speakers, microphones, and targets
        
        Args:
            use_filtered (bool): Whether to show filtered positions (True) or raw estimated positions (False)
            
        Returns:
            fig: Matplotlib figure object for saving
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set limits
        ax.set_xlim(0, self.room_dim[0])
        ax.set_ylim(0, self.room_dim[1])
        ax.set_zlim(0, self.room_dim[2])
        
        # Labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        title = 'High-Resolution Acoustic Multi-Target Tracking'
        if use_filtered:
            title += ' (Kalman Filtered)'
        ax.set_title(title)
        
        # Draw room
        # Just the edges for simplicity
        for x in [0, self.room_dim[0]]:
            for y in [0, self.room_dim[1]]:
                ax.plot([x, x], [y, y], [0, self.room_dim[2]], 'k-', alpha=0.3)
        for x in [0, self.room_dim[0]]:
            for z in [0, self.room_dim[2]]:
                ax.plot([x, x], [0, self.room_dim[1]], [z, z], 'k-', alpha=0.3)
        for y in [0, self.room_dim[1]]:
            for z in [0, self.room_dim[2]]:
                ax.plot([0, self.room_dim[0]], [y, y], [z, z], 'k-', alpha=0.3)
        
        # Draw speakers
        for i, pos in enumerate(self.speakers):
            ax.scatter(pos[0], pos[1], pos[2], color='red', marker='o', s=100, label=f'Speaker {i+1}')
        
        # Draw microphones
        for i, pos in enumerate(self.mics):
            ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='^', s=100, label=f'Mic {i+1}')
        
        # Draw targets and their trajectories
        for target in self.targets:
            # Latest position
            pos = target['position']
            ax.scatter(pos[0], pos[1], pos[2], color='green', marker='s', s=150, label=target['name'])
            
            # Trajectory (ground truth)
            if len(target['history']) > 1:
                history = np.array(target['history'])
                ax.plot(history[:, 0], history[:, 1], history[:, 2], 'g-', alpha=0.5)
            
            # Raw estimated or filtered positions
            if use_filtered:
                positions = [p for p in target['filtered_positions'] if p is not None]
                label = f"{target['name']} (Filtered)"
            else:
                positions = [p for p in target['estimated_positions'] if p is not None]
                label = f"{target['name']} (Est)"
            
            if len(positions) > 0:
                positions = np.array(positions)
                ax.scatter(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    color='orange', marker='x', s=80, label=label
                )
                
                # Draw estimated trajectory
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'orange', linestyle='--', alpha=0.5)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
                 if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_correlation(self, step=0):
        """Visualize the correlation and MTI filtering results for a specific step
        
        Args:
            step (int): Step to visualize
            
        Returns:
            fig: Matplotlib figure object for saving
        """
        if not hasattr(self, 'correlation_history') or not self.correlation_history:
            print("No correlation history available. Run tracking first.")
            return None
        
        # Only visualize if we have data
        if step >= len(self.correlation_history):
            step = len(self.correlation_history) - 1
            
        echo_data = self.correlation_history[step]
        
        # Create subplots for each speaker-mic pair
        n_pairs = len(echo_data)
        fig, axs = plt.subplots(n_pairs, 2, figsize=(14, 3*n_pairs))
        
        # If only one pair, make sure axs is 2D
        if n_pairs == 1:
            axs = np.array([axs])
            
        # Plot each pair
        for i, ((s_idx, m_idx), data) in enumerate(echo_data.items()):
            # Plot correlation
            axs[i, 0].plot(data['correlation'])
            axs[i, 0].set_title(f"Correlation: Speaker {s_idx+1} -> Mic {m_idx+1}")
            
            # Mark direct path
            direct_delay = self.direct_delays[(s_idx, m_idx)]
            axs[i, 0].axvline(x=direct_delay, color='r', linestyle='--', label='Direct Path')
            
            # Mark detected peaks
            for peak in data['peaks']:
                axs[i, 0].plot(peak, data['correlation'][peak], 'rx')
            
            # Plot difference (MTI filtered)
            axs[i, 1].plot(data['diff'])
            axs[i, 1].set_title(f"MTI Filtered: Speaker {s_idx+1} -> Mic {m_idx+1}")
            
            # Mark detected peaks
            for peak in data['peaks']:
                axs[i, 1].plot(peak, data['diff'][peak], 'rx')
                
            # Add distance axis (convert samples to meters)
            ax2 = axs[i, 0].twiny()
            max_samples = len(data['correlation'])
            max_dist = max_samples * self.c / self.fs
            ax2.set_xlim(0, max_dist)
            ax2.set_xlabel('Distance (m)')
            
            ax2 = axs[i, 1].twiny()
            max_samples = len(data['diff'])
            max_dist = max_samples * self.c / self.fs
            ax2.set_xlim(0, max_dist)
            ax2.set_xlabel('Distance (m)')
            
            # Mark the expected positions of each target
            for t_idx, target in enumerate(self.targets):
                if step < len(target['history']):
                    pos = target['history'][step]
                    
                    # Calculate the expected echo delay
                    speaker_pos = self.speakers[s_idx]
                    mic_pos = self.mics[m_idx]
                    
                    to_target = np.linalg.norm(pos - speaker_pos)
                    from_target = np.linalg.norm(mic_pos - pos)
                    total_dist = to_target + from_target
                    delay_samples = int(total_dist / self.c * self.fs)
                    
                    # Add a vertical line at the expected position for both plots
                    color = f'C{t_idx}'
                    axs[i, 0].axvline(x=delay_samples, color=color, linestyle=':', 
                                     label=f'Expected {target["name"]}')
                    axs[i, 1].axvline(x=delay_samples, color=color, linestyle=':', 
                                     label=f'Expected {target["name"]}')
        
        # Add legends to first row only to avoid clutter
        if n_pairs > 0:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            axs[0, 0].legend(*zip(*unique), loc='upper right')
            
            handles, labels = axs[0, 1].get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            axs[0, 1].legend(*zip(*unique), loc='upper right')
        
        fig.suptitle(f"Step {step}: Correlation and MTI Results")
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
        
        return fig
    
    def compare_tracking(self):
        """Compare raw estimated vs Kalman filtered tracking
        
        Returns:
            fig: Matplotlib figure object for saving
        """
        # Create a figure with two subplots
        fig = plt.figure(figsize=(14, 6))
        
        # Raw estimated positions
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Raw Estimated Positions')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_xlim(0, self.room_dim[0])
        ax1.set_ylim(0, self.room_dim[1])
        ax1.set_zlim(0, self.room_dim[2])
        
        # Kalman filtered positions
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('Kalman Filtered Positions')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_xlim(0, self.room_dim[0])
        ax2.set_ylim(0, self.room_dim[1])
        ax2.set_zlim(0, self.room_dim[2])
        
        # Draw room in both subplots
        for ax in [ax1, ax2]:
            for x in [0, self.room_dim[0]]:
                for y in [0, self.room_dim[1]]:
                    ax.plot([x, x], [y, y], [0, self.room_dim[2]], 'k-', alpha=0.3)
            for x in [0, self.room_dim[0]]:
                for z in [0, self.room_dim[2]]:
                    ax.plot([x, x], [0, self.room_dim[1]], [z, z], 'k-', alpha=0.3)
            for y in [0, self.room_dim[1]]:
                for z in [0, self.room_dim[2]]:
                    ax.plot([0, self.room_dim[0]], [y, y], [z, z], 'k-', alpha=0.3)
                    
            # Draw speakers and microphones
            for i, pos in enumerate(self.speakers):
                ax.scatter(pos[0], pos[1], pos[2], color='red', marker='o', s=50)
            for i, pos in enumerate(self.mics):
                ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='^', s=50)
        
        # Draw targets and their trajectories in both subplots
        colors = ['g', 'purple', 'orange', 'cyan', 'magenta']
        
        for t_idx, target in enumerate(self.targets):
            color = colors[t_idx % len(colors)]
            
            # Ground truth trajectory
            history = np.array(target['history'])
            ax1.plot(history[:, 0], history[:, 1], history[:, 2], color=color, linestyle='-', label=target['name'])
            ax2.plot(history[:, 0], history[:, 1], history[:, 2], color=color, linestyle='-', label=target['name'])
            
            # Raw estimated positions in first subplot
            est_positions = [p for p in target['estimated_positions'] if p is not None]
            if est_positions:
                est_positions = np.array(est_positions)
                ax1.scatter(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], color=color, marker='x')
                ax1.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], color=color, linestyle='--', alpha=0.5)
            
            # Filtered positions in second subplot
            filt_positions = np.array(target['filtered_positions'])
            ax2.scatter(filt_positions[:, 0], filt_positions[:, 1], filt_positions[:, 2], color=color, marker='x')
            ax2.plot(filt_positions[:, 0], filt_positions[:, 1], filt_positions[:, 2], color=color, linestyle='--', alpha=0.5)
        
        # Add legend
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print error metrics
        print("\nTracking Error Metrics:")
        print("=====================")
        
        error_data = {}
        for target in self.targets:
            # True positions
            true_pos = np.array(target['history'])
            
            # Raw estimated positions
            est_pos = []
            for i, pos in enumerate(target['estimated_positions']):
                if pos is not None:
                    est_pos.append((i, pos))
            
            # Filtered positions
            filt_pos = np.array(target['filtered_positions'])
            
            # Calculate errors
            est_errors = []
            est_errors_by_axis = {'x': [], 'y': [], 'z': []}
            for idx, pos in est_pos:
                est_errors.append(np.linalg.norm(pos - true_pos[idx]))
                
                # Calculate error by axis (to detect constant shift)
                error_x = pos[0] - true_pos[idx][0]
                error_y = pos[1] - true_pos[idx][1]
                error_z = pos[2] - true_pos[idx][2]
                est_errors_by_axis['x'].append(error_x)
                est_errors_by_axis['y'].append(error_y)
                est_errors_by_axis['z'].append(error_z)
            
            filt_errors = []
            filt_errors_by_axis = {'x': [], 'y': [], 'z': []}
            for i in range(len(filt_pos)):
                filt_errors.append(np.linalg.norm(filt_pos[i] - true_pos[i]))
                
                # Calculate error by axis (to detect constant shift)
                error_x = filt_pos[i][0] - true_pos[i][0]
                error_y = filt_pos[i][1] - true_pos[i][1]
                error_z = filt_pos[i][2] - true_pos[i][2]
                filt_errors_by_axis['x'].append(error_x)
                filt_errors_by_axis['y'].append(error_y)
                filt_errors_by_axis['z'].append(error_z)
            
            # Save error data
            error_data[target['name']] = {
                'raw': {
                    'total': est_errors,
                    'by_axis': est_errors_by_axis
                },
                'filtered': {
                    'total': filt_errors,
                    'by_axis': filt_errors_by_axis
                }
            }
            
            # Print metrics
            print(f"\n{target['name']}:")
            if est_errors:
                print(f"  Raw estimation - Mean error: {np.mean(est_errors):.3f} m, Max error: {np.max(est_errors):.3f} m")
                print(f"  Raw estimation - Available measurements: {len(est_errors)}/{len(true_pos)} ({len(est_errors)/len(true_pos)*100:.1f}%)")
                
                # Print axis-specific errors to help diagnose constant shift
                print(f"  Raw estimation - Axis errors - X: {np.mean(est_errors_by_axis['x']):.3f} m, Y: {np.mean(est_errors_by_axis['y']):.3f} m, Z: {np.mean(est_errors_by_axis['z']):.3f} m")
            else:
                print("  Raw estimation - No valid measurements")
                
            print(f"  Kalman filter  - Mean error: {np.mean(filt_errors):.3f} m, Max error: {np.max(filt_errors):.3f} m")
            print(f"  Kalman filter  - Axis errors - X: {np.mean(filt_errors_by_axis['x']):.3f} m, Y: {np.mean(filt_errors_by_axis['y']):.3f} m, Z: {np.mean(filt_errors_by_axis['z']):.3f} m")
            print(f"  Kalman filter  - Improvement: {np.mean(est_errors)/np.mean(filt_errors) if est_errors else 'N/A':.2f}x")
        
        # Create a new figure to show error over time
        fig_error = plt.figure(figsize=(14, 10))
        
        # Create subplot for each target
        for t_idx, target_name in enumerate(error_data.keys()):
            data = error_data[target_name]
            
            # Subplot for total error
            ax_total = fig_error.add_subplot(len(error_data), 2, t_idx*2+1)
            ax_total.set_title(f"{target_name} - Total Error")
            ax_total.set_xlabel("Step")
            ax_total.set_ylabel("Error (m)")
            
            # Plot total errors
            if data['raw']['total']:
                raw_indices = [i for i, _ in enumerate(data['raw']['total'])]
                ax_total.plot(raw_indices, data['raw']['total'], 'r-', label="Raw Estimation")
            
            filtered_indices = np.arange(len(data['filtered']['total']))
            ax_total.plot(filtered_indices, data['filtered']['total'], 'b-', label="Filtered")
            
            ax_total.legend()
            ax_total.grid(True, alpha=0.3)
            
            # Subplot for per-axis error
            ax_axis = fig_error.add_subplot(len(error_data), 2, t_idx*2+2)
            ax_axis.set_title(f"{target_name} - Error by Axis (Filtered)")
            ax_axis.set_xlabel("Step")
            ax_axis.set_ylabel("Error (m)")
            
            # Plot per-axis errors for filtered data
            ax_axis.plot(filtered_indices, data['filtered']['by_axis']['x'], 'r-', label="X-axis")
            ax_axis.plot(filtered_indices, data['filtered']['by_axis']['y'], 'g-', label="Y-axis")
            ax_axis.plot(filtered_indices, data['filtered']['by_axis']['z'], 'b-', label="Z-axis")
            
            # Add a horizontal line at zero for reference
            ax_axis.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            ax_axis.legend()
            ax_axis.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save the error plot as a separate file
        import os
        output_dir = "amt_debug_images"
        os.makedirs(output_dir, exist_ok=True)
        fig_error.savefig(f"{output_dir}/tracking_errors.png", dpi=300)
        
        return fig


    
    def visualize_correlation_animation(self, interval=200, save_path=None):
        """Create an animation of the correlation and MTI filtering results
        
        Args:
            interval (int): Animation interval in milliseconds
            save_path (str): Optional path to save the animation (e.g., 'correlation_animation.mp4')
                
        Returns:
            animation.FuncAnimation: Animation object
        """
        if not hasattr(self, 'correlation_history') or not self.correlation_history:
            print("No correlation history available. Run tracking first.")
            return None
            
        # Create figure and subplots
        n_pairs = len(self.correlation_history[0])
        fig, axs = plt.subplots(n_pairs, 2, figsize=(16, 3.5*n_pairs))
        
        # If only one pair, make sure axs is 2D
        if n_pairs == 1:
            axs = np.array([axs])
            
        # Initialize plots
        lines = []
        peak_plots = []
        target_markers = []
        
        # Get color map for different targets
        cmap = plt.cm.get_cmap('tab10', len(self.targets))
    
        for i, (s_idx, m_idx) in enumerate(self.correlation_history[0].keys()):
            # Correlation plot
            line, = axs[i, 0].plot([], [], 'b-', linewidth=1.5)
            lines.append(line)
            peak_plot = axs[i, 0].plot([], [], 'rx', markersize=8)[0]
            peak_plots.append(peak_plot)
            
            # Add markers for each target's predicted echo position - one color per target
            for t_idx in range(len(self.targets)):
                marker, = axs[i, 0].plot([], [], 'o', color=cmap(t_idx), markersize=10, alpha=0.6, 
                                      label=f"Target {t_idx+1}" if i == 0 else None)
                target_markers.append(marker)
            
            # Direct path line
            direct_delay = self.direct_delays[(s_idx, m_idx)]
            axs[i, 0].axvline(x=direct_delay, color='r', linestyle='--', label='Direct Path' if i == 0 else None)
            
            # MTI difference plot
            line, = axs[i, 1].plot([], [], 'g-', linewidth=1.5)
            lines.append(line)
            peak_plot = axs[i, 1].plot([], [], 'rx', markersize=8)[0]
            peak_plots.append(peak_plot)
            
            # Add markers for each target's predicted echo position
            for t_idx in range(len(self.targets)):
                marker, = axs[i, 1].plot([], [], 'o', color=cmap(t_idx), markersize=10, alpha=0.6)
                target_markers.append(marker)
            
            # Labels and styling
            axs[i, 0].set_title(f"Correlation: Speaker {s_idx+1} → Mic {m_idx+1}")
            axs[i, 1].set_title(f"MTI Filtered: Speaker {s_idx+1} → Mic {m_idx+1}")
            
            # Set y limits
            axs[i, 0].set_ylim(0, 1.1)
            axs[i, 1].set_ylim(0, 1.1)
        
            # Add distance axis (convert samples to meters)
            ax2 = axs[i, 0].twiny()
            max_samples = len(self.correlation_history[0][(s_idx, m_idx)]['correlation'])
            max_dist = max_samples * self.c / self.fs
            ax2.set_xlim(0, max_dist)
            ax2.set_xlabel('Distance (m)')
            
            ax2 = axs[i, 1].twiny()
            ax2.set_xlim(0, max_dist)
            ax2.set_xlabel('Distance (m)')
            
            # Add grid
            axs[i, 0].grid(True, alpha=0.3)
            axs[i, 1].grid(True, alpha=0.3)
        
        # Add legend to first subplot only
        if n_pairs > 0:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                     ncol=len(self.targets)+1, frameon=True)
        
        # Frame counter text
        frame_text = fig.text(0.02, 0.02, "", fontsize=12)
        
        # Animation update function
        def update(frame):
            if frame >= len(self.correlation_history):
                return lines + peak_plots + target_markers + [frame_text]
                    
            echo_data = self.correlation_history[frame]
            
            line_idx = 0
            peak_idx = 0
            marker_idx = 0
        
            for (s_idx, m_idx), data in echo_data.items():
                # Update correlation plot
                corr = data['correlation']
                lines[line_idx].set_data(np.arange(len(corr)), corr)
                
                # Set x limit based on data
                axs[line_idx//2, 0].set_xlim(0, len(corr))
                
                # Update peak plot
                peaks = data['peaks']
                if len(peaks) > 0:
                    peak_plots[peak_idx].set_data(peaks, corr[peaks])
                else:
                    peak_plots[peak_idx].set_data([], [])
                
                # Update target markers - show predicted echo positions
                speaker_pos = self.speakers[s_idx]
                mic_pos = self.mics[m_idx]
                
                for t_idx, target in enumerate(self.targets):
                    # Get the target position at this frame
                    if frame < len(target['history']):
                        pos = target['history'][frame]
                        
                        # Calculate the expected echo delay
                        to_target = np.linalg.norm(pos - speaker_pos)
                        from_target = np.linalg.norm(mic_pos - pos)
                        total_dist = to_target + from_target
                        delay_samples = int(total_dist / self.c * self.fs)
                        
                        # Adjust for direct path
                        direct_delay = self.direct_delays[(s_idx, m_idx)]
                        relative_delay = delay_samples - direct_delay
                        
                        # Show marker at expected echo position
                        if 0 <= relative_delay < len(corr):
                            target_markers[marker_idx].set_data([delay_samples], [corr[relative_delay]])
                        else:
                            target_markers[marker_idx].set_data([], [])
                    else:
                        target_markers[marker_idx].set_data([], [])
                    
                    marker_idx += 1
                
                line_idx += 1
                peak_idx += 1
            
                # Update difference plot
                diff = data['diff']
                lines[line_idx].set_data(np.arange(len(diff)), diff)
                
                # Set x limit based on data
                axs[line_idx//2, 1].set_xlim(0, len(diff))
                
                # Update peak plot
                if len(peaks) > 0:
                    peak_plots[peak_idx].set_data(peaks, diff[peaks])
                else:
                    peak_plots[peak_idx].set_data([], [])
                
                # Update target markers for difference plot
                for t_idx, target in enumerate(self.targets):
                    if frame < len(target['history']):
                        pos = target['history'][frame]
                        
                        to_target = np.linalg.norm(pos - speaker_pos)
                        from_target = np.linalg.norm(mic_pos - pos)
                        total_dist = to_target + from_target
                        delay_samples = int(total_dist / self.c * self.fs)
                        
                        direct_delay = self.direct_delays[(s_idx, m_idx)]
                        relative_delay = delay_samples - direct_delay
                        
                        if 0 <= relative_delay < len(diff):
                            target_markers[marker_idx].set_data([delay_samples], [diff[relative_delay]])
                        else:
                            target_markers[marker_idx].set_data([], [])
                    else:
                        target_markers[marker_idx].set_data([], [])
                    
                    marker_idx += 1
                
                line_idx += 1
                peak_idx += 1
            
            # Update frame counter
            frame_text.set_text(f"Frame: {frame}/{len(self.correlation_history)-1}")
            
            # Update title
            fig.suptitle(f"Step {frame}: Correlation and MTI Results", fontsize=16)
            
            return lines + peak_plots + target_markers + [frame_text]
    
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.correlation_history),
            interval=interval, blit=False
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save animation if requested
        if save_path:
            writer = animation.FFMpegWriter(fps=1000/interval, metadata=dict(artist='AMT+'), bitrate=5000)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        
        return anim


def analyze_constant_shift(tracker, output_dir):
    """Analyze whether there is a constant position shift in the tracking
    
    Args:
        tracker (AcousticTracker): The tracker with tracking data
        output_dir (str): Directory to save visualizations
    """
    print("\nAnalyzing potential constant position shift...")
    
    # Create figure for each target
    for target in tracker.targets:
        # Get actual and estimated positions
        true_positions = np.array(target['history'])
        
        # For raw estimates, filter out None values
        est_pos_with_idx = [(i, pos) for i, pos in enumerate(target['estimated_positions']) if pos is not None]
        if est_pos_with_idx:
            indices, est_positions = zip(*est_pos_with_idx)
            est_positions = np.array(est_positions)
            indices = np.array(indices)
            
            # Calculate errors
            errors = est_positions - true_positions[indices]
            
            # Create figure with 4 subplots (3 for each axis, 1 for 3D)
            fig, axs = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
            
            # Plot errors for each axis
            axs[0, 0].plot(indices, errors[:, 0], 'r-', label='X Error')
            axs[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axs[0, 0].axhline(y=np.mean(errors[:, 0]), color='r', linestyle=':', label=f'Mean: {np.mean(errors[:, 0]):.3f}m')
            axs[0, 0].set_xlabel('Step')
            axs[0, 0].set_ylabel('Error (m)')
            axs[0, 0].set_title(f'{target["name"]} - X-axis Error')
            axs[0, 0].grid(True, alpha=0.3)
            axs[0, 0].legend()
            
            axs[0, 1].plot(indices, errors[:, 1], 'g-', label='Y Error')
            axs[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axs[0, 1].axhline(y=np.mean(errors[:, 1]), color='g', linestyle=':', label=f'Mean: {np.mean(errors[:, 1]):.3f}m')
            axs[0, 1].set_xlabel('Step')
            axs[0, 1].set_ylabel('Error (m)')
            axs[0, 1].set_title(f'{target["name"]} - Y-axis Error')
            axs[0, 1].grid(True, alpha=0.3)
            axs[0, 1].legend()
            
            axs[1, 0].plot(indices, errors[:, 2], 'b-', label='Z Error')
            axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axs[1, 0].axhline(y=np.mean(errors[:, 2]), color='b', linestyle=':', label=f'Mean: {np.mean(errors[:, 2]):.3f}m')
            axs[1, 0].set_xlabel('Step')
            axs[1, 0].set_ylabel('Error (m)')
            axs[1, 0].set_title(f'{target["name"]} - Z-axis Error')
            axs[1, 0].grid(True, alpha=0.3)
            axs[1, 0].legend()
            
            # 3D plot of actual vs. estimated position
            ax_3d = axs[1, 1]
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            
            # Plot true trajectory
            ax_3d.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], 
                     'g-', label='True Position')
            
            # Plot estimated positions
            ax_3d.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 
                     'r--', label='Estimated Position')
            
            # Connect corresponding points with lines to show error vectors
            for i, idx in enumerate(indices):
                ax_3d.plot([true_positions[idx, 0], est_positions[i, 0]],
                         [true_positions[idx, 1], est_positions[i, 1]],
                         [true_positions[idx, 2], est_positions[i, 2]],
                         'k:', alpha=0.3)
            
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title(f'{target["name"]} - True vs. Estimated Position')
            ax_3d.legend()
            
            # Add a text box with error statistics
            mean_error = np.mean(np.linalg.norm(errors, axis=1))
            max_error = np.max(np.linalg.norm(errors, axis=1))
            std_error = np.std(np.linalg.norm(errors, axis=1))
            mean_vec = np.mean(errors, axis=0)
            
            stats_text = (f"Mean Error: {mean_error:.3f}m\n"
                        f"Max Error: {max_error:.3f}m\n"
                        f"Std Dev: {std_error:.3f}m\n"
                        f"Mean Error Vector:\n"
                        f"  X: {mean_vec[0]:.3f}m\n"
                        f"  Y: {mean_vec[1]:.3f}m\n"
                        f"  Z: {mean_vec[2]:.3f}m")
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Check if there's a significant constant shift
            mean_abs = np.abs(mean_vec)
            threshold = 0.1  # Threshold for significant shift in meters
            
            if np.any(mean_abs > threshold):
                shift_axes = []
                if mean_abs[0] > threshold:
                    shift_axes.append(f"X ({mean_vec[0]:.3f}m)")
                if mean_abs[1] > threshold:
                    shift_axes.append(f"Y ({mean_vec[1]:.3f}m)")
                if mean_abs[2] > threshold:
                    shift_axes.append(f"Z ({mean_vec[2]:.3f}m)")
                
                shift_text = f"DETECTED CONSTANT SHIFT in {', '.join(shift_axes)}"
                fig.suptitle(f"{target['name']} Position Errors - {shift_text}", fontsize=16, color='red')
            else:
                fig.suptitle(f"{target['name']} Position Errors - No significant constant shift detected", fontsize=16)
            
            plt.tight_layout()
            fig.subplots_adjust(top=0.92)
            
            # Save figure
            fig.savefig(f"{output_dir}/shift_analysis_{target['name'].replace(' ', '_')}.png", dpi=300)
            plt.close(fig)
            
            # Print findings
            print(f"\n{target['name']} Error Analysis:")
            print(f"  Mean Error: {mean_error:.3f}m")
            print(f"  Mean Error Vector: X: {mean_vec[0]:.3f}m, Y: {mean_vec[1]:.3f}m, Z: {mean_vec[2]:.3f}m")
            
            if np.any(mean_abs > threshold):
                print(f"  CONSTANT SHIFT DETECTED in {', '.join(shift_axes)}")
            else:
                print("  No significant constant shift detected")
        else:
            print(f"\n{target['name']}: No valid estimated positions for analysis")

def run_demo():
    """Run a demonstration of the high-resolution acoustic tracking system with performance optimizations"""
    import time
    import os
    
    # Create output directory for saving images
    output_dir = "amt_debug_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Performance measurement
    total_start_time = time.time()
    
    # Create tracker
    print("Initializing acoustic tracker...")
    tracker = AcousticTracker()
    
    # Add targets with perpendicular trajectories (to better demonstrate tracking)
    tracker.add_target((2.5, 3.0, 1.7), velocity=(0.3, 0, 0), name="Person 1")
    tracker.add_target((1.5, 4.0, 1.6), velocity=(0, 0.25, 0), name="Person 2")
    
    try:
        # Run tracking simulation with performance metrics
        print("\nRunning tracking simulation...")
        tracking_start = time.time()
        # Use fewer steps for faster execution while testing
        tracker.run_tracking(duration=5.0, steps=15)  # Reduced steps for faster execution
        tracking_time = time.time() - tracking_start
        print(f"Tracking completed in {tracking_time:.2f} seconds")
    except Exception as e:
        print(f"Error during tracking simulation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Selective visualization for better performance
    # Only generate the most important visualizations
    
    try:
        # Just a few key frames for correlation visualization
        frames_to_show = [4, 8]
        for frame in frames_to_show:
            if frame < len(tracker.correlation_history):
                print(f"\nGenerating correlation visualization for frame {frame}...")
                fig = tracker.visualize_correlation(step=frame)
                if fig:
                    fig.savefig(f"{output_dir}/correlation_frame{frame}.png", dpi=300)
    except Exception as e:
        print(f"Error generating correlation frames: {str(e)}")
    
    # Skip animation generation for better performance
    # Only create if explicitly needed
    if False:  # Set to True if animation is needed
        try:
            print("\nCreating correlation animation...")
            anim = tracker.visualize_correlation_animation(save_path=f"{output_dir}/correlation_animation.mp4")
        except Exception as e:
            print(f"Error generating correlation animation: {str(e)}")
    
    try:
        # Compare raw estimation vs Kalman filtered tracking and save the comparison
        print("\nGenerating tracking comparison...")
        fig = tracker.compare_tracking()
        if fig:
            fig.savefig(f"{output_dir}/tracking_comparison.png", dpi=300)
    except Exception as e:
        print(f"Error generating tracking comparison: {str(e)}")
    
    try:
        # Visualize final tracking results
        print("\nGenerating tracking visualizations...")
        # Only generate filtered visualization (more important than raw)
        fig = tracker.visualize(use_filtered=True)
        if fig:
            fig.savefig(f"{output_dir}/filtered_tracking.png", dpi=300)
    except Exception as e:
        print(f"Error generating tracking visualization: {str(e)}")
    
    try:
        # Analyze any potential constant shift issues
        print("\nAnalyzing position errors...")
        analyze_constant_shift(tracker, output_dir)
    except Exception as e:
        print(f"Error during constant shift analysis: {str(e)}")
    
    # Report total runtime
    total_time = time.time() - total_start_time
    print(f"\nDemo completed in {total_time:.2f} seconds")
    print(f"Debug images saved to '{output_dir}' directory")


# For installing required packages:
# pip install numpy matplotlib scipy pyroomacoustics filterpy

if __name__ == "__main__":
    run_demo()