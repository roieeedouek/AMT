import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm  # Import LogNorm from colors module
import matplotlib.animation as animation
from scipy import signal
import pyroomacoustics as pra
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
import time
import os

class AcousticTracker:
    """Acoustic multi-target tracking system using home theater setup with pyroomacoustics and Kalman filtering"""
    
    def __init__(self, room_dim=(5.0, 6.0, 2.4), speed_of_sound=343.0, debug_mode=True):
        """Initialize the acoustic tracker
        
        Args:
            room_dim (tuple): Room dimensions (width, length, height) in meters
            speed_of_sound (float): Speed of sound in m/s at room temperature
        """
        self.room_dim = room_dim
        self.c = speed_of_sound
        self.fs = 48000  # Increased sampling rate for better resolution
        
        self.debug_mode = debug_mode
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
    
    def simulate_echoes(self, block_length=2048, noise_snr=None):
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
                
                # Add Additive White Gaussian Noise if SNR is specified
                if noise_snr is not None:
                    # Calculate signal power
                    signal_power = np.mean(mic_signal**2)
                    
                    # Convert SNR from dB to linear scale
                    snr_linear = 10**(noise_snr/10)
                    
                    # Calculate noise power
                    noise_power = signal_power / snr_linear
                    
                    # Generate AWGN with the appropriate power
                    noise = np.random.normal(0, np.sqrt(noise_power), size=buffer_length)
                    
                    # Add noise to the signal
                    mic_signal += noise
                
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
    
    def _compute_error(self, point, ellipses, prev_point=None):
        """Helper function to compute error for a point
        
        Args:
            point (np.array): Position to evaluate
            ellipses (list): List of ellipses
            prev_point (np.array, optional): Previous position estimate for continuity
            
        Returns:
            float: Total error
        """
        # Basic error from ellipses
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
    
    def multilateration_twostage(self, ellipses, coarse_res=20, fine_res=50, save_error_map=False):
        """Three-stage multilateration with adaptive resolution and historical tracking
        
        Args:
            ellipses (list): List of ellipse dictionaries
            coarse_res (int): Resolution for coarse search
            fine_res (int): Resolution for fine search
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
        
        # For debugging: create error maps at fixed heights
        if save_error_map:
            try:
                # Choose a few z-slices to visualize
                z_slices = [0.5, 1.0, 1.5, 2.0]
                error_maps = {}
                
                for z_val in z_slices:
                    try:
                        # Find closest z in our grid
                        z_idx = np.argmin(np.abs(z - z_val))
                        z_actual = z[z_idx]
                        
                        # Create error map for this z-slice
                        error_map = np.zeros((len(y), len(x)))
                        
                        for i, yi in enumerate(y):
                            for j, xi in enumerate(x):
                                point = np.array([xi, yi, z_actual])
                                error_map[i, j] = self._compute_error(point, ellipses, prev_point)
                        
                        error_maps[z_actual] = error_map
                    except Exception as e:
                        print(f"Warning: Error creating error map for z={z_val}: {str(e)}")
                
                # Save the error maps
                # self._save_error_maps(error_maps, ellipses, "coarse")
            except Exception as e:
                print(f"Warning: Error creating coarse error maps: {str(e)}")
        
        # Enhancement 3 continued: Use trajectory information to guide search
        if prev_point is not None and trajectory_direction is not None:
            # Create a predicted position based on trajectory
            predicted_point = prev_point + trajectory_direction * 0.1  # Assume small movement
            
            # Make sure the predicted point is within room bounds
            predicted_point[0] = np.clip(predicted_point[0], 0, self.room_dim[0])
            predicted_point[1] = np.clip(predicted_point[1], 0, self.room_dim[1])
            predicted_point[2] = np.clip(predicted_point[2], 0, self.room_dim[2])
            
            # First check near the predicted point to potentially skip coarse search
            x_pred = np.linspace(max(0, predicted_point[0] - 0.4), 
                               min(self.room_dim[0], predicted_point[0] + 0.4), coarse_res//2)
            y_pred = np.linspace(max(0, predicted_point[1] - 0.4),
                               min(self.room_dim[1], predicted_point[1] + 0.4), coarse_res//2)
            z_pred = np.linspace(max(0, predicted_point[2] - 0.3),
                               min(self.room_dim[2] + 0.3, predicted_point[2] + 0.3), coarse_res * z_res_multiplier)
            
            # Search near predicted position first
            for xi in x_pred:
                for yi in y_pred:
                    for zi in z_pred:
                        point = np.array([xi, yi, zi])
                        total_error = self._compute_error(point, ellipses, prev_point)
                        
                        if total_error < min_error:
                            min_error = total_error
                            best_point = point.copy()
            
            # If error is low enough, we can skip the full coarse search
            skip_coarse = False
            if min_error < 0.05:  # Threshold for accepting predicted-area search
                skip_coarse = True
        
        # Perform coarse search if needed
        if best_point is None or (locals().get('skip_coarse', False) == False):
            # Search on coarse grid with continuity constraints
            for xi in x:
                for yi in y:
                    for zi in z:
                        point = np.array([xi, yi, zi])
                        total_error = self._compute_error(point, ellipses, prev_point)
                        
                        if total_error < min_error:
                            min_error = total_error
                            best_point = point.copy()
        
        if best_point is None:
            return None
        
        # Enhancement 2: Dynamic Resolution Adaptation
        # Analyze error landscape to detect potential ambiguities
        error_samples = []
        ambiguity_detected = False
        
        # Sample points around the best point to detect multiple minima
        if prev_point is not None:
            test_points = []
            # Sample around best point
            for dz in [-0.2, -0.1, 0, 0.1, 0.2]:
                test_point = best_point.copy()
                test_point[2] += dz
                test_points.append(test_point)
            
            # Compute errors for test points
            for point in test_points:
                error = self._compute_error(point, ellipses, prev_point)
                error_samples.append((point[2], error))
            
            # Check for multiple local minima
            error_samples.sort(key=lambda x: x[0])  # Sort by z-coordinate
            errors = np.array([e[1] for e in error_samples])
            
            # Simple detection of multiple minima
            for i in range(1, len(errors)-1):
                if errors[i] < errors[i-1] and errors[i] < errors[i+1]:
                    # Found a local minimum
                    if i > 0 and abs(error_samples[i][0] - best_point[2]) > 0.05:
                        # There's another minimum away from our current best point
                        ambiguity_detected = True
        
        # Stage 2: Fine search with adaptive resolution based on detected ambiguity
        # Define search region around best point
        x_range = max(0.5, self.room_dim[0] / coarse_res)  # Size of search region
        y_range = max(0.5, self.room_dim[1] / coarse_res)
        z_range = max(0.5, self.room_dim[2] / coarse_res)
        
        # Enhancement 2: If ambiguity is detected, increase resolution
        adaptive_fine_res = fine_res
        adaptive_z_multiplier = z_res_multiplier
        if ambiguity_detected:
            adaptive_fine_res = int(fine_res * 1.5)  # 50% more resolution
            adaptive_z_multiplier = z_res_multiplier * 2  # Double z-resolution
            z_range = z_range * 1.5  # Expand search range in z-direction
        
        x = np.linspace(max(0, best_point[0] - x_range/2), 
                       min(self.room_dim[0], best_point[0] + x_range/2), adaptive_fine_res)
        y = np.linspace(max(0, best_point[1] - y_range/2),
                       min(self.room_dim[1], best_point[1] + y_range/2), adaptive_fine_res)
        z = np.linspace(max(0, best_point[2] - z_range/2),
                       min(self.room_dim[2] + 0.3, best_point[2] + z_range/2), 
                       adaptive_fine_res * adaptive_z_multiplier)
        
        # For debugging: create error maps at fine resolution
        if save_error_map:
            try:
                # Use z-value closest to best point
                z_val = best_point[2]
                z_idx = np.argmin(np.abs(z - z_val))
                z_actual = z[z_idx]
                
                # Create error map for this z-slice
                error_map = np.zeros((len(y), len(x)))
                
                for i, yi in enumerate(y):
                    for j, xi in enumerate(x):
                        point = np.array([xi, yi, z_actual])
                        error_map[i, j] = self._compute_error(point, ellipses, prev_point)
                
                # Save the fine error map
                # self._save_error_maps({z_actual: error_map}, ellipses, "fine", best_point=best_point)
            except Exception as e:
                print(f"Warning: Error creating fine error map: {str(e)}")
        
        # Search on fine grid with continuity constraints
        for xi in x:
            for yi in y:
                for zi in z:
                    point = np.array([xi, yi, zi])
                    total_error = self._compute_error(point, ellipses, prev_point)
                    
                    if total_error < min_error:
                        min_error = total_error
                        best_point = point.copy()
        
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
    
    def run_tracking(self, duration=5.0, steps=50, noise_snr=0):
        """Run the complete tracking simulation
        
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
        
        for step in range(steps):
            print(f"Processing step {step+1}/{steps}...")
            
            # Update target positions
            self.update_targets(dt)
            
            # Simulate propagation and echoes
            
            received_signals = self.simulate_echoes(noise_snr=noise_snr)
            # Detect echoes
            echo_data = self.detect_echoes(received_signals)
            
            # Store echo data for visualization
            self.correlation_history.append(echo_data)
            
            # Create ranging ellipses for debugging
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
            self.ellipses_history.append(ellipses)
            
            # Visualize ellipses every 1 steps
            if step % 1 == 0 or step == steps - 1:
                self._visualize_ellipses(ellipses, step, output_dir)
            
            # Locate targets with Kalman filtering (pass step number for debugging)
            estimated_positions, filtered_positions = self.locate_targets(echo_data, dt, step=step)
        
        # Create animation of ellipses over time
        self._create_ellipses_animation(output_dir)
        
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
        fig.savefig(f"correlation/correlation_analysis_{target['movement_data']['type']}{step}.png", dpi=300)
        plt.close(fig)
        
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
        filename = f"trackingresults/tracking_comparison_{self.timestamp}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        
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
        
        # Save the error plot as a separate file
        import os
        output_dir = "amt_debug_images"
        os.makedirs(output_dir, exist_ok=True)
        fig_error.savefig(f"{output_dir}/tracking_errors.png", dpi=300)
        plt.close(fig_error)
        
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

    def _generate_test_summary_plots(self, results, output_dir):
        """
        Generate summary plots for all tests
        
        Args:
            results (dict): All test results
            output_dir (str): Output directory for plots
        """
        # 1. Microphone configuration comparison
        try:
            self._plot_mic_config_comparison(results['mic_configs'], output_dir)
        except Exception as e:
            print(f"Error generating mic config plots: {str(e)}")
        
        # 2. SNR level comparison
        try:
            self._plot_snr_comparison(results['snr_tests'], output_dir)
        except Exception as e:
            print(f"Error generating SNR plots: {str(e)}")
        
        # 3. Movement pattern comparison
        try:
            self._plot_movement_comparison(results['movement_tests'], output_dir)
        except Exception as e:
            print(f"Error generating movement plots: {str(e)}")

    def _plot_mic_config_comparison(self, mic_results, output_dir):
        """Plot comparison of microphone configurations"""
        if not mic_results:
            return
        
        # Extract data for plotting
        configs = []
        mean_errors = []
        axis_errors = []
        valid_percentages = []
        
        for config_name, result in mic_results.items():
            if result['is_valid'] and result['targets'] > 0:
                configs.append(config_name)
                
                # Get metrics for first target
                metrics = result['error_metrics'][0]
                
                mean_errors.append(metrics['filtered_mean_error'])
                axis_errors.append(metrics['filtered_axis_mean'])
                valid_percentages.append(metrics['valid_percentage'])
        
        if not configs:
            return
        
        # Convert to numpy arrays
        axis_errors = np.array(axis_errors)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot overall mean error
        bar_width = 0.8
        x = np.arange(len(configs))
        
        ax1.bar(x, mean_errors, width=bar_width, color='steelblue')
        ax1.set_xlabel('Microphone Configuration')
        ax1.set_ylabel('Mean Error (m)')
        ax1.set_title('Tracking Error by Microphone Configuration')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add valid measurement percentage as text on bars
        for i, v in enumerate(mean_errors):
            ax1.text(i, v + 0.01, f"{valid_percentages[i]:.1f}%", 
                    ha='center', va='bottom', fontsize=9)
        
        # Plot per-axis errors
        x = np.arange(len(configs))
        bar_width = 0.25
        
        ax2.bar(x - bar_width, axis_errors[:, 0], width=bar_width, color='red', label='X Error')
        ax2.bar(x, axis_errors[:, 1], width=bar_width, color='green', label='Y Error')
        ax2.bar(x + bar_width, axis_errors[:, 2], width=bar_width, color='blue', label='Z Error')
        
        ax2.set_xlabel('Microphone Configuration')
        ax2.set_ylabel('Mean Error (m)')
        ax2.set_title('Per-Axis Error by Microphone Configuration')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add title and adjust layout
        fig.suptitle('Microphone Configuration Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure
        filename = f"{output_dir}/mic_config_comparison.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def analyze_position_shift(self, num_steps=20):
        """Analyze whether there is a constant position shift in the tracking
        
        Args:
            num_steps (int): Number of tracking steps to analyze
            
        Returns:
            dict: Diagnostics including shift vectors and correction matrices
        """
        # Run a short tracking simulation if not already done
        if not self.tracking_data or not self.targets:
            print("No tracking data available. Running simulation...")
            self.run_tracking(duration=2.0, steps=num_steps)
        
        # Initialize diagnostic results
        diagnostics = {
            'per_target': {},
            'global': {
                'position_shifts': [],
                'has_consistent_shift': False,
                'correction_vector': None,
                'correction_matrix': None
            }
        }
        
        # Collect error data for each target
        for target in self.targets:
            target_id = target['id']
            
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
                
                # Calculate statistics
                mean_error = np.mean(errors, axis=0)
                std_error = np.std(errors, axis=0)
                median_error = np.median(errors, axis=0)
                
                # Check if the error is consistent (low standard deviation)
                error_consistency = np.all(std_error < 0.05)  # 5cm threshold
                
                # Check if there's a significant shift (mean error > 10cm)
                significant_shift = np.linalg.norm(mean_error) > 0.1
                
                # Store results for this target
                target_diagnostics = {
                    'error_vectors': errors,
                    'mean_error': mean_error,
                    'median_error': median_error, 
                    'std_error': std_error,
                    'error_consistency': error_consistency,
                    'significant_shift': significant_shift,
                    'num_valid_estimates': len(est_positions),
                    'correction_vector': -mean_error if error_consistency else None
                }
                
                diagnostics['per_target'][target_id] = target_diagnostics
                
                # Add to global results
                if error_consistency:
                    diagnostics['global']['position_shifts'].append(mean_error)
        
        # Analyze global shifts if we have data from multiple targets
        if len(diagnostics['global']['position_shifts']) > 0:
            # Calculate average shift across targets
            global_shift = np.mean(diagnostics['global']['position_shifts'], axis=0)
            
            # Check if shift is consistent across targets
            shifts = np.array(diagnostics['global']['position_shifts'])
            shift_std = np.std(shifts, axis=0)
            global_consistency = np.all(shift_std < 0.05)  # 5cm threshold
            
            diagnostics['global']['has_consistent_shift'] = global_consistency
            
            if global_consistency:
                # We've detected a consistent position shift!
                diagnostics['global']['correction_vector'] = -global_shift
                
                # Generate a correction matrix (for more complex transformations)
                # For now, just a simple translation
                correction_matrix = np.eye(4)  # 4x4 homogeneous transformation
                correction_matrix[:3, 3] = -global_shift
                diagnostics['global']['correction_matrix'] = correction_matrix
                
                print("\n==== POSITION SHIFT DETECTED ====")
                print(f"Detected a consistent position shift across targets:")
                print(f"  X shift: {global_shift[0]:.3f} m")
                print(f"  Y shift: {global_shift[1]:.3f} m")
                print(f"  Z shift: {global_shift[2]:.3f} m")
                print(f"Correction vector: {-global_shift}")
                print("===================================\n")
                
                # Create visualization of the position shift
                self._visualize_position_shift(diagnostics)
        
        # Store diagnostics for later reference
        self.position_shift_diagnostics = diagnostics
        
        return diagnostics
    
    def _visualize_position_shift(self, diagnostics):
        """
        Visualize the position shift for better understanding
        
        Args:
            diagnostics (dict): Position shift diagnostics
        """
        # Create a dedicated directory for position shift analysis
        output_dir = "amt_debug_images/position_shift"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Create a 3D visualization of error vectors
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set limits based on room dimensions
        ax.set_xlim(0, self.room_dim[0])
        ax.set_ylim(0, self.room_dim[1])
        ax.set_zlim(0, self.room_dim[2])
        
        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Position Shift Analysis')
        
        # Draw room boundaries
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
            ax.scatter(pos[0], pos[1], pos[2], color='red', marker='o', s=50, label=f'Speaker {i+1}' if i == 0 else None)
        for i, pos in enumerate(self.mics):
            ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='^', s=50, label=f'Mic {i+1}' if i == 0 else None)
        
        # Draw position shift vectors for each target
        colors = ['g', 'purple', 'orange', 'cyan', 'magenta']
        
        for target_id, target_diag in diagnostics['per_target'].items():
            if target_id < len(self.targets):
                target = self.targets[target_id]
                color = colors[target_id % len(colors)]
                
                # Only plot if we have a correction vector
                if target_diag['correction_vector'] is not None:
                    # Get a representative position
                    pos = target['position']
                    
                    # Draw position and error vector
                    ax.scatter(pos[0], pos[1], pos[2], color=color, marker='s', s=80, 
                            label=f'{target["name"]} Position')
                    
                    # Draw error vector (scaled for visibility)
                    error_vector = target_diag['mean_error']
                    ax.quiver(pos[0], pos[1], pos[2], 
                            error_vector[0], error_vector[1], error_vector[2], 
                            color=color, length=1.0, normalize=True,
                            label=f'{target["name"]} Error')
        
        # Draw global shift if available
        if diagnostics['global']['has_consistent_shift']:
            # Use room center as reference point
            room_center = np.array([
                self.room_dim[0]/2, 
                self.room_dim[1]/2, 
                self.room_dim[2]/2
            ])
            
            # Draw global shift vector
            global_shift = diagnostics['global']['position_shifts'][0]  # Use first shift for visualization
            ax.quiver(room_center[0], room_center[1], room_center[2],
                    global_shift[0], global_shift[1], global_shift[2],
                    color='red', length=1.0, linewidth=3, normalize=True,
                    label='Global Shift')
        
        # Add legend
        ax.legend()
        
        # Save figure
        filename = f"{output_dir}/position_shift_3d.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        
        # 2. Create a multi-panel figure showing error distribution per axis
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Collect all error vectors
        all_errors = []
        for target_diag in diagnostics['per_target'].values():
            if 'error_vectors' in target_diag:
                all_errors.extend(target_diag['error_vectors'])
        
        if all_errors:
            all_errors = np.array(all_errors)
            
            # Plot X errors
            axes[0].hist(all_errors[:, 0], bins=20, color='red', alpha=0.7)
            axes[0].set_title('X-axis Position Error Distribution')
            axes[0].set_xlabel('Error (m)')
            axes[0].set_ylabel('Count')
            axes[0].axvline(x=0, color='k', linestyle='--')
            if diagnostics['global']['has_consistent_shift']:
                axes[0].axvline(x=diagnostics['global']['correction_vector'][0], 
                            color='red', linestyle='-', linewidth=2,
                            label=f'Correction: {diagnostics["global"]["correction_vector"][0]:.3f}m')
                axes[0].legend()
            
            # Plot Y errors
            axes[1].hist(all_errors[:, 1], bins=20, color='green', alpha=0.7)
            axes[1].set_title('Y-axis Position Error Distribution')
            axes[1].set_xlabel('Error (m)')
            axes[1].set_ylabel('Count')
            axes[1].axvline(x=0, color='k', linestyle='--')
            if diagnostics['global']['has_consistent_shift']:
                axes[1].axvline(x=diagnostics['global']['correction_vector'][1], 
                            color='green', linestyle='-', linewidth=2,
                            label=f'Correction: {diagnostics["global"]["correction_vector"][1]:.3f}m')
                axes[1].legend()
            
            # Plot Z errors
            axes[2].hist(all_errors[:, 2], bins=20, color='blue', alpha=0.7)
            axes[2].set_title('Z-axis Position Error Distribution')
            axes[2].set_xlabel('Error (m)')
            axes[2].set_ylabel('Count')
            axes[2].axvline(x=0, color='k', linestyle='--')
            if diagnostics['global']['has_consistent_shift']:
                axes[2].axvline(x=diagnostics['global']['correction_vector'][2], 
                            color='blue', linestyle='-', linewidth=2,
                            label=f'Correction: {diagnostics["global"]["correction_vector"][2]:.3f}m')
                axes[2].legend()
        
        # Add summary text
        if diagnostics['global']['has_consistent_shift']:
            corr = diagnostics['global']['correction_vector']
            summary_text = (
                f"Position Shift Analysis\n"
                f"----------------------\n"
                f"Detected a consistent position shift:\n"
                f"  X shift: {-corr[0]:.3f} m\n"
                f"  Y shift: {-corr[1]:.3f} m\n"
                f"  Z shift: {-corr[2]:.3f} m\n"
                f"Applied correction: {corr}"
            )
            fig.text(0.1, 0.01, summary_text, fontsize=12, family='monospace', 
                bbox=dict(facecolor='white', alpha=0.5))
        
        # Save figure
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        filename = f"{output_dir}/position_shift_distributions.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def _update_direct_delays(self):
        """Update direct path delays when microphone configuration changes"""
        self.direct_delays = {}
        for s_idx, speaker_pos in enumerate(self.speakers):
            for m_idx, mic_pos in enumerate(self.mics):
                direct_dist = np.linalg.norm(mic_pos - speaker_pos)
                direct_delay_samples = int(direct_dist / self.c * self.fs)
                self.direct_delays[(s_idx, m_idx)] = direct_delay_samples

    def _clear_targets(self):
        """Clear all existing targets"""
        self.targets = []
        self.tracking_data = {}
        self.kalman_filters = {}

    def _setup_linear_movement(self, starting_pos, velocity):
        """Setup a target with linear movement"""
        self.add_target(
            position=starting_pos,
            velocity=velocity,
            name="LinearTarget"
        )

    def _setup_circular_movement(self):
        """Setup a target with circular movement in XY plane"""
        center = np.array([2.5, 3.0, 1.2])
        radius = 1.0
        angular_velocity = 0.5  # radians per second
        
        # Add target at starting position
        target = self.add_target(
            position=[center[0] + radius, center[1], center[2]],
            velocity=[0, radius * angular_velocity, 0],
            name="CircularTarget"
        )
        
        # Store additional data for updating movement
        target['movement_data'] = {
            'type': 'circular',
            'center': center,
            'radius': radius,
            'angular_velocity': angular_velocity,
            'angle': 0.0
        }
        
        # Override the standard update_targets method for this test
        self._original_update_targets = self.update_targets
        
        def circular_update(dt):
            for target in self.targets:
                if 'movement_data' in target and target['movement_data']['type'] == 'circular':
                    data = target['movement_data']
                    
                    # Update angle
                    data['angle'] += data['angular_velocity'] * dt
                    
                    # Calculate new position
                    new_x = data['center'][0] + data['radius'] * np.cos(data['angle'])
                    new_y = data['center'][1] + data['radius'] * np.sin(data['angle'])
                    
                    # Update position
                    target['position'] = np.array([new_x, new_y, data['center'][2]])
                    
                    # Update velocity (tangential)
                    vx = -data['radius'] * data['angular_velocity'] * np.sin(data['angle'])
                    vy = data['radius'] * data['angular_velocity'] * np.cos(data['angle'])
                    target['velocity'] = np.array([vx, vy, 0])
                    
                    # Update history and tracking data
                    target['history'].append(target['position'].copy())
                    self.tracking_data[target['id']]['true_positions'].append(target['position'].copy())
        
        # Replace update method
        self.update_targets = circular_update

    def _setup_zigzag_3d_movement(self):
        """Setup a target with zigzag movement in 3D space"""
        # Add target at starting position
        target = self.add_target(
            position=[1.0, 1.0, 1.0],
            velocity=[0.3, 0.2, 0.1],
            name="ZigzagTarget"
        )
        
        # Store additional data for updating movement
        target['movement_data'] = {
            'type': 'zigzag',
            'waypoints': [
                [1.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
                [4.0, 4.0, 1.5],
                [1.0, 4.0, 1.5],
                [1.0, 1.0, 0.5],
                [4.0, 1.0, 0.5]
            ],
            'current_waypoint': 1,
            'speed': 0.5  # meters per second
        }
        
        # Override the standard update_targets method for this test
        self._original_update_targets = self.update_targets
        
        def zigzag_update(dt):
            for target in self.targets:
                if 'movement_data' in target and target['movement_data']['type'] == 'zigzag':
                    data = target['movement_data']
                    
                    # Get current and next waypoint
                    current_pos = target['position']
                    next_pos = np.array(data['waypoints'][data['current_waypoint']])
                    
                    # Calculate direction and distance
                    direction = next_pos - current_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance < 0.1:  # Close enough to waypoint
                        # Move to next waypoint
                        data['current_waypoint'] = (data['current_waypoint'] + 1) % len(data['waypoints'])
                        next_pos = np.array(data['waypoints'][data['current_waypoint']])
                        direction = next_pos - current_pos
                        distance = np.linalg.norm(direction)
                    
                    # Normalize direction
                    if distance > 0:
                        direction = direction / distance
                    
                    # Calculate movement for this step
                    step_distance = min(data['speed'] * dt, distance)
                    step = direction * step_distance
                    
                    # Update position
                    target['position'] = current_pos + step
                    
                    # Update velocity
                    target['velocity'] = direction * data['speed']
                    
                    # Update history and tracking data
                    target['history'].append(target['position'].copy())
                    self.tracking_data[target['id']]['true_positions'].append(target['position'].copy())
        
        # Replace update method
        self.update_targets = zigzag_update

    def _run_single_test(self, steps=20, duration=4.0, noise_snr=None, test_name="test"):
        """
        Run a single test and gather performance metrics
        
        Args:
            steps (int): Number of tracking steps
            duration (float): Test duration in seconds
            noise_snr (float): Signal-to-noise ratio in dB (None for no noise)
            test_name (str): Name for this test
            
        Returns:
            dict: Test results and metrics
        """
        print(f"  Running test: {test_name}")
        
        # Initialize results
        results = {
            'test_name': test_name,
            'mics': [m.tolist() for m in self.mics],
            'steps': steps,
            'duration': duration,
            'noise_snr': noise_snr,
            'targets': len(self.targets),
            'error_metrics': {},
            'computation_time': None,
            'is_valid': True
        }
        
        try:
            # Run tracking
            start_time = time.time()
            self.run_tracking(duration=duration, steps=steps, noise_snr=noise_snr)
            end_time = time.time()
            
            # Calculate computation time
            computation_time = end_time - start_time
            results['computation_time'] = computation_time
            
            # Calculate error metrics for each target
            for target in self.targets:
                target_id = target['id']
                
                # Get ground truth positions
                true_positions = np.array(target['history'])
                
                # Get estimated positions (filtered)
                filtered_positions = np.array(target['filtered_positions'])
                
                # Get raw estimated positions (may contain None values)
                valid_estimates = []
                valid_indices = []
                for i, pos in enumerate(target['estimated_positions']):
                    if pos is not None:
                        valid_estimates.append(pos)
                        valid_indices.append(i)
                
                # Calculate metrics
                if len(valid_estimates) > 0:
                    valid_estimates = np.array(valid_estimates)
                    valid_true = np.array([true_positions[i] for i in valid_indices])
                    
                    # Raw estimation error
                    raw_errors = np.linalg.norm(valid_estimates - valid_true, axis=1)
                    raw_mean_error = np.mean(raw_errors)
                    raw_median_error = np.median(raw_errors)
                    raw_max_error = np.max(raw_errors)
                    
                    # Per-axis errors
                    raw_axis_errors = valid_estimates - valid_true
                    raw_axis_mean = np.mean(np.abs(raw_axis_errors), axis=0)
                    
                    # Percentage of valid measurements
                    valid_percentage = len(valid_estimates) / len(true_positions) * 100
                else:
                    raw_mean_error = None
                    raw_median_error = None
                    raw_max_error = None
                    raw_axis_mean = [None, None, None]
                    valid_percentage = 0
                
                # Filtered position error (always available)
                filtered_errors = np.linalg.norm(filtered_positions - true_positions, axis=1)
                filtered_mean_error = np.mean(filtered_errors)
                filtered_median_error = np.median(filtered_errors)
                filtered_max_error = np.max(filtered_errors)
                
                # Per-axis errors
                filtered_axis_errors = filtered_positions - true_positions
                filtered_axis_mean = np.mean(np.abs(filtered_axis_errors), axis=0)
                
                # Store metrics
                results['error_metrics'][target_id] = {
                    'raw_mean_error': raw_mean_error,
                    'raw_median_error': raw_median_error,
                    'raw_max_error': raw_max_error,
                    'raw_axis_mean': raw_axis_mean.tolist() if hasattr(raw_axis_mean, 'tolist') else raw_axis_mean,
                    'filtered_mean_error': filtered_mean_error,
                    'filtered_median_error': filtered_median_error,
                    'filtered_max_error': filtered_max_error,
                    'filtered_axis_mean': filtered_axis_mean.tolist(),
                    'valid_percentage': valid_percentage
                }
            
            # Print summary
            print(f"  Completed in {computation_time:.2f} seconds")
            
            if self.targets and 0 in results['error_metrics']:
                metrics = results['error_metrics'][0]  # First target
                print(f"  First target metrics:")
                print(f"    Filtered mean error: {metrics['filtered_mean_error']:.3f} m")
                print(f"    Raw measurements available: {metrics['valid_percentage']:.1f}%")
            
            # Save visualization
            filename = f"amt_test_results/{test_name}_visualization.png"
            fig = self.visualize()
            fig.savefig(filename, dpi=300)
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error during test: {str(e)}")
            results['is_valid'] = False
            results['error'] = str(e)
        
        # Restore original update method if it was replaced
        if hasattr(self, '_original_update_targets'):
            self.update_targets = self._original_update_targets
            delattr(self, '_original_update_targets')
        
        return results
    
    def _plot_snr_comparison(self, snr_results, output_dir):
        """Plot comparison of SNR levels"""
        if not snr_results:
            return
        
        # Extract data for plotting
        snr_levels = []
        mean_errors = []
        valid_percentages = []
        
        for snr_name, result in snr_results.items():
            if result['is_valid'] and result['targets'] > 0:
                snr_levels.append(snr_name)
                
                # Get metrics for first target
                metrics = result['error_metrics'][0]
                
                mean_errors.append(metrics['filtered_mean_error'])
                valid_percentages.append(metrics['valid_percentage'])
        
        if not snr_levels:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean error vs SNR
        bar_width = 0.8
        x = np.arange(len(snr_levels))
        
        # Main bars for error
        bars = ax.bar(x, mean_errors, width=bar_width, color='steelblue')
        
        # Add percentage text on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{valid_percentages[i]:.1f}%", ha='center', va='bottom', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_ylabel('Mean Error (m)')
        ax.set_title('Tracking Error vs. SNR Level')
        ax.set_xticks(x)
        ax.set_xticklabels(snr_levels)
        ax.grid(axis='y', alpha=0.3)
        
        # Add second y-axis for valid percentage
        ax2 = ax.twinx()
        ax2.plot(x, valid_percentages, 'ro-', linewidth=2)
        ax2.set_ylabel('Valid Measurements (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 105)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f"{output_dir}/snr_comparison.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def _plot_movement_comparison(self, movement_results, output_dir):
        """Plot comparison of movement patterns"""
        if not movement_results:
            return
        
        # Extract data for plotting
        patterns = []
        mean_errors = []
        axis_errors = []
        valid_percentages = []
        
        for pattern_name, result in movement_results.items():
            if result['is_valid'] and result['targets'] > 0:
                patterns.append(pattern_name)
                
                # Get metrics for first target
                metrics = result['error_metrics'][0]
                
                mean_errors.append(metrics['filtered_mean_error'])
                axis_errors.append(metrics['filtered_axis_mean'])
                valid_percentages.append(metrics['valid_percentage'])
        
        if not patterns:
            return
        
        # Convert to numpy arrays
        axis_errors = np.array(axis_errors)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot overall mean error
        bar_width = 0.8
        x = np.arange(len(patterns))
        
        ax1.bar(x, mean_errors, width=bar_width, color='steelblue')
        ax1.set_xlabel('Movement Pattern')
        ax1.set_ylabel('Mean Error (m)')
        ax1.set_title('Tracking Error by Movement Pattern')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patterns, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add valid measurement percentage as text on bars
        for i, v in enumerate(mean_errors):
            ax1.text(i, v + 0.01, f"{valid_percentages[i]:.1f}%", 
                    ha='center', va='bottom', fontsize=9)
        
        # Plot per-axis errors
        x = np.arange(len(patterns))
        bar_width = 0.25
        
        ax2.bar(x - bar_width, axis_errors[:, 0], width=bar_width, color='red', label='X Error')
        ax2.bar(x, axis_errors[:, 1], width=bar_width, color='green', label='Y Error')
        ax2.bar(x + bar_width, axis_errors[:, 2], width=bar_width, color='blue', label='Z Error')
        
        ax2.set_xlabel('Movement Pattern')
        ax2.set_ylabel('Mean Error (m)')
        ax2.set_title('Per-Axis Error by Movement Pattern')
        ax2.set_xticks(x)
        ax2.set_xticklabels(patterns, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add title and adjust layout
        fig.suptitle('Movement Pattern Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure
        filename = f"{output_dir}/movement_comparison.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def run_comprehensive_tests(self):
        """
        Run comprehensive tests of the AMT3D system with different configurations
        
        Tests various microphone configurations, SNR levels, and movement patterns
        to evaluate system performance and robustness.
        
        Returns:
            dict: Test results organized by configuration and scenario
        """
        # Create output directory for test results
        output_dir = "amt_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all test results
        all_results = {
            'mic_configs': {},
            'snr_tests': {},
            'movement_tests': {}
        }
        
        print("\n===== RUNNING COMPREHENSIVE TESTS =====\n")
        
        # Define the test configurations
        mic_configs = {
            "soundbar_2mic": {
                "description": "Soundbar with 2 mics in a row (horizontal plane)",
                "positions": [
                    [2.0, 0.3, 1.0],  # Left mic
                    [3.0, 0.3, 1.0]   # Right mic
                ]
            },
            "soundbar_3mic": {
                "description": "Soundbar with 3 mics in a row (horizontal plane)",
                "positions": [
                    [1.5, 0.3, 1.0],  # Left mic
                    [2.5, 0.3, 1.0],  # Center mic
                    [3.5, 0.3, 1.0]   # Right mic
                ]
            },
            "tv_plus": {
                "description": "55-inch TV with 4 mics in + configuration (frontal plane)",
                "positions": [
                    [2.5, 0.3, 1.0],  # Center mic
                    [1.5, 0.3, 1.0],  # Left mic
                    [3.5, 0.3, 1.0],  # Right mic
                    [2.5, 0.3, 1.5]   # Top mic
                ]
            },
            "tv_x": {
                "description": "55-inch TV with 4 mics in X configuration (frontal plane)",
                "positions": [
                    [1.5, 0.3, 0.7],  # Bottom-left mic
                    [3.5, 0.3, 0.7],  # Bottom-right mic
                    [1.5, 0.3, 1.3],  # Top-left mic
                    [3.5, 0.3, 1.3]   # Top-right mic
                ]
            },
            "boxy_soundbar_3mic": {
                "description": "Boxy soundbar with 3 mics (horizontal plane, not in a row)",
                "positions": [
                    [1.5, 0.3, 1.0],    # Left mic
                    [3.5, 0.3, 1.0],    # Right mic
                    [2.5, 0.7, 1.0]     # Back-center mic
                ]
            },
            "boxy_soundbar_4mic": {
                "description": "Boxy soundbar with 4 mics (horizontal plane, not in a row)",
                "positions": [
                    [1.5, 0.3, 1.0],    # Front-left mic
                    [3.5, 0.3, 1.0],    # Front-right mic
                    [1.8, 0.7, 1.0],    # Back-left mic
                    [3.2, 0.7, 1.0]     # Back-right mic
                ]
            }
        }
        
        # Define SNR levels to test
        snr_levels = [None, 20, 10, 5, 0]  # None means no noise
        
        # Define movement patterns to test
        movement_patterns = {
            "linear_x": {
                "description": "Linear movement along X axis",
                "starting_pos": [1.0, 3.0, 1.2],
                "velocity": [0.3, 0.0, 0.0]
            },
            "linear_y": {
                "description": "Linear movement along Y axis",
                "starting_pos": [2.5, 1.0, 1.2],
                "velocity": [0.0, 0.3, 0.0]
            },
            "circular": {
                "description": "Circular movement in XY plane",
                "setup": self._setup_circular_movement
            },
            "zigzag_3d": {
                "description": "Zigzag movement in 3D space",
                "setup": self._setup_zigzag_3d_movement
            }
        }
        
        # Start time measurement for full test suite
        test_start_time = time.time()
        
        # 1. Test with different microphone configurations
        print("\n1. TESTING MICROPHONE CONFIGURATIONS")
        print("------------------------------------")
        
        original_mics = self.mics.copy()  # Save original mic configuration
        
        for config_name, config in mic_configs.items():
            print(f"\nTesting {config_name}: {config['description']}")
            
            # Update microphone positions
            self.mics = [np.array(pos) for pos in config['positions']]
            
            # Update mic array in the room
            self.mic_array = np.array(self.mics).T
            self.room.mic_array.R = self.mic_array
            
            # Recalculate direct path delays
            self._update_direct_delays()
            
            # Run a simple test
            self._clear_targets()
            self._setup_linear_movement([2.5, 2.0, 1.2], [0.0, 0.4, 0.0])
            results = self._run_single_test(
                steps=20, 
                duration=4.0, 
                noise_snr=None, 
                test_name=f"mic_config_{config_name}"
            )
            
            # Store results
            all_results['mic_configs'][config_name] = results
        
        # Restore original mic configuration
        self.mics = original_mics
        self.mic_array = np.array(self.mics).T
        self.room.mic_array.R = self.mic_array
        self._update_direct_delays()
        
        # 2. Test with different SNR levels
        print("\n2. TESTING SNR LEVELS")
        print("--------------------")
        
        # Use the TV_X configuration as it should be most robust
        self.mics = [np.array(pos) for pos in mic_configs["tv_x"]["positions"]]
        self.mic_array = np.array(self.mics).T
        self.room.mic_array.R = self.mic_array
        self._update_direct_delays()
        
        for snr in snr_levels:
            snr_name = f"SNR_{snr}dB" if snr is not None else "No_Noise"
            print(f"\nTesting {snr_name}")
            
            # Run a simple test
            self._clear_targets()
            self._setup_linear_movement([2.5, 2.0, 1.2], [0.0, 0.4, 0.0])
            results = self._run_single_test(
                steps=20, 
                duration=4.0, 
                noise_snr=snr, 
                test_name=f"snr_{snr_name}"
            )
            
            # Store results
            all_results['snr_tests'][snr_name] = results
        
        # 3. Test with different movement patterns
        print("\n3. TESTING MOVEMENT PATTERNS")
        print("--------------------------")
        
        # Use the best configuration based on previous tests
        for pattern_name, pattern in movement_patterns.items():
            print(f"\nTesting {pattern_name}: {pattern['description']}")
            
            # Set up movement
            self._clear_targets()
            if "setup" in pattern:
                pattern["setup"]()
            else:
                self._setup_linear_movement(pattern["starting_pos"], pattern["velocity"])
            
            # Run a test with moderate noise
            results = self._run_single_test(
                steps=30, 
                duration=6.0, 
                noise_snr=10, 
                test_name=f"movement_{pattern_name}"
            )
            
            # Store results
            all_results['movement_tests'][pattern_name] = results
        
        # Restore original mic configuration
        self.mics = original_mics
        self.mic_array = np.array(self.mics).T
        self.room.mic_array.R = self.mic_array
        self._update_direct_delays()
        
        # Calculate and display overall test statistics
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        
        print("\n===== TEST RESULTS SUMMARY =====\n")
        print(f"Total test duration: {test_duration:.2f} seconds")
        
        # Generate summary plots
        self._generate_test_summary_plots(all_results, output_dir)
        
        # Generate comprehensive test report
        self._generate_test_report(all_results, output_dir)
        
        print(f"\nTest results saved to {output_dir}/")
        print("\n===== COMPREHENSIVE TESTS COMPLETE =====\n")
        
        return all_results

    def _generate_test_report(self, results, output_dir):
        """
        Generate comprehensive test report
        
        Args:
            results (dict): All test results
            output_dir (str): Output directory for report
        """
        # Create CSV report for each test category
        
        # 1. Microphone configuration report
        mic_data = []
        for config_name, result in results['mic_configs'].items():
            if result['is_valid'] and result['targets'] > 0:
                # Get metrics for first target
                metrics = result['error_metrics'][0]
                
                row = {
                    'Configuration': config_name,
                    'Mean Error (m)': metrics['filtered_mean_error'],
                    'Median Error (m)': metrics['filtered_median_error'],
                    'Max Error (m)': metrics['filtered_max_error'],
                    'X Error (m)': metrics['filtered_axis_mean'][0],
                    'Y Error (m)': metrics['filtered_axis_mean'][1],
                    'Z Error (m)': metrics['filtered_axis_mean'][2],
                    'Valid Measurements (%)': metrics['valid_percentage'],
                    'Computation Time (s)': result['computation_time']
                }
                mic_data.append(row)
        
        if mic_data:
            import pandas as pd
            df = pd.DataFrame(mic_data)
            df.to_csv(f"{output_dir}/mic_config_results.csv", index=False)
        
        # 2. SNR level report
        snr_data = []
        for snr_name, result in results['snr_tests'].items():
            if result['is_valid'] and result['targets'] > 0:
                # Get metrics for first target
                metrics = result['error_metrics'][0]
                
                row = {
                    'SNR Level': snr_name,
                    'Mean Error (m)': metrics['filtered_mean_error'],
                    'Median Error (m)': metrics['filtered_median_error'],
                    'Max Error (m)': metrics['filtered_max_error'],
                    'X Error (m)': metrics['filtered_axis_mean'][0],
                    'Y Error (m)': metrics['filtered_axis_mean'][1],
                    'Z Error (m)': metrics['filtered_axis_mean'][2],
                    'Valid Measurements (%)': metrics['valid_percentage'],
                    'Computation Time (s)': result['computation_time']
                }
                snr_data.append(row)
        
        if snr_data:
            import pandas as pd
            df = pd.DataFrame(snr_data)
            df.to_csv(f"{output_dir}/snr_results.csv", index=False)
        
        # 3. Movement pattern report
        movement_data = []
        for pattern_name, result in results['movement_tests'].items():
            if result['is_valid'] and result['targets'] > 0:
                # Get metrics for first target
                metrics = result['error_metrics'][0]
                
                row = {
                    'Movement Pattern': pattern_name,
                    'Mean Error (m)': metrics['filtered_mean_error'],
                    'Median Error (m)': metrics['filtered_median_error'],
                    'Max Error (m)': metrics['filtered_max_error'],
                    'X Error (m)': metrics['filtered_axis_mean'][0],
                    'Y Error (m)': metrics['filtered_axis_mean'][1],
                    'Z Error (m)': metrics['filtered_axis_mean'][2],
                    'Valid Measurements (%)': metrics['valid_percentage'],
                    'Computation Time (s)': result['computation_time']
                }
                movement_data.append(row)
        
        if movement_data:
            import pandas as pd
            df = pd.DataFrame(movement_data)
            df.to_csv(f"{output_dir}/movement_results.csv", index=False)
        
        # 4. Generate consolidated HTML report
        try:
            self._generate_html_report(results, output_dir)
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
        
        # Print summary of best configurations
        print("\nBest Configurations Summary:")
        
        if mic_data:
            best_mic = min(mic_data, key=lambda x: x['Mean Error (m)'])
            print(f"  Best microphone config: {best_mic['Configuration']} with mean error: {best_mic['Mean Error (m)']:.3f} m")
        
        if snr_data:
            valid_snr = [item for item in snr_data if item['Valid Measurements (%)'] > 50]
            if valid_snr:
                worst_tolerable_snr = min(valid_snr, key=lambda x: x['SNR Level'])
                print(f"  Worst tolerable SNR: {worst_tolerable_snr['SNR Level']} with valid measurements: {worst_tolerable_snr['Valid Measurements (%)']:.1f}%")
        
        if movement_data:
            challenging_movement = max(movement_data, key=lambda x: x['Mean Error (m)'])
            print(f"  Most challenging movement: {challenging_movement['Movement Pattern']} with mean error: {challenging_movement['Mean Error (m)']:.3f} m")

    def _visualize_ellipses(self, ellipses, step, output_dir="amt_debug_images/ellipses"):
        """Visualize the ellipses used for multilateration
        
        Args:
            ellipses (list): List of ellipse dictionaries
            step (int): Current step number
            output_dir (str): Directory to save images
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
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
            
            # Add suptitle to the figure
            fig.suptitle(f'Step {step}: Multilateration Ellipses', fontsize=16)
            
            # Save figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            filename = f"{output_dir}/ellipses_step{step:03d}.png"
            fig.savefig(filename, dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Error in _visualize_ellipses: {e}")

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
    tracker = AcousticTracker(debug_mode=True)
    
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
        tracker.analyze_position_shift(num_steps=15)
    except Exception as e:
        print(f"Error during constant shift analysis: {str(e)}")
    
    # Report total runtime
    total_time = time.time() - total_start_time
    print(f"\nDemo completed in {total_time:.2f} seconds")
    print(f"Debug images saved to '{output_dir}' directory")

# For installing required packages:
# pip install numpy matplotlib scipy pyroomacoustics filterpy

def main():
    """
    Main function to demonstrate the AMT3D system
    """
    print("Starting AMT3D: Acoustic Multi-target Tracking in 3D")
    
    # Create acoustic tracker with default room dimensions
    tracker = AcousticTracker(room_dim=(5.0, 6.0, 2.4), debug_mode=True)
    
    # Display system information
    print(f"Room dimensions: {tracker.room_dim}")
    print(f"Number of speakers: {len(tracker.speakers)}")
    print(f"Number of microphones: {len(tracker.mics)}")
    print(f"Speed of sound: {tracker.c} m/s")
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Run simple demo")
    print("2. Run comprehensive tests")
    print("3. Run position shift analysis")
    
    mode = input("Enter mode (1-3): ")
    
    if mode == "1":
        # Simple demo with targets moving in different patterns
        print("\nRunning simple demo...")
        
        # Add targets
        tracker.add_target(
            position=[1.0, 1.0, 1.2],
            velocity=[0.2, 0.2, 0.0],
            name="Target_1"
        )
        tracker.add_target(
            position=[4.0, 1.0, 1.0],
            velocity=[-0.1, 0.25, 0.05],
            name="Target_2"
        )
        
        # Run tracking
        tracker.run_tracking(duration=10.0, steps=50)
        
        # Visualize results
        fig = tracker.visualize()
        fig.savefig("amt_demo_results.png", dpi=300)
        print(f"Demo visualization saved to amt_demo_results.png")
        
        # Show visualization
        plt.show()
        
    elif mode == "2":
        # Run comprehensive tests
        print("\nRunning comprehensive tests (this may take some time)...")
        tracker.run_comprehensive_tests()
        
    elif mode == "3":
        # Run position shift analysis
        print("\nRunning position shift analysis...")
        
        # Add a target with known position for analysis
        tracker.add_target(
            position=[2.5, 3.0, 1.2],
            velocity=[0.0, 0.0, 0.0],
            name="Reference_Target"
        )
        
        # Analyze position shift
        diagnostics = tracker.analyze_position_shift(num_steps=15)
        
        if diagnostics['global']['has_consistent_shift']:
            correction = diagnostics['global']['correction_vector']
            print(f"Position shift detected: {-correction}")
            print(f"Consider applying this correction vector to calibrate your system")
        else:
            print("No consistent position shift detected")
    
    else:
        print("Invalid mode selected. Exiting.")
    
    print("\nAMT3D demonstration completed.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import os
    
    # Run the main function
    main()