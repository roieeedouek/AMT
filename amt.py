import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy import signal
import pyroomacoustics as pra
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

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
        self.fs = 96000  # Increased sampling rate for better resolution
        
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
        """Set up the room, speakers, and microphones"""
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
            np.array([width/2, 0.3, 1.0]),     # Center
            np.array([width - 0.2, 0.5, 1.0]), # Front Right
            # Surround speakers
            np.array([0.5, length - 0.5, 1.2]),         # Surround Left
            np.array([width - 0.5, length - 0.5, 1.2]), # Surround Right
        ]
        
        # Microphone positions - soundbar-like arrangement at the front
        self.mics = [
            np.array([width/2 - 0.4, 0.3, 1.0]),  # Left Mic
            np.array([width/2 + 0.4, 0.3, 1.0]),  # Right Mic
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
        """Initialize a Kalman filter for a target
        
        Args:
            target_id (int): Target ID
            pos (np.array): Initial position
            dt (float): Time step in seconds
            
        Returns:
            KalmanFilter: Initialized Kalman filter
        """
        # We'll track 3D position and velocity: [x, y, z, vx, vy, vz]
        dim_x = 6  # State dimension
        dim_z = 3  # Measurement dimension (x, y, z position)
        
        # Create Kalman filter
        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # State transition matrix (pos += vel * dt)
        kf.F = np.eye(dim_x)
        kf.F[:3, 3:] = np.eye(3) * dt
        
        # Measurement function (we only measure position)
        kf.H = np.zeros((dim_z, dim_x))
        kf.H[:3, :3] = np.eye(3)
        
        # Covariance matrices - tuned for better precision
        # Lower process noise for more stable tracking
        q = 0.001  # Reduced for smoother tracking
        q_dim = block_diag(q, q, q, q*10, q*10, q*10)  # More noise for velocity
        kf.Q = q_dim
        
        # Reduced measurement noise for higher weight on measurements
        r = 0.05  # Reduced for more precision when measurements are good
        kf.R = np.eye(dim_z) * r
        
        # Initial state
        kf.x = np.zeros(dim_x)
        kf.x[:3] = pos  # Position
        if target_id < len(self.targets):
            kf.x[3:] = self.targets[target_id]['velocity']  # Velocity
        
        # Initial covariance
        kf.P = np.eye(dim_x) * 0.1
        
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
    
    def locate_targets(self, echo_data, dt=0.1):
        """Locate targets using multilateration from detected echoes with Kalman filtering
        
        Args:
            echo_data (dict): Echo detection data
            dt (float): Time step in seconds
            
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
                position = self.multilateration_twostage(target_ellipses[i])
                
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
    
    def _compute_error(self, point, ellipses):
        """Helper function to compute error for a point
        
        Args:
            point (np.array): Position to evaluate
            ellipses (list): List of ellipses
            
        Returns:
            float: Total error
        """
        total_error = 0
        for ellipse in ellipses:
            speaker_to_point = np.linalg.norm(point - ellipse['speaker_pos'])
            point_to_mic = np.linalg.norm(ellipse['mic_pos'] - point)
            computed_path = speaker_to_point + point_to_mic
            error = (computed_path - ellipse['path_length'])**2
            total_error += error
        return total_error
    
    def multilateration_twostage(self, ellipses, coarse_res=20, fine_res=50):
        """Two-stage multilateration for higher resolution
        
        Args:
            ellipses (list): List of ellipse dictionaries
            coarse_res (int): Resolution for coarse search
            fine_res (int): Resolution for fine search
            
        Returns:
            np.array: Estimated position
        """
        if not ellipses:
            return None
            
        # Stage 1: Coarse search
        x = np.linspace(0, self.room_dim[0], coarse_res)
        y = np.linspace(0, self.room_dim[1], coarse_res)
        z = np.linspace(0, self.room_dim[2], coarse_res)
        
        best_point = None
        min_error = float('inf')
        
        # Search on coarse grid
        for xi in x:
            for yi in y:
                for zi in z:
                    point = np.array([xi, yi, zi])
                    total_error = self._compute_error(point, ellipses)
                    
                    if total_error < min_error:
                        min_error = total_error
                        best_point = point.copy()
        
        if best_point is None:
            return None
        
        # Stage 2: Fine search around best point
        # Define search region around best point
        x_range = max(0.5, self.room_dim[0] / coarse_res)  # Size of search region
        y_range = max(0.5, self.room_dim[1] / coarse_res)
        z_range = max(0.5, self.room_dim[2] / coarse_res)
        
        x = np.linspace(max(0, best_point[0] - x_range/2), 
                       min(self.room_dim[0], best_point[0] + x_range/2), fine_res)
        y = np.linspace(max(0, best_point[1] - y_range/2),
                       min(self.room_dim[1], best_point[1] + y_range/2), fine_res)
        z = np.linspace(max(0, best_point[2] - z_range/2),
                       min(self.room_dim[2], best_point[2] + z_range/2), fine_res)
        
        # Search on fine grid
        for xi in x:
            for yi in y:
                for zi in z:
                    point = np.array([xi, yi, zi])
                    total_error = self._compute_error(point, ellipses)
                    
                    if total_error < min_error:
                        min_error = total_error
                        best_point = point.copy()
        
        return best_point
    
    def run_tracking(self, duration=5.0, steps=50):
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
        
        for step in range(steps):
            print(f"Processing step {step+1}/{steps}...")
            
            # Update target positions
            self.update_targets(dt)
            
            # Simulate propagation and echoes
            received_signals = self.simulate_echoes()
            
            # Detect echoes
            echo_data = self.detect_echoes(received_signals)
            
            # Store echo data for visualization
            self.correlation_history.append(echo_data)
            
            # Locate targets with Kalman filtering
            estimated_positions, filtered_positions = self.locate_targets(echo_data, dt)
        
        return self.tracking_data
    
    def visualize(self, use_filtered=True):
        """Visualize the room, speakers, microphones, and targets
        
        Args:
            use_filtered (bool): Whether to show filtered positions (True) or raw estimated positions (False)
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
    
    def visualize_correlation(self, step=0):
        """Visualize the correlation and MTI filtering results for a specific step
        
        Args:
            step (int): Step to visualize
        """
        if not hasattr(self, 'correlation_history') or not self.correlation_history:
            print("No correlation history available. Run tracking first.")
            return
        
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
        
        fig.suptitle(f"Step {step}: Correlation and MTI Results")
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
    
    def compare_tracking(self):
        """Compare raw estimated vs Kalman filtered tracking"""
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
            for idx, pos in est_pos:
                est_errors.append(np.linalg.norm(pos - true_pos[idx]))
            
            filt_errors = []
            for i in range(len(filt_pos)):
                filt_errors.append(np.linalg.norm(filt_pos[i] - true_pos[i]))
            
            # Print metrics
            print(f"\n{target['name']}:")
            if est_errors:
                print(f"  Raw estimation - Mean error: {np.mean(est_errors):.3f} m, Max error: {np.max(est_errors):.3f} m")
                print(f"  Raw estimation - Available measurements: {len(est_errors)}/{len(true_pos)} ({len(est_errors)/len(true_pos)*100:.1f}%)")
            else:
                print("  Raw estimation - No valid measurements")
                
            print(f"  Kalman filter  - Mean error: {np.mean(filt_errors):.3f} m, Max error: {np.max(filt_errors):.3f} m")
            print(f"  Kalman filter  - Improvement: {np.mean(est_errors)/np.mean(filt_errors) if est_errors else 'N/A':.2f}x")


    def visualize_correlation_animation(self, interval=200):
        """Create an animation of the correlation and MTI filtering results
        
        Args:
            interval (int): Animation interval in milliseconds
            
        Returns:
            animation.FuncAnimation: Animation object
        """
        if not hasattr(self, 'correlation_history') or not self.correlation_history:
            print("No correlation history available. Run tracking first.")
            return None
            
        # Create figure and subplots
        n_pairs = len(self.correlation_history[0])
        fig, axs = plt.subplots(n_pairs, 2, figsize=(14, 3*n_pairs))
        
        # If only one pair, make sure axs is 2D
        if n_pairs == 1:
            axs = np.array([axs])
            
        # Initialize plots
        lines = []
        peak_plots = []
        
        for i, (s_idx, m_idx) in enumerate(self.correlation_history[0].keys()):
            # Correlation plot
            line, = axs[i, 0].plot([], [])
            lines.append(line)
            peak_plot = axs[i, 0].plot([], [], 'rx')[0]
            peak_plots.append(peak_plot)
            
            # Direct path line
            direct_delay = self.direct_delays[(s_idx, m_idx)]
            axs[i, 0].axvline(x=direct_delay, color='r', linestyle='--')
            
            # Difference plot
            line, = axs[i, 1].plot([], [])
            lines.append(line)
            peak_plot = axs[i, 1].plot([], [], 'rx')[0]
            peak_plots.append(peak_plot)
            
            # Labels
            axs[i, 0].set_title(f"Correlation: Speaker {s_idx+1} -> Mic {m_idx+1}")
            axs[i, 1].set_title(f"MTI Filtered: Speaker {s_idx+1} -> Mic {m_idx+1}")
            
            # Set y limits
            axs[i, 0].set_ylim(0, 1.1)
            axs[i, 1].set_ylim(0, 1.1)
        
        # Animation update function
        def update(frame):
            if frame >= len(self.correlation_history):
                return lines + peak_plots
                
            echo_data = self.correlation_history[frame]
            
            line_idx = 0
            peak_idx = 0
            
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
                
                line_idx += 1
                peak_idx += 1
            
            # Update title
            fig.suptitle(f"Step {frame}: Correlation and MTI Results")
            
            return lines + peak_plots
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.correlation_history),
            interval=interval, blit=False
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return anim

  
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


def run_demo():
    """Run a demonstration of the high-resolution acoustic tracking system"""
    # Create tracker
    tracker = AcousticTracker()
    
    # Add targets with perpendicular trajectories (to better demonstrate tracking)
    tracker.add_target((2.5, 3.0, 1.7), velocity=(0.3, 0, 0), name="Person 1")
    tracker.add_target((1.5, 4.0, 1.6), velocity=(0, 0.25, 0), name="Person 2")
    
    # Run tracking simulation
    tracker.run_tracking(duration=5.0, steps=40)  # More steps for smoother tracking
    
    # Visualize correlation for an early and late frame
    print("\nVisualization of correlation and MTI for frame 5:")
    tracker.visualize_correlation(step=5)
    
    print("\nVisualization of correlation and MTI for frame 30:")
    tracker.visualize_correlation(step=30)

    print("\nVisualization of correlation and MTI filtering animation:")
    anim = tracker.visualize_correlation_animation()
    
    # Compare raw estimation vs Kalman filtered tracking
    print("\nComparing raw estimated vs Kalman filtered tracking:")
    tracker.compare_tracking()
    
    # Visualize results with Kalman filtering
    print("\nVisualization of Kalman filtered tracking results:")
    tracker.visualize(use_filtered=True)


# For installing required packages:
# pip install numpy matplotlib scipy pyroomacoustics filterpy

if __name__ == "__main__":
    run_demo()