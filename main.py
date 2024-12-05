import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Union, Dict, Any
from scipy.special import gammaln
from scipy.io import loadmat
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import threading
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
import time
from PIL import Image, ImageTk  # Import Pillow for image display

# --------------------------- Constants and Configurations --------------------------- #

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Visualization parameters
VISUALIZATION_PARAMS = {
    'subsample_neurons': True,
    'subsample_fraction': 0.2,
    'use_raster_plot': True,
    'use_heatmap': False,
    'add_jitter': True,
    'jitter_amount': 0.2
}

# SNN parameters
SNN_PARAMS = {
    'nrep': 5,
    'dt': 0.1,
    'gain': 1.5,
    'tau_m': 10.0,
    'tau_s': 5.0,
    'v_th': 1.0,
    'lr': 0.01,
    'stdp_window': 20.0
}

# --------------------------- Data Handling Module --------------------------- #

def load_data(file_path: str) -> np.ndarray:
    """
    Loads data from various file formats (.npy, .mat, images).

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        np.ndarray: Loaded data.
    """
    try:
        if file_path.lower().endswith('.npy'):
            data = np.load(file_path)
        elif file_path.lower().endswith('.mat'):
            mat_contents = loadmat(file_path)
            data_key = next((key for key in mat_contents.keys() if not key.startswith('__')), None)
            if data_key:
                data = mat_contents[data_key]
            else:
                raise ValueError("No valid data found in .mat file.")
        else:
            # Attempt to load as an image
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            data = np.array(img)
        logger.debug(f"Loaded data from {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


# --------------------------- Feature Extraction Module --------------------------- #

def extract_features(data: np.ndarray, feature_type: str) -> np.ndarray:
    """
    Extracts features from the input data.

    Parameters:
        data (np.ndarray): Input data (images or words).
        feature_type (str): Type of features to extract ('image' or 'word').

    Returns:
        np.ndarray: Extracted features.
    """
    if feature_type == 'image':
        logger.debug("Extracting image features")
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=100)  # Adjust the number of components as needed
        features = pca.fit_transform(data.reshape(data.shape[0], -1))
        return features
    elif feature_type == 'word':
        logger.debug("Extracting word features")
        # Use one-hot encoding for word features
        vocab = np.unique(data)
        word_to_index = {word: index for index, word in enumerate(vocab)}
        features = np.eye(len(vocab))[np.vectorize(word_to_index.get)(data)]
        return features
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")


# --------------------------- SNN Module --------------------------- #

class SNN:
    """
    Spiking Neural Network class with LIF neurons, STDP learning, and multiple layers.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initializes the SNN with given parameters.

        Parameters:
            parameters (dict): Dictionary of SNN parameters.
                - nrep (int): Number of repetitions.
                - dt (float): Time step.
                - gain (float): Gain factor.
                - tau_m (float): Membrane time constant.
                - tau_s (float): Synaptic time constant.
                - v_th (float): Firing threshold.
                - lr (float): Learning rate for STDP.
                - stdp_window (float): STDP time window.
                - hidden_layers (list): List of hidden layer sizes.
        """
        self.parameters = parameters
        self.weights = []
        self.n_neurons = []
        self.hidden_layers = parameters.get('hidden_layers', [100])  # Default to one hidden layer with 100 neurons

        # Initialize weights for each layer
        input_size = None  # Will be determined when loading data
        for layer_size in self.hidden_layers:
            if input_size is None:
                # First layer (input layer)
                input_size = 784  # Assuming MNIST-like input size, adjust as needed
            self.weights.append(np.random.rand(layer_size, input_size) * 0.1)  # Initialize with small random values
            self.n_neurons.append(layer_size)
            input_size = layer_size  # Output of this layer becomes input for the next

    def load_weights(self, weights_path: str):
        """
        Loads synaptic weights from a file.

        Parameters:
            weights_path (str): Path to the weights file.
        """
        try:
            loaded_weights = load_data(weights_path)
            if isinstance(loaded_weights, list) and len(loaded_weights) == len(self.hidden_layers):
                self.weights = loaded_weights
                self.n_neurons = [w.shape[0] for w in self.weights]
                logger.debug(f"Loaded weights from {weights_path}")
            else:
                raise ValueError("Loaded weights do not match the network architecture.")
        except Exception as e:
            logger.error(f"Error loading weights from {weights_path}: {e}")
            raise

    def record(self, features: np.ndarray) -> np.ndarray:
        """
        Records neuronal activity for given input features across all layers.

        Parameters:
            features (np.ndarray): Input features.

        Returns:
            np.ndarray: Recorded activity matrix for the output layer.
        """
        n_rep = self.parameters.get('nrep', 1)
        activity = np.zeros((self.n_neurons[-1], features.shape[1], n_rep))  # Activity of the output layer

        for n in range(n_rep):
            layer_input = features
            for i, layer_weights in enumerate(self.weights):
                layer_output = self.simulate_layer(layer_input, layer_weights)
                layer_input = layer_output  # Output of this layer becomes input for the next
            activity[:, :, n] = layer_output  # Store the output layer activity
            logger.debug(f"Repetition {n + 1}/{n_rep} completed.")

        logger.debug(f"Activity recorded with shape: {activity.shape}")
        return activity

    def simulate_layer(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Simulates a single layer of the SNN for a single repetition.

        Parameters:
            features (np.ndarray): Input features to the layer.
            weights (np.ndarray): Synaptic weights for the layer.

        Returns:
            np.ndarray: Spike counts for each neuron in the layer.
        """
        dt = self.parameters.get('dt', 1.0)
        tau_m = self.parameters.get('tau_m', 10.0)
        tau_s = self.parameters.get('tau_s', 5.0)
        v_th = self.parameters.get('v_th', 1.0)

        n_neurons = weights.shape[0]
        v = np.zeros((n_neurons, features.shape[1]))
        s = np.zeros_like(v)
        spikes = np.zeros_like(v)

        for t in range(features.shape[0]):
            # Calculate synaptic input
            I_syn = weights @ features[t, :]

            # Update membrane potential
            dv = (dt / tau_m) * (-v + I_syn)
            v += dv

            # Update synaptic currents
            ds = (dt / tau_s) * (-s + v)
            s += ds

            # Check for spikes
            fired = v >= v_th
            spikes[fired, :] += 1
            v[fired, :] = 0  # Reset membrane potential

        return spikes

    def train_stdp(self, features: np.ndarray):
        """
        Trains the SNN using Spike-Timing-Dependent Plasticity (STDP) across all layers.

        Parameters:
            features (np.ndarray): Input features.
        """
        dt = self.parameters.get('dt', 1.0)
        lr = self.parameters.get('lr', 0.01)
        stdp_window = self.parameters.get('stdp_window', 20.0)

        layer_input = features
        for i, layer_weights in enumerate(self.weights):
            # Get spike timings for each neuron and sample in the layer
            spike_times = []
            for j in range(layer_weights.shape[0]):
                spike_times_j = []
                for k in range(layer_input.shape[1]):
                    spike_times_j.append(np.where(self.simulate_layer(layer_input[:, k].reshape(-1, 1), layer_weights[j, :].reshape(1, -1))[:, 0] > 0)[0] * dt)
                spike_times.append(spike_times_j)

            # Apply STDP rule to the layer
            for j in range(layer_weights.shape[0]):
                for k in range(layer_input.shape[1]):
                    for l in range(layer_input.shape[1]):
                        if k != l:
                            times_j = spike_times[j][k]
                            times_l = spike_times[j][l]
                            for t_j in times_j:
                                for t_l in times_l:
                                    delta_t = t_j - t_l
                                    if abs(delta_t) <= stdp_window:
                                        if delta_t > 0:
                                            layer_weights[j, k] += lr * np.exp(-abs(delta_t) / stdp_window)
                                        else:
                                            layer_weights[j, k] -= lr * np.exp(-abs(delta_t) / stdp_window)

            # Normalize weights for the layer
            layer_weights = np.clip(layer_weights, 0, None)
            layer_weights /= np.max(layer_weights)
            self.weights[i] = layer_weights  # Update the weights in the network

            layer_input = self.simulate_layer(layer_input, layer_weights)  # Output of this layer becomes input for the next

        logger.debug("STDP training completed.")


# --------------------------- Analysis and Visualization Module --------------------------- #

def confusion_matrix(posterior_probs: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the confusion matrix.

    Parameters:
        posterior_probs (np.ndarray): Posterior probabilities.
        num_classes (int): Number of classes.

    Returns:
        tuple: Confusion matrix and indices of maximum posterior probabilities.
    """
    if posterior_probs.shape[0] != posterior_probs.shape[1]:
        raise ValueError("Posterior probabilities matrix must be square.")

    samples_per_class = posterior_probs.shape[1] // num_classes
    cm = np.zeros((num_classes, num_classes))
    ind = np.argmax(posterior_probs, axis=0)

    for true_label in range(num_classes):
        start = true_label * samples_per_class
        end = (true_label + 1) * samples_per_class
        predicted_labels = ind[start:end] // samples_per_class
        counts = np.bincount(predicted_labels, minlength=num_classes)
        cm[true_label] = counts / samples_per_class

    logger.debug(f"Confusion matrix calculated with shape: {cm.shape}")
    return cm, ind


def show_confusion_matrix(cm: np.ndarray):
    """
    Displays the confusion matrix.

    Parameters:
        cm (np.ndarray): Confusion matrix.
    """
    if not isinstance(cm, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    plt.figure(figsize=(8, 6))
    plt.imshow(100 * cm, cmap='jet')
    plt.xticks(np.arange(cm.shape[0]), labels=np.arange(cm.shape[0]))
    plt.yticks(np.arange(cm.shape[1]), labels=np.arange(cm.shape[1]))
    plt.xlabel('Decoded Category')
    plt.ylabel('Presented Category')
    cbar = plt.colorbar()
    cbar.set_label('Frequency (%)')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def log_likelihood(activity: np.ndarray, features: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
    """
    Computes the log-likelihood.

    Parameters:
        activity (np.ndarray): Neuronal activity.
        features (np.ndarray): Input features.
        parameters (dict): SNN parameters.

    Returns:
        np.ndarray: Log-likelihood matrix.
    """
    mean_activity = np.ceil(np.mean(activity, axis=2))
    num_samples = features.shape[1]

    if num_samples > 1000:
        nn = NearestNeighbors(n_neighbors=num_samples, metric='euclidean', algorithm='auto')
        nn.fit(mean_activity.T)
        distances, _ = nn.kneighbors(mean_activity.T)
        ll = -distances
    else:
        distances = cdist(mean_activity.T, mean_activity.T, metric='euclidean')
        sigma = parameters.get('sigma', 1.0)  # Gaussian kernel width
        ll = np.exp(-0.5 * (distances / sigma) ** 2)

    logger.debug("Log-likelihood calculated.")
    return ll


def posterior_probabilities(ll: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """
    Calculates posterior probabilities.

    Parameters:
        ll (np.ndarray): Log-likelihood matrix.
        log_prior (np.ndarray): Log prior probabilities.

    Returns:
        np.ndarray: Posterior probabilities.
    """
    if ll.shape[0] != ll.shape[1]:
        raise ValueError("Log-likelihood matrix must be square.")
    if log_prior.shape[0] != ll.shape[0]:
        raise ValueError("Log prior dimensions mismatch.")

    log_posterior = ll + log_prior[:, np.newaxis]

    # Log-sum-exp trick for numerical stability
    max_log_posterior = np.max(log_posterior, axis=0, keepdims=True)
    posterior = np.exp(log_posterior - max_log_posterior)
    posterior = posterior / posterior.sum(axis=0, keepdims=True)

    logger.debug("Posterior probabilities calculated.")
    return posterior


def posterior_averaged_images(images: np.ndarray, posterior: np.ndarray, navg: int) -> np.ndarray:
    """
    Computes posterior-averaged images.

    Parameters:
        images (np.ndarray): Input images.
        posterior (np.ndarray): Posterior probabilities.
        navg (int): Number of top images to average.

    Returns:
        np.ndarray: Posterior-averaged images.
    """
    if images.shape[1] != posterior.shape[0]:
        raise ValueError("Images and posterior dimensions mismatch.")
    if not (1 <= navg <= posterior.shape[1]):
        raise ValueError("Invalid navg value.")

    num_samples = images.shape[1]
    pa = np.zeros_like(images, dtype=np.float64)
    sorted_posterior = np.sort(posterior, axis=0)[::-1]
    indices = np.argsort(posterior, axis=0)[::-1]

    for i in range(num_samples):
        top_indices = indices[:navg, i]
        top_values = sorted_posterior[:navg, i]
        top_images = images[:, top_indices]
        pa[:, i] = np.sum(top_images * top_values, axis=1)
        max_value = pa[:, i].max()
        if max_value > 0:
            pa[:, i] /= max_value

    logger.debug("Posterior-averaged images computed.")
    return pa


def log_prior(class_prior: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Calculates log prior probabilities.

    Parameters:
        class_prior (np.ndarray): Class prior probabilities.
        num_samples (int): Total number of samples.

    Returns:
        np.ndarray: Log prior probabilities.
    """
    if not isinstance(class_prior, np.ndarray) or not np.isclose(class_prior.sum(), 1) or np.any(class_prior == 0):
        raise ValueError("Invalid class prior probabilities.")

    num_classes = len(class_prior)
    samples_per_class = num_samples // num_classes
    log_prior = np.zeros(num_samples)

    for c in range(num_classes):
        start = c * samples_per_class
        end = (c + 1) * samples_per_class
        log_prior[start:end] = np.log(class_prior[c] / samples_per_class)

    logger.debug("Log prior probabilities calculated.")
    return log_prior


def visualize_spikes(spikes: np.ndarray, ax: plt.Axes, params: Dict[str, Any] = VISUALIZATION_PARAMS):
    """
    Visualizes spike trains with various options.

    Parameters:
        spikes (np.ndarray): Spike data (neurons x timesteps).
        ax (plt.Axes): Matplotlib Axes object.
        params (dict): Visualization parameters.
    """
    ax.clear()
    num_neurons, timesteps = spikes.shape

    subsample_neurons = params.get('subsample_neurons', True)
    subsample_fraction = params.get('subsample_fraction', 0.2)
    use_raster_plot = params.get('use_raster_plot', True)
    use_heatmap = params.get('use_heatmap', False)
    add_jitter = params.get('add_jitter', True)
    jitter_amount = params.get('jitter_amount', 0.2)

    if subsample_neurons:
        neurons_to_plot = np.random.choice(num_neurons, int(subsample_fraction * num_neurons), replace=False)
    else:
        neurons_to_plot = np.arange(num_neurons)

    if use_heatmap:
        heatmap_data = spikes[neurons_to_plot, :]
        im = ax.imshow(heatmap_data, cmap='hot', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron Index')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Spike Count')
        ax.set_title('Spike Heatmap')
    else:
        for neuron_idx in neurons_to_plot:
            spike_times = np.where(spikes[neuron_idx, :] == 1)[0]
            if add_jitter:
                y_pos = np.full_like(spike_times, neuron_idx) + \
                        np.random.uniform(-jitter_amount, jitter_amount, size=len(spike_times))
            else:
                y_pos = np.full_like(spike_times, neuron_idx)

            if use_raster_plot:
                ax.plot(spike_times, y_pos, '.', color='black', markersize=2)
            else:
                ax.plot(spike_times, y_pos, '|k', markersize=8, alpha=0.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron Index')
        ax.set_title('Spike Raster Plot' if use_raster_plot else 'Spike Visualization (Subsampled)')

    ax.set_xlim(0, timesteps)
    ax.set_ylim(-1, num_neurons)


def visualize_spikes_over_time(spikes: np.ndarray, ax: plt.Axes, time_window: int = None):
    """
    Visualizes the total number of spikes over time with an optional rolling time window.

    Parameters:
        spikes (np.ndarray): Spike data (neurons x timesteps).
        ax (plt.Axes): Matplotlib Axes object for the plot.
        time_window (int): Optional rolling time window for smoothing the spike counts.
    """
    ax.clear()
    total_spikes_over_time = np.sum(spikes, axis=0)

    if time_window:
        # Apply a rolling window to smooth the spike counts
        window = np.ones(time_window) / time_window
        total_spikes_over_time = np.convolve(total_spikes_over_time, window, mode='same')

    # Create the initial plot
    line, = ax.plot(total_spikes_over_time)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Spikes')
    ax.set_title('Total Spikes Over Time')

    # Update function for the animation
    def update(frame):
        line.set_data(np.arange(frame), total_spikes_over_time[:frame])
        return line,

    # Create the animation
    ani = FuncAnimation(ax.figure, update, frames=len(total_spikes_over_time), interval=50, blit=True)


# --------------------------- GUI Module --------------------------- #

class SNNApp:
    """
    GUI application for the Spiking Neural Network processor.
    """

    def __init__(self, master):
        """
        Initializes the GUI application.
        """
        self.master = master
        master.title("Spiking Neural Network Processor")

        self.images_path = tk.StringVar()
        self.weights_path = tk.StringVar()
        self.params = SNN_PARAMS.copy()  # Use a copy of the default parameters
        self.time_window = tk.IntVar(value=0)  # Variable for the time window
        self.snn = None

        self.create_widgets()
        self.setup_logging()

    def create_widgets(self):
        """
        Creates and arranges the GUI widgets.
        """
        # File Selection Frame
        file_frame = ttk.LabelFrame(self.master, text="Select Input Files")
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(file_frame, text="Images File (.npy/.mat/image):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.images_entry = ttk.Entry(file_frame, textvariable=self.images_path, width=50)
        self.images_entry.grid(row=0, column=1, padx=5, pady=5)
        self.images_button = ttk.Button(file_frame, text="Browse", command=self.browse_images)
        self.images_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="Weights File (.npy/.mat):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.weights_entry = ttk.Entry(file_frame, textvariable=self.weights_path, width=50)
        self.weights_entry.grid(row=1, column=1, padx=5, pady=5)
        self.weights_button = ttk.Button(file_frame, text="Browse", command=self.browse_weights)
        self.weights_button.grid(row=1, column=2, padx=5, pady=5)

        # Parameters Frame
        params_frame = ttk.LabelFrame(self.master, text="Set Parameters")
        params_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Create entry fields for each parameter in SNN_PARAMS
        self.param_entries = {}
        for i, (param_name, param_value) in enumerate(self.params.items()):
            ttk.Label(params_frame, text=f"{param_name.replace('_', ' ').title()}:").grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(params_frame, width=10)
            entry.insert(0, str(param_value))
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.param_entries[param_name] = entry

        # Execution Frame
        exec_frame = ttk.LabelFrame(self.master, text="Execute Processing")
        exec_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.execute_button = ttk.Button(exec_frame, text="Run SNN Processing", command=self.run_processing)
        self.execute_button.grid(row=0, column=0, padx=5, pady=5)

        self.progress = ttk.Progressbar(exec_frame, orient='horizontal', mode='determinate', length=400)
        self.progress.grid(row=0, column=1, padx=5, pady=5)

        # Logs Frame
        logs_frame = ttk.LabelFrame(self.master, text="Logs")
        logs_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        self.log_text = tk.Text(logs_frame, height=10, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(logs_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Visualization Frame
        viz_frame = ttk.LabelFrame(self.master, text="Spike Visualization")
        viz_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Time Window Control
        time_window_frame = ttk.LabelFrame(self.master, text="Time Window")
        time_window_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(time_window_frame, text="Window Size:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.time_window_entry = ttk.Entry(time_window_frame, textvariable=self.time_window, width=10)
        self.time_window_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.set_time_window_button = ttk.Button(time_window_frame, text="Set Time Window", command=self.set_time_window)
        self.set_time_window_button.grid(row=0, column=2, padx=5, pady=5)

        # Image Display Frame
        image_frame = ttk.LabelFrame(self.master, text="Input Image")
        image_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        self.image_label = ttk.Label(image_frame)
        self.image_label.pack()

        # Configure grid weights for resizing
        self.master.grid_rowconfigure(4, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

    def browse_images(self):
        """
        Opens a file dialog for the user to select the images file.
        """
        file_path = filedialog.askopenfilename(
            title="Select Images File",
            filetypes=[("All Files", "*.*")]
        )
        if file_path:
            self.images_path.set(file_path)
            logger.debug(f"Selected images file: {file_path}")

            # Display the selected image
            try:
                img = Image.open(file_path)
                img.thumbnail((200, 200))  # Resize the image for display
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo  # Keep a reference to avoid garbage collection
            except Exception as e:
                logger.error(f"Error displaying image: {e}")

    def browse_weights(self):
        """
        Opens a file dialog for the user to select the weights file.
        """
        file_path = filedialog.askopenfilename(
            title="Select Weights File",
            filetypes=[("All Files", "*.*")]
        )
        if file_path:
            self.weights_path.set(file_path)
            logger.debug(f"Selected weights file: {file_path}")

    def setup_logging(self):
        """
        Redirects logger output to the text widget in the GUI.
        """

        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record) + '\n'
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg)
                self.text_widget.configure(state='disabled')
                self.text_widget.see(tk.END)

        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)

    def run_processing(self):
        """
        Starts the SNN processing in a separate thread.
        """
        self.execute_button.config(state='disabled')

        # Update parameters from the entry fields
        for param_name, entry in self.param_entries.items():
            try:
                param_value = float(entry.get())  # Convert to float
                self.params[param_name] = param_value
            except ValueError:
                messagebox.showerror("Invalid Input", f"Invalid value for {param_name}. Please enter a number.")
                self.execute_button.config(state='normal')
                return

        processing_thread = threading.Thread(target=self.process_snn)
        processing_thread.start()

    def process_snn(self):
        """
        Executes the SNN processing steps.
        """
        try:
            images_path = self.images_path.get()
            weights_path = self.weights_path.get()
            navg = self.params['navg']  # Get navg from the updated parameters
            num_classes = self.params['num_classes']  # Get num_classes from the updated parameters

            if not images_path or not weights_path:
                messagebox.showerror("Input Error", "Please select both images and weights files.")
                logger.error("Missing input files.")
                self.execute_button.config(state='normal')
                return

            self.progress['value'] = 0
            self.master.update_idletasks()

            self.snn = SNN(self.params)  # Create SNN with updated parameters
            self.snn.load_weights(weights_path)

            logger.info("Loading and preprocessing image data...")
            images_data = load_data(images_path)
            images_features = extract_features(images_data, feature_type='image')
            self.progress['value'] = 20
            self.master.update_idletasks()

            logger.info("Recording neuronal activity...")
            activity = self.snn.record(images_features)
            logger.info(f"Activity recorded with shape: {activity.shape}")
            self.progress['value'] = 40
            self.master.update_idletasks()

            # Visualize spikes (first repetition)
            self.visualize_spikes(activity[:, :, 0])
            self.progress['value'] = 60
            self.master.update_idletasks()

            logger.info("Computing log-likelihood...")
            LL = log_likelihood(activity, images_features, self.params)  # Use updated parameters
            logger.info(f"Log-likelihood matrix computed with shape: {LL.shape}")
            self.progress['value'] = 70
            self.master.update_idletasks()

            # Define class prior probabilities
            class_prior = np.array([1 / num_classes] * num_classes)

            logger.info("Calculating log prior probabilities...")
            LP = log_prior(class_prior, num_samples=LL.shape[0])
            logger.info(f"Log prior probabilities calculated with shape: {LP.shape}")
            self.progress['value'] = 80
            self.master.update_idletasks()

            logger.info("Calculating posterior probabilities...")
            POS = posterior_probabilities(LL, LP)
            logger.info(f"Posterior probabilities calculated with shape: {POS.shape}")
            self.progress['value'] = 90
            self.master.update_idletasks()

            logger.info("Calculating confusion matrix...")
            CM, IND = confusion_matrix(POS, num_classes=num_classes)
            logger.info(f"Confusion matrix calculated with shape: {CM.shape}")

            # Show confusion matrix (optional)
            # show_confusion_matrix(CM)

            logger.info("Computing posterior-averaged images...")
            PA = posterior_averaged_images(images=images_data, posterior=POS, navg=navg)
            logger.info(f"Posterior-averaged images computed with shape: {PA.shape}")
            self.progress['value'] = 100
            self.master.update_idletasks()

            messagebox.showinfo("Processing Complete", "SNN processing completed successfully.")

        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            messagebox.showerror("Processing Error", f"An error occurred: {ve}")
        except Exception as e:
            logger.error(f"Exception: {e}")
            messagebox.showerror("Processing Error", f"An unexpected error occurred: {e}")
        finally:
            self.execute_button.config(state='normal')

    def visualize_spikes(self, spikes: np.ndarray):
        """
        Visualizes the spike trains in two subplots.

        Parameters:
            spikes (np.ndarray): Spikes matrix (neurons x timesteps).
        """
        visualize_spikes(spikes, self.axs[0])
        visualize_spikes_over_time(spikes, self.axs[1], time_window=self.time_window.get())
        self.canvas.draw()

    def set_time_window(self):
        """
        Gets the time window size from the user and updates the visualization.
        """
        try:
            time_window = simpledialog.askinteger("Time Window", "Enter time window size:")
            if time_window is not None and time_window > 0:
                self.time_window.set(time_window)
                if self.snn is not None:
                    self.visualize_spikes(self.snn.record(extract_features(load_data(self.images_path.get()), 'image'))[:, :, 0])
            else:
                messagebox.showwarning("Invalid Input", "Please enter a positive integer for the time window size.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


# --------------------------- Main Execution --------------------------- #

def main():
    """
    Initializes and runs the SNN processing GUI application.
    """
    root = tk.Tk()
    app = SNNApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
