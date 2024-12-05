import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import logging  # For logging debug and error messages
from typing import Tuple, Union  # For type annotations
from scipy.special import gammaln  # For log-factorial calculations in Poisson PMF
from scipy.io import loadmat  # For loading MATLAB .mat files
import os  # For file path operations
import tkinter as tk  # For creating the GUI
from tkinter import filedialog, messagebox  # For file dialogs and message boxes
from tkinter import ttk  # For advanced GUI widgets like progress bars
import threading  # For running tasks in the background
from scipy.spatial.distance import cdist  # For efficient distance calculations
from sklearn.neighbors import NearestNeighbors  # For approximate nearest neighbors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding matplotlib plots in Tkinter


# --------------------------- Logger Configuration --------------------------- #

# Configure the logger to display INFO-level messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Create a logger for this module


# --------------------------- Data Loading Function --------------------------- #

def load_data(file_path: str) -> np.ndarray:
    """
    Loads data from a .npy or .mat file.

    Parameters:
        file_path (str): Path to the .npy or .mat file.

    Returns:
        np.ndarray: Loaded data as a NumPy array.

    Raises:
        ValueError: If the file extension is not .npy or .mat, or if loading fails.
    """
    logger.debug(f"Attempting to load data from {file_path}")

    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.npy':
        try:
            data = np.load(file_path)
            logger.debug(f"Loaded .npy file: {file_path} with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load .npy file: {file_path}. Error: {e}")
            raise ValueError(f"Failed to load .npy file: {file_path}. Error: {e}")
    elif ext == '.mat':
        try:
            mat_contents = loadmat(file_path)
            data_keys = [key for key in mat_contents.keys() if not key.startswith('__')]
            if not data_keys:
                logger.error(f"No valid variables found in .mat file: {file_path}")
                raise ValueError(f"No valid variables found in .mat file: {file_path}")
            if len(data_keys) > 1:
                logger.warning(f"Multiple variables found in .mat file: {file_path}. Using the first variable: {data_keys[0]}")
            data = mat_contents[data_keys[0]]
            logger.debug(f"Loaded .mat file: {file_path} variable '{data_keys[0]}' with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load .mat file: {file_path}. Error: {e}")
            raise ValueError(f"Failed to load .mat file: {file_path}. Error: {e}")
    else:
        logger.error(f"Unsupported file extension: {ext}. Only .npy and .mat are supported.")
        raise ValueError(f"Unsupported file extension: {ext}. Only .npy and .mat are supported.")


# --------------------------- Feature Extraction Functions --------------------------- #

def extract_image_features(images: np.ndarray) -> np.ndarray:
    """
    Extracts features from images (replace this with your chosen method).

    This is a placeholder function. Replace it with your actual feature extraction method.
    For example, you can use:
    - Histogram of Oriented Gradients (HOG)
    - Local Binary Patterns (LBP)
    - A simple convolutional layer

    Parameters:
        images (np.ndarray): Input images.

    Returns:
        np.ndarray: Extracted image features.
    """
    logger.debug("Extracting image features")
    # Placeholder: Replace with your actual feature extraction method
    # Here, we simply flatten the images as a basic example
    return images.reshape(images.shape[0], -1)


def extract_word_features(words: np.ndarray) -> np.ndarray:
    """
    Extracts features from words (replace this with your chosen method).

    This is a placeholder function. Replace it with your actual word embedding method.
    For example, you can use:
    - One-hot encoding
    - Word embeddings like Word2Vec or GloVe

    Parameters:
        words (np.ndarray): Input words.

    Returns:
        np.ndarray: Extracted word features.
    """
    logger.debug("Extracting word features")
    # Placeholder: Replace with your actual word embedding method
    # Here, we use a simple one-hot encoding as an example
    vocab = np.unique(words)
    word_to_index = {word: index for index, word in enumerate(vocab)}
    features = np.eye(len(vocab))[np.vectorize(word_to_index.get)(words)]
    return features


# --------------------------- Neural Network Processing Functions --------------------------- #

class SNN:
    """
    Spiking Neural Network class.
    """

    def __init__(self, parameters: dict):
        """
        Initializes the SNN with the given parameters.

        Parameters:
            parameters (dict): Dictionary containing SNN parameters.
        """
        logger.debug("Initializing SNN")
        self.parameters = parameters
        self.weights = None  # Initialize weights (to be loaded later)

    def load_weights(self, weights_path: str):
        """
        Loads synaptic weights from a file.

        Parameters:
            weights_path (str): Path to the weights file (.npy or .mat).
        """
        logger.debug(f"Loading weights from {weights_path}")
        self.weights = load_data(weights_path)

    def record(self, images: np.ndarray) -> np.ndarray:
        """
        Records neuronal activity.

        Parameters:
            images (np.ndarray): Input images (or word features).

        Returns:
            np.ndarray: Recorded activity matrix.
        """
        logger.debug("Starting to record neuronal activity")

        if self.weights is None:
            logger.error("Weights not loaded. Please load weights first.")
            raise ValueError("Weights not loaded.")

        n_rep = self.parameters.get('nrep', 1)
        activity = np.zeros((self.weights.shape[0], images.shape[1], n_rep))
        logger.debug(f"Initialized activity matrix with shape: {activity.shape}")

        for n in range(n_rep):
            activity[:, :, n] = self.spikes(images)
            non_zero = np.count_nonzero(activity[:, :, n])
            logger.debug(f"Repetition {n + 1}/{n_rep}. Non-zero activity: {non_zero}")

        logger.debug(f"Activity matrix recorded. Shape: {activity.shape}")
        return activity

    def spikes(self, images: np.ndarray) -> np.ndarray:
        """
        Generates spikes for the neuron population.

        Parameters:
            images (np.ndarray): Input images (or word features).

        Returns:
            np.ndarray: Spikes matrix.
        """
        logger.debug("Generating spikes for the neuron population")

        R = self.rates(images)
        dt = self.parameters.get('dt', 1.0)
        S = np.random.poisson(dt * R)

        logger.debug(f"Spikes matrix generated. Max: {np.max(S)}, Min: {np.min(S)}, Non-zero spikes: {np.count_nonzero(S)}")
        return S

    def rates(self, images: np.ndarray) -> np.ndarray:
        """
        Computes the firing rates.

        Parameters:
            images (np.ndarray): Input images (or word features).

        Returns:
            np.ndarray: Firing rates.
        """
        logger.debug("Starting rates calculation")

        R = self.weights @ images
        gain = self.parameters.get('gain', 1.0)
        R = linrectify(R) * gain

        logger.debug(f"Firing rates (R) calculated. Non-zero elements: {np.count_nonzero(R)}")
        return R


def linrectify(X: np.ndarray) -> np.ndarray:
    """
    Applies linear rectification to the input data.

    Parameters:
        X (np.ndarray): Input data.

    Returns:
        np.ndarray: Rectified data.
    """
    logger.debug("Applying linear rectification")
    return np.maximum(X, 0)


def confusion(POS: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the confusion matrix.

    Parameters:
        POS (np.ndarray): Posterior probabilities matrix.
        num_classes (int): Number of classes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Confusion matrix and indices of maximum posterior probabilities.
    """
    logger.debug("Starting confusion matrix calculation")

    if POS.shape[0] != POS.shape[1]:
        logger.error(f"POS must be a square array, but got {POS.shape}")
        raise ValueError("Invalid POS shape")

    samples_per_class = POS.shape[1] // num_classes
    CM = np.zeros((num_classes, num_classes))
    IND = np.argmax(POS, axis=0)

    for true_label in range(num_classes):
        start = true_label * samples_per_class
        end = (true_label + 1) * samples_per_class
        predicted_labels = IND[start:end] // samples_per_class
        counts = np.bincount(predicted_labels, minlength=num_classes)
        CM[true_label] = counts / samples_per_class

    logger.debug(f"Confusion matrix calculated with shape: {CM.shape}")
    return CM, IND


def show_cm(cm: np.ndarray):
    """
    Displays the confusion matrix.

    Parameters:
        cm (np.ndarray): Confusion matrix.
    """
    logger.debug("Displaying confusion matrix")

    if not isinstance(cm, np.ndarray):
        logger.error("CM must be a NumPy array representing the confusion matrix.")
        raise ValueError("CM must be a NumPy array representing the confusion matrix.")

    plt.figure(figsize=(8, 6))
    plt.imshow(100 * cm, cmap='jet')
    plt.xticks(np.arange(cm.shape[0]), labels=np.arange(cm.shape[0]))
    plt.yticks(np.arange(cm.shape[1]), labels=np.arange(cm.shape[1]))
    plt.xlabel('Decoded Image Category')
    plt.ylabel('Presented Image Category')
    cbar = plt.colorbar()
    cbar.set_label('Categorization Frequency (%)')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def loglikelihood(activity: np.ndarray, images: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Computes the log-likelihood for the spiking neural network.

    Parameters:
        activity (np.ndarray): Neuronal activity data.
        images (np.ndarray): Input images (or word features).
        parameters (dict): Parameters for the SNN.

    Returns:
        np.ndarray: Log-likelihood matrix.
    """
    logger.debug("Starting log likelihood calculation")

    meanact = np.ceil(np.mean(activity, axis=2))
    logger.debug(f"Mean activity calculated. Non-zero elements: {np.count_nonzero(meanact)}")

    num_samples = images.shape[1]

    # Use approximate nearest neighbors for large datasets
    if num_samples > 1000:  # Adjust threshold as needed
        nn = NearestNeighbors(n_neighbors=num_samples, metric='euclidean', algorithm='auto')
        nn.fit(meanact.T)
        distances, indices = nn.kneighbors(meanact.T)
        LL = -distances  # Use negative distances as likelihoods
    else:
        # Calculate pairwise Euclidean distances between all samples
        distances = cdist(meanact.T, meanact.T, metric='euclidean')

        # Convert distances to likelihoods (e.g., using a Gaussian kernel)
        sigma = 1.0  # Adjust this parameter
        LL = np.exp(-0.5 * (distances / sigma)**2)

    logger.debug("Log likelihood calculation completed")
    return LL


def posterior(LL: np.ndarray, LP: np.ndarray) -> np.ndarray:
    """
    Calculates the posterior probabilities.

    Parameters:
        LL (np.ndarray): Log-likelihood matrix.
        LP (np.ndarray): Log prior probabilities.

    Returns:
        np.ndarray: Posterior probabilities matrix.
    """
    logger.debug("Starting posterior calculation")

    if LL.shape[0] != LL.shape[1]:
        logger.error(f"LL must be a square matrix, but got shape {LL.shape}.")
        raise ValueError("Invalid LL shape.")

    if LP.shape[0] != LL.shape[0]:
        logger.error(f"LP must match the number of samples in LL, but got shape {LP.shape}.")
        raise ValueError("Invalid LP shape.")

    LPOS = LL + LP[:, np.newaxis]
    logger.debug(f"Log posterior (LPOS) calculated. Max: {np.max(LPOS)}, Min: {np.min(LPOS)}")

    # Use the log-sum-exp trick for numerical stability
    max_LPOS = np.max(LPOS, axis=0, keepdims=True)
    POS = np.exp(LPOS - max_LPOS)
    POS_sum = POS.sum(axis=0, keepdims=True)
    POS = POS / POS_sum

    logger.debug(f"Posterior matrix (POS) calculated. Max: {np.max(POS)}, Min: {np.min(POS)}, Shape: {POS.shape}")
    return POS


def posaverage(images: np.ndarray, POS: np.ndarray, navg: int) -> np.ndarray:
    """
    Computes the posterior-averaged images.

    Parameters:
        images (np.ndarray): Input images.
        POS (np.ndarray): Posterior probabilities matrix.
        navg (int): Number of top images to average.

    Returns:
        np.ndarray: Posterior-averaged images.
    """
    logger.debug("Computing posterior-averaged images")

    if images.shape[1] != POS.shape[0]:
        logger.error("IMAGES and POS dimensions mismatch.")
        raise ValueError("Invalid images or POS shape.")

    if not (1 <= navg <= POS.shape[1]):
        logger.error(f"NAVG must be an integer between 1 and {POS.shape[1]}, but got {navg}.")
        raise ValueError("Invalid navg value.")

    num_samples = images.shape[1]
    PA = np.zeros_like(images, dtype=np.float64)
    spos = np.sort(POS, axis=0)[::-1]
    ipos = np.argsort(POS, axis=0)[::-1]

    for i in range(num_samples):
        top_indices = ipos[:navg, i]
        top_values = spos[:navg, i]
        top_images = images[:, top_indices]
        PA[:, i] = np.sum(top_images * top_values, axis=1)
        max_value = PA[:, i].max()
        if max_value > 0:
            PA[:, i] /= max_value

    logger.debug("Posterior-averaged images computed successfully")
    return PA


def logprior(cp: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Calculates the log prior probabilities.

    Parameters:
        cp (np.ndarray): Class prior probabilities.
        num_samples (int): Total number of samples.

    Returns:
        np.ndarray: Log prior probabilities.
    """
    logger.debug("Calculating log prior probabilities")

    if not isinstance(cp, np.ndarray) or not np.isclose(cp.sum(), 1) or np.any(cp == 0):
        logger.error("CP must be a vector that sums to 1 and does not contain any zeros.")
        raise ValueError("CP must be a vector that sums to 1 and does not contain any zeros.")

    num_classes = len(cp)
    samples_per_class = num_samples // num_classes
    LP = np.zeros(num_samples)

    for c in range(num_classes):
        start = c * samples_per_class
        end = (c + 1) * samples_per_class
        LP[start:end] = np.log(cp[c] / samples_per_class)

    logger.debug("Log prior probabilities calculated successfully")
    return LP


# --------------------------- GUI Implementation --------------------------- #

class SNNApp:
    """
    Spiking Neural Network (SNN) Processing Application with GUI.
    """

    def __init__(self, master):
        """
        Initializes the GUI components.
        """
        self.master = master
        master.title("Spiking Neural Network (SNN) Processor")

        self.images_path = tk.StringVar()
        self.weights_path = tk.StringVar()
        self.nrep = tk.IntVar(value=5)
        self.dt = tk.DoubleVar(value=0.1)
        self.gain = tk.DoubleVar(value=1.5)
        self.navg = tk.IntVar(value=3)
        self.num_classes = tk.IntVar(value=10)
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

        ttk.Label(file_frame, text="Images File (.npy/.mat):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
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

        ttk.Label(params_frame, text="Number of Repetitions (nrep):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.nrep_entry = ttk.Entry(params_frame, textvariable=self.nrep, width=10)
        self.nrep_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Time Step (dt):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dt_entry = ttk.Entry(params_frame, textvariable=self.dt, width=10)
        self.dt_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Gain:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.gain_entry = ttk.Entry(params_frame, textvariable=self.gain, width=10)
        self.gain_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Number of Classes:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.num_classes_entry = ttk.Entry(params_frame, textvariable=self.num_classes, width=10)
        self.num_classes_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Number of Top Images to Average (navg):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.navg_entry = ttk.Entry(params_frame, textvariable=self.navg, width=10)
        self.navg_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

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

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Configure grid weights for resizing
        self.master.grid_rowconfigure(4, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

    def browse_images(self):
        """
        Opens a file dialog for the user to select the images file.
        """
        file_path = filedialog.askopenfilename(
            title="Select Images File",
            filetypes=[("NumPy Files", "*.npy"), ("MATLAB Files", "*.mat")])
        if file_path:
            self.images_path.set(file_path)
            logger.debug(f"Selected images file: {file_path}")

    def browse_weights(self):
        """
        Opens a file dialog for the user to select the weights file.
        """
        file_path = filedialog.askopenfilename(
            title="Select Weights File",
            filetypes=[("NumPy Files", "*.npy"), ("MATLAB Files", "*.mat")])
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
        processing_thread = threading.Thread(target=self.process_snn)
        processing_thread.start()

    def process_snn(self):
        """
        Executes the SNN processing steps.
        """
        try:
            images_path = self.images_path.get()
            weights_path = self.weights_path.get()
            parameters = {
                'nrep': self.nrep.get(),
                'dt': self.dt.get(),
                'gain': self.gain.get()
            }
            navg = self.navg.get()
            num_classes = self.num_classes.get()

            if not images_path or not weights_path:
                messagebox.showerror("Input Error", "Please select both images and weights files.")
                logger.error("Missing input files.")
                self.execute_button.config(state='normal')
                return

            self.snn = SNN(parameters)  # Create an SNN object
            self.snn.load_weights(weights_path)

            logger.info("Recording neuronal activity...")
            images_data = load_data(images_path)
            images_features = extract_image_features(images_data)
            activity = self.snn.record(images_features)
            logger.info(f"Activity recorded with shape: {activity.shape}")

            # Visualize spikes (first repetition)
            self.visualize_spikes(activity[:, :, 0])

            logger.info("Computing log-likelihood...")
            LL = loglikelihood(activity, images_features, parameters)
            logger.info(f"Log-likelihood matrix computed with shape: {LL.shape}")

            # Define class prior probabilities (replace with your method)
            class_prior = np.array([1 / num_classes] * num_classes)

            logger.info("Calculating log prior probabilities...")
            LP = logprior(class_prior, num_samples=LL.shape[0])
            logger.info(f"Log prior probabilities calculated with shape: {LP.shape}")

            logger.info("Calculating posterior probabilities...")
            POS = posterior(LL, LP)
            logger.info(f"Posterior probabilities calculated with shape: {POS.shape}")

            logger.info("Calculating confusion matrix...")
            CM, IND = confusion(POS, num_classes=num_classes)
            logger.info(f"Confusion matrix calculated with shape: {CM.shape}")

            # Show confusion matrix (optional)
            # show_cm(CM)

            logger.info("Computing posterior-averaged images...")
            PA = posaverage(images=images_data, POS=POS, navg=navg)
            logger.info(f"Posterior-averaged images computed with shape: {PA.shape}")

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
        Visualizes the spike trains with more advanced options.

        Parameters:
            spikes (np.ndarray): Spikes matrix (neurons x timesteps).
        """
        logger.debug("Visualizing spikes")

        self.ax.clear()
        num_neurons, timesteps = spikes.shape

        # --- Visualization Options ---
        subsample_neurons = True  # Whether to subsample neurons
        subsample_fraction = 0.2  # Fraction of neurons to sample
        use_raster_plot = True   # Use raster plot instead of individual lines
        use_heatmap = False      # Use heatmap instead of lines/raster
        add_jitter = True       # Add jitter to spike positions
        jitter_amount = 0.2     # Amount of jitter

        # --- Subsampling ---
        if subsample_neurons:
            sampled_neurons = np.random.choice(num_neurons, int(subsample_fraction * num_neurons), replace=False)
        else:
            sampled_neurons = np.arange(num_neurons)

        # --- Visualization Logic ---
        if use_heatmap:
            # Create a heatmap of spike activity
            heatmap_data = spikes[sampled_neurons, :]
            im = self.ax.imshow(heatmap_data, cmap='hot', aspect='auto', interpolation='nearest')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Neuron Index')
            cbar = self.fig.colorbar(im, ax=self.ax)
            cbar.set_label('Spike Count')
            self.ax.set_title('Spike Heatmap')

        else:
            # Raster plot or individual lines
            for neuron_idx in sampled_neurons:
                spike_times = np.where(spikes[neuron_idx, :] == 1)[0]
                if add_jitter:
                    y_pos = np.full_like(spike_times, neuron_idx) + np.random.uniform(-jitter_amount, jitter_amount, size=len(spike_times))
                else:
                    y_pos = np.full_like(spike_times, neuron_idx)

                if use_raster_plot:
                    # Plot spikes as dots in a raster plot
                    self.ax.plot(spike_times, y_pos, '.', color='black', markersize=2)
                else:
                    # Plot spikes as vertical lines
                    self.ax.plot(spike_times, y_pos, '|k', markersize=8, alpha=0.5)

            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Neuron Index')
            if use_raster_plot:
                self.ax.set_title('Spike Raster Plot')
            else:
                self.ax.set_title('Spike Visualization (Subsampled)')

        self.ax.set_xlim(0, timesteps)
        self.ax.set_ylim(-1, num_neurons)
        self.canvas.draw()


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
