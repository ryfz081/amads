'''Plots expectation trajectories'''

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from amads.expectation.predictions import SequencePrediction
from amads.expectation.tokenizer import Token
from amads.expectation.metrics import NegativeLogLikelihood, Entropy
from amads.utils import check_python_package_installed
import warnings

def plot_expectation_trajectory(
    predictions: SequencePrediction,
    tokens: List[Token],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    show_points: bool = True,
    fit_gp: bool = False,
    gp_params: Optional[dict] = None,
    moving_window: bool = False,
    window_size: float = 5.0,
    window_type: str = 'average',
    show_piano_roll: bool = False,
    show_waveform: bool = False,
    midi_file: Optional[str] = None,
    audio_file: Optional[str] = None
) -> tuple:
    """Plot expectation metrics over time for a sequence of predictions.
    
    Parameters
    ----------
    predictions : SequencePrediction
        Model predictions for the sequence
    tokens : List[Token]
        The sequence of tokens that were predicted (including timestamps)
    metrics : List[str], optional
        List of metrics to plot. Default is ['nll', 'entropy']
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    show_points : bool, optional
        Whether to show individual data points as a scatter plot
    fit_gp : bool, optional
        Whether to fit and plot a Gaussian Process regression
    gp_params : dict, optional
        Parameters for the Gaussian Process regression
    moving_window : bool, optional
        Whether to plot moving window statistics
    window_size : float, optional
        Size of the moving window in seconds
    window_type : str, optional
        Type of window statistics to plot. Options: 'average' or 'sum'
    show_piano_roll : bool, optional
        Whether to include a piano roll visualization of the MIDI file. 
        Requires midi_file to be provided.
    show_waveform : bool, optional
        Whether to include a waveform visualization. Can be used with either
        audio_file or midi_file.
    midi_file : str, optional
        Path to the MIDI file for visualization. Required if show_piano_roll is True
        or if show_waveform is True and audio_file is not provided.
    audio_file : str, optional
        Path to the audio file for waveform visualization. Required if show_waveform 
        is True and midi_file is not provided.
        
    Returns
    -------
    tuple
        (fig, axes) tuple containing the figure and axes objects
    """
    if metrics is None:
        metrics = ['nll', 'entropy']
    
    # Initialize metric calculators
    metric_calculators = {
        'nll': NegativeLogLikelihood(),
        'entropy': Entropy()
    }
    
    # Calculate metrics
    metric_values = {}
    for metric_name in metrics:
        if metric_name not in metric_calculators:
            raise ValueError(f"Unknown metric: {metric_name}")
        calculator = metric_calculators[metric_name]
        values = calculator.compute(predictions, tokens[1:])
        metric_values[metric_name] = values
    
    # Get timestamps for x-axis (skip first token as it's context)
    timestamps = []
    last_known_time = 0  # Default to 0 if no timestamps are available
    for token in tokens[1:]:
        if token.start_time is not None:
            last_known_time = token.start_time
        timestamps.append(last_known_time)
    
    # Use the plot_expectation_metrics function for consistency
    return plot_expectation_metrics(
        timestamps=timestamps,
        metric_values=metric_values,
        metrics=metrics,
        title=title,
        figsize=figsize,
        show_points=show_points,
        fit_gp=fit_gp,
        gp_params=gp_params,
        moving_window=moving_window,
        window_size=window_size,
        window_type=window_type,
        show_piano_roll=show_piano_roll,
        show_waveform=show_waveform,
        midi_file=midi_file,
        audio_file=audio_file
    )

def plot_expectation_metrics(timestamps, metric_values, metrics=None, title=None, figsize=(12, 8), 
                             show_points=True, fit_gp=False, gp_params=None, 
                             moving_window=False, window_size=5.0, window_type='average',
                             show_piano_roll=False, show_waveform=False,
                             midi_file=None, audio_file=None):
    """Plot expectation metrics over time with various smoothing options.
    
    Parameters
    ----------
    timestamps : list
        List of time points for the metrics
    metric_values : dict
        Dictionary mapping metric names to lists of values
    metrics : list, optional
        List of metric names to plot. If None, all metrics in metric_values are plotted.
    title : str, optional
        Title for the figure
    figsize : tuple, optional
        Figure size
    show_points : bool, optional
        Whether to show individual data points as a scatter plot
    fit_gp : bool, optional
        Whether to fit and plot a Gaussian Process regression (requires scikit-learn)
    gp_params : dict, optional
        Parameters for the Gaussian Process regression. If None, default parameters are used.
    moving_window : bool, optional
        Whether to plot moving window statistics
    window_size : float, optional
        Size of the moving window in seconds
    window_type : str, optional
        Type of window statistics to plot. Options: 'average' or 'sum'
    show_piano_roll : bool, optional
        Whether to include a piano roll visualization of the MIDI file. 
        Requires midi_file to be provided.
    show_waveform : bool, optional
        Whether to include a waveform visualization. Can be used with either
        audio_file or midi_file.
    midi_file : str, optional
        Path to the MIDI file for visualization. Required if show_piano_roll is True
        or if show_waveform is True and audio_file is not provided.
    audio_file : str, optional
        Path to the audio file for waveform visualization. Required if show_waveform 
        is True and midi_file is not provided.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    # Ensure at least one visualization method is active
    if not (show_points or fit_gp or moving_window or show_piano_roll or show_waveform):
        warnings.warn("No visualization method selected. Defaulting to show_points=True.")
        show_points = True
    
    # Validate window_type
    if window_type not in ['average', 'sum']:
        raise ValueError(f"Invalid window_type: {window_type}. Must be 'average' or 'sum'.")
    
    # Validate inputs
    if show_piano_roll and midi_file is None:
        raise ValueError("midi_file must be provided when show_piano_roll is True.")
    
    if show_waveform:
        if audio_file is None and midi_file is None:
            raise ValueError("Either audio_file or midi_file must be provided when show_waveform is True.")
    
    # Determine if we should show a visualization
    show_visualization = show_piano_roll or show_waveform
    
    # Convert to numpy arrays for calculations
    timestamps = np.array(timestamps)
    
    # If no metrics specified, use all available metrics
    if metrics is None:
        metrics = list(metric_values.keys())
    
    # Create subplots
    num_extra_plots = 1 if show_visualization else 0
    fig, axes = plt.subplots(len(metrics) + num_extra_plots, 1, figsize=figsize, sharex=True)
    if len(metrics) + num_extra_plots == 1:
        axes = [axes]  # Make sure axes is always a list
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Plot visualization if requested
    current_ax_idx = 0
    if show_visualization:
        if show_piano_roll:
            # Piano roll mode - always use MIDI file as piano roll
            _plot_piano_roll(axes[current_ax_idx], midi_file, timestamps.min(), timestamps.max())
        elif show_waveform:
            # Waveform mode - use audio file if provided, otherwise synthesize MIDI
            if audio_file is not None:
                _plot_waveform(axes[current_ax_idx], audio_file, timestamps.min(), timestamps.max())
            else:
                _plot_midi_as_waveform(axes[current_ax_idx], midi_file, timestamps.min(), timestamps.max())
        current_ax_idx += 1
    
    # Plot each metric
    for i, metric_name in enumerate(metrics):
        ax = axes[current_ax_idx + i]
        values = np.array(metric_values[metric_name])
        
        # Group and average values at the same timestamp for cleaner visualization
        unique_times, averaged_values = _get_averaged_values(timestamps, values)
        
        # Plot the averaged data points if requested
        if show_points:
            _plot_data_points(ax, unique_times, averaged_values, metric_name)
        
        # Fit and plot Gaussian Process if requested
        if fit_gp:
            _plot_gaussian_process(ax, unique_times, averaged_values, timestamps, metric_name, gp_params)
        
        # Plot moving window statistics if requested
        if moving_window and len(timestamps) > 1:
            _plot_moving_window(ax, unique_times, averaged_values, timestamps, 
                              window_size, window_type, metric_name)
        
        # Add legend if we didn't already handle it with the special case for moving sum
        if not (moving_window and window_type == 'sum'):
            ax.legend(loc='upper right')
        
        ax.set_ylabel(metric_name.upper())
        ax.grid(True)
        
        # Calculate and display mean
        mean_value = np.mean(values)
        ax.axhline(y=mean_value, color='r', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.95, f'Mean: {mean_value:.2f}', 
                transform=ax.transAxes, verticalalignment='top')
    
    # Set the x-axis label on the bottom subplot
    axes[-1].set_xlabel('Time (seconds)')
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.92)  # Make room for the title
    
    return fig

def _get_averaged_values(timestamps, values):
    """Group and average values at the same timestamp for cleaner visualization.
    
    Parameters
    ----------
    timestamps : np.ndarray
        Array of time points
    values : np.ndarray
        Array of metric values corresponding to timestamps
        
    Returns
    -------
    tuple
        (unique_times, averaged_values) arrays
    """
    unique_times, unique_indices = np.unique(timestamps, return_inverse=True)
    averaged_values = np.zeros_like(unique_times)
    count_values = np.zeros_like(unique_times)
    
    # Compute averages at each unique timestamp
    for i, idx in enumerate(unique_indices):
        averaged_values[idx] += values[i]
        count_values[idx] += 1
    
    # Avoid division by zero
    averaged_values = averaged_values / np.maximum(count_values, 1)
    
    return unique_times, averaged_values

def _plot_data_points(ax, times, values, metric_name):
    """Plot data points on the given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    times : np.ndarray
        Array of time points
    values : np.ndarray
        Array of values to plot
    metric_name : str
        Name of the metric for the legend
    """
    ax.plot(times, values, 'o', alpha=0.7, 
           label=f'{metric_name.upper()} (avg at each time)')

def _plot_gaussian_process(ax, times, values, all_timestamps, metric_name, gp_params=None):
    """Fit and plot a Gaussian Process regression.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    times : np.ndarray
        Array of unique time points
    values : np.ndarray
        Array of averaged values corresponding to unique time points
    all_timestamps : np.ndarray
        Full array of all timestamps (for determining plot range)
    metric_name : str
        Name of the metric for the legend
    gp_params : dict, optional
        Parameters for the Gaussian Process regression
    """
    try:
        check_python_package_installed('sklearn')
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel
        
        # Default GP parameters - more responsive to local variations
        default_gp_params = {
            'kernel': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
            'alpha': 1e-6,  # Slight regularization
            'n_restarts_optimizer': 3,  # Number of optimizer restarts
            'normalize_y': True  # Normalize the target values
        }
        
        # Use default GP parameters if none provided
        if gp_params is None:
            gp_params = default_gp_params
        
        if len(times) > 5:  # Need enough points for a meaningful fit
            # Prepare X for GP (reshape for scikit-learn)
            X = times.reshape(-1, 1)
            
            # Create and fit the GP model
            gp = GaussianProcessRegressor(**gp_params)
            gp.fit(X, values)
            
            # Predict on a fine grid for smooth visualization
            X_pred = np.linspace(all_timestamps.min(), all_timestamps.max(), 500).reshape(-1, 1)
            y_pred, sigma = gp.predict(X_pred, return_std=True)
            
            # Plot the GP fit
            ax.plot(X_pred, y_pred, '-', color='blue', linewidth=2, 
                    label=f'{metric_name.upper()} (GP fit)')
            
            # Plot uncertainty band (± 2 standard deviations, 95% confidence interval)
            ax.fill_between(X_pred.ravel(), y_pred - 2*sigma, y_pred + 2*sigma,
                            color='blue', alpha=0.2)
            
    except ImportError:
        warnings.warn("scikit-learn is required for Gaussian Process fitting. "
                     "Please install it with 'pip install scikit-learn'. "
                     "Continuing without GP fitting.")
    except Exception as e:
        warnings.warn(f"GP fitting failed for {metric_name}: {e}")

def _plot_moving_window(ax, times, values, all_timestamps, window_size, window_type, metric_name):
    """Plot moving window statistics.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    times : np.ndarray
        Array of unique time points
    values : np.ndarray
        Array of averaged values corresponding to unique time points
    all_timestamps : np.ndarray
        Full array of all timestamps (for determining plot range)
    window_size : float
        Size of the moving window in seconds
    window_type : str
        Type of window statistics to plot ('average' or 'sum')
    metric_name : str
        Name of the metric for the legend
    """
    # Calculate the indices that fall within the window for each timestamp
    half_window = window_size / 2
    output_times = np.linspace(all_timestamps.min(), all_timestamps.max(), 200)
    moving_values = np.zeros_like(output_times)
    
    # Use unique_times and averaged_values for window calculations
    for i, t in enumerate(output_times):
        # Find all points within a centered window around t
        window_indices = (times >= t - half_window) & (times <= t + half_window)
        window_values = values[window_indices]
        
        if len(window_values) > 0:
            if window_type == 'average':
                moving_values[i] = np.mean(window_values)
            else:  # window_type == 'sum'
                moving_values[i] = np.sum(window_values)
        else:
            # No points in window
            moving_values[i] = np.nan
    
    # Plot moving average
    if window_type == 'average':
        ax.plot(output_times, moving_values, '-', color='green', linewidth=2, 
               label=f'{metric_name.upper()} (Moving Avg, {window_size}s)')
    
    # Plot moving sum - use a second y-axis for clarity
    elif window_type == 'sum':
        # Create a twin y-axis for the sum
        ax_sum = ax.twinx()
        ax_sum.plot(output_times, moving_values, '-', color='purple', linewidth=2, 
                  label=f'{metric_name.upper()} (Moving Sum, {window_size}s)')
        ax_sum.set_ylabel(f'{metric_name.upper()} SUM', color='purple')
        ax_sum.tick_params(axis='y', labelcolor='purple')
        
        # Add the moving sum to the legend of the main axis
        # Get the last line from the twin axis
        sum_line = ax_sum.get_lines()[-1]
        # Add this line to the main axis's legend
        ax.legend(handles=ax.get_lines() + [sum_line], loc='upper right')

def _plot_piano_roll(ax, midi_file, min_time, max_time):
    """Plot a piano roll visualization of a MIDI file.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    midi_file : str
        Path to the MIDI file
    min_time : float
        Minimum time to display (seconds)
    max_time : float
        Maximum time to display (seconds)
    """
    try:
        # Check if pretty_midi is installed
        check_python_package_installed('pretty_midi')
        import pretty_midi
        
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        
        # Get the end time of the MIDI file
        midi_end_time = midi_data.get_end_time()
        
        # Adjust plot range if needed
        plot_min = max(0, min_time)
        plot_max = min(midi_end_time, max_time)
        
        # Initialize empty piano roll
        min_pitch = 128
        max_pitch = 0
        
        # Plot each instrument
        for i, instrument in enumerate(midi_data.instruments):
            # Different color for each instrument (cycling through a colormap)
            cmap = plt.get_cmap('tab10')
            color = cmap(i % 10)
            
            # Plot notes as rectangles
            for note in instrument.notes:
                # Skip notes outside our time range
                if note.end < plot_min or note.start > plot_max:
                    continue
                    
                # Update min/max pitch for y-axis scaling
                min_pitch = min(min_pitch, note.pitch)
                max_pitch = max(max_pitch, note.pitch)
                
                # Scale alpha by velocity
                alpha = 0.4 + 0.6 * (note.velocity / 127)
                
                # Create rectangle for the note
                rect = plt.Rectangle(
                    (note.start, note.pitch - 0.4),
                    note.end - note.start,
                    0.8,
                    color=color,
                    alpha=alpha
                )
                ax.add_patch(rect)
        
        # Add a margin to pitch range for better visualization
        pitch_margin = 3
        min_pitch = max(0, min_pitch - pitch_margin)
        max_pitch = min(127, max_pitch + pitch_margin)
        
        # Set axis limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(min_pitch, max_pitch)
        
        # Add piano keys on y-axis
        yticks = []
        yticklabels = []
        
        # Add C notes for each octave
        for octave in range(11):  # MIDI octaves 0-10
            for pitch_class, name in enumerate(['C', 'D', 'E', 'F', 'G', 'A', 'B']):
                pitch = 12 * octave + pitch_class
                if min_pitch <= pitch <= max_pitch:
                    if name == 'C':  # Only label C notes for each octave
                        yticks.append(pitch)
                        yticklabels.append(f"{name}{octave}")
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        # Add grid lines for better readability
        ax.grid(True, alpha=0.3)
        
        # Add labels
        ax.set_ylabel('MIDI Note')
        
    except ImportError:
        warnings.warn("pretty_midi is required for piano roll visualization. "
                     "Please install it with 'pip install pretty_midi'. "
                     "Continuing without piano roll.")
        ax.text(0.5, 0.5, "Piano roll visualization requires pretty_midi package",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('MIDI Note')
    except Exception as e:
        warnings.warn(f"Error plotting piano roll: {e}")
        ax.text(0.5, 0.5, f"Error loading MIDI file: {str(e)}",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('MIDI Note')

def _plot_midi_as_waveform(ax, midi_file, min_time, max_time):
    """Synthesize and plot a MIDI file as a waveform.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    midi_file : str
        Path to the MIDI file
    min_time : float
        Minimum time to display (seconds)
    max_time : float
        Maximum time to display (seconds)
    """
    try:
        # Check if pretty_midi is installed
        check_python_package_installed('pretty_midi')
        import pretty_midi
        
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        
        # Synthesize the MIDI file to a waveform
        sample_rate = 44100  # Standard audio sample rate
        audio_data = midi_data.synthesize(fs=sample_rate)
        
        # Adjust plot range if needed
        audio_duration = len(audio_data) / sample_rate
        plot_min = max(0, min_time)
        plot_max = min(audio_duration, max_time)
        
        # Convert times to samples
        start_sample = int(plot_min * sample_rate)
        end_sample = int(plot_max * sample_rate)
        
        # Slice audio to the desired range
        audio_segment = audio_data[start_sample:end_sample]
        
        # Create time array for x-axis
        times = np.linspace(plot_min, plot_max, len(audio_segment))
        
        # Plot waveform
        ax.plot(times, audio_segment, color='blue', alpha=0.7)
        
        # Set axis limits
        ax.set_xlim(plot_min, plot_max)
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        # Add labels
        ax.set_ylabel('Amplitude')
        
    except Exception as e:
        warnings.warn(f"Error synthesizing MIDI as waveform: {e}")
        ax.text(0.5, 0.5, f"Error synthesizing MIDI file: {str(e)}",
              ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Amplitude')

def _plot_waveform(ax, audio_file, min_time, max_time):
    """Plot a waveform visualization of an audio file.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    audio_file : str
        Path to the audio file
    min_time : float
        Minimum time to display (seconds)
    max_time : float
        Maximum time to display (seconds)
    """
    try:
        # Check if librosa is installed
        check_python_package_installed('librosa')
        import librosa
        
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Adjust plot range if needed
        audio_duration = len(y) / sr
        plot_min = max(0, min_time)
        plot_max = min(audio_duration, max_time)
        
        # Convert times to samples
        start_sample = int(plot_min * sr)
        end_sample = int(plot_max * sr)
        
        # Slice audio to the desired range
        y_segment = y[start_sample:end_sample]
        
        # Create time array for x-axis
        times = np.linspace(plot_min, plot_max, len(y_segment))
        
        # Plot waveform
        ax.plot(times, y_segment, color='blue', alpha=0.7)
        
        # Set axis limits
        ax.set_xlim(plot_min, plot_max)
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        # Add labels
        ax.set_ylabel('Amplitude')
        
    except ImportError:
        warnings.warn("librosa is required for waveform visualization. "
                    "Please install it with 'pip install librosa'. "
                    "Continuing without waveform.")
        ax.text(0.5, 0.5, "Waveform visualization requires librosa package",
              ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Amplitude')
    except Exception as e:
        warnings.warn(f"Error plotting waveform: {e}")
        ax.text(0.5, 0.5, f"Error loading audio file: {str(e)}",
              ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Amplitude')

