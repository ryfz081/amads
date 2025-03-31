from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import os
import torch
from pathlib import Path
from amads.core.basics import Score, Note
from amads.expectation.tokenizers.base_tokenizer import Token, Tokenizer
from amads.pitch.ismonophonic import ismonophonic
from amads.utils import check_python_package_installed
import sys
import shutil
import importlib
import inspect
import pkg_resources


class AudioTokenizer(Tokenizer):
    """Base class for audio tokenizers.
    
    This class handles common functionality for audio processing
    like loading audio files. Specific tokenization strategies
    should be implemented in subclasses.
    """
    
    def tokenize(self, audio_path: str) -> List[Token]:
        """Convert an audio file into a sequence of tokens.
        
        This base method loads the audio and delegates to _tokenize,
        which should be implemented by subclasses.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio file to tokenize
            
        Returns
        -------
        List[Token]
            A list of Token objects
        """
        # Load the audio file
        y, sr = self._load_audio(audio_path)
        
        # Delegate to the subclass implementation
        return self._tokenize(y, sr)
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load an audio file using torchaudio or librosa.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio file
            
        Returns
        -------
        Tuple[np.ndarray, int]
            Tuple containing the audio data and sample rate
        """
        try:
            check_python_package_installed('torchaudio')
            import torchaudio
            wav, sr = torchaudio.load(audio_path)
            return wav.numpy(), sr
        except ImportError:
            check_python_package_installed('librosa')
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return y, sr
    
    def _tokenize(self, y: np.ndarray, sr: int) -> List[Token]:
        """Tokenize the audio data into a sequence of tokens.
        
        This method should be implemented by subclasses.
        
        Parameters
        ----------
        y : np.ndarray
            Audio data
        sr : int
            Sample rate
            
        Returns
        -------
        List[Token]
            A list of Token objects
        """
        raise NotImplementedError("Subclasses must implement _tokenize method")


class FrameAmplitudeTokenizer(AudioTokenizer):
    """Simple tokenizer that segments audio into frames and quantizes amplitude."""
    
    def __init__(self, frame_size: float = 0.025, hop_size: float = 0.010, n_bins: int = 100):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_bins = n_bins
    
    def _tokenize(self, y: np.ndarray, sr: int) -> List[Token]:
        """Tokenize audio based on amplitude in fixed-size frames."""
        frame_samples = int(self.frame_size * sr)
        hop_samples = int(self.hop_size * sr)
        
        tokens = []
        for i in range(0, len(y) - frame_samples, hop_samples):
            frame = y[i:i+frame_samples]
            start_time = i / sr
            end_time = (i + frame_samples) / sr
            feature = np.mean(np.abs(frame))
            token_value = min(int(feature * self.n_bins), self.n_bins - 1)
            token = Token(token_value, start_time, end_time)
            tokens.append(token)
        
        return tokens


class WavTokenizerAudio(AudioTokenizer):
    """Audio tokenizer using WavTokenizer.
    
    WavTokenizer is a neural audio tokenizer that converts audio signals into discrete tokens
    and can also reconstruct audio from these tokens. This implementation provides access to 
    various pre-trained WavTokenizer models with different characteristics.
    
    Available Models:
    -----------------
    - small-600-24k-4096: Small model trained on LibriTTS, 40 tokens/second, suitable for speech
    - small-320-24k-4096: Small model trained on LibriTTS, 75 tokens/second, suitable for speech
    - medium-320-24k-4096: Medium model trained on 10,000 hours of data, 75 tokens/second, 
                          handles speech, audio, and music
    - medium-music-audio-320-24k-4096: Specialized medium model optimized for music and audio,
                                      75 tokens/second
    - large-600-24k-4096: Large model trained on 80,000 hours of data, 40 tokens/second, 
                         handles speech, audio, and music
    - large-320-24k-4096: Large model trained on 80,000 hours of data, 75 tokens/second, 
                         handles speech, audio, and music
    
    You can also specify custom URLs for any new models released in the future.
    
    Model Selection Guide:
    ---------------------
    - Size (small/medium/large): Larger models capture more complex audio characteristics but 
      require more memory. Small models are best for speech only, while larger models handle 
      music and complex audio better.
    
    - Segment size (600/320): Determines tokens per second - 600 gives 40 tokens/sec, 
      320 gives 75 tokens/sec. Higher token rate (320) provides more temporal detail at the 
      cost of increased token sequence length.
    
    - Domain: Small models focus on speech, medium and large models cover speech, music, and 
      general audio. The specialized medium-music-audio model is optimized for music and audio tasks.
    
    Parameters
    ----------
    model_name : str, default='small-600-24k-4096'
        Name of the WavTokenizer model to use. See above for available options.
    device : str, default='cpu'
        Device to run the model on ('cpu' or 'cuda:0', etc.)
    model_url : str, optional
        Custom URL to download a model checkpoint from. If provided, this overrides the 
        built-in URLs for the standard models.
    config_url : str, optional
        Custom URL to download a configuration file from. If provided, this overrides the
        default config selection.
    tokens_per_second : int, optional
        Number of tokens per second (40 or 75). Only needed when using custom URLs to ensure
        the correct config file is used. If not specified, will attempt to determine from 
        the model name or default to 40.
    
    Examples
    --------
    >>> # Basic usage with a standard model
    >>> tokenizer = WavTokenizerAudio(model_name='small-600-24k-4096')
    >>> tokens = tokenizer.tokenize('speech_sample.wav')
    >>> 
    >>> # Using the specialized music model
    >>> music_tokenizer = WavTokenizerAudio(model_name='medium-music-audio-320-24k-4096')
    >>> music_tokens = music_tokenizer.tokenize('music_sample.wav')
    >>> 
    >>> # Using custom URLs for a new model release
    >>> custom_tokenizer = WavTokenizerAudio(
    ...     model_name='custom-model',
    ...     model_url='https://example.com/new_model.ckpt',
    ...     config_url='https://example.com/new_config.yaml',
    ...     tokens_per_second=75
    ... )
    >>> custom_tokens = custom_tokenizer.tokenize('audio_sample.wav')
    >>> 
    >>> # Reconstructing audio from tokens
    >>> reconstructed_audio = tokenizer.decode_tokens(tokens)
    """
    
    def __init__(self, model_name='small-600-24k-4096', device='cpu', 
                 model_url=None, config_url=None, tokens_per_second=None):
        """Initialize the WavTokenizer model.
        
        Sets up the directory structure, downloads the model if necessary,
        and initializes the tokenizer.
        
        Parameters
        ----------
        model_name : str, default='small-600-24k-4096'
            Name of the WavTokenizer model to use, or a custom name for a model from custom URLs.
        device : str, default='cpu'
            Device to run the model on ('cpu' or 'cuda:0', etc.)
        model_url : str, optional
            Custom URL to download a model checkpoint from. If provided, this overrides the 
            built-in URLs for the standard models.
        config_url : str, optional
            Custom URL to download a configuration file from. If provided, this overrides the
            default config selection.
        tokens_per_second : int, optional
            Number of tokens per second (40 or 75). Only needed when using custom URLs to ensure
            the correct config file is used. If not specified, will attempt to determine from 
            the model name or default to 40.
        """
        # Check dependencies
        check_python_package_installed('torch')
        check_python_package_installed('torchaudio')
        
        # Store parameters
        self.model_name = model_name
        self.device = device
        self.custom_model_url = model_url
        self.custom_config_url = config_url
        
        # Get paths and ensure directory structure
        current_dir = Path(os.path.abspath(__file__)).parent
        self.wavtokenizer_dir = current_dir / 'wavtokenizer'
        self.repo_dir = self.wavtokenizer_dir / 'repo'
        self.models_dir = self.wavtokenizer_dir / 'models'
        self.configs_dir = self.wavtokenizer_dir / 'configs'
        
        # Create directory structure if needed
        self._ensure_directory_structure()
        
        # Add repo to path
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        
        # Parse model name for tokens_per_second and set file paths
        if self.model_name.startswith('medium-music-audio'):
            # Special case for the music model
            self.tokens_per_second = 75
            model_filename = "wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
            self.model_path = self.models_dir / model_filename
        else:
            # Parse model name
            model_parts = self.model_name.split('-')
            model_size = model_parts[0]  # small, medium, large
            segment = model_parts[1] if len(model_parts) > 1 else "600"
            sr = model_parts[2] if len(model_parts) > 2 else "24k"
            vq = model_parts[3] if len(model_parts) > 3 else "4096"
            
            # Set tokens_per_second if not explicitly specified
            if tokens_per_second is None:
                self.tokens_per_second = 75 if segment == "320" else 40
            else:
                self.tokens_per_second = tokens_per_second
                
            # Set model path based on model name
            if model_url:
                # For custom URL, extract filename from URL
                model_filename = os.path.basename(model_url)
                self.model_path = self.models_dir / model_filename
            else:
                # For standard models
                self.model_path = self.models_dir / f"WavTokenizer_{model_size}_{segment}_{sr}_{vq}.ckpt"
        
        # Handle custom config URL
        if config_url:
            config_filename = os.path.basename(config_url)
            self.config_path = self.configs_dir / config_filename
            
            # Download config if needed
            if not self.config_path.exists():
                self._download_file(config_url, self.config_path, "config")
        else:
            # Find the appropriate config file
            self.config_path = self._find_config_file()
            
            # Special handling for medium music model
            if self.model_name.startswith('medium-music-audio'):
                # Look specifically for the music config
                music_config = self.configs_dir / "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
                if music_config.exists():
                    self.config_path = music_config
                    print(f"Using specialized music config: {self.config_path}")
        
        # Download model if needed
        if not self.model_path.exists():
            if model_url:
                self._download_file(model_url, self.model_path, "model")
            else:
                self._download_model()
        
        # Load model
        self._initialize_wavtokenizer()
    
    def _ensure_directory_structure(self):
        """Ensure the directory structure is correctly set up."""
        # Create main directories if they don't exist
        os.makedirs(self.wavtokenizer_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        
        # Check if repo directory exists and has necessary files
        repo_initialized = (self.repo_dir.exists() and 
                           (self.repo_dir / 'encoder').exists() and 
                           (self.repo_dir / 'decoder').exists())
        
        if not repo_initialized:
            print(f"WavTokenizer repository not found at {self.repo_dir}. Cloning from GitHub...")
            self._clone_wavtokenizer_repo()
        
        # Check for config files, download sample if none exist
        yaml_files = list(self.configs_dir.glob('*.yaml'))
        if not yaml_files:
            print("No configuration files found. Downloading sample configs...")
            self._download_sample_configs()
    
    def _clone_wavtokenizer_repo(self):
        """Clone the WavTokenizer repository."""
        try:
            check_python_package_installed('git')
            import subprocess
            
            # Create repo directory
            os.makedirs(self.repo_dir, exist_ok=True)
            
            # Clone the repository
            repo_url = "https://github.com/jishengpeng/WavTokenizer.git"
            subprocess.run(["git", "clone", repo_url, str(self.repo_dir)], 
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"WavTokenizer repository cloned successfully to {self.repo_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to clone WavTokenizer repository: {e}. "
                              f"Please manually clone {repo_url} to {self.repo_dir}")
    
    def _download_sample_configs(self):
        """Download sample configuration files."""
        try:
            check_python_package_installed('requests')
            import requests
            
            # Define configs to download
            configs = [
                # 40 tokens/sec config (600 frame)
                ("https://raw.githubusercontent.com/jishengpeng/WavTokenizer/main/configs/small_600_24k_4096.yaml", 
                 self.configs_dir / "small_600_24k_4096.yaml"),
                
                # 75 tokens/sec config (320 frame)
                ("https://raw.githubusercontent.com/jishengpeng/WavTokenizer/main/configs/small_320_24k_4096.yaml", 
                 self.configs_dir / "small_320_24k_4096.yaml"),
                
                # Specialized music config
                ("https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/raw/main/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                 self.configs_dir / "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
            ]
            
            for url, path in configs:
                print(f"Downloading config from {url}...")
                response = requests.get(url)
                response.raise_for_status()
                
                with open(path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded config to {path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download sample configs: {e}. "
                              f"Please manually download configuration files to {self.configs_dir}")
    
    def _find_config_file(self):
        """Find the appropriate config file for the selected model."""
        yaml_files = list(self.configs_dir.glob('*.yaml'))
        
        if not yaml_files:
            raise FileNotFoundError(
                f"No configuration files found in {self.configs_dir}. "
                f"Please download appropriate config files for {self.model_name} "
                f"from the WavTokenizer repository."
            )
        
        # Try to find a config file that matches the tokens per second
        config_path = None
        for file_path in yaml_files:
            # For 40 tokens/second (600 segment), look for "600" or "frame40" in filename
            if self.tokens_per_second == 40 and ("600" in file_path.name or "frame40" in file_path.name):
                config_path = file_path
                break
            # For 75 tokens/second (320 segment), look for "320" or "frame75" in filename  
            elif self.tokens_per_second == 75 and ("320" in file_path.name or "frame75" in file_path.name):
                config_path = file_path
                break
        
        # If no match found, raise a specific error
        if not config_path:
            expected_filename = "600" if self.tokens_per_second == 40 else "320"
            raise FileNotFoundError(
                f"No matching configuration file found for {self.model_name} "
                f"which requires {self.tokens_per_second} tokens/second. "
                f"Please add a config file containing '{expected_filename}' in the filename to "
                f"{self.configs_dir}."
            )
        
        print(f"Using config file: {config_path} for {self.tokens_per_second} tokens/second model")
        return config_path
    
    def _download_model(self):
        """Download the pre-trained model from Hugging Face.
        
        If custom_model_url is provided, that URL will be used instead of the 
        built-in URLs for the standard models.
        
        Each model has different characteristics:
        - small-600-24k-4096: 40 tokens/sec, trained on LibriTTS, speech only
        - small-320-24k-4096: 75 tokens/sec, trained on LibriTTS, speech only
        - medium-320-24k-4096: 75 tokens/sec, trained on 10k hours, general audio
        - medium-music-audio-320-24k-4096: 75 tokens/sec, specialized for music/audio
        - large-600-24k-4096: 40 tokens/sec, trained on 80k hours, general audio
        - large-320-24k-4096: 75 tokens/sec, trained on 80k hours, general audio
        """
        # If a custom model URL is provided, use it
        if self.custom_model_url:
            self._download_file(self.custom_model_url, self.model_path, "model")
            return
        
        # Otherwise, use the standard URLs based on model name
        check_python_package_installed('requests')
        import requests
        
        # Determine URL based on model name
        model_parts = self.model_name.split('-')
        model_size = model_parts[0]
        
        # Handle special case for medium-music-audio model
        if self.model_name.startswith('medium-music-audio'):
            url = "https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/resolve/main/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
            print("Downloading specialized medium model for music and audio (75 tokens/sec)")
            
            # Also download the specialized config file if it doesn't exist
            config_url = "https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/raw/main/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
            config_path = self.configs_dir / "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
            
            if not config_path.exists():
                self._download_file(config_url, config_path, "config")
                
                # Update the config path to use this specialized config
                self.config_path = config_path
        else:
            # Handle regular models
            segment = model_parts[1] if len(model_parts) > 1 else "600"
            
            url = None
            if model_size == 'small' and segment == '600':
                url = "https://huggingface.co/novateur/WavTokenizer/resolve/main/WavTokenizer_small_600_24k_4096.ckpt"
                print("Downloading small model (40 tokens/sec, speech only)")
            elif model_size == 'small' and segment == '320':
                url = "https://huggingface.co/novateur/WavTokenizer/resolve/main/WavTokenizer_small_320_24k_4096.ckpt"
                print("Downloading small model (75 tokens/sec, speech only)")
            elif model_size == 'medium' and segment == '320':
                # Use the specialized medium model link
                url = "https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/resolve/main/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
                print("Downloading medium model (75 tokens/sec, speech/audio/music)")
            elif model_size == 'large' and segment == '600':
                url = "https://huggingface.co/novateur/WavTokenizer-large-unify-40token/resolve/main/WavTokenizer_large_600_24k_4096.ckpt"
                print("Downloading large model (40 tokens/sec, speech/audio/music)")
            elif model_size == 'large' and segment == '320':
                url = "https://huggingface.co/novateur/WavTokenizer/resolve/main/WavTokenizer_large_320_24k_4096.ckpt"
                print("Downloading large model (75 tokens/sec, speech/audio/music)")
            else:
                raise ValueError(
                    f"No download URL available for model {self.model_name}. "
                    f"Valid options include: small-600-24k-4096 (40 tokens/sec, speech), "
                    f"small-320-24k-4096 (75 tokens/sec, speech), "
                    f"medium-320-24k-4096 (75 tokens/sec, general audio), "
                    f"medium-music-audio-320-24k-4096 (75 tokens/sec, optimized for music), "
                    f"large-600-24k-4096 (40 tokens/sec, general audio), "
                    f"large-320-24k-4096 (75 tokens/sec, general audio). "
                    f"Or provide a custom URL with the model_url parameter."
                )
        
        # Download the model
        self._download_file(url, self.model_path, "model")
    
    def _initialize_wavtokenizer(self):
        """Initialize WavTokenizer exactly as in the README."""
        try:
            # This follows the README exactly
            from encoder.utils import convert_audio
            import torchaudio
            import torch
            from decoder.pretrained import WavTokenizer
            
            print(f"Loading model with config={self.config_path}, checkpoint={self.model_path}")
            
            # Create the model exactly as shown in the README
            self.model = WavTokenizer.from_pretrained0802(
                str(self.config_path),
                str(self.model_path)
            )
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Store convert_audio for use in tokenize
            self.convert_audio = convert_audio
            
            print("Model loaded successfully!")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to initialize WavTokenizer: {e}\n"
                f"Make sure the WavTokenizer repository is properly set up at {self.repo_dir} "
                f"and the config file {self.config_path} is compatible with the model {self.model_path}."
            )
    
    def _tokenize(self, y: np.ndarray, sr: int) -> List[Token]:
        """Tokenize audio using WavTokenizer.
        
        This method converts audio data to discrete tokens using the WavTokenizer model.
        Each token represents a fixed time segment in the audio based on the model's token rate:
        - 40 tokens/second for 600-segment models
        - 75 tokens/second for 320-segment models
        
        Parameters
        ----------
        y : np.ndarray
            Audio data as a numpy array
        sr : int
            Sample rate of the audio
            
        Returns
        -------
        List[Token]
            A list of Token objects with accurate timing information based on the token rate
        """
        import torch
        
        # Convert to tensor
        if y.ndim == 1:
            y = y.reshape(1, -1)
        wav_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Convert to 24kHz if needed
        if sr != 24000:
            wav_tensor = self.convert_audio(wav_tensor, sr, 24000, 1)
            # Update y to reflect the resampled audio
            y = wav_tensor.cpu().numpy()
            sr = 24000
        
        # Move to device
        wav_tensor = wav_tensor.to(self.device)
        bandwidth_id = torch.tensor([0], device=self.device)
        
        # Get tokens - exactly as in README
        with torch.no_grad():
            _, discrete_codes = self.model.encode_infer(wav_tensor, bandwidth_id=bandwidth_id)
        
        # Extract codes
        if discrete_codes.ndim == 3:
            codes = discrete_codes[0, 0].cpu().numpy()
        else:
            codes = discrete_codes[0].cpu().numpy()
        
        # Create token objects with accurate timing
        tokens = []
        total_duration = len(y[0]) / sr if y.ndim > 1 else len(y) / sr
        
        # Method 1: Calculate time_per_token based on tokens_per_second
        expected_tokens = self.tokens_per_second * total_duration
        
        # Sanity check - make sure the number of tokens is approximately what we expect
        # If there's a significant discrepancy, log a warning but proceed
        if len(codes) > 0 and abs(len(codes) - expected_tokens) > self.tokens_per_second:
            print(f"Warning: Expected ~{expected_tokens:.1f} tokens but got {len(codes)}. " 
                  f"Timing information may not be accurate.")
        
        # Create tokens with timing based on the actual token rate
        if len(codes) > 0:
            time_per_token = 1.0 / self.tokens_per_second
            for i, code in enumerate(codes):
                start_time = i * time_per_token
                end_time = (i + 1) * time_per_token
                tokens.append(Token(int(code), start_time, end_time))
        
        return tokens
    
    def decode_tokens(self, tokens: List[Token]) -> np.ndarray:
        """Decode tokens back to audio."""
        import torch
        
        # Extract token values
        codes = [token.value for token in tokens]
        
        # Prepare codes tensor
        n_q = 1  # First quantizer only
        token_tensor = torch.zeros((n_q, 1, len(codes)), dtype=torch.long, device=self.device)
        token_tensor[0, 0, :] = torch.tensor(codes, dtype=torch.long, device=self.device)
        
        # Decode - following README example
        bandwidth_id = torch.tensor([0], device=self.device)
        features = self.model.codes_to_features(token_tensor)
        audio = self.model.decode(features, bandwidth_id=bandwidth_id)
        
        # Return as numpy array
        return audio.cpu().numpy()[0]

    def _download_file(self, url, save_path, file_type="file"):
        """Download a file from a URL and save it to the specified path.
        
        Parameters
        ----------
        url : str
            URL to download the file from
        save_path : Path or str
            Path where the downloaded file will be saved
        file_type : str, default="file"
            Type of file being downloaded, for logging purposes
        """
        try:
            check_python_package_installed('requests')
            import requests
            
            print(f"Downloading {file_type} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Downloaded {file_type} to {save_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download {file_type} from {url}: {e}")