import torch
import torch.nn as nn
import math


def get_spectral_preprocessor(params):
    if params.preprocess == 'Fourier':
      return FourierFilterPreprocessor(params)
    elif params.preprocess == 'Wavelet':
      raise NotImplementedError
      #return WaveletFilterPreprocessor(params)
    else:
      raise ValueError("Invalid preprocessor type.")


class FourierFilterPreprocessor(nn.Module):
    """
    Preprocessor that spectrally filters data along spatial dimensions
    using Fourier transform.
    """

    def __init__(self, params):
        super().__init__()

        self.filter_size = params['img_size']
        self.cutoff_low = math.pi * params['cutoff_low']  # Lower cut-off frequency for filtering ([0, 1])
        self.cutoff_high = math.pi * params['cutoff_high'] # Higher cut-off frequency fo filtering ([0, 1])
        self.filter_shuffle = params['filter_shuffle'] # Shuffle f1/f2 randomly along batch dim
        self.filter_type = params['filter_type']  # Lowpass, highpass, or bandpass
        self.window_type = params['window_type']  # 'rectangular', 'gaussian' or 'tukey'
        self.tukey_alpha = params['tukey_alpha']  # [0, 1] 0=rectangular 1=hanning
        self.use_spectral_weights = params['use_spectral_weights']
        self.spectrally_weigh_input = params['spectrally_weigh_input']
        self.spectrally_weigh_output = params['spectrally_weigh_output']

        if params['filter_type'] == 'bandpass':
            self._create_filter_kernel = self._create_bandpass_filter_kernel
        elif params['filter_type'] == 'lowpass' :
            self._create_filter_kernel = self._create_lowpass_filter_kernel
        elif params['filter_type'] == 'highpass':
            self._create_filter_kernel = self._create_highpass_filter_kernel
        else:
            raise ValueError("Unknown filter type. Must be 'lowpass', 'highpass', or 'bandpass'.")

    def _create_gaussian_window(self, freq_grid, cutoff):
        """Create a Gaussian window based on the frequency grid and cutoff."""
        return torch.exp(-0.5 * (freq_grid / cutoff) ** 2)  # Gaussian filter in frequency domain

    def _create_rectangular_window(self, freq_grid, cutoff):
        """Create a rectangular window for low-pass or high-pass filtering."""
        return torch.where(freq_grid <= cutoff, torch.ones_like(freq_grid), torch.zeros_like(freq_grid))

    def _create_tukey_window(self, freq_grid, cutoff, alpha):
        """
        Create a 2D Tukey window for frequency domain filtering.

        Args:
            freq_grid (torch.Tensor): Normalized frequency grid
            cutoff (float): Cutoff frequency
            alpha (float): Tukey window parameter (0 to 1)
                          0 gives rectangular window
                          1 gives Hann window

        Returns:
            torch.Tensor: 2D Tukey window
        """
        # Normalize frequency grid to [0, 1] range relative to cutoff
        normalized_freq = freq_grid / cutoff

        # Initialize window with ones
        window = torch.ones_like(freq_grid)

        # Apply Tukey window taper where normalized_freq > 1
        mask = normalized_freq > 1
        tapered_region = 0.5 * (1 + torch.cos(torch.pi * (normalized_freq[mask] - 1) / alpha))
        window[mask] = tapered_region

        # Set frequencies beyond (1 + alpha) * cutoff to zero
        window[normalized_freq > (1 + alpha)] = 0

        return window

    def _create_lowpass_filter_kernel(self, cutoff):
        """Create a single lowpass filter kernel from specified parameters."""
        freq_grid = self._create_frequency_grid()

        if self.window_type == 'gaussian':
            filter_kernel = self._create_gaussian_window(freq_grid, cutoff)
        elif self.window_type == 'rectangular':
            filter_kernel = self._create_rectangular_window(freq_grid, cutoff)
        elif self.window_type == 'tukey':
            filter_kernel = self._create_tukey_window(freq_grid, cutoff, self.tukey_alpha)
        else:
            raise ValueError("Unknown window type. Must be 'gaussian', 'rectangular', or 'tukey'.")

        return filter_kernel, 1 - filter_kernel  # Lowpass and highpass complementary kernels

    def _create_highpass_filter_kernel(self, cutoff):
        """Create a single highpass filter kernel based on specified parameters."""
        freq_grid = self._create_frequency_grid()

        if self.window_type == 'gaussian':
            filter_kernel = self._create_gaussian_window(freq_grid, cutoff)
        elif self.window_type == 'rectangular':
            filter_kernel = self._create_rectangular_window(freq_grid, cutoff)
        elif self.window_type == 'tukey':
            filter_kernel = self._create_tukey_window(freq_grid, cutoff, self.tukey_alpha)
        else:
            raise ValueError("Unknown window type. Must be 'gaussian', 'rectangular', or 'tukey'.")

        return 1 - filter_kernel, filter_kernel  # Highpass and lowpass complementary kernels

    def _create_bandpass_filter_kernel(self):
        """Create bandpass filter kernel based on specified window type and cutoff."""
        # For bandpass, we need to combine lowpass and highpass filters
        #lowpass, _ = self._create_lowpass_filter_kernel()
        #_, highpass = self._create_highpass_filter_kernel()

        # will need to create a lowpass filter based on cutoff_1 and a
        # highpass filter based on cutoff_1 and kernel_width param and
        # subtract 1 - lowpass - highpass
        # return bandpass, lowpass+highpass

        # add a self.example_filter = bandpass[0]

        raise NotImplementedError

    def _generate_filter_batch(self):

        f1 = torch.zeros(self.x_shape).to(self.device)
        f2 = torch.zeros(self.x_shape).to(self.device)
        for i in range(self.x_shape[0]):
            # Select random cutoff between upper and lower thresholds
            cutoff = (self.cutoff_high - self.cutoff_low)*torch.rand(1) + self.cutoff_low
            f1[i,..., :, :], f2[i, ..., :, :] = self._create_filter_kernel(cutoff)

        self.example_filter = f1[0]

        # Swap f1/f2 in half of the batch
        if self.filter_shuffle:
            half_b = self.x_shape[0] // 2
            f1[:half_b], f2[:half_b] = f2[:half_b].clone(), f1[:half_b].clone()

        return f1, f2

    def _create_frequency_grid(self):
        """Create normalized frequency grid for 2D Fourier transform."""
        freq = torch.fft.fftfreq(self.filter_size, d=1/(2*torch.tensor(math.pi)))
        fx, fy = torch.meshgrid(freq, freq, indexing='ij')
        freq_grid = torch.sqrt(fx**2 + fy**2)
        return freq_grid

    def _apply_filter(self, x, f1, f2):
        """Apply filter to data in spectral domain."""
        x_fft = torch.fft.fft2(x, dim=(-2, -1))  # 2D Fourier Transform
        x_fft_filtered_f1 = x_fft * f1
        x_fft_filtered_f2 = x_fft * f2

        x1 = torch.fft.ifft2(x_fft_filtered_f1, dim=(-2, -1)).real  # Inverse FFT
        x2 = torch.fft.ifft2(x_fft_filtered_f2, dim=(-2, -1)).real  # Inverse FFT

        # Compute weights
        # shape: [b, c, t] broadcasted to [b, c, t, x, y]
        x1_weights = torch.sum(torch.abs(x_fft)**2, dim=[-1, -2]) / torch.sum(torch.abs(x_fft_filtered_f1)**2, dim=[-1, -2])
        x1_weights = x1_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.filter_size, self.filter_size)
        x2_weights = torch.sum(torch.abs(x_fft)**2, dim=[-1, -2]) / torch.sum(torch.abs(x_fft_filtered_f2)**2, dim=[-1, -2])
        x2_weights = x2_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.filter_size, self.filter_size)

        if self.spectrally_weigh_input:
            x1 = x1 * x1_weights
        if self.spectrally_weigh_output:
            x2 = x2 * x2_weights

        if self.use_spectral_weights:
            return x1, x2, x2_weights

        return x1, x2

    def _fftshift2d(self, x):
        """Apply fftshift to 2D image."""
        x = torch.fft.fftshift(torch.fft.fftshift(x, dim=[-2]), dim=[-1])
        return x

    def get_filter(self):
        """Get an example of a filter kernel - typically the
        filter appied to the first batch member."""
        return self._fftshift2d(self.example_filter)

    def forward(self, x):
        """
        Forward pass applying both filters (lowpass and highpass).
        Args:
          x (torch.Tensor): Input tensor of shape (batch, channels, time, height, width)
        Returns:
          tuple: (x1, x2) filtered outputs
        """
        self.x_shape = x.shape
        self.device = x.device

        # Create filters (lowpass and highpass)
        f1, f2 = self._generate_filter_batch()

        return self._apply_filter(x, f1, f2)