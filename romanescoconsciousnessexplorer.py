"""
ROMANESCO CONSCIOUSNESS VISUALIZER
===================================
Multi-scale eigenmode dynamics with hierarchical coupling.

Demonstrates how brain rhythms (delta→theta→alpha→beta→gamma) form
a fractal cascade where each scale modulates the next, creating
the "crumbling" pattern you see in romanesco broccoli.

The mathematics:
- Each band extracts 8 spatial eigenmodes (K≈2 critical regime)
- Each creates a radial mandala (like your wtf.json pipeline)
- Slow bands modulate amplitude of fast bands (phase-amplitude coupling)
- Result: Nested structure at all scales simultaneously
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, butter, filtfilt
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# SYNTHETIC EEG GENERATOR (Simulates 64-channel brain activity)
# =============================================================================

class SyntheticBrain:
    """
    Generates realistic multi-scale oscillatory data.
    Each frequency band has its own spatial pattern and temporal dynamics.
    """
    
    def __init__(self, n_channels=64, fs=250):
        self.n_channels = n_channels
        self.fs = fs
        self.t = 0
        
        # Create 2D electrode layout (8×8 grid approximating scalp)
        self.positions = self._create_electrode_grid()
        
        # Compute graph Laplacian eigenmodes (spatial harmonics)
        self.eigenmodes = self._compute_spatial_eigenmodes()
        
        # Each band gets different spatial modes + temporal dynamics
        self.band_config = {
            'delta': {'freq': (1, 4), 'modes': [0, 1], 'amp': 2.0, 'speed': 0.1},
            'theta': {'freq': (4, 8), 'modes': [1, 2, 3], 'amp': 1.5, 'speed': 0.3},
            'alpha': {'freq': (8, 13), 'modes': [2, 3, 4], 'amp': 3.0, 'speed': 0.5},
            'beta': {'freq': (13, 30), 'modes': [3, 4, 5, 6], 'amp': 1.0, 'speed': 0.8},
            'gamma': {'freq': (30, 80), 'modes': [5, 6, 7], 'amp': 0.5, 'speed': 1.5}
        }
        
    def _create_electrode_grid(self):
        """8×8 grid of electrode positions"""
        n = int(np.sqrt(self.n_channels))
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])
    
    def _compute_spatial_eigenmodes(self):
        """
        Graph Laplacian eigenmodes = spatial harmonics of the electrode array.
        Mode 0 = global average
        Mode 1 = anterior-posterior gradient
        Mode 2 = left-right gradient
        Mode 3+ = higher spatial frequencies
        """
        # Distance matrix
        D = squareform(pdist(self.positions))
        
        # Adjacency (Gaussian kernel)
        sigma = 0.5
        A = np.exp(-D**2 / (2 * sigma**2))
        np.fill_diagonal(A, 0)
        
        # Laplacian
        deg = np.sum(A, axis=1)
        L = np.diag(deg) - A
        
        # Eigenmodes (sorted by eigenvalue)
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Return first 8 modes (smooth to rough)
        return eigvecs[:, :8]
    
    def generate_frame(self, duration=1.0):
        """
        Generate one window of synthetic EEG data.
        Each band has oscillations in specific spatial modes.
        """
        n_samples = int(duration * self.fs)
        time = np.arange(n_samples) / self.fs + self.t
        
        # Initialize with noise
        eeg = np.random.randn(self.n_channels, n_samples) * 0.1
        
        # Add each frequency band with spatial structure
        for band_name, config in self.band_config.items():
            freq_range = config['freq']
            modes = config['modes']
            amp = config['amp']
            speed = config['speed']
            
            # Temporal oscillation (with slow amplitude modulation)
            center_freq = np.mean(freq_range)
            bandwidth = freq_range[1] - freq_range[0]
            
            # Carrier frequency + slow envelope
            carrier = np.sin(2 * np.pi * center_freq * time)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * speed * time)
            
            # Add frequency jitter (bandwidth)
            jitter = np.sin(2 * np.pi * (center_freq + np.random.randn() * bandwidth/4) * time)
            
            signal = carrier * envelope + jitter * 0.3
            
            # Apply spatial pattern (sum of selected eigenmodes)
            spatial_pattern = np.zeros(self.n_channels)
            for mode_idx in modes:
                phase = self.t * speed * mode_idx  # Each mode drifts
                weight = np.sin(phase) * 0.5 + 0.5
                spatial_pattern += self.eigenmodes[:, mode_idx] * weight
            
            # Broadcast to all timepoints
            eeg += amp * spatial_pattern[:, np.newaxis] * signal[np.newaxis, :]
        
        self.t += duration
        return eeg

# =============================================================================
# MULTI-SCALE EIGENMODE EXTRACTOR
# =============================================================================

def extract_band_eigenmodes(eeg, fs, band_range, eigenmodes, n_modes=8):
    """
    Filter EEG to frequency band, project onto spatial eigenmodes.
    Returns: 8D vector (power in each spatial mode)
    """
    # Bandpass filter
    nyq = fs / 2
    low, high = band_range
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    filtered = filtfilt(b, a, eeg, axis=1)
    
    # Analytic signal (Hilbert transform for instantaneous amplitude)
    analytic = hilbert(filtered, axis=1)
    amplitude = np.abs(analytic)
    
    # Average amplitude over time
    avg_amplitude = np.mean(amplitude, axis=1)  # (n_channels,)
    
    # Project onto first 8 eigenmodes
    mode_powers = []
    for i in range(n_modes):
        mode = eigenmodes[:, i]
        power = np.abs(np.dot(mode, avg_amplitude))
        mode_powers.append(power)
    
    # Normalize to 0-1
    mode_powers = np.array(mode_powers)
    if np.max(mode_powers) > 0:
        mode_powers = mode_powers / np.max(mode_powers)
    
    return mode_powers

# =============================================================================
# MANDALA GENERATOR (Radial projection like wtf.json)
# =============================================================================

def vector_to_mandala(vector_8d, size=128):
    """
    Convert 8-element vector to radial mandala.
    This is your EigenToImageNode logic.
    """
    mandala = np.zeros((size, size))
    center = size / 2
    
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Map radius to vector index
    max_r = size / 2
    r_norm = r / max_r  # 0 to 1
    
    # Interpolate vector values
    indices = r_norm * (len(vector_8d) - 1)
    indices = np.clip(indices, 0, len(vector_8d) - 1)
    
    # Bilinear interpolation
    idx_low = np.floor(indices).astype(int)
    idx_high = np.ceil(indices).astype(int)
    weight = indices - idx_low
    
    mandala = (1 - weight) * vector_8d[idx_low] + weight * vector_8d[idx_high]
    
    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter
    mandala = gaussian_filter(mandala, sigma=1.5)
    
    return mandala

# =============================================================================
# ROMANESCO CASCADE (Hierarchical coupling)
# =============================================================================

def create_romanesco_cascade(band_vectors):
    """
    Generate nested mandalas with cross-scale modulation.
    Each scale modulates the amplitude of the next (phase-amplitude coupling).
    
    band_vectors: dict with keys 'delta', 'theta', 'alpha', 'beta', 'gamma'
    """
    size = 128
    
    # Generate individual scale mandalas
    mandalas = {}
    for band, vector in band_vectors.items():
        mandalas[band] = vector_to_mandala(vector, size)
    
    # Normalize each
    for band in mandalas:
        m = mandalas[band]
        if np.max(m) > 0:
            mandalas[band] = m / np.max(m)
    
    # Hierarchical modulation (slow controls fast amplitude)
    # Delta modulates theta
    theta_modulated = mandalas['theta'] * (0.5 + 0.5 * mandalas['delta'])
    
    # Theta modulates alpha
    alpha_modulated = mandalas['alpha'] * (0.5 + 0.5 * theta_modulated)
    
    # Alpha modulates beta
    beta_modulated = mandalas['beta'] * (0.5 + 0.5 * alpha_modulated)
    
    # Beta modulates gamma
    gamma_modulated = mandalas['gamma'] * (0.5 + 0.5 * beta_modulated)
    
    # Store modulated versions
    mandalas_modulated = {
        'delta': mandalas['delta'],
        'theta': theta_modulated,
        'alpha': alpha_modulated,
        'beta': beta_modulated,
        'gamma': gamma_modulated
    }
    
    # Create composite (color-coded by scale)
    composite = np.zeros((size, size, 3))
    
    # Red channel: slow rhythms (delta + theta)
    composite[:,:,0] = mandalas_modulated['delta'] * 0.3 + mandalas_modulated['theta'] * 0.3
    
    # Green channel: medium (alpha)
    composite[:,:,1] = mandalas_modulated['alpha'] * 0.5
    
    # Blue channel: fast rhythms (beta + gamma)
    composite[:,:,2] = mandalas_modulated['beta'] * 0.3 + mandalas_modulated['gamma'] * 0.3
    
    # Normalize to 0-1
    if np.max(composite) > 0:
        composite = composite / np.max(composite)
    
    return composite, mandalas, mandalas_modulated

# =============================================================================
# MOIRÉ STRESS DETECTOR
# =============================================================================

def compute_moire_stress(mandala):
    """
    Measure high-frequency content (aliasing stress).
    This is the "crumbling" signature.
    """
    fft = np.fft.fft2(mandala)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    h, w = magnitude.shape
    center_h, center_w = h//2, w//2
    
    # High frequency = outer region
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Stress = power in outer half of frequency space
    high_freq_mask = r > (min(h, w) / 4)
    stress = np.sum(magnitude[high_freq_mask]) / np.sum(magnitude)
    
    return stress

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization():
    """
    Real-time animated display of the romanesco consciousness cascade.
    """
    # Initialize
    brain = SyntheticBrain(n_channels=64, fs=250)
    
    # Setup figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('ROMANESCO CONSCIOUSNESS: Multi-Scale Eigenmode Dynamics', 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.3)
    
    # Top row: Individual band mandalas
    ax_delta = fig.add_subplot(gs[0, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_alpha = fig.add_subplot(gs[0, 2])
    ax_beta = fig.add_subplot(gs[0, 3])
    ax_gamma = fig.add_subplot(gs[0, 4])
    
    band_axes = {
        'delta': ax_delta,
        'theta': ax_theta,
        'alpha': ax_alpha,
        'beta': ax_beta,
        'gamma': ax_gamma
    }
    
    for band, ax in band_axes.items():
        ax.set_title(f'{band.upper()} ({"1-4" if band=="delta" else "4-8" if band=="theta" else "8-13" if band=="alpha" else "13-30" if band=="beta" else "30-80"} Hz)')
        ax.axis('off')
    
    # Middle row: Modulated versions
    ax_delta_mod = fig.add_subplot(gs[1, 0])
    ax_theta_mod = fig.add_subplot(gs[1, 1])
    ax_alpha_mod = fig.add_subplot(gs[1, 2])
    ax_beta_mod = fig.add_subplot(gs[1, 3])
    ax_gamma_mod = fig.add_subplot(gs[1, 4])
    
    mod_axes = {
        'delta': ax_delta_mod,
        'theta': ax_theta_mod,
        'alpha': ax_alpha_mod,
        'beta': ax_beta_mod,
        'gamma': ax_gamma_mod
    }
    
    for ax in mod_axes.values():
        ax.set_title('Modulated')
        ax.axis('off')
    
    # Stress meters
    ax_stress = fig.add_subplot(gs[1, 5])
    ax_stress.set_title('Moiré Stress\n(Crumbling)')
    ax_stress.set_ylim(0, 0.5)
    ax_stress.set_xlim(-0.5, 4.5)
    ax_stress.grid(True, alpha=0.3)
    
    # Bottom row: Composite romanesco + coupling diagram
    ax_composite = fig.add_subplot(gs[2, :3])
    ax_composite.set_title('ROMANESCO COMPOSITE (RGB = Slow/Medium/Fast)', fontweight='bold')
    ax_composite.axis('off')
    
    ax_coupling = fig.add_subplot(gs[2, 3:])
    ax_coupling.set_title('Hierarchical Coupling Flow', fontweight='bold')
    ax_coupling.axis('off')
    
    # Coupling diagram (static)
    coupling_text = """
    DELTA (1-4 Hz)
       ↓ modulates amplitude
    THETA (4-8 Hz)
       ↓ modulates amplitude
    ALPHA (8-13 Hz)
       ↓ modulates amplitude
    BETA (13-30 Hz)
       ↓ modulates amplitude
    GAMMA (30-80 Hz)
    
    Each scale reads the scale below
    and writes to the scale above.
    
    This is PHASE-AMPLITUDE COUPLING:
    the mathematics of romanesco.
    """
    ax_coupling.text(0.1, 0.5, coupling_text, 
                     fontsize=11, family='monospace',
                     verticalalignment='center')
    
    # Animation state
    im_composite = None
    stress_bars = None
    
    def update(frame):
        nonlocal im_composite, stress_bars
        
        # Generate new brain data
        eeg = brain.generate_frame(duration=1.0)
        
        # Extract eigenmodes for each band
        band_vectors = {}
        for band, config in brain.band_config.items():
            vector = extract_band_eigenmodes(
                eeg, brain.fs, config['freq'], brain.eigenmodes, n_modes=8
            )
            band_vectors[band] = vector
        
        # Create romanesco cascade
        composite, mandalas_raw, mandalas_mod = create_romanesco_cascade(band_vectors)
        
        # Display individual band mandalas (raw)
        for band, ax in band_axes.items():
            ax.clear()
            ax.imshow(mandalas_raw[band], cmap='twilight', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(f'{band.upper()}')
        
        # Display modulated versions
        for band, ax in mod_axes.items():
            ax.clear()
            ax.imshow(mandalas_mod[band], cmap='hot', vmin=0, vmax=1)
            ax.axis('off')
        
        # Compute and display stress
        stresses = {band: compute_moire_stress(mandalas_mod[band]) 
                    for band in band_vectors.keys()}
        
        ax_stress.clear()
        ax_stress.set_ylim(0, 0.5)
        ax_stress.set_title('Moiré Stress\n(Crumbling)')
        colors = ['red', 'orange', 'yellow', 'cyan', 'blue']
        bands_list = list(stresses.keys())
        stress_vals = [stresses[b] for b in bands_list]
        ax_stress.bar(range(len(bands_list)), stress_vals, color=colors, alpha=0.7)
        ax_stress.set_xticks(range(len(bands_list)))
        ax_stress.set_xticklabels([b[0].upper() for b in bands_list])
        ax_stress.grid(True, alpha=0.3)
        ax_stress.axhline(y=0.3, color='white', linestyle='--', alpha=0.5, label='Critical')
        
        # Display composite
        ax_composite.clear()
        ax_composite.imshow(composite)
        ax_composite.axis('off')
        ax_composite.set_title('ROMANESCO COMPOSITE (RGB = Slow/Medium/Fast)', 
                               fontweight='bold', fontsize=12)
        
        # Add info text
        fig.text(0.5, 0.02, 
                 f'Frame {frame} | Red=Delta+Theta | Green=Alpha | Blue=Beta+Gamma | '
                 f'High stress = crumbling (Nyquist violation)',
                 ha='center', fontsize=10, color='white',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Animate
    anim = FuncAnimation(fig, update, frames=200, interval=100, repeat=True)
    
    plt.tight_layout()
    return fig, anim

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ROMANESCO CONSCIOUSNESS VISUALIZER")
    print("=" * 70)
    print()
    print("Demonstrating multi-scale eigenmode dynamics:")
    print("  • Each frequency band extracts 8 spatial eigenmodes (K≈2 regime)")
    print("  • Each creates a radial mandala (like romanesco spirals)")
    print("  • Slow bands modulate fast bands (hierarchical coupling)")
    print("  • Result: Fractal structure at all scales simultaneously")
    print()
    print("The 'Moiré Stress' shows where the system is 'crumbling':")
    print("  • High stress = Nyquist violation (can't represent at this scale)")
    print("  • This stress is INFORMATION about the next scale")
    print()
    print("Watch the composite (bottom): nested mandalas like romanesco broccoli.")
    print("=" * 70)
    print()
    print("Generating visualization... (close window to exit)")
    print()
    
    fig, anim = create_visualization()
    plt.show()
    
    print()
    print("Visualization closed.")
    print()
    print("KEY INSIGHTS:")
    print("  1. Each band operates at K≈2 (8D = 2³ bits of structure)")
    print("  2. Cross-scale modulation creates phase-amplitude coupling")
    print("  3. Moiré stress reveals where information needs next scale")
    print("  4. DNA does this across 5+ scales (atoms→molecules→cells→tissue→organ)")
    print("  5. Your brain does this across frequency bands (delta→gamma)")
    print()
    print("=" * 70)