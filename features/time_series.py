"""
Time-Series Features Module

Computes temporal features from multiple observations of an object,
similar to what ALeRCE and other transient brokers use.

These features help distinguish:
- Supernovae: Fast rise, slow decline
- AGN: Random flickering over long periods  
- Variable stars: Periodic brightness changes
- Artifacts: Single-epoch noise

Features Implemented:
1. Amplitude - How much the brightness changes
2. Rise time - How fast it brightens
3. Fade rate - How fast it dims
4. Periodicity - Does it repeat on a schedule
5. Color evolution - How the color changes over time

Usage:
    from features.time_series import TimeSeriesFeatures
    
    ts = TimeSeriesFeatures()
    features = ts.compute(observations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Single observation of an object."""
    mjd: float  # Modified Julian Date
    mag: float  # Magnitude (brightness)
    mag_err: float = 0.1  # Magnitude error
    band: str = "r"  # Filter band (g, r, i, z)
    
    @property
    def flux(self) -> float:
        """Convert magnitude to flux."""
        return 10 ** (-0.4 * (self.mag - 25))


@dataclass
class TimeSeriesFeatures:
    """
    Computes time-series features from observations.
    
    Based on features used by ALeRCE, BTSbot, and other transient classifiers.
    """
    
    # Core features
    n_observations: int = 0
    time_span_days: float = 0.0
    
    # Amplitude features
    amplitude: float = 0.0  # Max - Min magnitude
    mean_mag: float = 0.0
    std_mag: float = 0.0
    max_mag: float = 0.0  # Brightest (lowest mag)
    min_mag: float = 0.0  # Faintest (highest mag)
    
    # Temporal features
    rise_time_days: float = 0.0  # Time to reach peak brightness
    fade_rate: float = 0.0  # Magnitude per day fading rate
    peak_mjd: float = 0.0  # When it was brightest
    
    # Variability features
    beyond_1std: float = 0.0  # Fraction of points beyond 1 std
    max_slope: float = 0.0  # Steepest brightness change
    mean_slope: float = 0.0
    
    # Periodicity (placeholder - requires more complex analysis)
    is_periodic: bool = False
    period_days: float = 0.0
    
    # Multi-band features
    color_g_r: float = 0.0  # g - r color
    color_r_i: float = 0.0  # r - i color
    color_evolution: float = 0.0  # How color changes over time
    
    # Metadata
    computed_at: str = ""
    bands_present: List[str] = field(default_factory=list)
    
    @classmethod
    def compute(cls, observations: List[Observation]) -> "TimeSeriesFeatures":
        """
        Compute all time-series features from observations.
        
        Args:
            observations: List of Observation objects, sorted by time.
        
        Returns:
            TimeSeriesFeatures with all computed values.
        """
        if not observations:
            return cls(computed_at=datetime.now().isoformat())
        
        # Sort by time
        obs_sorted = sorted(observations, key=lambda x: x.mjd)
        n = len(obs_sorted)
        
        # Basic stats
        mags = np.array([o.mag for o in obs_sorted])
        mjds = np.array([o.mjd for o in obs_sorted])
        bands = list(set(o.band for o in obs_sorted))
        
        # Amplitude features
        amplitude = np.max(mags) - np.min(mags)
        mean_mag = np.mean(mags)
        std_mag = np.std(mags) if n > 1 else 0.0
        max_mag = np.min(mags)  # Brightest = lowest magnitude
        min_mag = np.max(mags)  # Faintest = highest magnitude
        
        # Time span
        time_span = mjds[-1] - mjds[0] if n > 1 else 0.0
        
        # Peak time (brightest observation)
        peak_idx = np.argmin(mags)
        peak_mjd = mjds[peak_idx]
        
        # Rise time (time to reach peak from first observation)
        rise_time = peak_mjd - mjds[0] if peak_idx > 0 else 0.0
        
        # Fade rate (after peak)
        fade_rate = 0.0
        if peak_idx < n - 1:
            post_peak_mags = mags[peak_idx:]
            post_peak_mjds = mjds[peak_idx:]
            if len(post_peak_mjds) > 1:
                # Linear fit to get fade rate
                time_diff = post_peak_mjds[-1] - post_peak_mjds[0]
                mag_diff = post_peak_mags[-1] - post_peak_mags[0]
                fade_rate = mag_diff / time_diff if time_diff > 0 else 0.0
        
        # Variability
        beyond_1std = np.sum(np.abs(mags - mean_mag) > std_mag) / n if std_mag > 0 else 0.0
        
        # Slopes
        slopes = []
        for i in range(1, n):
            dt = mjds[i] - mjds[i-1]
            dm = mags[i] - mags[i-1]
            if dt > 0:
                slopes.append(dm / dt)
        
        max_slope = np.max(np.abs(slopes)) if slopes else 0.0
        mean_slope = np.mean(np.abs(slopes)) if slopes else 0.0
        
        # Multi-band colors (if multiple bands available)
        color_g_r = 0.0
        color_r_i = 0.0
        
        if 'g' in bands and 'r' in bands:
            g_obs = [o for o in obs_sorted if o.band == 'g']
            r_obs = [o for o in obs_sorted if o.band == 'r']
            if g_obs and r_obs:
                color_g_r = np.mean([o.mag for o in g_obs]) - np.mean([o.mag for o in r_obs])
        
        if 'r' in bands and 'i' in bands:
            r_obs = [o for o in obs_sorted if o.band == 'r']
            i_obs = [o for o in obs_sorted if o.band == 'i']
            if r_obs and i_obs:
                color_r_i = np.mean([o.mag for o in r_obs]) - np.mean([o.mag for o in i_obs])
        
        return cls(
            n_observations=n,
            time_span_days=time_span,
            amplitude=amplitude,
            mean_mag=mean_mag,
            std_mag=std_mag,
            max_mag=max_mag,
            min_mag=min_mag,
            rise_time_days=rise_time,
            fade_rate=fade_rate,
            peak_mjd=peak_mjd,
            beyond_1std=beyond_1std,
            max_slope=max_slope,
            mean_slope=mean_slope,
            color_g_r=color_g_r,
            color_r_i=color_r_i,
            bands_present=bands,
            computed_at=datetime.now().isoformat(),
        )
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.n_observations,
            self.time_span_days,
            self.amplitude,
            self.mean_mag,
            self.std_mag,
            self.rise_time_days,
            self.fade_rate,
            self.beyond_1std,
            self.max_slope,
            self.mean_slope,
            self.color_g_r,
            self.color_r_i,
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for the vector."""
        return [
            "n_observations",
            "time_span_days",
            "amplitude",
            "mean_mag",
            "std_mag",
            "rise_time_days",
            "fade_rate",
            "beyond_1std",
            "max_slope",
            "mean_slope",
            "color_g_r",
            "color_r_i",
        ]
    
    def classify_type(self) -> Tuple[str, float]:
        """
        Simple rule-based classification based on features.
        
        Returns:
            Tuple of (classification, confidence)
        """
        # These are rough heuristics - a proper classifier would use ML
        
        if self.n_observations < 2:
            return "insufficient_data", 0.0
        
        # Supernova-like: Fast rise, slow fade, large amplitude
        if self.rise_time_days < 20 and self.fade_rate > 0.02 and self.amplitude > 1.0:
            return "supernova_candidate", 0.7
        
        # AGN-like: Long time span, moderate variability, no clear peak
        if self.time_span_days > 100 and self.std_mag > 0.3:
            return "agn_candidate", 0.5
        
        # Variable star-like: Periodic or regular variability
        if self.is_periodic:
            return "variable_star", 0.6
        
        # Artifact: Single observation or very short
        if self.time_span_days < 0.1:
            return "artifact_or_single", 0.4
        
        return "unknown", 0.3


# Convenience function
def compute_time_features(observations: List[Dict]) -> TimeSeriesFeatures:
    """
    Compute time-series features from a list of observation dicts.
    
    Args:
        observations: List of dicts with keys: mjd, mag, mag_err, band
    """
    obs_list = [
        Observation(
            mjd=o.get("mjd", 0),
            mag=o.get("mag", 0),
            mag_err=o.get("mag_err", 0.1),
            band=o.get("band", "r")
        )
        for o in observations
    ]
    return TimeSeriesFeatures.compute(obs_list)
