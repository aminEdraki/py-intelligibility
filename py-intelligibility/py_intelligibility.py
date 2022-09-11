# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:08:00 2022

@author: Amin Edraki
"""

from utils import log_mel_spectrogram, sgbfb
import numpy as np


def pystgi(clean_speech, degraded_speech, Fs):
    """
    Implementation of the Spectro-Temporal Glimpsing Index (STGI) 
    predictor, described in [1].
 
    Parameters
    ----------
    clean_speech : numpy array - float
        clean reference time domain signal.
    degraded_speech : numpy array - float
        noisy/processed time domain signal.
    Fs : float/int
        sampling frequency in Hz.

    Returns
    -------
    rho : float
        intelligibility index.


    References:
    [1] A. Edraki, W.-Y. Chan, J. Jensen, & D. Fogerty, 
    “A Spectro-Temporal Glimpsing Index (STGI) for Speech Intelligibility Prediction," 
    Proc. Interspeech, 5 pages, Aug 2021.
    """
    assert clean_speech.size == degraded_speech.size
    assert Fs == 10000
    STM_channels = np.ones((11, 4))
    thresholds   = [[0.252,0.347,0.275,0.189],
                    [0.502,0.495,0.404,0.279],
                    [0.486,0.444,0.357,0.247],
                    [0.456,0.405,0.332,0.229],
                    [0.426,0.361,0.287,0.191],
                    [0.357,0.299,0.229,0.150],
                    [0.269,0.228,0.175,0.114],
                    [0.185,0.158,0.118,0.075],
                    [0.119,0.103,0.073,0.047],
                    [0.081,0.067,0.047,0.030],
                    [0.050,0.043,0.031,0.020]]

    win_length  = 25.6
    win_shift   = win_length/2
    freq_range  = [64, min(np.floor(Fs/2), 12000)]
    num_bands   = 130
    band_factor = 1
    N           = 40
    eps = 1e-15
    
    X, _    = log_mel_spectrogram(clean_speech, fs=Fs, win_shift=win_shift, win_length=win_length, freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)
    Y, _    = log_mel_spectrogram(degraded_speech, fs=Fs, win_shift=win_shift, win_length=win_length, freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)

    X_hat     = sgbfb(X, STM_channels)
    Y_hat     = sgbfb(Y, STM_channels)
    
    intell    = np.zeros((X_hat.shape[3] - N + 1,))
    
    
    for n in range(N-1, X_hat.shape[3]):
        X_seg = X_hat[..., (n-N+1):n+1]
        Y_seg = Y_hat[..., (n-N+1):n+1]
        
        X_seg = X_seg + eps*np.random.rand(*X_seg.shape)
        Y_seg = Y_seg + eps*np.random.rand(*Y_seg.shape)
        
        X_seg = X_seg - np.expand_dims(np.mean(X_seg, axis=3), axis=3)
        Y_seg = Y_seg - np.expand_dims(np.mean(Y_seg, axis=3), axis=3)
        X_seg = X_seg / np.expand_dims(np.sqrt(np.sum(X_seg*X_seg, axis=3)), axis=3)
        Y_seg = Y_seg / np.expand_dims(np.sqrt(np.sum(Y_seg*Y_seg, axis=3)), axis=3)
        
        X_seg = X_seg - np.expand_dims(np.mean(X_seg, axis=2), axis=2)
        Y_seg = Y_seg - np.expand_dims(np.mean(Y_seg, axis=2), axis=2)
        X_seg = X_seg / np.expand_dims(np.sqrt(np.sum(X_seg*X_seg, axis=2)), axis=2)
        Y_seg = Y_seg / np.expand_dims(np.sqrt(np.sum(Y_seg*Y_seg, axis=2)), axis=2)
        
        d = np.squeeze(np.sum(X_seg * Y_seg, axis=2));
        d = np.squeeze(np.mean(d, axis=2));
        g = d > thresholds
        intell[n-N+1] = np.mean(g[:])

    rho = np.mean(intell[:])
    return rho






def pywstmi(clean_speech, degraded_speech, Fs):
    """
    Implementation of the weighted Spectro-Temporal Modulation Index (wSTMI) 
    predictor, described in [1].
 
    Parameters
    ----------
    clean_speech : numpy array - float
        clean reference time domain signal.
    degraded_speech : numpy array - float
        noisy/processed time domain signal.
    Fs : float/int
        sampling frequency in Hz.

    Returns
    -------
    rho : float
        intelligibility index.


    References:
    [1] A. Edraki, W.-Y. Chan, J. Jensen, & D. Fogerty, 
    Speech Intelligibility Prediction Using Spectro-Temporal Modulation
    Analysis. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 210-225.
    """
    assert clean_speech.size == degraded_speech.size
    assert Fs == 10000
    
    win_length   = 25.6
    win_shift    = win_length/2
    freq_range  = [64, min(np.floor(Fs/2), 12000)]
    num_bands    = 130
    band_factor  = 1
    eps = 1e-15
    omega_s = np.array([0.081,0.128,0.326,0.518])
    omega_t = np.array([0,0.389,0.619])
    
    w   = [[0.000,0.031,0.140],
           [0.013,0.041,0.055],
           [0.459,0.528,0.000],
           [0.151,0.000,0.000]]
    b   = 0.16;
    
    X_spec, _    = log_mel_spectrogram(clean_speech, fs=Fs, win_shift=win_shift, win_length=win_length, freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)
    Y_spec, _    = log_mel_spectrogram(degraded_speech, fs=Fs, win_shift=win_shift, win_length=win_length, freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)

    X     = sgbfb(X_spec, omega_s=omega_s, omega_t=omega_t)
    Y     = sgbfb(Y_spec, omega_s=omega_s, omega_t=omega_t)

    X = X + eps*np.random.rand(*X.shape)
    Y = Y + eps*np.random.rand(*Y.shape)
    
    X   = X - np.expand_dims(np.mean(X, axis=3), axis=3)
    Y   = Y - np.expand_dims(np.mean(Y, axis=3), axis=3)
    X   = X / np.expand_dims(np.sqrt(np.sum(X*X, axis=3)), axis=3)
    Y   = Y / np.expand_dims(np.sqrt(np.sum(Y*Y, axis=3)), axis=3)
    rho = np.mean(np.sum(X*Y, axis=3), axis=2)
    d   = np.sum(w*rho) + b

    return d











