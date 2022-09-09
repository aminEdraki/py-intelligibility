# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:14:21 2022

@author: Amin Edraki

Separated spectro-temporal modulation Gabor filter bank for wstmi.
This script extracts the separable spectro-temporal modulation
represntation from a log Mel-spectrogram as described in [1, 2].
Modified from the reference [1, 3] implementation.

[1] Schädler, M. R., & Kollmeier, B. (2015). Separable spectro-temporal
Gabor filter bank features: Reducing the complexity of robust features
for automatic speech recognition. The Journal of the Acoustical 
Society of America, 137(4), 2047-2059.
[2] Edraki, A., Chan, W. Y., Jensen, J., & Fogerty, D. (2020). 
Speech Intelligibility Prediction Using Spectro-Temporal Modulation
Analysis. IEEE/ACM Transactions on Audio, Speech, and Language 
Processing, 29, 210-225.
[3] Schädler, M. R., Meyer, B. T., & Kollmeier, B. (2012). 
Spectro-temporal modulation subspace-spanning filter bank features 
for robust automatic speech recognition. The Journal of the Acoustical
Society of America, 131(5), 4134-4151.

"""

import numpy as np


def hann_win(width):
    x_center = 0.5
    step = 1/width
    right = np.arange(x_center, 1+step, step=step)
    left  = np.arange(x_center, 0-step, step=-step)
    
    x_values = np.transpose(np.concatenate((np.flip(left, axis=0), right[1:]), axis=0))
    valid_values_mask = (x_values > 0) & (x_values < 1)
    valid_values_mask[0] = False
    valid_values_mask[-1] = False
    window_function = 0.5 * (1 - (np.cos(2*np.pi*x_values[valid_values_mask])))
    
    return window_function


def gfilter_gen(omega, nu, phi, size_max):
    w = np.Inf;
    if omega > 0:
        w = 2*np.pi / abs(omega) * nu / 2
        
    if w > size_max:
        w = size_max
        omega = 0
    
    envelope = hann_win(w)
    win_size = envelope.size
    x_0 = (win_size+1) / 2
    x = np.arange(1, win_size+1, 1)
    sinusoid = np.exp(1j * (omega*(x - x_0) + phi))
    gfilter  = np.real(envelope * sinusoid)    
    envelope_mean = np.mean(envelope)
    gfilter_mean = np.mean(gfilter)
    
    if omega != 0:    
        gfilter = gfilter - envelope/envelope_mean * gfilter_mean
    
    gfilter = gfilter / max(abs(np.fft.fft(gfilter)))
    return gfilter


def gbfb1d(X, size_max, nu, phase, omega):
    out = []
    for i in range(omega.size):
        gfilter = gfilter_gen(omega[i], nu, phase, size_max)
        in_filtered = np.apply_along_axis(lambda m: np.convolve(m, gfilter, mode='full'), axis=0, arr=X)
        
        M = gfilter.size
        N = X.shape[0]
        idx = int(np.floor((M - 1)/2))
        in_filtered = in_filtered[idx:idx+N]
        out.append(in_filtered)
    return out


def gbfb_axis(omega_max, size_max, nu, distance):
    omega_min = (np.pi * nu) / size_max;
    c = distance * 8 / nu;
    space = (1 + c/2) / (1 - c/2);
    omega = [];
    count = 0
    last_omega = omega_max
    
    while last_omega/space > omega_min:
        omega.append(omega_max/(space**count))
        count = count + 1
        last_omega = omega[-1]
    
    omega = np.flip(omega)
    omega = np.insert(omega, 0, 0)
    return omega


def sgbfb(log_mel_spec, stm_channels=np.ones((11, 5)), omega_s=None, omega_t=None):
    omega_max = np.array([2*np.pi/3, np.pi/2])
    num_bands = log_mel_spec.shape[0]
    nu = [3.5, 3.5]
    size_max = [3*num_bands, 40]
    phases = [0, 0];
    distance = [0.2, 0.2];
    context = int(np.floor(size_max[1]/2))
    
    if omega_s is None and omega_t is None:
        row, col = np.nonzero(stm_channels)
        row = np.unique(row)
        col = np.unique(col)
        omega_s = gbfb_axis(omega_max[0], size_max[0], nu[0], distance[0])
        omega_t = gbfb_axis(omega_max[1], size_max[1], nu[1], distance[1])
        omega_s = omega_s[row];
        omega_t = omega_t[col];
    
    
    left_context = np.repeat(np.expand_dims(log_mel_spec[:, 0], axis=1), repeats=context, axis=1)
    right_context = np.repeat(np.expand_dims(log_mel_spec[:, -1], axis=1), repeats=context, axis=1)
    
    log_mel_spec = np.concatenate((left_context, log_mel_spec, right_context), axis=1)
    
    features_spec = gbfb1d(log_mel_spec, size_max[0], nu[0], phases[0], omega_s)
    features_spec = np.concatenate(features_spec, axis=0)
    features_spec = gbfb1d(np.transpose(features_spec), size_max[1], nu[1], phases[1], omega_t)
    features_spec = np.concatenate(features_spec, axis=0)
    features_spec = np.transpose(features_spec)
    
    time_frames   = int(log_mel_spec.shape[1])
    freq_bins     = int(log_mel_spec.shape[0])
    
    STM_dec       = np.zeros((omega_s.size, omega_t.size, freq_bins, time_frames-2*context))
    for s in range(omega_s.size):
        for r in range(omega_t.size):
            filtered_spec = features_spec[s*freq_bins:(s+1)*freq_bins, r*time_frames:(r+1)*time_frames]
            STM_dec[s, r, ...] = filtered_spec[:, context:-context]

    return STM_dec
    


    