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
    


def log_mel_spectrogram(signal, fs, win_shift=10, win_length=25, freq_range=None, num_bands=130, band_factor=1):
    signal = np.squeeze(signal)
    assert signal.ndim == 1
    if freq_range is None:
        freq_range = [64.0, min(np.floor(fs/2), 12000)]
        
    M = round(win_shift/1000*fs)
    N = round(win_length/1000*fs)
    num_coeff = 1024
    
    
    num_frames = int(1 + np.floor((len(signal) - N) / M))
    
    
    
    
    frames = np.zeros((N, num_frames))
    for i in range(num_frames):
        frames[:, i] = signal[i*M:(i*M+N)]
    

    window_function = np.hamming(N)
    window_function = window_function / np.sqrt(np.mean(window_function**2))
    window_function = np.expand_dims(window_function, axis=1)
    signal_frame = frames*window_function
    
   
    spec = (1/num_coeff) * abs(np.fft.fft(signal_frame, num_coeff, axis=0))
    freq_centers = mel2hz(np.linspace(hz2mel(freq_range[0]), hz2mel(freq_range[1]), (num_bands+1)*band_factor+1))
    mel_spec, _ = triafbmat(fs, num_coeff, freq_centers, [1, 1]*band_factor)
    
    
    
    mel_spec = np.matmul(mel_spec, spec)
    

    freq_centers = freq_centers[band_factor:-band_factor]
    log_mel_spec = np.maximum(-20, np.minimum(0, 20*np.log10(np.maximum(mel_spec, 0))) + 130)
    return log_mel_spec, freq_centers  









def triafbmat(fs, num_coeff, freq_centers, width):
    width_left = width[0]
    width_right = width[1]
    freq_centers_idx = np.round(freq_centers/fs * num_coeff) - 1
    num_bands = len(freq_centers)-(width_left+width_right)
    transmat = np.zeros((num_bands, num_coeff))
    for i in range(num_bands):
        left = int(freq_centers_idx[i])
        center = int(freq_centers_idx[i+width_left])
        right = int(freq_centers_idx[i+width_left+width_right])
        start_raise = 0
        stop_raise = 1
        start_fall = 1
        stop_fall = 0
        if left >= 1:
            transmat[i, left:center+1] = np.linspace(start_raise, stop_raise, center-left+1)
    
        if right <= num_coeff:      
            transmat[i, center:right+1] = np.linspace(start_fall, stop_fall, right-center+1)
        
    
    return transmat, freq_centers_idx


def mel2hz(m):
    f = 700*((10.**(m/2595))-1)
    return f

def hz2mel(f):
    m = 2595*np.log10(1+f/700)
    return m

    