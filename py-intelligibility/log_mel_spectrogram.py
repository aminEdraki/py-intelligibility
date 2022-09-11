# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:01:32 2022

@author: 18ae5
"""

import numpy as np




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

