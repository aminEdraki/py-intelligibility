# py-intelligibility
Python implementation of the following speech intelligibility prediction methods:
weighted Spectro-Temporal Modulation Index (wSTMI)
Spectro-Temporal Glimpsing Index (STGI)


## Usage
The functions ```wstmi``` and ```stgi``` take three inputs:
```Python
d = pywstmi(clean_speech, degraded_speech, sampling_frequency)
d = pystgi(clean_speech, degraded_speech, sampling_frequency)
```

* ```clean_speech```: A numpy array containing a single-channel clean (reference) speech signal.
* ```degraded_speech```: A numpy array containing a single-channel degraded/processed speech signal.
* ```sampling_frequency```: The sampling frequency of the input signals in ```Hz```.

Note that the clean and degraded speech signals must be time-aligned and of the same length. The algorithms only support 10 KHz sampling rate.

## Missing Features
Voice activity detection is not implemented yet.

## References
If you use pywstmi or pystgi, please cite the references [1] and [2], respectively:
```
[1] A. Edraki, W.-Y. Chan, J. Jensen, & D. Fogerty, “Speech Intelligibility Prediction Using Spectro-Temporal Modulation Analysis,” IEEE/ACM Trans. Audio, Speech, & Language Processing, vol. 29, pp. 210-225, 2021.
[2] A. Edraki, W.-Y. Chan, J. Jensen, & D. Fogerty, “A Spectro-Temporal Glimpsing Index (STGI) for Speech Intelligibility Prediction," Proc. Interspeech, 5 pages, Aug 2021.
```
