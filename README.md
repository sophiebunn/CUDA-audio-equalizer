# CUDA-audio-equalizer
A GPU-accelerated 3-band audio equalizer built with a custom CUDA kernel and FFT.

The pipeline loads a stereo WAV file and computes an FFT on the GPU using PyTorch. The resulting frequency-domain representation is transferred to a PyCUDA gpuarray, where a custom CUDA C++ kernel applies gain multipliers to each frequency band. The filtered spectrum is then converted back to the time domain via inverse FFT and saved as a new .wav file.

<p align="center"><img width="600" height="480" alt="signal_compare" src="https://github.com/user-attachments/assets/80fe6d81-8c18-4569-952c-0559402d3090" /></p>

