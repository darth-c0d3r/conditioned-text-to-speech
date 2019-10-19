## Abstract

We aim to implement a Neural Network system for generating the speech corresponding to a piece of text, with a small caveat: the generated speech should be directed towards an input emotion, such as angry, sad, happy, shocked etc. The preliminary idea is to join two networks: the first one for a general text to speech conversion and the second one for adding emotions to the waveform output by the first network. Our initial plan is to have the first network (text to speech) take as input, a voice sample and generate the output that sounds similar to the input voice sample.

## Team

1. Drumil Trivedi
2. Maitrey Gandopadhye
3. Gurparkash Singh

## References

### Text to Speech Papers
1. [Tacotron 2 (Blog)](https://ai.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html)
2. [Tacotron 2 (Paper)](https://arxiv.org/pdf/1712.05884.pdf)
3. [Tacotron 1 (Paper)](https://arxiv.org/abs/1703.10135.pdf)
4. [Wavenet (Blog)](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
5. [Wavenet (Paper)](https://arxiv.org/pdf/1609.03499.pdf)
6. [Conditional PixelCNN (Paper)](https://arxiv.org/pdf/1606.05328.pdf)
7. [PixelRNN & PixelCNN (Paper)](https://arxiv.org/pdf/1601.06759.pdf)
8. [Useful Paper for Wavenet](https://arxiv.org/pdf/1702.07825.pdf)
9. [Wavenet Tutorial Blog](http://sergeiturukin.com/2017/03/02/wavenet.html)
10. [Wavenet (Keras Implementation)](https://github.com/basveeling/wavenet)
11. [Wavenet (Tensorflow Implementation)](https://github.com/ibab/tensorflow-wavenet)
12. [Wavenet (PyTorch Implementation)](https://github.com/vincentherrmann/pytorch-wavenet)

### Conditioned Speech Conversion
1. [Nonparallel Emotional Speech Conversion](https://arxiv.org/pdf/1811.01174.pdf)

### Alternate Project
1. [Voice Impersonation using GANs](https://arxiv.org/pdf/1802.06840.pdf)

### Datasets
1. [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)