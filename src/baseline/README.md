
## A benchmark classification using HTS-AT model

This baseline is an attempt to integrate and adapt code from the original author's source code from: https://github.com/retrocirce/hts-audio-transformer

## The original author of this model
Ke Chen
knutchen@ucsd.edu
HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
The configuration for training the model will be adapted as required for this experiment.

## Approach for Project Echo

The idea is the benchmark the classification performance against our project Echo audio dataset using an algorithm with known good performance in audio classification tasks.

The HTS-AT algorithm is one of the top performing algorithms on the baseline ESC-50 audio dataset as shown here: https://paperswithcode.com/sota/audio-classification-on-esc-50

The code in this folder is an adaptation of the original author's code with the eventual aim to leverage pre-computed melspectrograms rather than calculating them inline (which the original code does).

The original code was implemented in pytorch and leverages the pytorch lightening framework.  This approach will be continued for this benchmark investigation.  If proved to be successful the project may move to re-writing this code for tensorflow (the preferred framework for Project Echo)