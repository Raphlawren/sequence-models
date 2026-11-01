# Sequence learning: text classification and trigger-word detection

Two notebook projects explore text classification with pretrained word vectors and audio trigger-word detection. Each records the implementation scope, saved results and limitations. They build on supplied notebook scaffolding and data.

## Sentence-to-emoji classification

[Open the text-classification notebook](W2_ANSWERED/W2A2/Emoji_v3a.ipynb)

Averaged pretrained GloVe embeddings with a softmax classifier are compared with an LSTM sentence model.

| Saved run | Test accuracy |
| --- | ---: |
| Averaged-embedding baseline | 91.07% |
| LSTM | 83.93% |

The supplied split contains 132 training and 56 test examples. The simpler baseline performs better on this small test set; broader performance is not established. Next, inspect class-level errors, out-of-vocabulary text and performance on a representative independent set.

## Audio trigger-word detection

[Open the trigger-word notebook](W3_ANSWERED/W3A2/Trigger_word_detection_v2a.ipynb)

Synthetic audio examples and spectrograms feed a sequence model that detects “activate” and triggers a chime.

![Saved spectrogram and predicted trigger probabilities](assets/readme/trigger-word-detection.png)

The saved run reports approximately 92.39% development-set accuracy. Time-step accuracy alone does not establish event-level reliability. A meaningful next evaluation would measure missed triggers and false activations per hour across speakers, noise and recording conditions.

## Reproduction requirements

The original environment has not been pinned or revalidated. The text notebook expects `data/glove.6B.50d.txt` in addition to the supplied CSV files. The audio notebook expects raw and example audio, development arrays and model weights. These large inputs are not all included. Python, NumPy, TensorFlow/Keras, Matplotlib and audio-processing dependencies are required; compatible versions and inputs must be restored before running.

## Attribution

Adapted from DeepLearning.AI’s Deep Learning Specialization. Original notebook scaffolding, utilities, datasets, diagrams and pretrained embeddings or model weights belong to their respective authors. Saved outputs have not been regenerated for this presentation update.
