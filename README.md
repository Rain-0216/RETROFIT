# RETROFIT: Continual Learning with Bounded Forgetting for Security Applications

This repository contains the official implementation of our paper on achieving Bounded Forgetting in Security Applications through continual learning techniques.

## MALWARE DETECTION

### Model

Our model is defined in **Malware_Detection/model.py**.

### Data

We adopt the dataset from [Transcendent](https://ieeexplore.ieee.org/abstract/document/9833659)
, which spans five years (from 2014 to 2018). It contains 232,848 benign and 26,387 malicious Android applications collected from [AndroZoo](https://androzoo.uni.lu/). 

Following prior work, we extract the widely used [Drebin](https://media.telefonicatech.com/telefonicatech/uploads/2021/1/4915_2014-ndss.pdf) features.

### Train & Merge

We adopt a multilayer perceptron (MLP) classifier as the backbone model. First, a base model is trained on 2014 data using a cold-start strategy; then, a low-rank update is performed on the new task dataset using a warm-start strategy, followed by a merging operation, ultimately obtaining a target model adapted to the new task.

The following commands are required to execute training and merging:

```bash
python Malware_Detection/train_ours.py
```

### Eval

First, we use the built-in f1_score function imported from sklearn.metrics to sequentially evaluate the F1 score of each task's model on the task itself, as well as the F1 scores across all previous tasks, and then calculate the AUT and PTR values.


## BINARY ANALYSIS

### Model
Our [models](..) will be available on the HF Hub soon.

### Setup
You need to first refer to the [BinT5](https://github.com/AISE-TUDelft/Capybara-BinT5/tree/main) setup, download and properly configure the official CodeT5 code and the [Capybara dataset](https://huggingface.co/datasets/AISE-TUDelft/Capybara) .

### Train
Our method starts from the fully fine-tuned CodeT5-C model. To launch training with our LoRI Adapter, simply replace the official CodeT5 models.py and run_gen.py with our supplied models_low_rank.py and run_gen_low_rank.py, then run the standard CodeT5 training script. We follow the base configuration of the BinT5 project and set the learning rate to 1e-4.
```bash
bash CodeT5_train.sh
```

### Merge
After training the adapter for the new task, set the correct paths and run our merge script to train the soft-mask adapter to be merged, Once trained, merge the adapter into the base model to serve as the new base model for the next task's training.
```bash
bash CodeT5_merge.sh
```

### Eval
We present the inference results of the BINARY ANALYSIS section in the Output folder. You can run the official test code of CodeT5 (with do_train=False and do_test=True) to obtain the outputs of each model and the BLEU and EM scores directly calculated by the official CodeT5 implementation. 
```bash
bash CodeT5_test.sh
```
If you wish to compute additional metrics, you can directly use the following command:
```bash
python Eval_Metrics.py --base_dir /output_file_path
```
After running, you will obtain the METEOR, ROUGE-L, and BERTScore for all outputs.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
