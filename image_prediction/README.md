# Image Captioning
The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this tutorial, we used [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) network. 

#### Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is **"Giraffes standing next to each other"**, the source sequence is a list containing **['\<start\>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']** and the target sequence is a list containing **['Giraffes', 'standing', 'next', 'to', 'each', 'other', '\<end\>']**. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector.

#### Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using [encoder.eval()](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py#L37). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a [for-loop](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py#L48).

## Usage 


#### 1. Clone the repositories

```bash
git clone git@iwisher.ddns.net:JoshuaZe/img2tagsserving.git
cd image_prediction/
pip install -r requirements.txt
```

#### 2. Upload or Download the dataset

```bash
ssh -i ~/Workspace/applesay rsong@iwisher.ddns.net
cd /home/rsong/img2tagsserving/image_prediction
scp -i ~/Workspace/applesay data/A.zip rsong@iwisher.ddns.net:/home/rsong/img2tagsserving/image_prediction/data/
unzip A.zip
```

#### 3. Train Model

```bash
cd /home/rsong/img2tagsserving/image_prediction
python 01-build_vocab.py
nohup python -u 02-train.py > nohup_train.out 2>&1 &

top -d 5
tail -f nohup_train.out
nvidia-smi

scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/img2tagsserving/image_prediction/models/*-220.ckpt /Users/zezzhang/Workspace/img2tags_serving/image_prediction/models
scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/img2tagsserving/image_prediction/models/*-final.ckpt /Users/zezzhang/Workspace/img2tags_serving/image_prediction/models
scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/img2tagsserving/image_prediction/models/vocab.pkl /Users/zezzhang/Workspace/img2tags_serving/image_prediction/models/vocab.pkl
```

#### 4. Test Model 
```bash
nohup python -u 03-prediction.py > nohup_prediction.out 2>&1 &
tail -f nohup_prediction.out
scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/img2tagsserving/image_prediction/data/A/evaluation/image_objects_prediction.csv /Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/evaluation/image_objects_prediction.csv


```

#### 5. Evaluate Model 
```bash
python 06-evaluation.py

scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/image2tags/tutorials/03-advanced/image_captioning/source_data/images_evaluation.csv /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/source_data/images_evaluation.csv
scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/image2tags/tutorials/03-advanced/image_captioning/source_data/images_prediction.csv /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/source_data/images_prediction.csv
scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/image2tags/tutorials/03-advanced/image_captioning/source_data/instances_evaluation.csv /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/source_data/instances_evaluation.csv
scp -i ~/Workspace/applesay rsong@iwisher.ddns.net:/home/rsong/image2tags/tutorials/03-advanced/image_captioning/source_data/tags_evaluation.csv /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/source_data/tags_evaluation.csv
```

#### 6. Deploy Model 
```bash
ssh -i ~/Workspace/applesay applesay@149.129.125.190
scp -i ~/Workspace/applesay -r /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/models applesay@149.129.125.190:/home/applesay/image2tags/
scp -i ~/Workspace/applesay -r applesay@149.129.125.190:/home/applesay/image2tags/models /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/models
scp -i ~/Workspace/applesay -r ./docker-compose.yml applesay@149.129.125.190:/home/applesay/image2tags/

ssh -i ~/Workspace/new_ai_box root@139.224.19.230
ssh -i ~/Workspace/applesay applesay@139.224.19.230
scp -i ~/Workspace/applesay -r /Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/models applesay@139.224.19.230:/home/applesay/image2tags
scp -i ~/Workspace/applesay -r ./docker-compose.yml applesay@139.224.19.230:/home/applesay/image2tags

docker-compose down
docker-compose up -d
docker stats
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and the vocabulary file [here](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0). You should extract pretrained_model.zip to `./models/` and vocab.pkl to `./data/` using `unzip` command.

https://github.com/kabbi159/awesome-image-tagging

https://github.com/conda/conda/issues/2463
