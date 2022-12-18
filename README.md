# CLIP prefix for video captioning.

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/platform-%20linux%20-green.svg)
![Cuda version](https://img.shields.io/badge/cuda-%3E%3D11.6-blue)
![py version](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![pip](https://img.shields.io/badge/pip-%3E%3D21-blue)
![transformer](https://img.shields.io/badge/transformers-4.10.2-blue)




## implementation for the report ["ClIP Prefix for Video Caption Generation"](https://drive.google.com/file/d/1-RpgQarWn7NpS5WhvOd6aTO5dLWxGZXN/view?usp=share_link)


## Description  
Contrastive models like CLIP have demonstrated impressive ability in learning robust and high quality visual represetations and have sparked many promising application directions. In this work, we try to leverage the visual embeddings produced by CLIP to tackle the problem of video caption generation. Video captioning is a fundamental task for vision-language understanding, where the model is asked to generate a piece of text description for an input video clip. This task is challenging as it requires wisdom from both video understanding and natural language generation. Therefore, we take advantage of both the high quality visual features produced by CLIP and a pre-trained language generation model, GPT2, to create a simple and light weight model for the video caption generation task. In our model, representation of video frames encoded by CLIP are transformed into prefixes of a sentence and sent to the language model to generate the corresponding caption. Experiments on a public video captioning dataset demonstrated the promising results of our simple method. 
## Demos for our Video Captioning  
<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/41601003/207977160-190ba8cb-a1ff-434e-a40c-c6f3990832ac.gif" ></video></td>
    <td><img src="https://user-images.githubusercontent.com/41601003/207977795-8cc11699-e1b2-4323-bf50-849039caeb8e.gif" ></td>
    <td><img src="https://user-images.githubusercontent.com/41601003/207977869-03a6a42f-ac20-4488-b2aa-d09d3c04db4a.gif" ></td>
  </tr>
  <tr>
    <td>a girl is talking about how to make a mask</td>
     <td>a man is driving a car in a car and</td>
     <td>a band is performing a song on stage and a</td>
  </tr>
 </table>

## Training prerequisites
Clone, create environment and install dependencies:  
```
git clone https://github.com/juexZZ/NYUFall22-CVProject-CLIPVideoCap.git && cd CLIP_prefix_caption NYUFall22-CVProject-CLIPVideoCap
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## MSR_VTT training

Download [video dataset](https://www.mediafire.com/folder/h14iarbs62e7p/shared)


Extract CLIP features
```
Run 'CLIP_feature_extraction.ipynb'
```

Train only the feature transformation module
```
python train_vtt.py --mapping_type transformer --num_layers 8 --prefix_length_clip 28 --bs 40 --only_prefix --save_every 10 --epochs 10 \
--cross --out_dir cross_length20 --prefix_length 20
```

To fine-tune the GPT-2
```
python train_vtt.py --mapping_type transformer --num_layers 8 --prefix_length_clip 28 --bs 40 --save_every 10 --epochs 10 \
--cross --out_dir cross_length20 --prefix_length 20
```

To do inference with a trained model:
```
python inference_vtt.py --model_dir cross_length20 --prefix_length 20 --mapping_type transformer --cross --entry_length 10 \
--num_layers 8 --epoch 9
```



## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP), [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [MSR_VTT dataset](https://github.com/nasib-ullah/video-captioning-models-in-Pytorch/tree/main/MSRVTT)


