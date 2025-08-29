# HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition

> [[arXiv]](https://arxiv.org/abs/2401.05698) | [[Information Fusion]](https://doi.org/10.1016/j.inffus.2024.102382)<br>
> [Licai Sun](https://sunlicai.github.io/), [Zheng Lian](https://scholar.google.com/citations?user=S34nWz0AAAAJ&hl=en), [Bin Liu](https://scholar.google.com/citations?user=UEB_5QEAAAAJ&hl=en), and [Jianhua Tao](https://scholar.google.com/citations?user=781jbHMAAAAJ&hl=en)<br>
> University of Chinese Academy of Sciences & Institute of Automation, Chinese Academy of Sciences & Tsinghua University<br>

## 📰 News

**[2024.10.21]** We upload the fine-tuned models on CREMA-D and MAFW. 

**[2024.04.11]** We upload the pre-training code. 

**[2024.03.20]** Our paper is accepted by Information Fusion. 

**[2024.01.11]** We upload the initial code and pre-trained model. 


## ✨ Overview

![HiCMAE](figs/hicmae.png)

Abstract:
Audio-Visual Emotion Recognition (AVER) has garnered increasing attention in recent years for its critical role in creating emotion-ware 
intelligent machines. Previous efforts in this area are dominated by the supervised learning paradigm. Despite significant
progress, supervised learning is meeting its bottleneck due to the longstanding data scarcity issue in AVER. Motivated by recent advances 
in self-supervised learning, we propose Hierarchical Contrastive Masked Autoencoder (HiCMAE), a novel self-supervised
framework that leverages large-scale self-supervised pre-training on vast unlabeled audio-visual data to promote the advancement
of AVER. Following prior arts in self-supervised audio-visual representation learning, HiCMAE adopts two primary forms of self-supervision 
for pre-training, namely masked data modeling and contrastive learning. Unlike them which focus exclusively on top-layer representations 
while neglecting explicit guidance of intermediate layers, HiCMAE develops a *three-pronged* strategy to foster *hierarchical* audio-visual 
feature learning and improve the overall quality of learned representations. Firstly, it incorporates *hierarchical skip connections* between 
the encoder and decoder to encourage intermediate layers to learn more meaningful representations and bolster masked audio-visual reconstruction. 
Secondly, *hierarchical cross-modal contrastive learning* is also exerted on intermediate representations to narrow the audio-visual 
modality gap progressively and facilitate subsequent cross-modal fusion. Finally, during downstream fine-tuning, HiCMAE employs 
*hierarchical feature fusion* to comprehensively integrate multi-level features from different layers. To verify the effectiveness 
of HiCMAE, we conduct extensive experiments on 9 datasets covering both categorical and dimensional AVER tasks. Experimental results 
show that our method significantly outperforms state-of-the-art supervised and self-supervised audio-visual methods, which indicates that 
**HiCMAE is a powerful audio-visual emotion representation learner**.


## 🚀 Main Results

<p align="center">
  <img src="figs/radar_plot.png" width=65%> <br>
   Comparison with state-of-the-art audio-visual methods on 9 datasets.
</p>

Please check our arXiv paper to see detailed results on each dataset.


## 👀 Visualization

### ✨ Reconstruction 

![Reconstruction](figs/reconstruction.png)

Please check our arXiv paper to see details better.

### ✨ t-SNE on CREMA-D


![t-SNE_on_CREMA-D](figs/t-SNE.png)



## 🔨 Installation

Main prerequisites:

* `Python 3.8`
* `PyTorch 1.10.1 (cuda 11.3), torchvision==0.11.2, torchaudio==0.10.1`
* `timm==0.4.12`
* `einops==0.6.1`
* `decord==0.6.0`
* `openmim==0.3.6, mmcv==1.7.1`
* `scikit-learn=1.2.1, scipy=1.10.0, pandas==1.5.3, numpy=1.23.5`
* `opencv-python=4.7.0.72`
* `tensorboardX=2.6.1`
* `soundfile==0.12.1`

If some are missing, please refer to [requirements.txt](requirements.txt) for more details.


## ➡️ Data Preparation

1. If the original dataset does not provide extracted faces, we use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to extract them from videos. Please see [extract_face.py](preprocess/face_and_audio/extract_face.py) in [preprocess](preprocess) for details.

2. If the original dataset does not provide extracted audios (sampling rate: 16k), we use FFmpeg to extract them from videos. Please see [extract_audio.py](preprocess/face_and_audio/extract_audio.py) in [preprocess](preprocess) for details.

3. Follow the files (e.g., [cremad_av.py](preprocess/creamd_av.py)) in [preprocess](preprocess) to prepare your annotation files.

    Specifically, you need to enerate annotations for dataloader ("<path_to_video> <path_to_audio> <class_idx>" in annotations). 
The annotation usually includes `train.csv` and `test.csv`. The format of `*.csv` file is typically like this:

    ```
    dataset_root/video_1 dataset_root/audio_1 label_1
    dataset_root/video_2 dataset_root/audio_2 label_2
    dataset_root/video_3 dataset_root/audio_3 label_3
    ...
    dataset_root/video_N dataset_root/audio_N label_N
    ```

    An example of [train.csv](saved/data/crema-d/audio_visual/split01/train.csv) for CREMA-D (6-class) fold1 (fd1) is shown as follows:

    ```
    /data/ycs/AC/Dataset/CREMA-D/face_aligned/1019_DFA_ANG_XX /data/ycs/AC/Dataset/CREMA-D/AudioWAV/1019_DFA_ANG_XX.wav 0
    /data/ycs/AC/Dataset/CREMA-D/face_aligned/1019_DFA_DIS_XX /data/ycs/AC/Dataset/CREMA-D/AudioWAV/1019_DFA_DIS_XX.wav 1
    /data/ycs/AC/Dataset/CREMA-D/face_aligned/1019_DFA_FEA_XX /data/ycs/AC/Dataset/CREMA-D/AudioWAV/1019_DFA_FEA_XX.wav 2
    ```

## 📍Pre-train HiCMAE

- VoxCeleb2

    ```
    sh scripts/voxceleb2/audio_visual/hicmae_pretrain_base/pretrain_base.sh
    ```
  You can download our pre-trained model on VoxCeleb2 from [this link](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/ESULFR9UGilPuLkGkc7dN1gBTNcE_p0JhtrZwshwzkbZwg?e=gZdf7Y) and put it into [this folder](saved/model/pretraining/voxceleb2/audio_visual/hicmae_pretrain_base).

## ⤴️ Fine-tuning with pre-trained models

- CREMA-D

    ```
    sh scripts/voxceleb2/audio_visual/hicmae_pretrain_base/cream-d/finetune_170.sh
    ```
    
    The fine-tuned checkpoints and logs across five folds on CREMA-D are provided as follows: 
    |  Fold    | UAR        | WR       |      Fine-tuned   Model            |
    | :------: | :--------: | :------: | :-----------------------:          |
    |  1       | 86.66      | 86.67    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EXf8jT_e7uFIpUIiuQISAGEBDuMdeZXM2L7pTN7VbDA_DA?e=QAPZ0n) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/ERtJlWmrrk5LuKycdrmlXFYBbGFdmvjI6oNvLf-6OCYZWA?e=j7YXkP) | 
    |  2       | 83.27      | 83.20    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EbU6et8b9DpBpyWI26lDhjsB2JdDwRt8yHOA0k5JNX0L-A?e=nnkPp6) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/Ef_tG1Eh7nNOj5BV3ljvKMcBDWkzt17ytaUknSPgg28jBQ?e=dqTvez) | 
    |  3       | 87.23      | 87.19    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EelLU_bTRsZDkoNXDbw1q18BI9Yes-3TKt9zcCwqGJPsQg?e=QyymZS) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EUIvB5E1029Atp0kmIaNsOEBIjFItoyI_B8WCCWxNCTRyw?e=aT7Ha0) | 
    |  4       | 83.70      | 83.79    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EY8mercoIS9Koa18BiKlOesBGco7EkKW8It7y-3mUkWyeA?e=8Y2B0Q) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EelA-2vHrOdBj2VFX5xWBwgBXGeHrLCKRCwC-ToLkQqVPA?e=UqhN17) | 
    |  5       | 83.88      | 83.79    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EU5dZKG1mHVHpN_W6-j9cwcBYk89-51rzGRGLkt4vMp3CA?e=sflC8v) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EaS5Lz94qLdErJPkXnIaHUUBO24c56cuaMIcMRRxppPAdw?e=TJ5ysA) |
    |  Total   | 84.94      | 84.91    | - |

- MAFW

    ```
    sh scripts/voxceleb2/audio_visual/hicmae_pretrain_base/mafw/finetune_170.sh
    ```
    
    The fine-tuned checkpoints and logs across five folds on MAFW are provided as follows: 
    |  Fold    | UAR        | WR       |      Fine-tuned   Model            |
    | :------: | :--------: | :------: | :-----------------------:          |
    |  1       | 36.02      | 47.60    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EXPu0Nlku05LkOyXD7UrgKIBFEWMKpXee9JkQ9Tp3597kw?e=uDV3l0) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/ES-eTQiMgSZKno9KyUZy1cgB5irFyRkGDIoLsChy7G9VmQ?e=li2MbK) | 
    |  2       | 41.57      | 55.10    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EcKoXaBqwG9IhRukKstH_4IBc8dpIWWl-fQQSiRx2B0OdA?e=XF2DhJ) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EVESyhEmuwJKpswfhfkWtsIBHVjsIoIVgi9BIP1LRl_KSw?e=fSwwWk) | 
    |  3       | 46.46      | 60.10    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/Ec1DpC4iekxHsbn3Ixf1-QMB1wcTRv7R-Ba7XpKvtg5idg?e=hj5Od9) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/ER74xfvkeFFGtfPzPH9CHhAB_qtuDI_rdKKKN-f4ep05ZA?e=5Gz7Iw) | 
    |  4       | 47.50      | 63.09    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EbA-glMilZBDmE3oytwHdWoBWF9yUD9-rMzyiwosnDbnwg?e=eOji11) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EXgUG2jtGopElcHJKA3bSokB4p6gv1ds-CfXhE2AaAdEuQ?e=5YTm4T) | 
    |  5       | 41.88      | 55.22    | [log](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EVuvCgeTE9xFlFP21ZCAFaYBQRS7LSeXKVCvLG6MT77NuA?e=hIkdLe) / [checkpoint](https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi/EWpqPBE74IlIooG0JoOPcrABjxsS3iIsLiDeEbLJnllidA?e=EhoobV) |
    |  Total   | 42.69      | 56.21    | - |


## ☎️ Contact 

If you have any questions, please feel free to reach me out at `licai.sun@oulu.fi`.

## 👍 Acknowledgements

This project is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [AudioMAE](https://github.com/facebookresearch/AudioMAE). Thanks for their great codebase.

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@article{sun2024hicmae,
  title={HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition},
  author={Sun, Licai and Lian, Zheng and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2401.05698},
  year={2024}
}
```




