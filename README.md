<h1 align="center">Hand-Object Reconstruction Method with Feature Enhancement and
 Physical Constraints</h1>

<p align="center">
Jia Liu, Zirui Jiang, Jiahui Zhang, Lina Wei, Chengcheng Hua and Dapeng Chen,∗<br>
Nanjing University of Information Science and Technology
</p>

---
<h2 align="center">ABSTRACT</h2>
Reconstructing hand-object meshes from a single RGB image is challenging due to severe occlusions and close contacts during interactions, leading to missing information and feature entanglement. This paper presents a method that incorporates feature enhancement and physical constraints to address these issues. A dual-stream residual feature extraction module is used to separately model hand and object features, reducing competition in feature extraction. A dual-branch feature augmentation module employs local contact graph convolution and global keypoint topology constraints for structured guidance of intermediate features. A cross-feature supplementation module enhances feature robustness through cross-branch attention interaction. A signed distance field loss enforces physical plausibility, reducing geometric penetrations. Evaluated on HO3D V2 and Dex-YCB datasets, the method achieves significant improvements in hand joint and mesh errors, object center error, and closest point distance. On HO3D V2, the average joint error is reduced to 8.8 mm, and on Dex-YCB, the mesh error drops to 9.8 mm. Additionally, it outperforms existing methods in interaction metrics like penetration depth and contact ratio, confirming the framework's effectiveness. Ablation studies validate the necessity of each module and loss function, and dynamic interaction experiments highlight its strong applicability in real-time scenarios.

<h1 align="center">Overall working frame diagram</h1>
Our method primarily consists of three com
ponents: the MCS-ResNet50, the D-BFAM, and the CFSM.
 Theoverall frameworkisillustrated in Fig. 1. Firstly, a single
 RGB image is input into the system. The MCS-ResNet50
 and D-BFAM modules are employed to extract the primary
 and secondary features of both the hand and the object.
 Subsequently, the primary and secondary features of the
 hand and object are respectively fed into the CFSM for
 cross-feature supplementation. Finally, through dedicated
 decoders for the hand and the object, the supplemented
 features are upsampled to regress the mesh representations
 of the hand and the object separately.
<p align="center">
</p>


## Install
Install packages in requirments.txt,We trained the model on a single 4090 gpu with 11.8 CUDA,We recommend using a Linux server for reproduction  
```bash
conda create -n <environment name>
conda activate <environment name>
pip install -r requirments.txt
```

Install tiny-cuda-nn
```bash
git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Install CLIP
```bash
git+https://github.com/openai/CLIP.git
```
Download checkpoints [here](https://pan.baidu.com/s/1SZ51OcqDAk68VJOPMnZ_gQ) 
## RUN
1.Place the downloaded checkpoint in the corresponding folder
```bash
pzh3d
|-- ckpt
    |-- ViT-L-14.ckpt
    |-- pzh3d-pretrain.ckpt
|-- models
    |-- pzhadapter_sketch_sd15v2.pth
    |-- sd-v1-4.ckpt
```
2.Combine sketches with text to generate images
```bash
python test_adapter.py --which_cond sketch --cond_path examples/sketch/sofa3.png --cond_inp_type sketch --prompt "A purple sofa." --sd_ckpt models/sd-v1-4.ckpt --resize_short_edge 512 --cond_tau 0.8 --cond_weight 0.8 --n_samples 5 --adapter_ckpt models/t2iadapter_sketch_sd15v2.pth
```
3.3D generation
```bash
3.1 Generate multi-view images
python generate.py --ckpt ckpt/syncdreamer-pretrain.ckpt --input testset/sofa3.png --output output/sofa3 --sample_num 4 --cfg_scale 1.5 --elevation 30 --crop_size 200
3.2 Use the Nues algorithm to generate rendered videos as well as meshes
python train_renderer.py -i output/sofa3/0.png -n sofa3-neus -b configs/neus.yaml -l output/renderer 
```
## Results
<p align="center">
  <img src="assets/github图.jpg" width="400">
</p>


