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
 <img width="500" height="194" alt="image" src="https://github.com/1nightzhangjh1/nightzhangjh1-hand-object-with-fep/blob/main/OVERALL.png" />
<p align="center">
</p>
## Install
Install packages from requirements.txt. We trained our models on RTX 4090 GPUs with CUDA 11.8. We recommend using Linux for reproduction.

```bash
conda create -n ho3d python=3.8
conda activate ho3d
pip install -r requirements.txt
```
## Install additional dependencies 

```bash
pip install "pytorch3d" -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt112/download.html
pip install neural-renderer-pytorch
pip install torch-geometric
```

1.Generate reconstructions and visualizations:
```bash
python test.py --gpu 0 --model_path

```
## Dataset Preparation
1.Download HO3D v3 from official website and preprocess the data
```bash
python scripts/preprocess_ho3d.py --data_path /path/to/ho3d --output_path local_data/ho3d_simple
```
2.Download DexYCB from official website and preprocess the data
```bash
python scripts/preprocess_dexycb.py --data_path /path/to/dexycb --output_path local_data/dex_simple
```

## Results
<img width="200" height="194" alt="image" src="https://github.com/1nightzhangjh1/nightzhangjh1-hand-object-with-fep/blob/main/ho3d1.png" />
<img width="500" height="194" alt="image" src="https://github.com/1nightzhangjh1/nightzhangjh1-hand-object-with-fep/blob/main/dexycb1.png" />
