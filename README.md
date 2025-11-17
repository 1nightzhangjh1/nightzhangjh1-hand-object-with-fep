<h1 align="center">Hand-Object Reconstruction Method with Feature Enhancement and
 Physical Constraints</h1>

<p align="center">
Jia Liu, Zirui Jiang, Jiahui Zhang, Lina Wei, Chengcheng Hua and Dapeng Chen,∗<br>
Nanjing University of Information Science and Technology
</p>

---
<h2 align="center">ABSTRACT</h2>
Reconstructing hand-object meshes from a single RGB image is challenging due to severe occlusions and close contacts during interactions, leading to missing information and feature entanglement. This paper presents a method that incorporates feature enhancement and physical constraints to address these issues. A dual-stream residual feature extraction module is used to separately model hand and object features, reducing competition in feature extraction. A dual-branch feature augmentation module employs local contact graph convolution and global keypoint topology constraints for structured guidance of intermediate features. A cross-feature supplementation module enhances feature robustness through cross-branch attention interaction. A signed distance field loss enforces physical plausibility, reducing geometric penetrations. Evaluated on HO3D V2 and Dex-YCB datasets, the method achieves significant improvements in hand joint and mesh errors, object center error, and closest point distance. On HO3D V2, the average joint error is reduced to 8.8 mm, and on Dex-YCB, the mesh error drops to 9.8 mm. Additionally, it outperforms existing methods in interaction metrics like penetration depth and contact ratio, confirming the framework's effectiveness. Ablation studies validate the necessity of each module and loss function, and dynamic interaction experiments highlight its strong applicability in real-time scenarios.
