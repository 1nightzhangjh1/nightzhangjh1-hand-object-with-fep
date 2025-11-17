import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import cfg
from common.base import Tester
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights,
    BlendParams, AlphaCompositor, TexturesVertex
)

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_path', type=str, dest='model_path')
    return parser.parse_args()

def render_overlay(original_img, mesh, renderer):
    rendered = renderer(mesh)[0, ..., :3].cpu().numpy()
    rendered = (rendered * 255).astype(np.uint8)
    original_resized = cv2.resize(original_img, (rendered.shape[1], rendered.shape[0]))
    blended = cv2.addWeighted(original_resized, 0.5, rendered, 0.5, 0)
    return blended

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, test_set='HO3D')
    cudnn.benchmark = True

    save_frame_ids = [100, 200, 300]
    os.makedirs("demo_outputs", exist_ok=True)

    tester = Tester()
    tester._make_batch_generator()
    tester._make_model(args.model_path)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=FoVPerspectiveCameras(device='cuda'),
            raster_settings=RasterizationSettings(image_size=512)
        ),
        shader=SoftPhongShader(
            device='cuda',
            cameras=FoVPerspectiveCameras(device='cuda'),
            lights=PointLights(device='cuda', location=[[0.0, 0.0, -3.0]]),
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))
        )
    )

    for i, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

            if i not in save_frame_ids:
                continue

            hand_verts = out['hand_verts_out'][0].cpu()
            obj_verts = out['obj_verts_refine'][0].cpu()
            hand_faces = tester.model.module.face.T.cpu()
            obj_faces = out['obj_faces'].cpu()

            # 保存 obj
            save_obj(f"demo_outputs/hand_{i}.obj", hand_verts, hand_faces)
            save_obj(f"demo_outputs/object_{i}.obj", obj_verts, obj_faces)

            # 获取原始图像
            img_path = inputs['img_path'][0]
            orig_img = cv2.imread(img_path)[..., ::-1]  # BGR to RGB

            # 准备 mesh 渲染
            hand_mesh = Meshes(verts=[hand_verts.cuda()],
                               faces=[hand_faces.cuda()],
                               textures=TexturesVertex(verts_features=torch.ones_like(hand_verts[None].cuda())))
            obj_mesh = Meshes(verts=[obj_verts.cuda()],
                              faces=[obj_faces.cuda()],
                              textures=TexturesVertex(verts_features=torch.ones_like(obj_verts[None].cuda()) * 0.5))

            # 渲染叠加
            rendered_hand = render_overlay(orig_img, hand_mesh, renderer)
            rendered_obj = render_overlay(rendered_hand, obj_mesh, renderer)

            # 保存图片
            plt.imsave(f"demo_outputs/render_{i}.png", rendered_obj)

if __name__ == "__main__":
    main()
