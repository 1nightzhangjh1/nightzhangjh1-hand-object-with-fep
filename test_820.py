import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from tqdm import tqdm
import json
import cv2

from common.base import Tester
from common.utils.ho3deval import dump
from common.utils.pointmertic import get_point_metrics
from pytorch3d.io import save_obj, load_obj  # use pytorch3d to read/write obj

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

save_path = "demo_outputs"
os.makedirs(save_path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_path', type=str, dest='model_path')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(str, range(*gpus)))
    return args


def get_inter_metrics(verts_person, verts_object, faces_person, faces_object):
    import common.utils.scenesdf as scenesdf
    sdfl = scenesdf.SDFSceneLoss([faces_person[0], faces_object[0]])
    _, sdf_meta = sdfl([verts_person, verts_object])
    max_depths = sdf_meta['dist_values'][(1, 0)].mean(1)[0]
    has_contact = (max_depths > 0)
    return max_depths.item(), has_contact


# ----------------- Headless OBJ -> PNG (flat gray) -----------------

def _faces_F3(faces_idx: torch.Tensor) -> torch.Tensor:
    """
    Ensure faces shape=(F,3), 0-based long tensor on CPU.
    `faces_idx` could be from pytorch3d.io.load_obj which returns (F, 3) long.
    """
    f = faces_idx
    if f.dim() == 3:
        f = f[0]
    if f.shape[0] == 3 and f.shape[1] != 3:
        f = f.transpose(0, 1).contiguous()
    f = f.long().cpu()
    if f.numel() > 0 and f.min().item() >= 1:
        f = f - 1
    return f


def _project_pinhole(verts_xyz: np.ndarray, fx, fy, cx, cy):
    """
    verts_xyz: (V,3) numpy, camera coords (x,y,z)
    return: uv (V,2), z (V,)
    """
    z = np.maximum(verts_xyz[:, 2], 1e-6)
    u = fx * (verts_xyz[:, 0] / z) + cx
    v = fy * (verts_xyz[:, 1] / z) + cy
    return np.stack([u, v], axis=1), z


def _unique_edges_from_faces(f_np: np.ndarray):
    edges = set()
    for a, b, c in f_np:
        a, b, c = int(a), int(b), int(c)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return edges


def obj_to_png_headless(
    obj_path: str,
    png_path: str,
    W: int,
    H: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tri_gray=(180, 180, 180),
    pt_gray=(150, 150, 150),
    pt_radius: int = 1,
    draw_edges: bool = False,
    edge_gray=(130, 130, 130),
    edge_thick: int = 1,
):
    """
    Load OBJ (verts in camera coords), project by intrinsics, paint triangles as flat gray on white background.
    No OpenGL, no pyglet required. Pure CPU with OpenCV.
    """
    try:
        # load_obj returns: (verts, faces_idx, aux)
        verts, faces_idx, _ = load_obj(obj_path, load_textures=False, device="cpu")
        verts = verts.cpu().numpy()  # (V,3)
        faces = faces_idx.verts_idx  # (F,3) long
        faces = _faces_F3(faces).numpy()

        uv, z = _project_pinhole(verts, fx, fy, cx, cy)

        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Painter's algorithm: sort by avg depth (far to near). Depending on your z convention, flip if needed.
        face_depth = z[faces].mean(axis=1)
        order = np.argsort(-face_depth)  # far->near; switch to np.argsort(face_depth) if occlusion looks inverted

        for idx in order:
            tri = faces[idx]
            pts = uv[tri].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(img, pts, color=tri_gray, lineType=cv2.LINE_AA)

        # draw points
        pts_all = uv.astype(np.int32)
        for p in pts_all:
            cv2.circle(img, (int(p[0]), int(p[1])), pt_radius, pt_gray, thickness=-1, lineType=cv2.LINE_AA)

        # optional edges
        if draw_edges:
            for i, j in _unique_edges_from_faces(faces):
                p1 = tuple(pts_all[i])
                p2 = tuple(pts_all[j])
                cv2.line(img, p1, p2, color=edge_gray, thickness=edge_thick, lineType=cv2.LINE_AA)

        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[WARN] Headless OBJ->PNG failed for {obj_path}: {e}")


# -------------------------------------------------------------------

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, test_set='HO3D')
    cudnn.benchmark = True

    tester = Tester()
    tester._make_batch_generator()
    tester._make_model(args.model_path)

    sequences = ["SM1", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "AP10",
                 "AP11", "AP12", "AP13", "AP14"]

    chamer_dist, add_s, verts_dists = [], [], []
    all_hands, all_joints, avg_fps = [], [], []
    mean_penetration_depth, contacts = [], []
    pred_joints, pred_hands = {seq: [] for seq in sequences}, {seq: [] for seq in sequences}

    with open(
        "/home/ubuntu/ZHANGJH/DenseMutualAttention-main/DenseMutualAttention-main/local_data/ho3d/annotations/HO3D_evaluation_data.json",
        "r"
    ) as f:
        ho3d_json = json.load(f)

    images_info = {img["id"]: img for img in ho3d_json["images"]}
    annots_info = {ann["image_id"]: ann for ann in ho3d_json["annotations"]}

    save_records = True
    camextr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    for _, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        with torch.no_grad():
            start = time.time()
            out = tester.model(inputs, targets, meta_info, 'test')
            end = time.time()
            avg_fps.append((1 / (end - start)))

            hand_verts_out = out['hand_verts_out'][0]
            hand_joints_out = out['hand_joints_out'][0]
            obj_verts_out = out['obj_verts_refine']

            hand_faces = tester.model.module.face.T.cpu()
            obj_faces = out['obj_faces'].cpu()

            img_path = inputs['img_path'][0]
            parts = img_path.split('/')
            seq_name = parts[4]
            frame_name = os.path.splitext(parts[-1])[0]

            # predicted verts
            hand_verts = hand_verts_out.cpu().float()
            obj_verts_aligned = torch.bmm(
                targets['R'].cuda(),
                out['obj_verts_template'].transpose(1, 2)
            ).transpose(1, 2) + targets['T'].unsqueeze(1).cuda()
            object_verts = obj_verts_aligned[0].cpu().float()

            combined_verts = torch.cat([hand_verts, object_verts], dim=0)
            combined_faces = torch.cat([hand_faces, obj_faces + hand_verts.shape[0]], dim=0)

            # save obj files
            combined_obj = os.path.join(save_path, f"{seq_name}_combined_{frame_name}.obj")
            gt_hand_obj = os.path.join(save_path, f"{seq_name}_gt_hand_{frame_name}.obj")
            gt_obj_obj = os.path.join(save_path, f"{seq_name}_gt_obj_{frame_name}.obj")

            save_obj(combined_obj, combined_verts, combined_faces)

            gt_hand_verts = targets['fit_mesh_cam'].cpu().float().squeeze(0)
            gt_obj_verts = targets['obj_verts'].cpu().float()
            gt_obj_verts = torch.bmm(
                targets['R'],
                gt_obj_verts.transpose(1, 2)
            ).transpose(1, 2) + targets['T'].unsqueeze(1)
            gt_obj_verts = gt_obj_verts.squeeze(0).cpu().float()

            gt_hand_faces = tester.model.module.face.T.cpu()
            gt_obj_faces = out['obj_faces'].cpu()

            save_obj(gt_hand_obj, gt_hand_verts, gt_hand_faces)
            save_obj(gt_obj_obj, gt_obj_verts, gt_obj_faces)

            # camera intrinsics from HO3D json
            img_id = int(os.path.splitext(parts[-1])[0])
            img_meta = images_info[img_id]
            ann_meta = annots_info[img_id]
            fx, fy = ann_meta["cam_param"]["focal"]
            cx, cy = ann_meta["cam_param"]["princpt"]
            W, H = img_meta["width"], img_meta["height"]

            # Headless OBJ->PNG snapshots (flat gray faces + gray points)
            combined_png = os.path.join(save_path, f"{seq_name}_combined_{frame_name}.png")
            gt_hand_png = os.path.join(save_path, f"{seq_name}_gt_hand_{frame_name}.png")
            gt_obj_png = os.path.join(save_path, f"{seq_name}_gt_obj_{frame_name}.png")

            obj_to_png_headless(combined_obj, combined_png, W, H, fx, fy, cx, cy,
                                tri_gray=(180,180,180), pt_gray=(150,150,150), pt_radius=1,
                                draw_edges=False)

            obj_to_png_headless(gt_hand_obj, gt_hand_png, W, H, fx, fy, cx, cy,
                                tri_gray=(180,180,180), pt_gray=(150,150,150), pt_radius=1,
                                draw_edges=False)

            obj_to_png_headless(gt_obj_obj, gt_obj_png, W, H, fx, fy, cx, cy,
                                tri_gray=(180,180,180), pt_gray=(150,150,150), pt_radius=1,
                                draw_edges=False)

            # metrics (unchanged)
            obj_verts_gt = torch.bmm(
                targets['R'].cuda(), out['obj_verts_template'].transpose(1, 2)
            ).transpose(1, 2) + targets['T'].unsqueeze(1).cuda()

            pred_hands[seq_name].append(hand_verts.numpy().dot(camextr[:3, :3]))
            pred_joints[seq_name].append(hand_joints_out.cpu().numpy().dot(camextr[:3, :3]))

            obj_metrics = get_point_metrics(obj_verts_out.float(), obj_verts_gt.float())
            verts_dists.append(obj_metrics['verts_dists'][0])
            chamer_dist.append(obj_metrics['chamfer_dists'][0])
            add_s.append(obj_metrics['add-s'][0])

            pd, has_contact = get_inter_metrics(
                hand_verts_out.unsqueeze(0).float().cuda(),
                obj_verts_out.float().cuda(),
                tester.model.module.face.transpose(0, 1).unsqueeze(0).cuda(),
                out['obj_faces'].unsqueeze(0).cuda()
            )
            mean_penetration_depth.append(pd)
            contacts.append(has_contact.item())

    if save_records:
        for seq in sequences:
            for k in range(len(pred_hands[seq])):
                all_hands.append(pred_hands[seq][k])
                all_joints.append(pred_joints[seq][k])

        print("Object MME (cm):", (sum(verts_dists) / len(verts_dists) * 100))
        print("Object ADD-S (cm):", (sum(add_s) / len(add_s)) * 100)
        print("PD (mm):", (sum(mean_penetration_depth) / len(mean_penetration_depth)) * 1000)
        print("CP (%):", (sum(contacts) / len(contacts)) * 100)
        print("Avg FPS:", (sum(avg_fps) / len(avg_fps)))

        os.makedirs('../ho3d_preds/', exist_ok=True)
        dump(f'../ho3d_preds/pred.json', all_joints, all_hands)


if __name__ == "__main__":
    args = parse_args()
    main()
