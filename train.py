import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from common.base import Trainer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True, help='GPU ids, e.g., 0 or 0-3')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    return parser.parse_args()


def move_to_cuda(data):
    if isinstance(data, dict):
        return {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
    elif isinstance(data, (list, tuple)):
        return [v.cuda() for v in data if isinstance(v, torch.Tensor)]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    return data


def main():
    args = parse_args()
    cfg.trainset_3d = ['HO3D']
    cfg.set_args(args.gpu, stage='param', test_set='HO3D', continue_train=args.resume)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.logger.info(f"========== Epoch {epoch} ==========")
        trainer.set_lr(epoch)
        trainer.model.train()

        total_loss = 0.0
        total_coord = 0.0
        total_normal = 0.0
        total_edge = 0.0
        start_time = time.time()

        loop = tqdm(enumerate(trainer.batch_generator), total=trainer.itr_per_epoch, ncols=120, desc=f"Epoch {epoch}")

        for itr, (inputs, targets, meta_info) in loop:
            inputs = move_to_cuda(inputs)
            targets = move_to_cuda(targets)
            meta_info = move_to_cuda(meta_info)

            outputs = trainer.model(inputs, targets, meta_info, 'train')
            loss, loss_dict = trainer.compute_loss(outputs, targets)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            coord_loss = outputs.get('coord_loss', torch.tensor(0.0).cuda()).item()
            normal_loss = outputs.get('normal_loss', torch.tensor(0.0).cuda()).item()
            edge_loss = outputs.get('edge_loss', torch.tensor(0.0).cuda()).item()

            total_loss += loss.item()
            total_coord += coord_loss
            total_normal += normal_loss
            total_edge += edge_loss

            loop.set_postfix({
                "total": f"{loss.item():.4f}",
                "coord": f"{loss_dict['coord_loss'].item():.4f}",
                "joint": f"{loss_dict['joint_loss'].item():.4f}",
                "edge": f"{loss_dict['edge_loss'].item():.4f}",
                "norm": f"{loss_dict['normal_loss'].item():.4f}",
                "lr": f"{trainer.get_lr():.1e}"
            })

        avg_loss = total_loss / trainer.itr_per_epoch
        trainer.logger.info(f"[Epoch {epoch}] Total Loss: {avg_loss:.4f} | "
                            f"Coord: {total_coord / trainer.itr_per_epoch:.4f} | "
                            f"Normal: {total_normal / trainer.itr_per_epoch:.4f} | "
                            f"Edge: {total_edge / trainer.itr_per_epoch:.4f} | "
                            f"Time: {time.time() - start_time:.2f}s")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.end_epoch:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)
            trainer.logger.info(f"Model snapshot saved for epoch {epoch}")


if __name__ == "__main__":
    main()
