import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, BinaryF1Score, F1Score
from torchinfo import summary

import wandb
from wandb_osh.hooks import TriggerWandbSyncHook


from dataset import HumanDetectionDataset
from interruptible_utils import (
    EXIT,
    REQUEUE,
    get_requeue_state,
    init_handlers,
    save_and_requeue,
    save_state,
)
from models import *
from tqdm import tqdm


class DetectionModel:
    def __init__(self, model, args, ckpt_path, device, logger):
        super().__init__()
        self.model = model
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        if self.args.multi_class:
            self.accuracy = Accuracy(task="multiclass", num_classes=5)
            self.f1 = F1Score(task="multiclass", num_classes=5, average="macro")
        else:
            self.accuracy = Accuracy(task="binary")
            self.f1 = BinaryF1Score()

        if self.args.binary_depth:
            self.depth_accuracy = Accuracy(task="binary")
            self.depth_f1 = BinaryF1Score()
        elif self.args.multiclass_depth:
            self.depth_accuracy = Accuracy(
                task="multiclass", num_classes=args.num_depth_bins
            )
            self.depth_f1 = F1Score(
                task="multiclass", num_classes=args.num_depth_bins, average="macro"
            )

        self.device = device
        self.logger = logger

        self.model = self.model.to(self.device)

        # Load checkpoint
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.current_epoch = ckpt["epoch"] + 1
            # Load best metric if it exists
            if "best_metric" in ckpt.keys():
                self.best_metric = ckpt["best_metric"]
            print("Loaded checkpoint from epoch " + str(self.current_epoch))
        else:
            self.current_epoch = 0

        summary(self.model)

        self.best_metric = (
            0 if any(x in self.args.monitor_metric for x in ["acc", "f1"]) else np.inf
        )

        # Get criterion
        # Initialize criterion
        if self.args.multi_class:
            self.criterion_cls = nn.CrossEntropyLoss()
        else:
            self.criterion_cls = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.args.cls_posw]).to(self.device)
            )

        if self.args.binary_depth:
            self.criterion_depth = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=torch.tensor([self.args.depth_posw]).to(self.device),
            )
        elif self.args.multiclass_depth:
            self.criterion_depth = nn.CrossEntropyLoss(ignore_index=-1)
        elif args.depth_criterion == "l1":
            self.criterion_depth = nn.L1Loss(reduction="none")
        elif args.depth_criterion == "l2":
            self.criterion_depth = nn.MSELoss(reduction="none")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        if ckpt_path is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.args.multiclass_depth:
            csv_path = os.path.join(self.args.data_dir, "data.csv")
            df = pd.read_csv(csv_path)
            df = df[~df["path"].str.contains("empty")]
            df = df[df["depth"] != -1]

            # Get depth values
            depth = df["depth"].values

            # Split depth into num_depth_bins bins of equal counts
            self.bins = np.quantile(
                depth, np.linspace(0, 1, self.args.num_depth_bins + 1)
            )
            self.bins = np.round(self.bins, 2)

            # Convert bins to torch tensor
            self.bins = torch.from_numpy(self.bins).to(self.device)

    def check_preempt(self):
        if EXIT.is_set() or REQUEUE.is_set():
            print("requeuing job " + os.environ["SLURM_JOB_ID"])
            os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
            exit()

    def forward(self, spectrograms, classes, empty_spectrograms=None):
        if self.args.learn_backsub:
            pred_sin, pred_cos, pred_depth, pred_cls = self.model(
                spectrograms, empty_spectrograms, classes
            )
        else:
            pred_sin, pred_cos, pred_depth, pred_cls = self.model(spectrograms, classes)
        pred_sin = pred_sin.squeeze(1)
        pred_cos = pred_cos.squeeze(1)
        pred_cls = pred_cls.squeeze(1)
        pred_depth = pred_depth.squeeze(1)

        return pred_sin, pred_cos, pred_depth, pred_cls

    def get_loss_and_pred(
        self,
        pred_sin,
        pred_cos,
        pred_depth,
        pred_cls,
        centerpoint_sin,
        centerpoint_cos,
        centerpoint_x,
        depth,
        target_cls,
        classes,
        subdirs,
    ):
        class_idx_dict = {
            "empty_static": [],
            "empty_dynamic": [],
            "still_static": [],
            "sneaky_static": [],
            "normal_static": [],
            "loud_static": [],
            "still_dynamic": [],
            "sneaky_dynamic": [],
            "normal_dynamic": [],
            "loud_dynamic": [],
        }

        if self.args.nostill:
            class_idx_dict.pop("still_static")
            class_idx_dict.pop("still_dynamic")

        ret_dict = {}

        pred_x = torch.atan2(pred_sin, pred_cos)
        pred_x[pred_x < 0] += 2 * torch.pi
        pred_x *= 1440 / (2 * torch.pi)

        loss_reg = torch.minimum(
            torch.abs(pred_x - centerpoint_x), 1440 - torch.abs(pred_x - centerpoint_x)
        )

        if self.args.reg_loss_fcn == "l2":
            loss_reg = loss_reg**2

        loss_reg_cpy = loss_reg.detach().clone().cpu().numpy().astype(np.int32)

        # Zero out loss_reg for all points where target is 0
        loss_reg[target_cls == 0] *= 0
        loss_reg = loss_reg.sum() / (target_cls.sum() + 1e-8)

        if self.args.multi_class:
            loss_cls = self.criterion_cls(pred_cls, classes.long())
        else:
            loss_cls = self.criterion_cls(pred_cls, target_cls)

        loss_depth = self.criterion_depth(pred_depth, depth)

        # Zero out loss_depth for all points where target is -1
        if not self.args.multiclass_depth:
            loss_depth[depth == -1] *= 0
            num_valid_depth = len(depth[depth != -1])
            loss_depth = loss_depth.sum() / (num_valid_depth + 1e-8)

        if not self.args.binary_depth and not self.args.multiclass_depth:
            depth_l1 = torch.abs(pred_depth - depth)
            depth_l1 = depth_l1.detach().cpu().numpy()

        loss = (
            (self.args.reg_w * loss_reg)
            + (self.args.depth_w * loss_depth)
            + (self.args.cls_w * loss_cls)
        )
        ret_dict["loss"] = loss

        # Determine indices for different classes
        for i, subdir in enumerate(subdirs):
            if target_cls[i] == 0:
                # Empty sample
                (
                    date,
                    location,
                    _,
                    robot_class,
                    take,
                    time_idx,
                ) = subdir.split("_")
                class_idx_dict[f"empty_{robot_class}"].append(i)
            else:
                (
                    date,
                    location,
                    participant,
                    move_class,
                    robot_class,
                    take,
                    time_idx,
                ) = subdir.split("_")
                class_idx_dict[f"{move_class}_{robot_class}"].append(i)

        # Store predictions and targets
        for class_name, class_idxs in class_idx_dict.items():
            if len(class_idxs) == 0:
                continue
            ret_dict[f"{class_name}_pred_cls"] = pred_cls.detach().cpu()[class_idxs]
            ret_dict[f"{class_name}_targ_cls"] = target_cls.detach().cpu()[class_idxs]
            ret_dict[f"{class_name}_reg_l1"] = loss_reg_cpy[class_idxs]

            if self.args.binary_depth or self.args.multiclass_depth:
                ret_dict[f"{class_name}_pred_depth"] = pred_depth.detach().cpu()[
                    class_idxs
                ]
                ret_dict[f"{class_name}_targ_depth"] = depth.detach().cpu()[class_idxs]

            else:
                ret_dict[f"{class_name}_depth_l1"] = depth_l1[class_idxs]

        return ret_dict

    def iter_step(self, batch, batch_idx):
        if self.args.learn_backsub:
            (
                spectrograms,
                empty_spectrograms,
                person,
                classes,
                centerpoint_sin,
                centerpoint_cos,
                centerpoint_x,
                depth,
                raw_depth,
                subdir,
            ) = batch

            empty_spectrograms = [
                empty_spectrograms[idx] for idx in self.args.mic_channels
            ]
            empty_spectrograms = [x.to(self.device) for x in empty_spectrograms]
        else:
            (
                spectrograms,
                person,
                classes,
                centerpoint_sin,
                centerpoint_cos,
                centerpoint_x,
                depth,
                raw_depth,
                subdir,
            ) = batch

        # Keep spectrograms of mics that we're training on
        spectrograms = [spectrograms[idx] for idx in self.args.mic_channels]

        # Move to device
        spectrograms = [x.to(self.device) for x in spectrograms]
        centerpoint_sin, centerpoint_cos = centerpoint_sin.to(
            self.device
        ), centerpoint_cos.to(self.device)
        centerpoint_x = centerpoint_x.to(self.device)
        if not self.args.multiclass_depth:
            depth = depth.to(self.device)
            depth = depth.float()
        else:
            depth = depth.long().to(self.device)
            depth[depth != -1] = torch.bucketize(depth[depth != -1], self.bins) - 1
        person = person.to(self.device)
        classes = classes.to(self.device)

        # Training step
        if self.args.condition_depth:
            classes = F.one_hot(classes.long(), num_classes=5).float()

        if self.args.learn_backsub:
            pred_sin, pred_cos, pred_depth, pred_cls = self.forward(
                spectrograms, classes, empty_spectrograms
            )
        else:
            pred_sin, pred_cos, pred_depth, pred_cls = self.forward(
                spectrograms, classes
            )

        ret_dict = self.get_loss_and_pred(
            pred_sin,
            pred_cos,
            pred_depth,
            pred_cls,
            centerpoint_sin,
            centerpoint_cos,
            centerpoint_x,
            depth,
            person,
            classes,
            subdir,
        )

        preds_dict = {
            "pred_sin": pred_sin.detach().clone().cpu().numpy(),
            "pred_cos": pred_cos.detach().clone().cpu().numpy(),
            "pred_depth": pred_depth.detach().clone().cpu().numpy(),
            "pred_cls": pred_cls.detach().clone().cpu().numpy(),
            "raw_depth": raw_depth.detach().clone().cpu().numpy(),
            "centerpoint_x": centerpoint_x.detach().clone().cpu().numpy(),
            "subdir": subdir,
        }

        if self.args.slurm_job_id is not None:
            self.check_preempt()

        return ret_dict, preds_dict

    def run_dataloader(self, dataloader, split):
        classes = [
            "empty_static",
            "empty_dynamic",
            "still_static",
            "sneaky_static",
            "normal_static",
            "loud_static",
            "still_dynamic",
            "sneaky_dynamic",
            "normal_dynamic",
            "loud_dynamic",
        ]

        if self.args.nostill:
            classes.pop(2)
            classes.pop(5)

        # Create dicts for storing results
        metric_dict = {}
        metric_dict[f"{split}_loss"] = []
        preds_dict = {}

        for class_name in classes:
            preds_dict[f"{class_name}_pred_cls"] = []
            preds_dict[f"{class_name}_targ_cls"] = []
            preds_dict[f"{class_name}_reg_l1"] = []
            if self.args.binary_depth or self.args.multiclass_depth:
                preds_dict[f"{class_name}_pred_depth"] = []
                preds_dict[f"{class_name}_targ_depth"] = []
            else:
                preds_dict[f"{class_name}_depth_l1"] = []

        raw_preds_dict = {
            "pred_sin": [],
            "pred_cos": [],
            "pred_depth": [],
            "pred_cls": [],
            "raw_depth": [],
            "centerpoint_x": [],
            "subdir": [],
        }
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            ret, preds = self.iter_step(batch, batch_idx)
            loss = ret[f"loss"]

            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if split == "test":
                # Store raw predictions
                for key in raw_preds_dict.keys():
                    raw_preds_dict[key].append(preds[key])

            # Store metrics and predictions
            metric_dict[f"{split}_loss"].append(loss.item())

            # Iterate through keys in ret except for loss
            for key in ret.keys():
                if key == "loss":
                    continue
                preds_dict[key].append(ret[key])

        # Calculate epoch metrics
        overall_cls_acc = 0
        overall_depth_acc = 0
        total_cls = 0
        total_depth = 0
        for class_name in classes:
            pred_cls = torch.cat(preds_dict[f"{class_name}_pred_cls"], dim=0)
            targ_cls = torch.cat(preds_dict[f"{class_name}_targ_cls"], dim=0)

            # Get binary metrics
            cls_acc = self.accuracy(pred_cls, targ_cls)
            cls_f1 = self.f1(pred_cls, targ_cls)

            metric_dict[f"{split}_{class_name}_cls_acc"] = cls_acc.item()
            overall_cls_acc += cls_acc.item() * len(pred_cls)
            total_cls += len(pred_cls)

            metric_dict[f"{split}_{class_name}_cls_f1"] = cls_f1.item()

            reg_l1 = np.concatenate(preds_dict[f"{class_name}_reg_l1"], axis=0)
            reg_mae = np.mean(reg_l1)

            metric_dict[f"{split}_{class_name}_reg_mae"] = reg_mae

            # Get regression metrics if class is not empty
            if "empty" not in class_name:
                if self.args.binary_depth or self.args.multiclass_depth:
                    pred_depth = torch.cat(
                        preds_dict[f"{class_name}_pred_depth"], dim=0
                    )
                    targ_depth = torch.cat(
                        preds_dict[f"{class_name}_targ_depth"], dim=0
                    )

                    pred_depth = pred_depth[targ_depth != -1]
                    targ_depth = targ_depth[targ_depth != -1]

                    depth_acc = self.depth_accuracy(pred_depth, targ_depth)
                    depth_f1 = self.depth_f1(pred_depth, targ_depth)
                    metric_dict[f"{split}_{class_name}_depth_acc"] = depth_acc.item()
                    metric_dict[f"{split}_{class_name}_depth_f1"] = depth_f1.item()
                    overall_depth_acc += depth_acc.item() * len(pred_depth)
                    total_depth += len(pred_depth)

                else:
                    depth_l1 = np.concatenate(
                        preds_dict[f"{class_name}_depth_l1"], axis=0
                    )
                    depth_mae = np.mean(depth_l1)
                    metric_dict[f"{split}_{class_name}_depth_mae"] = depth_mae

        # Average the loss
        metric_dict[f"{split}_loss"] = np.mean(metric_dict[f"{split}_loss"])

        # Calculate overall cls acc
        overall_cls_acc /= total_cls

        # Calculate overall depth acc
        if self.args.binary_depth or self.args.multiclass_depth:
            overall_depth_acc /= total_depth

        # Add overall cls acc and depth acc to metric dict
        metric_dict[f"{split}_overall_cls_acc"] = overall_cls_acc
        metric_dict[f"{split}_overall_depth_acc"] = overall_depth_acc

        return metric_dict, raw_preds_dict, preds_dict

    def fit(self, trainloader, testloader):
        if self.logger is not None:
            trigger_sync = TriggerWandbSyncHook()

        for i in range(self.current_epoch, self.args.num_epochs, 1):
            # Train
            print("*" * 20 + f" Epoch: {i} " + "*" * 20)
            print("Training...")
            self.model.train()
            train_metric_dict, _, _ = self.run_dataloader(trainloader, "train")

            # Test
            print("Testing...")
            self.model.eval()
            test_metric_dict, raw_preds_dict, _ = self.run_dataloader(
                testloader, "test"
            )

            # Save latest checkpoint
            exp_id = (
                self.args.slurm_job_id
                if self.args.slurm_job_id is not None
                else self.args.exp_name
            )
            if self.args.log:
                dirpath = f"{wandb.run.project}/{exp_id}"
            else:
                dirpath = f"{'checkpoints'}/{exp_id}"
            os.makedirs(dirpath, exist_ok=True)
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    **test_metric_dict,
                },
                os.path.join(dirpath, "last.ckpt"),
            )

            # Save every 5 epochs
            if i % 5 == 0:
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        **test_metric_dict,
                    },
                    os.path.join(dirpath, f"epoch_{i:02d}.ckpt"),
                )

            # Save best checkpoint
            save_best = False
            metric_key = f"test_{self.args.monitor_metric}"

            # If acc or f1, higher is better
            if any(x in metric_key for x in ["acc", "f1"]):
                if test_metric_dict[metric_key] > self.best_metric:
                    save_best = True
            # If mae or loss, lower is better
            elif any(x in metric_key for x in ["mae", "loss"]):
                if test_metric_dict[metric_key] < self.best_metric:
                    save_best = True
            else:
                raise ValueError(f"Invalid metric key: {metric_key}")

            if save_best:
                self.best_metric = test_metric_dict[metric_key]
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        **test_metric_dict,
                    },
                    os.path.join(dirpath, f"best_{metric_key}.ckpt"),
                )

            # Log metrics
            if self.logger is not None:
                metric_dict = {**train_metric_dict, **test_metric_dict}
                metric_dict["epoch"] = i
                self.logger.log(metric_dict)
                trigger_sync()

    def evaluate(self, testloader):
        print("Testing...")
        self.model.eval()
        test_metric_dict, raw_preds_dict, preds_dict = self.run_dataloader(
            testloader, "test"
        )

        # Log metrics
        if self.logger is not None:
            metric_dict = test_metric_dict
            metric_dict["epoch"] = self.current_epoch
            self.logger.log(metric_dict)
        else:
            print(test_metric_dict)

        self.vis_step(raw_preds_dict)

    def vis_step(self, raw_preds_dict):
        vis_subdir = os.path.join(
            self.args.vis_dir,
            self.args.exp_name,
            f"epoch_{self.current_epoch}",
        )

        # Get outputs
        for key in raw_preds_dict.keys():
            raw_preds_dict[key] = np.concatenate(raw_preds_dict[key], axis=0)

        # Get indices where raw_preds_dict["subdir"] does not contain empty
        indices = []
        for i in range(len(raw_preds_dict["subdir"])):
            if "empty" not in raw_preds_dict["subdir"][i]:
                indices.append(i)

        for idx in indices:
            subdir = raw_preds_dict["subdir"][idx]
            pred_sin = raw_preds_dict["pred_sin"][idx]
            pred_cos = raw_preds_dict["pred_cos"][idx]

            # Get the predicted azimuth
            reg_pred = np.arctan2(pred_sin, pred_cos)

            if reg_pred < 0:
                reg_pred += 2 * np.pi

            reg_pred *= 1440 / (2 * np.pi)

            # Create a black image with a white vertical line at the predicted azimuth
            dir_path = os.path.join(vis_subdir, subdir)
            os.makedirs(dir_path, exist_ok=True)

            pred_img = np.zeros((720, 1440), dtype=np.uint8)
            pred_img[:, int(reg_pred)] = 255
            vert_line = Image.fromarray(pred_img)
            vert_line.save(os.path.join(dir_path, f"vert_line_pred.jpg"))


if __name__ == "__main__":
    # For handling preemption if using slurm
    if "SLURM_JOB_ID" in os.environ.keys():
        init_handlers()

    def dilation_rates(s):
        try:
            x, y = map(int, s.split(","))
            return [x, y]
        except:
            raise argparse.ArgumentTypeError("Coordinates must be x,y")

    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/coc/flash9/myang415/detection_dataset_depth",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--empty_data_dir",
        type=str,
        default="/coc/flash9/myang415/empty_aug_wav",
        help="Path to the empty augmentation dataset",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="human detection",
        help="Name of the experiment. Used for wandb logging purposes.",
    )
    parser.add_argument(
        "--proj_name",
        type=str,
        default="human_detection_0123",
        help="Name of the project. Used for wandb logging purposes.",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="train_vis",
        help="Local directory to for saving visualization frames. These frames are used to create the videos visualizing model predictions.",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint if resuming training. Otherwise None.",
    )

    # Spectrogram params
    parser.add_argument(
        "--use_mel", action="store_true", help="Whether or not to use mel spectrograms"
    )
    parser.add_argument("--n_mels", type=int, default=64, help="Number of mel bins")
    parser.add_argument(
        "--fmin", type=int, default=0, help="Minimum frequency for spectrogram"
    )
    parser.add_argument(
        "--fmax", type=int, default=22050, help="Maximum frequency for spectrogram"
    )
    parser.add_argument(
        "--spec_in_channels",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of channels for the input spectrograms. If 1, only use magnitude channel. If 2, use both magnitude and phase channels.",
    )
    parser.add_argument("--n_fft", type=int, default=512, help="n_fft for spectrogram")
    parser.add_argument(
        "--hop_length", type=int, default=160, help="hop_length for spectrogram"
    )
    parser.add_argument(
        "--win_length", type=int, default=400, help="win_length for spectrogram"
    )
    parser.add_argument(
        "--log_spec", action="store_true", help="Whether or not to log the spectrogram"
    )

    # Network hyperparameters
    parser.add_argument("--model", type=str, default="deepwv3")
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="normal_static_reg_mae",
        help="Metric to monitor for saving best checkpoint.",
    )
    parser.add_argument(
        "--test_room",
        type=str,
        nargs="+",
        default=["cherry-322"],
        help="Rooms to test on. Can list more than one.",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Skip training and directly run eval."
    )
    parser.add_argument(
        "--spec_feat_nc",
        type=int,
        default=256,
        help="Number of channels for each spectrogram feature.",
    )

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive weight for binary cross entropy loss.",
    )
    parser.add_argument(
        "--enc_kernel", type=int, default=4, help="Encoder kernel size."
    )
    parser.add_argument(
        "--dilation_rates",
        nargs="+",
        type=dilation_rates,
        default=[[2, 2], [4, 4], [8, 8]],
        help="Dilation rates for the dilated convolutions. Must be a list of lists where each sublist is of length 2. The sublist is the dilation rate for the height and width directions, respectively.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument(
        "--no_dilation",
        action="store_true",
        help="Whether or not to use dilated convolution in the encoder.",
    )
    parser.add_argument(
        "--mic_channels",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Which mic channels to use. Can list more than 1.",
    )
    parser.add_argument(
        "--reg_loss_fcn",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="Regression loss function",
    )
    parser.add_argument(
        "--depth_criterion",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="Loss function for depth if doing regression. Gets overwritten if binary_depth or multiclass_depth is used.",
    )
    parser.add_argument(
        "--multi_class",
        action="store_true",
        help="Whether to do multiclass classification (empty, standing still, ...) instead of binary (empty, person)",
    )
    parser.add_argument("--binary_depth", action="store_true")
    parser.add_argument("--multiclass_depth", action="store_true")
    parser.add_argument(
        "--condition_depth",
        action="store_true",
        help="Whether to condition the depth prediction on the data class.",
    )
    parser.add_argument("--depth_threshold", type=float, default=1.7)
    parser.add_argument("--depth_posw", type=float, default=1.0)
    parser.add_argument("--cls_posw", type=float, default=1.0)
    parser.add_argument("--num_depth_bins", type=int, default=4)

    # Misc
    parser.add_argument(
        "--log", action="store_true", help="Whether or not to log experiments to wandb"
    )
    parser.add_argument("--data_subset", type=float, default=1.0)

    parser.add_argument("--train_still", action="store_true")
    parser.add_argument("--no_empty", action="store_true")
    parser.add_argument("--normalize_point", action="store_true")

    parser.add_argument("--activation", type=str, default="linear")

    parser.add_argument("--clip_len", type=float, default=1.0)
    parser.add_argument("--acc_thresh", type=int, default=60)
    parser.add_argument("--reg_w", type=float, default=0.01)
    parser.add_argument("--depth_w", type=float, default=0.1)
    parser.add_argument("--data_repeat", type=int, default=1)
    parser.add_argument(
        "--nostill", action="store_true", help="Don't use standing still data"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsample rate of non-empty data. The dataset is sampled at 4Hz (every 0.25s). Subsample rate of 2 means that we sample every 0.5s.",
    )
    parser.add_argument(
        "--subsample_empty",
        type=int,
        default=1,
        help="Subsample rate of empty data. The dataset is sampled at 4Hz (every 0.25s). The non-empty dataset is approximately twice as large as the empty. So subsample_empty is by default set to half of subsample for class balance.",
    )
    parser.add_argument("--cls_w", type=float, default=1.0)
    parser.add_argument("--reg_empty_loss", action="store_true")

    # Background subtraction
    parser.add_argument("--background_sub", action="store_true")
    parser.add_argument("--backsub_w", type=float, default=1.0)
    parser.add_argument("--mean_col", action="store_true")
    parser.add_argument("--learn_backsub", action="store_true")

    # Empty room augmentation
    parser.add_argument(
        "--empty_aug", action="store_true", help="Whether to use empty augmentation"
    )
    parser.add_argument(
        "--empty_aug_prob",
        type=float,
        default=0.5,
        help="Probability of applying empty augmentation to training sample",
    )
    parser.add_argument(
        "--empty_w",
        type=float,
        default=0.5,
        help="Weight for augmentation sample, between 0-1.",
    )
    parser.add_argument(
        "--rand_empty_w",
        action="store_true",
        help="Whether to randomly generate args.empty_w",
    )

    # Other data augmentation
    parser.add_argument(
        "--aug_wave",
        action="store_true",
        help="Whether to scale the waveform by random factor",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Whether to add Gaussian noise to the waveform",
    )
    parser.add_argument(
        "--noise_factor", type=float, default=1e-4, help="Scale of Gaussian noise"
    )

    args = parser.parse_args()

    if args.use_mel:
        args.spec_in_channels = 1

    # Check if slurm is being used
    if "SLURM_JOB_ID" in os.environ.keys():
        args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    else:
        # If not, use current date and time as unique experiment identifier
        args.slurm_job_id = None

    # Initialize network
    if args.model == "deepwv3":
        if args.learn_backsub:
            net = DeepWV3PlusMultilossLearnBacksubDepth(
                args=args,
                rates=args.dilation_rates,
                input_nc=args.spec_in_channels,
                encoder_kernel_size=args.enc_kernel,
                no_dilation=args.no_dilation,
                spec_feat_nc=args.spec_feat_nc,
            )
        else:
            net = DeepWV3PlusMultilossDepth(
                args=args,
                rates=args.dilation_rates,
                input_nc=args.spec_in_channels,
                encoder_kernel_size=args.enc_kernel,
                no_dilation=args.no_dilation,
                spec_feat_nc=args.spec_feat_nc,
            )
    else:
        raise NotImplementedError

    # Don't use empty augmentation by setting augmentation probability to 0
    if not args.empty_aug:
        args.empty_aug_prob = 0.0

    # Initialize dataloaders
    train_dataset = HumanDetectionDataset(args, "train", depth=True)
    test_dataset = HumanDetectionDataset(args, "test", depth=True)

    # Person/no-person classification is not necessarily class-balanced so we can weight the loss to mitigate this
    args.cls_posw = train_dataset.cls_posw

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        drop_last=True,
    )

    # This code logs the experiment with wandb offline and uses wandb_osh to asynchronously sync the logs to wandb
    if args.log:
        proj_name = args.proj_name
        wandb.init(config=args, project=proj_name, group=args.exp_name, mode="offline")
        logger = wandb.run
        logger.config.update(args)
    else:
        logger = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.get_device_name() == "NVIDIA A40":
        torch.set_float32_matmul_precision("high")

    # If using slurm, checks if there is an interrupted job and resumes from there
    ckpt_path = None
    if args.slurm_job_id is not None:
        ckpt_root_dir = wandb.run.project if args.log else "checkpoints"
        subdirectories = [os.path.basename(x[0]) for x in os.walk(ckpt_root_dir)]
        if str(args.slurm_job_id) in subdirectories:
            print("FOUND INTERRUPTED JOB, RESUMING...")
            ckpt_path = glob.glob(f"{ckpt_root_dir}/{args.slurm_job_id}/last.ckpt")
            assert len(ckpt_path) == 1
            ckpt_path = ckpt_path[0]

    if args.resume_ckpt:
        ckpt_path = args.resume_ckpt

    trainer = DetectionModel(net, args, ckpt_path, device, logger)

    if args.eval:
        trainer.evaluate(test_loader)
    else:
        # Train the model
        trainer.fit(
            train_loader,
            test_loader,
        )
