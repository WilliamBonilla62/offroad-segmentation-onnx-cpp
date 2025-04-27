import os
import torch
from torch.utils.data import DataLoader
import super_gradients as sg
from model.factory import ModelFactory, DeepLabV3Wrapper
from super_gradients.training.metrics.segmentation_metrics import IoU
from dataset.goose import goose_create_dataDict, GOOSE_SemanticDataset
from PIL import Image
import numpy as np
class GooseTrainer:
    """
    Trainer class for semantic segmentation on GOOSE dataset using SuperGradients library.

    Attributes:
        dataset_path (str): Path to the GOOSE dataset root directory.
        experiment_name (str): Name of the training experiment.
        batch_size (int): Batch size for training and validation.
        num_epochs (int): Number of training epochs.
        resize_size (tuple): Target resize size for images and masks.
        num_classes (int): Number of segmentation classes.
         model_name (str): Which model to use ("deeplabv3_resnet50", "deeplabv3_resnet101", "maskrcnn_resnet50").
    """

    def __init__(self,
                 dataset_path: str,
                 experiment_name: str = "GOOSE_experiment",
                 batch_size: int = 5,
                 num_epochs: int = 10,
                 resize_size: tuple = (768, 768),
                 num_classes: int = 64, 
                 model_name: str = "deeplabv3_resnet50"):

        self.dataset_path = dataset_path
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.resize_size = resize_size
        self.num_classes = num_classes
        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = os.path.join(os.getcwd(), 'output', 'ckpts')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        sg.setup_device(device=self.device.type)

        self.trainer = sg.Trainer(experiment_name=self.experiment_name,
                                  ckpt_root_dir=self.checkpoint_dir)

    def load_data(self):
        """
        Load training and validation datasets and create DataLoaders.
        """
        test_dict, train_dict, val_dict, _ = goose_create_dataDict(self.dataset_path)

        self.train_dataset = GOOSE_SemanticDataset(train_dict, crop=False, resize_size=self.resize_size)
        self.val_dataset = GOOSE_SemanticDataset(val_dict, crop=False, resize_size=self.resize_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)

    def load_model(self):
        """
        Load the segmentation model.
        """
        base_model = ModelFactory.create_model(name=self.model_name,
                                num_classes=self.num_classes,
                                pretrained_weights='cityscapes')
        # Wrap it if needed
        if 'deeplabv3' in self.model_name:
            self.model = DeepLabV3Wrapper(base_model)
        else:
            self.model = base_model
        self.model.eval()

    def train(self):
        """
        Set up training parameters and run the training loop.
        """
        lr_updates = [int(.3 * self.num_epochs), int(.6 * self.num_epochs), int(.9 * self.num_epochs)]

        train_params = {
            "max_epochs": self.num_epochs,
            "initial_lr": 0.005,
            "optimizer": "sgd",
            "loss": "cross_entropy",
            "lr_mode": "step",
            "lr_updates": lr_updates,
            "lr_decay_factor": 0.1,
            "average_best_models": False,
            "greater_metric_to_watch_is_better": True,
            "metric_to_watch": "IoU",
            "train_metrics_list": [IoU(num_classes=self.num_classes)],
            "valid_metrics_list": [IoU(num_classes=self.num_classes)],
            "loss_logging_items_names": ["loss"],
            "drop_last": True,
        }

        self.trainer.train(
            model=self.model,
            training_params=train_params,
            train_loader=self.train_loader,
            valid_loader=self.val_loader
        )
        self.save_model()


    def inference_and_save(self):
        """
        Run inference on test set and save predicted masks.
        """
        # Load test set
        test_dict, _, _, _ = goose_create_dataDict(self.dataset_path)
        test_dataset = GOOSE_SemanticDataset(test_dict, crop=False, resize_size=self.resize_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create output folder
        output_dir = os.path.join(self.checkpoint_dir, "test_predictions")
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for idx, (images, _) in enumerate(test_loader):
                images = images.to(self.device)

                preds = self.model(images)  # (B, num_classes, H, W)
                preds = torch.argmax(preds, dim=1)  # (B, H, W)

                pred_mask = preds.squeeze(0).cpu().numpy().astype(np.uint8)

                # Save as PNG
                pred_pil = Image.fromarray(pred_mask)
                pred_pil.save(os.path.join(output_dir, f"prediction_{idx:04d}.png"))

        print(f"✅ Inference done. Saved predictions to {output_dir}")

    def save_model(self):
        """
        Save the trained model weights.
        """
        save_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

def main():
    """
    Main function to start GOOSE dataset training.
    """
    # Change this to your real dataset path
    dataset_path = "./dataset"

    goose_trainer = GooseTrainer(
        dataset_path=dataset_path,
        experiment_name="goose_deeplabv3_training",
        batch_size=2,
        num_epochs=10,
        resize_size=(768, 768),
        num_classes=64
    )

    goose_trainer.load_data()
    goose_trainer.load_model()
    goose_trainer.train()
    goose_trainer.inference_and_save()

if __name__ == "__main__":
    main()
