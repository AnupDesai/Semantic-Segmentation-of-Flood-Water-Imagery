class FloodSemSegModel(pl.LightningModule):
    def __init__(self, training_data, val_metadata, hparams):
        super(FloodSemSegModel, self).__init__()
        self.hparams.update(hparams)
        self.feature_id = self.hparams.get("feature_id")
        self.label_id = self.hparams.get("label_id")
        self.train_img_ids = self.hparams.get("train_img_ids")
        self.val_img_ids = self.hparams.get("val_img_ids")
        self.transform = train_transform
        self.backbone = self.hparams.get("backbone", "resnet34")
        self.weights = self.hparams.get("weights", "imagenet")
        self.learning_rate = self.hparams.get("lr", 0.0001)
        self.max_epochs = self.hparams.get("max_epochs", 1000)
        self.min_epochs = self.hparams.get("min_epochs", 10)
        self.patience = self.hparams.get("patience", 4)
        self.num_workers = self.hparams.get("num_workers", 2)
        self.batch_size = self.hparams.get("batch_size", 32)
        self.fast_dev_run = self.hparams.get("fast_dev_run", False)
        self.val_sanity_checks = self.hparams.get("val_sanity_checks", 0)
        self.gpu = self.hparams.get("gpu", False)

        self.output_path = Path.cwd() / self.hparams.get("output_path", "model_outputs")
        self.output_path.mkdir(exist_ok=True)
        self.log_path = Path.cwd() / hparams.get("log_path", "tensorboard_logs")
        self.log_path.mkdir(exist_ok=True)

        
        self.train_dataset = self.config_dataframe(training_data, group="train")
        self.val_dataset = self.config_dataframe(val_metadata, group="val")

        
        self.model = self.build_model()
        self.trainer_params = self.access_training_param()

        # Track validation IOU globally (reset each epoch)
        self.intersection = 0
        self.union = 0


    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        self.model.train()
        torch.set_grad_enabled(True)

        
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        
        preds = self.forward(x)
        criterion = CrossDiceLoss()
        xe_dice_loss = criterion(preds, y)

        self.log(
            "xe_dice_loss",
            xe_dice_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return xe_dice_loss

    def validation_step(self, batch, batch_idx):
        
        self.model.eval()
        torch.set_grad_enabled(False)

        
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        
        preds = self.forward(x)
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1

        
        intersection, union = IoU_fn(preds, y)
        self.intersection += intersection
        self.union += union

        
        batch_iou = intersection / union
        self.log(
            "iou", batch_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return batch_iou

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):     
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "iou_epoch",
        }  
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        self.intersection = 0
        self.union = 0

    def config_dataframe(self, metadata, group):
        if group == "train":
            df = FloodDataLoader(
                metadata=metadata,
                feature_id=self.feature_id,
                label_id=self.label_id,
                torchTransforms=self.transform,
            )
        elif group == "val":
            df = FloodDataLoader(
                metadata=metadata,
                feature_id=self.feature_id,
                label_id=self.label_id,
                torchTransforms=None,
            )
        return df

    def build_model(self): 
      ResUnetModel= ResUnetPlusPlus(channel=3)
      if self.gpu:
        ResUnetModel.cuda()
      return ResUnetModel
    

    def access_training_param(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="iou_epoch",
            mode="max",
            verbose=True,
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="iou_epoch",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )
      
        logger = pl.loggers.TensorBoardLogger(self.log_path, name="benchmark-model")
        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "default_root_dir": self.output_path,
            "logger": logger,
            "gpus": None if not self.gpu else 1,
            "fast_dev_run": self.fast_dev_run,
            "num_sanity_val_steps": self.val_sanity_checks,
        }
        return trainer_params

    def fit(self):
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)
