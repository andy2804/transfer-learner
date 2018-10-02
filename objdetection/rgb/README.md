# RGB Daytime Only Trainer

This package contains code to retrain a pretrained network to detect daytime objects only.

## Run the rgb2ir pipeline

2. Run `train_rgb_from_dayonly.sh` shell script to launch training.
   Be sure to have set up tensorflow models correctly for retraining.
3. After training run `export_rgb_dayonly.sh` script to derivate the frozen inference graph from
   the model.