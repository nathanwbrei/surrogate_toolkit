
# PHASM magnetic field map example

Once you have built PHASM with examples, you can run the GlueX magnetic field map example as follows:

```bash

# Set up environment variable pointing to your current work directory.
# This is where all your data files, python scripts, and Jupyter notebooks will live
export WORK_DIR=`pwd`

# Set up environment variable pointing to the directory where you downloaded PHASM. 
export PHASM_SOURCE_DIR=~/projects/phasm

# Set up environment variable pointing to the directory where you installed PHASM
# If you didn't specify a CMAKE_INSTALL_PREFIX, it installs here by default:
export PHASM_INSTALL_DIR=$PHASM_SOURCE_DIR/build/install

# Copy the python scripts you will be using into your work dir
cd $WORK_DIR
cp $PHASM_SOURCE_DIR/examples/magnetic_field_map/train_model.py $WORK_DIR
cp $PHASM_SOURCE_DIR/examples/magnetic_field_map/validate_model.py $WORK_DIR

# Intercept calls to GlueX's magnetic field map to capture training data. 
# This produces `training_captures.csv`
$PHASM_INSTALL_DIR/bin/phasm-example-magfieldmap 

# Create a model in Pytorch and train it using the captured data.
# This produces `gluex_mfield_mlp.pt`
python3 train_model.py training_captures.csv

# Run the field map again, this time using the trained model instead of the original function.
# This produces `validation_captures.csv`
$PHASM_INSTALL_DIR/bin/phasm-example-magfieldmap gluex_mfield_mlp.pt

# Generate a plot of the model performance
python3 validate_model.py training_captures.csv validation_captures.csv

# Analyze the performance of the model interactively
jupyter trust validate_model.ipynb
jupyter lab validate_model.ipynb
```

