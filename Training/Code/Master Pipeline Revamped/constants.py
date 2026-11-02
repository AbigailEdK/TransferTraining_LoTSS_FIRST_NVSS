NUM_TO_AUGMENT = 10000

BATCH_SIZE = 16
SIZE_PER_TYPE = 10000

# > On-the-fly augmentation: a fresh random flip and rotation of every training image each epoch (inactive at inference)
ONLINE_AUGMENTATION = True

# > Early stopping and learning-rate schedule. Both monitor the validation cross-entropy without the L2 penalty, which Keras
#   adds to val_loss and which would otherwise penalise any config with a regularisation above 0.
MONITOR = "val_crossentropy"
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7
