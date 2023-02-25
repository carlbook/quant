#!/home/carl/miniconda3/envs/trading/bin/python
import os
import sys
import numpy as np
# 3080 TI is at BusID 11 (0b) whereas 1050 TI is at BusID 4. See lspci and nvidia-settings. Can BusID change?
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # did not help me reduce "Kernel Launch" time
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
sys.path.extend(['/home/carl/trading/quant'])  # necessary for dataloaders import when this file is not in parent dir
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras import layers, callbacks
from dataloaders import SequenceCRSPdsf62InMemory

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


# noinspection PyMethodOverriding
class PositionalEncoder(layers.Layer):

    def __init__(self, sequence_len, embed_dim):
        super().__init__()
        self.sequence_len = sequence_len
        self.embed_dim = embed_dim
        w = self.compute_weights()
        self.pos_embd = layers.Embedding(input_dim=sequence_len, output_dim=embed_dim, weights=[w], trainable=False)
        idxs = tf.range(start=0, limit=self.sequence_len, delta=1)
        self.embd_idxs = self.pos_embd(idxs)

    def compute_weights(self, n=10000):
        # n=10000 is the value from the original Attention paper
        weights = np.zeros((self.sequence_len, self.embed_dim))
        for k in range(self.sequence_len):
            for i in np.arange(int(self.embed_dim / 2)):
                scale = np.power(n, 2 * i / self.embed_dim)
                weights[k, 2*i] = np.sin(k / scale)
                weights[k, 2*i + 1] = np.cos(k / scale)
        return weights

    def call(self, vect):
        # the input vect will be of shape (sequence_len, embed_dim)
        vect = vect + self.embd_idxs
        return vect


# @tf.function(jit_compile=True)
def mlp(layr, layer_sizes, dropout_rate=0.2):
    for num_units in layer_sizes:
        layr = layers.Dense(num_units, activation=tf.nn.swish)(layr)
        layr = layers.Dropout(dropout_rate)(layr)
    return layr


# @tf.function
def stockvector_xformer(input_layer):
    encoded_inputs = PositionalEncoder(sv_seq_len, sv_embd_dim)(input_layer)
    for _ in range(num_xformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_inputs)
        attn = layers.MultiHeadAttention(
            num_heads=num_xformer_heads, key_dim=sv_embd_dim, attention_axes=2, dropout=0.15)(x1, x1)
        x2 = layers.Add()([attn, encoded_inputs])  # skip connection
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x4 = layers.Flatten()(x3)
        x4 = mlp(x4, xformer_mlp_layer_sizes, dropout_rate=0.15)
        x5 = layers.Reshape((sv_seq_len, sv_embd_dim))(x4)
        encoded_inputs = layers.Add()([x5, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_inputs)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)
    return representation


def full_model(with_sp500=False):
    svect_input = tf.keras.Input(shape=(sv_seq_len, sv_embd_dim))
    cl_hist_input = tf.keras.Input(shape=(num_cl_dct_classes,))
    model_inputs = [svect_input, cl_hist_input]  # data inputs to the overall model
    fc_inputs = [stockvector_xformer(svect_input), cl_hist_input]  # inputs to fully-connected NN
    if with_sp500:
        sp500_input = tf.keras.Input(shape=(1,))
        sp500_input = layers.CategoryEncoding(
            num_tokens=num_sp500_classes, output_mode='one_hot', dtype='float32')(sp500_input)
        model_inputs.append(sp500_input)
        fc_inputs.append(layers.Flatten()(sp500_input))
    merged = layers.concatenate(fc_inputs)
    fc_dims = [1024, 512]
    fc_drop = 0.2
    # fully connected MLP for chosen training signal
    fc_net = layers.Dense(fc_dims[0], activation=tf.nn.swish)(merged)
    fc_net = layers.Dropout(fc_drop)(fc_net)
    fc_net = layers.Dense(fc_dims[1], activation=tf.nn.swish)(fc_net)
    fc_net = layers.Dropout(fc_drop)(fc_net)
    fc_net = layers.Dense(num_output_classes, activation=tf.nn.swish,
                          name=trn_signal_name, dtype='float32')(fc_net)
    mdl = tf.keras.Model(
        inputs=model_inputs,
        outputs=fc_net)
    return mdl


lrn_rates = {
    0: 0.0010,
    1: 0.0012,
    2: 0.0016,
    3: 0.0020,
    4: 0.0024,
    5: 0.0028,
    6: 0.0032,
    7: 0.0034,
    8: 0.0036,
    9: 0.0038}


def lr_schedule(epoch, lr):
    if epoch in lrn_rates:
        return lrn_rates[epoch]
    elif lr > 0.001:
        return max(lr * np.exp(-0.05), 0.001)
    else:
        return lr


if __name__ == '__main__':
    '''When running this file from the terminal, you must do "./model.py argv1 argv2" where argv1 indicates the fold, e.g
    Fold2, and argv2 is an optional argument corresponding to with_sp500'''
    num_args = len(sys.argv)
    if num_args > 1 and sys.argv[1][:4] != 'Fold' or num_args == 1:
        raise ValueError('You must indicate which Fold to use')
    fold = sys.argv[1]
    include_sp5 = True if sys.argv[-1] == 'sp500' else False
    trgt, trn_signal_name, attr = 'md', 'Multiday', 'num_md_classes'
    trn_batch = 512
    val_batch = 2**13
    num_epochs = 50
    num_xformer_heads = 16
    num_xformer_layers = 8
    fname_chkpt = f'mdl_chkpt_{fold}'
    trn_args = ('/mnt/data/trading/datasets/CRSPdsf62_trn_1657667869.hdf5', trn_batch)
    val_args = ('/mnt/data/trading/datasets/CRSPdsf62_val_1657667869.hdf5', val_batch)
    trn_kwargs = {'fold': fold, 'with_sp500': include_sp5, 'target': trgt}  # | {'num_samples': 20000}
    val_kwargs = trn_kwargs | {'epoch_shuffle': False}
    trn_seq = SequenceCRSPdsf62InMemory(*trn_args, **trn_kwargs)
    trn_seq.on_epoch_end()
    val_seq = SequenceCRSPdsf62InMemory(*val_args, **val_kwargs)
    sv_seq_len, sv_embd_dim = trn_seq.attrs['vector_dims']
    num_cl_dct_classes = trn_seq.attrs['cl_dct_coeffs_seq_len']  # number of class labels for retained DCT coefficients
    num_sp500_classes = trn_seq.attrs['num_sp500_cc_classes']  # number of labels for SP500 returns training input
    num_output_classes = trn_seq.attrs[attr]
    xformer_mlp_layer_sizes = [2 * sv_seq_len * sv_embd_dim, sv_seq_len * sv_embd_dim]
    model = full_model(with_sp500=include_sp5)
    tf.keras.utils.plot_model(model, to_file=os.path.join(WORKING_DIR, 'model.png'), show_shapes=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(),
        loss={trn_signal_name: tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)},
        metrics={
            trn_signal_name: [
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='TopKAccuracy'),
                tf.keras.metrics.CategoricalAccuracy(name='CatAccuracy'),
                tf.keras.metrics.CosineSimilarity(name='CosineSim')]},
        run_eagerly=None)
    # profiler = callbacks.TensorBoard(log_dir='./logs', profile_batch=(10, 15))
    # might not want early stopping. See trilium notes "classical stats to modern ML"
    cbks = [
        callbacks.TerminateOnNaN(),
        callbacks.EarlyStopping(monitor='val_loss', patience=5),
        callbacks.ModelCheckpoint(
            filepath=fname_chkpt, monitor='val_loss', mode='auto', save_best_only=True, save_weights_only=False),
        callbacks.CSVLogger(f'training_log_{fold}.csv'),
        callbacks.LearningRateScheduler(lr_schedule)]
    # note model.fit returns the training history...might want to log this to a file
    model.fit(
        trn_seq,
        epochs=num_epochs,
        callbacks=cbks,
        verbose=2,
        validation_data=val_seq,
        validation_steps=None,
        shuffle='batch',
        max_queue_size=64)
