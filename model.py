import tensorflow as tf
from tensorflow.keras import layers


class AdjacencyMatrixFromMothers(tf.keras.layers.Layer):
    """
    Create an adjacency matrix from mother particle indices (one-hot encoding)
    and optionally symmetrize, normalize and add missing self-loops to it
    (according to https://arxiv.org/abs/1609.02907)

    The normalized version of the symmetrized adjacency matrix A with self-loops is given by

    D^(-1/2)AD^(-1/2)

    with D being the degree matrix of A. Therefore the components (ij) are given by:

    deg(i)^(-1/2) * deg(j)^(-1/2)

    for non-zero components in A and 0 otherwise.
    """

    def __init__(self, add_diagonal=True, symmetrize=True, normalize=True):
        super().__init__()
        self.add_diagonal = add_diagonal
        self.symmetrize = symmetrize
        self.normalize = normalize

    def call(self, inputs):
        shape = tf.shape(inputs)
        N = shape[1]
        bs = shape[0]

        # adjaceny matrix is created by one-hot-encoding the mother indices
        idx = tf.where(inputs < 0, tf.cast(N, dtype=tf.float32), inputs)
        # last dimension corresponds to entries without mother particle (or padded slots)
        adj = tf.one_hot(tf.cast(idx, dtype=tf.int32), N + 1)[:, :, :-1]

        if self.symmetrize:
            adj = adj + tf.linalg.matrix_transpose(adj)

        if self.add_diagonal:
            diagonal = tf.broadcast_to(tf.eye(N), (bs, N, N))
            diagonal = tf.where(
                tf.repeat(tf.reshape(inputs != -1, (bs, N, 1)), N, axis=2),
                diagonal,
                tf.zeros_like(adj),
            )
            adj = adj + diagonal

        # have values either 0 or 1
        # (the adding above could produce higher values than 1)
        adj = tf.cast(adj != 0, dtype=tf.float32)

        if self.normalize:
            # calculate outer product of degree vector and multiply with adjaceny matrix
            deg_diag = tf.reduce_sum(adj, axis=2)
            deg12_diag = tf.where(deg_diag > 0, deg_diag  ** -0.5, 0)
            adj = tf.matmul(
                tf.expand_dims(deg12_diag, axis=2),
                tf.expand_dims(deg12_diag, axis=1),
            ) * adj

        return adj


class SimpleGCN(tf.keras.layers.Layer):
    """
    Simple graph convolution. Should be equivalent to Kipf & Welling (https://arxiv.org/abs/1609.02907)
    when fed a normalized adjacency matrix.
    """

    def __init__(self, units, activation="relu"):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        feat, adjacency = inputs
        return self.activation(tf.matmul(adjacency, self.dense(feat)))


def get_model(units=128, num_nodes=100, num_features=8, num_pdg=540, emb_size=8):
    adjacency_input = layers.Input(shape=(num_nodes,), name='x_adjacency')
    feature_input = layers.Input(shape=(num_nodes, num_features), name='x_feature')
    pdg_input = layers.Input(shape=(num_nodes,), name='x_pdg')

    pdg_l = layers.Embedding(
        input_dim=num_pdg + 1,
        output_dim=emb_size,
        name='embedding',
    )(pdg_input)

    adjacency_l = AdjacencyMatrixFromMothers(
        add_diagonal=True, symmetrize=True, normalize=True
    )(adjacency_input)

    feat_pdg_input = layers.Concatenate()([pdg_l, feature_input])

    # particle-level transformations
    p = feat_pdg_input
    for i in range(3):
        p = layers.Dense(units, activation="relu")(p)

    for i in range(3):
        p = SimpleGCN(units, activation="relu")([p, adjacency_l])

    x = layers.GlobalAveragePooling1D()(p)

    # event-level transformations
    for i in range(3):
        x = layers.Dense(units, activation="relu")(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(
        inputs=[adjacency_input, feature_input, pdg_input],
        outputs=[output]
    )
