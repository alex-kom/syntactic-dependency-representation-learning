import tensorflow as tf


class DependencyLSTM:
    """
    LSTM classifier for sequences of combined word and dependency features.

    Arguments:

        embedding_projection_dim : int
            Dimension of the embedding projection layer.

        dense_layers : array-like of int
            Dimensions of the dense layers before the output layer.

        sent_attention_layers : array-like of int
            Dimensions of the sentence attention layers.

        node_attention_layers : array-like of int
            Dimensions of the node attention layers.

        lstm_dim : int
            Dimension of the lstm.

        dropout_prob : float
            Dropout probability applied on the dense layers before the output
            layer.

        batch_size : int
            Batch size used for training and prediction.

        bidirectional : bool
            Whether to use a bidirectional lstm.

        embedding_projection_activation : activation function or str
            Activation of the embedding projection layer.

        dense_activation : activation function or str
            Activation of the dense layers before the output layer.

        sent_attention_activation : activation function or str
            Activation of the sentence attention layers.

        node_attention_activation : activation function or str
            Activation of the node attention layers.

        dep_pooling : 'attention' or 'mean'
            Whether to pool the dependency embeddings by attention or mean.

        sent_pooling : 'attention' or 'mean'
            Whether to pool the contextualized sentence tokens by attention
            or mean.

        optimizer : tf.keras.optimizers object or str
            The optimizer used for training.

        train_embedding : bool
            Whether to update the embedding layer weights during training.

        metrics : array-like of tf.keras.metrics objects or str
            The metrics to use for training and evaluation.

        monitor : str
            Monitor parameter of tf.keras.callbacks.EarlyStopping.

        mode : 'min', 'max' or 'auto'
            Mode parameter of tf.keras.callbacks.EarlyStopping.

        patience : int
            Patience parameter of tf.keras.callbacks.EarlyStopping.

        max_epochs : int
            The maximum number of epochs for training.

        save_fname : str or None
            If not None, the fitted model will be saved on the save_fname path.
    """
    def __init__(self,
                 embedding_projection_dim=100,
                 dense_layers=(100,),
                 sent_attention_layers=(50,),
                 node_attention_layers=(50,),
                 lstm_dim=100,
                 dropout_prob=0,
                 batch_size=32,
                 bidirectional=True,
                 embedding_projection_activation='linear',
                 dense_activation='relu',
                 sent_attention_activation='tanh',
                 node_attention_activation='tanh',
                 dep_pooling='attention',
                 sent_pooling='attention',
                 optimizer='adam',
                 train_embedding=False,
                 metrics=('accuracy',),
                 monitor='val_accuracy',
                 mode='max',
                 patience=10,
                 max_epochs=100,
                 save_fname=None):
        self.embedding_projection_dim = embedding_projection_dim
        self.embedding_projection_activation = embedding_projection_activation
        self.node_attention_layers = node_attention_layers
        self.node_attention_activation = node_attention_activation
        self.sent_attention_layers = sent_attention_layers
        self.sent_attention_activation = sent_attention_activation
        self.dense_layers = dense_layers
        self.dense_activation = dense_activation
        self.lstm_dim = lstm_dim
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.dep_pooling = dep_pooling
        self.sent_pooling = sent_pooling
        self.optimizer = optimizer
        self.train_embedding = train_embedding
        self.metrics = metrics
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.max_epochs = max_epochs
        self.save_fname = save_fname

    def _build(self):
        embedding = tf.keras.layers.Embedding(
                    self.embeddings.shape[0],
                    self.embeddings.shape[1],
                    weights=[self.embeddings],
                    trainable=self.train_embedding
        )
        word_input = tf.keras.layers.Input(shape=(None,), dtype='int32')
        dep_input = tf.keras.layers.Input(shape=(None, None), dtype='int32')

        word_mask = tf.cast(tf.not_equal(word_input, 0), tf.float32)
        dep_mask = tf.cast(tf.not_equal(dep_input, 0), tf.float32)

        word_vecs = embedding(word_input)
        dep_vecs = embedding(dep_input)

        if self.embedding_projection_dim:
            projection = tf.keras.layers.Dense(
                            self.embedding_projection_dim,
                            activation=self.embedding_projection_activation
            )
            word_vecs = projection(word_vecs)
            dep_vecs = projection(dep_vecs)

        if self.dep_pooling == 'mean':
            dep_vecs = masked_mean(dep_vecs, dep_mask)
        if self.dep_pooling == 'attention':
            dep_attention = self.dep_attention_pooling()
            dep_vecs = dep_attention([word_vecs, dep_vecs, dep_mask])

        h_sent = tf.concat([word_vecs, dep_vecs], axis=-1)
        h_sent *= tf.expand_dims(word_mask, axis=-1)
        h_sent = tf.keras.layers.Masking()(h_sent)

        lstm = tf.keras.layers.LSTM(self.lstm_dim, return_sequences=True)
        if self.bidirectional:
            lstm = tf.keras.layers.Bidirectional(lstm)

        h_sent = lstm(h_sent)

        if self.sent_pooling == 'mean':
            h_sent = masked_mean(h_sent, word_mask)
        if self.sent_pooling == 'attention':
            sent_attention = self.sent_attention_pooling()
            h_sent = sent_attention([h_sent, word_mask])

        h_sent = tf.math.l2_normalize(h_sent, axis=-1)

        for dim in self.dense_layers:
            h_sent = tf.keras.layers.Dense(
                        dim,
                        activation=self.dense_activation
            )(h_sent)

            if self.dropout_prob:
                h_sent = tf.keras.layers.Dropout(self.dropout_prob)(h_sent)

        if self.num_classes == 2:
            output = tf.keras.layers.Dense(1, activation='sigmoid')(h_sent)
        else:
            output = tf.keras.layers.Dense(
                        self.num_classes,
                        activation='softmax'
            )(h_sent)

        self.model = tf.keras.Model(
                        inputs=[word_input, dep_input],
                        outputs=[output]
        )

    def dep_attention_pooling(self):
        """
        Attention layers for pooling the dependency embeddings of each node.

        Attention weights are computed by an mlp with masked softmax activation.
        The queries are the word embeddings (x1), keys and values are the
        dependency embeddings (x2).
        """
        input_dim = (
                self.embedding_projection_dim
                if self.embedding_projection_dim
                else self.embeddings.shape[1]
        )
        x1 = tf.keras.layers.Input(shape=(None, input_dim), dtype='float32')
        x2 = tf.keras.layers.Input(shape=(None, None, input_dim),
                                   dtype='float32')
        mask = tf.keras.layers.Input(shape=(None, None), dtype='float32')

        s = tf.shape(x2)
        x1_tiled = tf.tile(tf.expand_dims(x1, axis=-2), [1, 1, s[2], 1])
        x = tf.concat([x1_tiled, x2], axis=-1)

        for dim in self.node_attention_layers:
            x = tf.keras.layers.Dense(
                    dim,
                    activation=self.node_attention_activation
            )(x)

        w = tf.keras.layers.Dense(1, activation='linear')(x)
        w = masked_softmax(w, mask)
        y = tf.reduce_sum(w * x2, axis=-2)

        return tf.keras.Model(inputs=[x1, x2, mask], outputs=[y])

    def sent_attention_pooling(self):
        """
        Attention layers for pooling the contextualized sentence tokens.

        Attention weights are computed by an mlp with masked softmax activation.
        The query is constant and baked in the mlp layers. Keys and values
        are the vectors returned by the lstm.
        """
        input_dim = 2*self.lstm_dim if self.bidirectional else self.lstm_dim
        x = tf.keras.layers.Input(shape=(None, input_dim), dtype='float32')
        mask = tf.keras.layers.Input(shape=(None,), dtype='float32')

        z = x
        for dim in self.sent_attention_layers:
            z = tf.keras.layers.Dense(
                    dim,
                    activation=self.sent_attention_activation
            )(z)

        w = tf.keras.layers.Dense(1, activation='linear')(z)
        w = masked_softmax(w, mask)
        y = tf.reduce_sum(w * x, axis=-2)

        return tf.keras.Model(inputs=[x, mask], outputs=[y])

    def train(self, data):
        """
        Fit the model parameters on the train data.

        Arguments:

            data : dict
                Dictionary created by DatasetCreator containing
                training, validation and optionally test arrays,
                the pre-trained embeddings matrix,
                and a map from targets to indexes.
        """

        self.num_classes = len(data['target2idx'])
        self.embeddings = data['embeddings']
        self.train_data = data['train_arrays']
        self.val_data = data['val_arrays']
        self.test_data = data.get('test_arrays', None)
        self._build()

        loss = ('binary_crossentropy' if self.num_classes == 2
                else 'categorical_crossentropy')
        self.model.compile(
                loss=loss,
                optimizer=self.optimizer,
                metrics=self.metrics
        )

        callback = tf.keras.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    mode=self.mode,
                    restore_best_weights=True,
                    patience=self.patience
        )

        self.log = self.model.fit(
                    self.train_data[0],
                    self.train_data[1],
                    validation_data=self.val_data,
                    batch_size=self.batch_size,
                    epochs=self.max_epochs,
                    callbacks=[callback]
        )

        if self.save_fname is not None:
            self.model.save(self.save_fname)

    def evaluate(self, test_data=None):
        """Return the model metrics on test data predictions."""
        if test_data is not None:
            self.test_data = test_data
        return self.model.evaluate(self.test_data[0], self.test_data[1])

    def predict(self, X):
        """Return model predictions on X."""
        return self.model.predict(X)


def masked_mean(x, mask):
    """Mean reduction along the penultimate axis ignoring masked values."""
    x_masked = x * tf.expand_dims(mask, axis=-1)
    x_sum = tf.reduce_sum(x_masked, axis=-2)
    n_items = tf.maximum(tf.reduce_sum(mask, axis=-1, keepdims=True), 1.)
    return x_sum / n_items

def masked_softmax(x, mask):
    """Compute softmax activation ignoring masked values."""
    x = tf.exp(x)
    x *= tf.expand_dims(mask, axis=-1)
    x /= tf.maximum(
            tf.reduce_sum(x, axis=-2, keepdims=True),
            tf.keras.backend.epsilon()
    )
    return x
