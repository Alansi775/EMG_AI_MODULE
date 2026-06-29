import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class DeepLearningModel:
    def __init__(self, input_shape, n_classes, learning_rate=0.001):
        """
        input_shape: (window_length, n_channels)
        n_classes: number of gesture classes
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Define a simple but effective CNN for EMG classification.
        """
        model = Sequential()
        
        # Conv block 1
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        
        # Conv block 2
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        
        # Flatten + Dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.n_classes, activation='softmax'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, epochs=30, batch_size=32, validation_split=0.2):
        """
        Train the CNN model.
        X: numpy array of shape (samples, window_len, n_channels)
        y: class labels (integer encoded)
        """
        y_cat = to_categorical(y, num_classes=self.n_classes)
        
        history = self.model.fit(
            X, y_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """
        Predict gesture labels.
        X: numpy array of shape (samples, window_len, n_channels)
        """
        probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        return preds
    
    def evaluate(self, X, y):
        """
        Evaluate model accuracy.
        """
        y_cat = to_categorical(y, num_classes=self.n_classes)
        return self.model.evaluate(X, y_cat, verbose=0)
    
    def save(self, filepath):
        """
        Save trained model.
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load trained model.
        """
        self.model = tf.keras.models.load_model(filepath)
