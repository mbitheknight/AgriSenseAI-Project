import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DataUnderstanding:
    def __init__(self, df):
        self.df = df
        
    def check_head_tail(self):
        print("\n First 5 Rows:")
        print(self.df.head())

        print("\n Last 5 Rows:")
        print(self.df.tail())
    
    def check_shape(self):
        print(f"\n Dataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def check_columns(self):
        print("\n Column Names:")
        print(list(self.df.columns))
    
    def check_dtypes(self):
        print("\n Data Types:")
        print(self.df.dtypes)
    
    def check_info(self):
        print("\n Dataset Info:")
        print(self.df.info())
    
    def check_summary(self):
        print("\n Summary Statistics:")
        print(self.df.describe())
    
    def full_report(self):
        self.check_head_tail()
        self.check_shape()
        self.check_columns()
        self.check_dtypes()
        self.check_info()
        self.check_summary()
        
        
class EDA:
    def __init__(self, df):
        self.df = df
        
    def histplot(self, columns):
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()
            
    def boxplot(self, columns):
        for col in columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()
            
    def heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
        plt.title("Feature Correlation Heatmap")
        plt.show()
        
    def scatterplot(self, columns, column1, column2):
        for col in columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=self.df[col], y=self.df[column1])
            plt.title(f"{column1} vs {col}")
            plt.xlabel(col)
            plt.ylabel(f"{column1}")
            plt.show()
            
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=self.df[col], y=self.df[column2])
            plt.title(f"{column2} vs {col}")
            plt.xlabel(col)
            plt.ylabel(f"{column2}")
            plt.show()
            
    def pairplot(self):
        sns.pairplot(self.df, corner=True, height=2, aspect=1)
        plt.title(f"Multivariate analysis using Pairplots")
        plt.show()
        
    def countplot(self,column):
        plt.figure(figsize=(12, 5))
        sns.countplot(y=self.df[column], order=self.df[column].value_counts().index[:10])
        plt.title(f"Top 10 Most Frequent {column}")
        plt.show()


class PricePredictionTuner:
    def __init__(self, data_path, tuner_dir="C:\\temp\\keras_tuner_dir"):
        # Suppress TensorFlow excessive logs
        tf.get_logger().setLevel("ERROR")
        
        # Set up tuner directory
        if os.path.exists(tuner_dir):
            shutil.rmtree(tuner_dir)
        os.makedirs(tuner_dir, exist_ok=True)
        self.tuner_dir = tuner_dir
        
        # Load dataset
        self.df = pd.read_csv(data_path)
        
        # Define features & target variables
        self.X = self.df.drop(columns=["Wholesale", "Retail"])
        self.y_wholesale = self.df["Wholesale"]
        self.y_retail = self.df["Retail"]
        
        # Split data for both wholesale & retail prices
        self.X_train, self.X_test, self.y_train_wholesale, self.y_test_wholesale = train_test_split(
            self.X, self.y_wholesale, test_size=0.2, random_state=42
        )
        _, _, self.y_train_retail, self.y_test_retail = train_test_split(
            self.X, self.y_retail, test_size=0.2, random_state=42
        )
    
    def build_model(self, hp):
        model = Sequential()
        model.add(Dense(
            units=hp.Int("units1", min_value=32, max_value=256, step=32),
            activation='relu',
            input_shape=(self.X_train.shape[1],)
        ))
        
        if hp.Boolean("use_dropout1"):
            model.add(Dropout(rate=hp.Float("dropout_rate1", 0.1, 0.5, step=0.1)))

        model.add(Dense(
            units=hp.Int("units2", min_value=16, max_value=128, step=16),
            activation='relu'
        ))

        if hp.Boolean("use_dropout2"):
            model.add(Dropout(rate=hp.Float("dropout_rate2", 0.1, 0.5, step=0.1)))

        if hp.Boolean("extra_layer"):
            model.add(Dense(
                units=hp.Int("units3", min_value=16, max_value=128, step=16),
                activation='relu'
            ))
            if hp.Boolean("use_dropout3"):
                model.add(Dropout(rate=hp.Float("dropout_rate3", 0.1, 0.5, step=0.1)))

        model.add(Dense(1, activation='linear'))

        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='mse', metrics=['mae'])
        
        return model
    
    def tune_and_train(self, X_train, y_train, X_test, y_test, price_type):
        print(f"\n Tuning Model for {price_type} Prices...")

        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0)

        tuner = kt.Hyperband(
            self.build_model,
            objective="val_mae",
            max_epochs=50,
            factor=3,
            directory=self.tuner_dir,
            project_name=f"price_prediction_{price_type.lower()}"
        )

        tuner.search(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, lr_reduce],
            verbose=0
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        model = tuner.hypermodel.build(best_hp)
        
        print(f" Training best model for {price_type} prices...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop, lr_reduce],
            verbose=0
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, price_type):
        y_pred = model.predict(X_test, batch_size=32, verbose=0).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n **{price_type} Price Model Metrics**")
        print(f" MAE: {mae:.2f}")
        print(f" RMSE: {rmse:.2f}")
        print(f" R² Score: {r2:.2f}")
    
    def run(self):
        model_wholesale = self.tune_and_train(self.X_train, self.y_train_wholesale, self.X_test, self.y_test_wholesale, "Wholesale")
        model_retail = self.tune_and_train(self.X_train, self.y_train_retail, self.X_test, self.y_test_retail, "Retail")
        
        self.evaluate_model(model_wholesale, self.X_test, self.y_test_wholesale, "Wholesale")
        self.evaluate_model(model_retail, self.X_test, self.y_test_retail, "Retail")



class MarketRecommendationModel:
    def __init__(self, data_path, mappings_path):
        self.data_path = data_path
        self.mappings_path = mappings_path
        self.model = None
        self.preprocessor = None
        self.market_mapping = None
        self.num_classes = None
        
    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=['Market_ID'])
        y = data['Market_ID']
        
        with open(self.mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        self.market_mapping = mappings["Market_Reverse"]
        
        self.num_classes = len(np.unique(y))
        return X, y
    
    def get_safe_k(self, y_train):
        min_class_size = y_train.value_counts().min()
        return max(1, min(5, min_class_size - 1))
    
    def preprocess_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        smote_k = self.get_safe_k(y_train)
        smote = SMOTE(sampling_strategy='auto', k_neighbors=smote_k, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train_resampled), y=y_train_resampled
        )
        class_weight_dict = {i: class_weights[i] for i in np.unique(y_train_resampled)}
        
        pca = PCA()
        pca.fit(StandardScaler().fit_transform(X_train_resampled))
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_components = np.argmax(explained_variance >= 0.95) + 1
        pca_components = min(optimal_components, X_train.shape[1])
        
        self.preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_components))
        ])
        
        X_train_transformed = self.preprocessor.fit_transform(X_train_resampled)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        y_train_cat = keras.utils.to_categorical(y_train_resampled, num_classes=self.num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes=self.num_classes)
        
        return X_train_transformed, X_test_transformed, y_train_cat, y_test_cat, class_weight_dict, y_test
    
    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def train_model(self, X_train, y_train, X_test, y_test, class_weights):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_data=(X_test, y_test),
                       class_weight=class_weights, callbacks=[early_stopping])
    
    def evaluate_model(self, X_test, y_test, y_test_original):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        y_pred_market_names = [self.market_mapping.get(mid, "Unknown Market") for mid in y_pred_classes]
        y_test_market_names = [self.market_mapping.get(mid, "Unknown Market") for mid in y_test_original]
        
        accuracy = accuracy_score(y_test_market_names, y_pred_market_names)
        print(f"Model Accuracy (Market Names): {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test_market_names, y_pred_market_names))
        
        return y_pred_classes
    
    def predict_market(self, sample_index):
        y_pred_classes = self.evaluate_model()
        return self.market_mapping.get(y_pred_classes[sample_index], "Unknown Market")
    
    def run(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test, class_weights, y_test_original = self.preprocess_data(X, y)
        self.build_model(X_train.shape[1])
        self.train_model(X_train, y_train, X_test, y_test, class_weights)
        self.evaluate_model(X_test, y_test, y_test_original)

