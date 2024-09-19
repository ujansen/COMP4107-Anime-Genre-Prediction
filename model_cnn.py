import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class DataPreprocessor:
    def __init__(self, filepath):
        self._filepath = filepath
        self._df = self.read_format_csv()
        self._GENRES = ['Comedy', 'Action', 'Fantasy', 'Adventure', 'Sci-Fi', 'Drama', 'Romance', 'Slice of Life']
        self._filter_genres()
        print(self._df.shape[0])

    def return_num_genres(self):
        return len(self._GENRES)

    def genre_count(self):
        return self._create_genre_count(self._find_genres())

    def plot_genre_hist(self):
        return self._create_genre_hist(self.genre_count())

    def read_format_csv(self):
        return self._read_csv()

    def _read_csv(self):
        df = pd.read_csv(self._filepath)
        filtered_df = self._format_df(df)

        return filtered_df

    def _find_genres(self):
        genre_list = []
        for row in self._df['Genres']:
            genres = row.split(',')
            for genre in genres:
                genre = genre.strip()
                if genre not in genre_list and genre != '-':
                    genre_list.append(genre)

        return genre_list

    def _create_genre_count(self, genres):
        genre_count = {genre: 0 for genre in genres}
        for row in self._df['Genres']:
            genre = row.split(',')
            for genre in genre:
                genre = genre.strip()
                if genre != '-':
                    genre_count[genre] += 1

        sorted_genre_count = {k: v for k, v in sorted(genre_count.items(), key=lambda item: item[1], reverse=True)}

        return sorted_genre_count

    @staticmethod
    def _create_genre_hist(genre_count):
        plt.bar(list(genre_count.keys()), list(genre_count.values()))

        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.title('Genre Distribution')

        plt.xticks(rotation=45)

        plt.show()

    @staticmethod
    def _format_df(df):
        df = df.dropna()
        df = df.drop(columns=['Type', 'Aired', 'Premiered', 'Status', 'Producers', 'Licensors', 'Source',
                              'Duration', 'Episodes'])

        def _choose_title(row):
            if row['English name'] == 'UNKNOWN':
                return row['Name']
            return row['English name']

        df['Title'] = df.apply(_choose_title, axis=1)
        df.insert(1, 'Title', df.pop('Title'))
        df = df.drop(columns=['Other name', 'Name', 'English name'])

        df = df[df['Genres'] != 'UNKNOWN']
        df = df[df['Studios'] != 'UNKNOWN']
        df['Score'] = df['Score'].apply(lambda x: -1 if x == 'UNKNOWN' else x)
        df['Scored By'] = df['Scored By'].apply(lambda x: -1 if x == 'UNKNOWN' else x)
        df['Rank'] = df['Rank'].apply(lambda x: -1 if x == 'UNKNOWN' else x)

        return df

    def _filter_genres(self):

        def _custom_filter(row):
            genre_list = ''
            for genre in row['Genres'].split(', '):
                if genre.strip() in self._GENRES:
                    genre_list = f'{genre_list}{genre.strip()}, '

            return genre_list[:-2]

        self._df['Genres'] = self._df.apply(_custom_filter, axis=1)
        self._df = self._df[self._df['Genres'] != '']

    def create_data_for_cnn(self):

        X, y = [], []

        genre_to_index = {}
        for i, genre in enumerate(self._GENRES):
            genre_to_index[genre] = i

        for index, row in self._df.iterrows():
            img = self.download_image(row['Image URL'])
            if img is not None:
                X.append(img)
                genres = row['Genres'].split(', ')
                label = np.zeros(len(self._GENRES))
                for genre in genres:
                    if genre in genre_to_index:
                        index = genre_to_index[genre]
                        label[index] = 1
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def download_image(self, url):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            return None


class CNN:
    def __init__(self):
        self._data_preprocessor = DataPreprocessor(r'./dataset/anime-dataset-2023.csv')
        self._X, self._y = self._data_preprocessor.create_data_for_cnn()
        self._X = self._X.astype('float32')
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test = self.split_data()

        if os.path.exists('./CNN-model.keras'):
            self._trained_model = load_model('CNN-model.keras')
        else:
            self._trained_model = self.create_model()
            self._trained_model = self.train_model(self._trained_model)
            save_model(self._trained_model, 'CNN-model.keras')

        self._compute_metrics()

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_model(self):
        set_global_policy('mixed_float16')
        num_genres = len(self._data_preprocessor._GENRES)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        cnn = base_model.output
        cnn = GlobalAveragePooling2D()(cnn)
        cnn = Dense(256, activation='relu')(cnn)
        predictions = Dense(num_genres, activation='sigmoid', dtype='float32')(cnn)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['binary_accuracy'])

        return model

    def train_model(self, model, epochs=15):
        model.fit(self._X_train, self._y_train, epochs=epochs, batch_size=16,
                  validation_data=(self._X_val, self._y_val))
        return model

    def test_model(self, model):
        loss, binary_accuracy = model.evaluate(self._X_test, self._y_test)
        print("Test Binary Accuracy:", binary_accuracy)

    def _compute_metrics(self):
        threshold = 0.3
        y_pred = self._trained_model.predict(self._X_test)
        y_pred_binary = (y_pred > threshold).astype(int)

        label_accuracies, label_precisions, label_recalls, label_f1_scores = [], [], [], []

        for label_idx in range(self._data_preprocessor.return_num_genres()):
            label_accuracy = accuracy_score(self._y_test[:, label_idx],
                                            y_pred_binary[:, label_idx])
            label_precision = precision_score(self._y_test[:, label_idx],
                                              y_pred_binary[:, label_idx])
            label_recall = recall_score(self._y_test[:, label_idx],
                                        y_pred_binary[:, label_idx])
            label_f1_score = f1_score(self._y_test[:, label_idx],
                                      y_pred_binary[:, label_idx])

            label_accuracies.append(label_accuracy)
            label_precisions.append(label_precision)
            label_recalls.append(label_recall)
            label_f1_scores.append(label_f1_score)

        for i in range(self._data_preprocessor.return_num_genres()):
            print(f'Label {self._data_preprocessor._GENRES[i]}')
            print(f'Accuracy {label_accuracies[i]:.4f}')
            print(f'Precision = {label_precisions[i]:.4f}')
            print(f'Recall = {label_recalls[i]:.4f}')
            print(f'F1 Score = {label_f1_scores[i]:.4f}\n')


class ImagePrediction:
    def __init__(self):
        self._trained_model = load_model('CNN-model.keras')
        self._GENRES = ['Comedy', 'Action', 'Fantasy', 'Adventure', 'Sci-Fi', 'Drama', 'Romance', 'Slice of Life']

    def get_genres(self, data, tolerance=0.3):
        result = []
        prediction = data[0]
        for i, score in enumerate(prediction):
            if prediction[i] > tolerance:
                result.append(self._GENRES[i])
        return result

    def predict_genre(self, imagepath):
        img = load_img(img_path, target_size=(224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self._trained_model.predict(img_array)
        predicted_genre = self.get_genres(predictions)
        return predicted_genre


if __name__ == '__main__':
    # img_path = './Sci-Fi.jpg'
    # predictor = ImagePrediction()
    # predicted_genres = predictor.predict_genre(img_path)
    # print("Predicted Genres:", predicted_genres)
    cnn = CNN()
