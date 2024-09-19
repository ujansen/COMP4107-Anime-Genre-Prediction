import os.path
import pandas as pd
import numpy as np

from data_preprocessor import DataPreprocessor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf

from keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv1D, Reshape, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras import Input, Model
from keras.models import save_model, load_model

from transformers import AutoTokenizer, TFBertModel


class FCN:
    def __init__(self):
        self._data_preprocessor = DataPreprocessor('./dataset/anime-dataset-2023.csv')
        self._X, self._y = self._data_preprocessor.create_data_for_fcn()
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test = self._split_data()

        if os.path.exists('./fcn-model.keras'):
            self._model = load_model('fcn-model.keras')
            # self._compute_metrics()
        else:
            self._model = self._create_model()
            self._train_model()
            save_model(self._model, 'fcn-model.keras')

        self._test_model()

    def _split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_model(self):
        input_layer_metadata = Input(shape=(self._X_train.shape[1],))
        model = Dense(512)(input_layer_metadata)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        # model = Dropout(0.2)(model)
        model = Dense(1024)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Activation('relu')(model)
        # model = Dropout(0.2)(model)
        model = Dense(512)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        output_layer = Dense(self._y_train.shape[1], activation='sigmoid')(model)

        model = Model(inputs=input_layer_metadata, outputs=output_layer)
        optimizer = Adam(learning_rate=1e-3)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

        return model

    def _train_model(self):
        self._model.fit(self._X_train,
                        self._y_train,
                        epochs=25,
                        validation_data=(self._X_val, self._y_val))

    def _test_model(self):
        self._model.evaluate(self._X_test, self._y_test)

    def _compute_metrics(self):
        threshold = 0.3
        y_pred = self._model.predict(self._X_test)
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
            print(f'Label {self._data_preprocessor.GENRES[i]}')
            print(f'Accuracy {label_accuracies[i]:.4f}')
            print(f'Precision = {label_precisions[i]:.4f}')
            print(f'Recall = {label_recalls[i]:.4f}')
            print(f'F1 Score = {label_f1_scores[i]:.4f}\n')
        print(f'Mean Precision = {np.mean(label_precisions):.4f}')
        print(f'Mean Recall = {np.mean(label_recalls):.4f}')
        print(f'Mean F1 Score = {np.mean(label_f1_scores):.4f}\n')


class Synopsis:

    def __init__(self, saved_df, synopsis_text):
        self._data_preprocessor = DataPreprocessor('./dataset/anime-dataset-2023.csv')
        self._synopsis_text = synopsis_text
        self._X, self._y = self._data_preprocessor.create_synopsis_data()
        self._model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
        self._bert_model = TFBertModel.from_pretrained(self._model_name)
        if os.path.exists(f'./{saved_df}.json'):
            self._X = pd.read_json(f'./{saved_df}.json')
        else:
            self._X['Embedding'] = self._X['Synopsis'].apply(lambda x:
                                                             self._generate_embeddings(str(x),
                                                                                       self._bert_model).numpy())
            self._X.to_json(f'./{saved_df}.json')
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test = self._split_data()

        if os.path.exists('./synopsis-model.keras'):
            self._trained_model = load_model('synopsis-model.keras')
        else:
            self._trained_model = self._create_model()
            save_model(self._trained_model, 'synopsis-model.keras')
            self._compute_metrics()

        self.final_prediction, self.final_result = self._test_model()

    def _split_data(self):
        X_train_exploded = np.array(self._X['Embedding'].tolist()).reshape(self._X.shape[0], -1)
        X_train, X_test, y_train, y_test = train_test_split(X_train_exploded,
                                                            self._y,
                                                            test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _generate_embeddings(self, sentence, model):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        input_ids = tokenizer(sentence, padding='max_length',
                              truncation=True,
                              return_tensors='tf',
                              max_length=256)['input_ids']
        attention_mask = tf.cast(input_ids != tokenizer.pad_token_id, tf.int32)
        last_hidden_state = model(input_ids, attention_mask=attention_mask)[0]
        return self._mean_pooling(attention_mask, last_hidden_state)

    @staticmethod
    def _mean_pooling(attention_mask, last_hidden_state):
        pre_mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.cast(tf.broadcast_to(pre_mask, tf.shape(last_hidden_state)), dtype=tf.float32)
        masked = tf.math.multiply(mask, last_hidden_state)
        mean_pooled = tf.math.divide(tf.reduce_sum(masked, 1),
                                     tf.clip_by_value(tf.reduce_sum(mask, 1),
                                                      clip_value_min=1e-9, clip_value_max=100))

        return mean_pooled

    def _create_model(self):
        input_layer = Input(shape=(768,))
        reshaped_input = Reshape((1, 768))(input_layer)
        cnn = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(reshaped_input)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = GlobalAveragePooling1D()(cnn)

        # lstm = LSTM(8, return_sequences=True)(reshaped_input)
        # lstm = BatchNormalization()(lstm)
        # lstm = LSTM(16, return_sequences=False)(lstm)
        # lstm = BatchNormalization()(lstm)
        # lstm = LSTM(64, return_sequences=True)(lstm)
        # lstm = BatchNormalization()(lstm)
        # lstm = LSTM(128, return_sequences=False)(lstm)

        # model = Concatenate()([cnn, lstm])
        model = Dense(512, activation='relu')(cnn)
        model = BatchNormalization()(model)
        model = Dropout(0.8)(model)
        model = Dense(1024, activation='relu')(model)
        model = BatchNormalization()(model)
        model = Dropout(0.5)(model)
        # model = Dense(512, activation='relu')(model)
        # model = BatchNormalization()(model)

        output_layer = Dense(self._data_preprocessor.return_num_genres(), activation='sigmoid')(model)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        model.fit(self._X_train, self._y_train,
                  epochs=25,
                  batch_size=128,
                  validation_data=(self._X_val, self._y_val))

        model.evaluate(self._X_test, self._y_test)

        return model

    def _test_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        input_ids = tokenizer(self._synopsis_text, padding='max_length',
                              truncation=True,
                              return_tensors='tf',
                              max_length=256)['input_ids']
        attention_mask = tf.cast(input_ids != tokenizer.pad_token_id, tf.int32)
        last_hidden_state = TFBertModel.from_pretrained(self._model_name)(input_ids, attention_mask=attention_mask)[0]

        pre_mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.cast(tf.broadcast_to(pre_mask, tf.shape(last_hidden_state)), dtype=tf.float32)
        masked = tf.math.multiply(mask, last_hidden_state)
        mean_pooled = tf.math.divide(tf.reduce_sum(masked, 1),
                                     tf.clip_by_value(tf.reduce_sum(mask, 1),
                                                      clip_value_min=1e-9, clip_value_max=100))
        embedding = mean_pooled

        prediction = self._trained_model.predict(embedding)[0]
        result = self._get_genres(prediction)

        return prediction, result

    def _get_genres(self, data, tolerance=0.3):
        result = []
        for i in range(self._data_preprocessor.return_num_genres()):
            if data[i] > tolerance:
                result.append(self._data_preprocessor.GENRES[i])
        return result

    def _compute_metrics(self):
        threshold = 0.3
        print(self._trained_model.evaluate(self._X_test, self._y_test))
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
            print(f'Label {self._data_preprocessor.GENRES[i]}')
            print(f'Accuracy {label_accuracies[i]:.4f}')
            print(f'Precision = {label_precisions[i]:.4f}')
            print(f'Recall = {label_recalls[i]:.4f}')
            print(f'F1 Score = {label_f1_scores[i]:.4f}\n')
        print(f'Mean Precision = {np.mean(label_precisions):.4f}')
        print(f'Mean Recall = {np.mean(label_recalls):.4f}')
        print(f'Mean F1 Score = {np.mean(label_f1_scores):.4f}\n')


class Name:
    def __init__(self, saved_df, title):
        self._data_preprocessor = DataPreprocessor('./dataset/anime-dataset-2023.csv')
        self._title_text = title
        self._X, self._y = self._data_preprocessor.create_title_data()
        self._model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
        self._bert_model = TFBertModel.from_pretrained(self._model_name)
        if os.path.exists(f'./{saved_df}.json'):
            self._X = pd.read_json(f'./{saved_df}.json')
        else:
            self._X['Embedding'] = self._X['Title'].apply(lambda x:
                                                          self._generate_embeddings(str(x), self._bert_model).numpy())
            self._X.to_json(f'./{saved_df}.json')
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test = self._split_data()

        if os.path.exists('./name-model.keras'):
            self._trained_model = load_model('./name-model.keras')
            # self._compute_metrics()
        else:
            self._trained_model = self._create_model()
            save_model(self._trained_model, './name-model.keras')
            self._compute_metrics()

        self.final_prediction, self.final_result = self._test_model()

    def _split_data(self):
        X_train_exploded = np.array(self._X['Embedding'].tolist()).reshape(self._X.shape[0], -1)
        X_train, X_test, y_train, y_test = train_test_split(X_train_exploded,
                                                            self._y,
                                                            test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _generate_embeddings(self, sentence, model):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        input_ids = tokenizer(sentence, padding='max_length',
                              truncation=True,
                              return_tensors='tf',
                              max_length=150)['input_ids']
        attention_mask = tf.cast(input_ids != tokenizer.pad_token_id, tf.int32)
        last_hidden_state = model(input_ids, attention_mask=attention_mask)[0]
        return self._mean_pooling(attention_mask, last_hidden_state)

    @staticmethod
    def _mean_pooling(attention_mask, last_hidden_state):
        pre_mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.cast(tf.broadcast_to(pre_mask, tf.shape(last_hidden_state)), dtype=tf.float32)
        masked = tf.math.multiply(mask, last_hidden_state)
        mean_pooled = tf.math.divide(tf.reduce_sum(masked, 1),
                                     tf.clip_by_value(tf.reduce_sum(mask, 1),
                                                      clip_value_min=1e-9, clip_value_max=100))

        return mean_pooled

    def _create_model(self):
        input_layer = Input(shape=(768,))
        reshaped_input = Reshape((1, 768))(input_layer)
        cnn = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(reshaped_input)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = GlobalAveragePooling1D()(cnn)

        model = Dense(128, activation='relu')(cnn)
        model = BatchNormalization()(model)
        model = Dropout(0.8)(model)
        model = Dense(256, activation='relu')(model)
        model = BatchNormalization()(model)
        model = Dropout(0.5)(model)
        # model = Dense(512, activation='relu')(model)
        # model = BatchNormalization()(model)

        output_layer = Dense(self._data_preprocessor.return_num_genres(), activation='sigmoid')(model)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        model.fit(self._X_train, self._y_train,
                  epochs=25,
                  batch_size=128,
                  validation_data=(self._X_val, self._y_val))

        model.evaluate(self._X_test, self._y_test)

        return model

    def _test_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        input_ids = tokenizer(self._title_text, padding='max_length',
                              truncation=True,
                              return_tensors='tf',
                              max_length=150)['input_ids']
        attention_mask = tf.cast(input_ids != tokenizer.pad_token_id, tf.int32)
        last_hidden_state = TFBertModel.from_pretrained(self._model_name)(input_ids, attention_mask=attention_mask)[0]

        pre_mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.cast(tf.broadcast_to(pre_mask, tf.shape(last_hidden_state)), dtype=tf.float32)
        masked = tf.math.multiply(mask, last_hidden_state)
        mean_pooled = tf.math.divide(tf.reduce_sum(masked, 1),
                                     tf.clip_by_value(tf.reduce_sum(mask, 1),
                                                      clip_value_min=1e-9, clip_value_max=100))
        embedding = mean_pooled

        prediction = self._trained_model.predict(embedding)[0]
        result = self._get_genres(prediction)

        return prediction, result

    def _get_genres(self, data, tolerance=0.3):
        result = []
        for i in range(self._data_preprocessor.return_num_genres()):
            if data[i] > tolerance:
                result.append(self._data_preprocessor.GENRES[i])
        return result

    def _compute_metrics(self):
        threshold = 0.3
        print(self._trained_model.evaluate(self._X_test, self._y_test))
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
            print(f'Label {self._data_preprocessor.GENRES[i]}')
            print(f'Accuracy {label_accuracies[i]:.4f}')
            print(f'Precision = {label_precisions[i]:.4f}')
            print(f'Recall = {label_recalls[i]:.4f}')
            print(f'F1 Score = {label_f1_scores[i]:.4f}\n')
        print(f'Mean Precision = {np.mean(label_precisions):.4f}')
        print(f'Mean Recall = {np.mean(label_recalls):.4f}')
        print(f'Mean F1 Score = {np.mean(label_f1_scores):.4f}\n')


class SynopsisName:
    def __init__(self, saved_df_synopsis, saved_df_name, title, synopsis):
        self._data_preprocessor = DataPreprocessor('./dataset/anime-dataset-2023.csv')
        self._title_text = title
        self._synopsis_text = synopsis
        self._data_preprocessor.create_synopsis_title_data()
        self._X, self._y = pd.DataFrame(), self._data_preprocessor.create_synopsis_title_data()
        self._model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
        self._bert_model = TFBertModel.from_pretrained(self._model_name)

        self._X_1 = pd.read_json(f'./{saved_df_synopsis}.json')
        self._X_2 = pd.read_json(f'./{saved_df_name}.json')

        self._X_1 = self._X_1.drop(columns='Synopsis')
        self._X_2 = self._X_2.loc[self._X_1.index].drop(columns='Title')
        self._X = pd.concat([self._X_1, self._X_2], axis=1)

        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test = self._split_data()

        if os.path.exists('./name-synopsis-model.keras'):
            self._trained_model = load_model('./name-synopsis-model.keras')
            self._compute_metrics()
        else:
            self._trained_model = self._create_model()
            save_model(self._trained_model, './name-synopsis-model.keras')
            self._compute_metrics()
        #
        self.final_prediction, self.final_result = self._test_model()

    def _split_data(self):
        X_train_1, X_train_2 = self._X.iloc[:, 0].values, self._X.iloc[:, 1].values
        X_train_1 = np.array([value[0] for value in X_train_1])
        X_train_2 = np.array([value[0] for value in X_train_2])

        X_concatenated = np.hstack((X_train_1, X_train_2))

        X_train, X_test, y_train, y_test = train_test_split(X_concatenated,
                                                            self._y,
                                                            test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_model(self):
        input_layer = Input(shape=(1536,))
        reshaped_input = Reshape((1, 1536))(input_layer)
        cnn = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(reshaped_input)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = GlobalAveragePooling1D()(cnn)

        model = Dense(128, activation='relu')(cnn)
        model = BatchNormalization()(model)
        model = Dropout(0.8)(model)
        model = Dense(256, activation='relu')(model)
        model = BatchNormalization()(model)
        model = Dropout(0.5)(model)
        # model = Dense(512, activation='relu')(model)
        # model = BatchNormalization()(model)

        output_layer = Dense(self._data_preprocessor.return_num_genres(), activation='sigmoid')(model)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        model.fit(self._X_train, self._y_train,
                  epochs=25,
                  batch_size=128,
                  validation_data=(self._X_val, self._y_val))

        model.evaluate(self._X_test, self._y_test)

        return model

    def _test_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        input_ids_title = tokenizer(self._title_text, padding='max_length',
                                    truncation=True,
                                    return_tensors='tf',
                                    max_length=150)['input_ids']
        attention_mask = tf.cast(input_ids_title != tokenizer.pad_token_id, tf.int32)
        last_hidden_state = TFBertModel.from_pretrained(self._model_name)(input_ids_title,
                                                                          attention_mask=attention_mask)[0]

        pre_mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.cast(tf.broadcast_to(pre_mask, tf.shape(last_hidden_state)), dtype=tf.float32)
        masked = tf.math.multiply(mask, last_hidden_state)
        mean_pooled = tf.math.divide(tf.reduce_sum(masked, 1),
                                     tf.clip_by_value(tf.reduce_sum(mask, 1),
                                                      clip_value_min=1e-9, clip_value_max=100))
        embedding_title = mean_pooled

        input_ids_synopsis = tokenizer(self._synopsis_text, padding='max_length',
                                       truncation=True,
                                       return_tensors='tf',
                                       max_length=150)['input_ids']
        attention_mask = tf.cast(input_ids_synopsis != tokenizer.pad_token_id, tf.int32)
        last_hidden_state = TFBertModel.from_pretrained(self._model_name)(input_ids_synopsis,
                                                                          attention_mask=attention_mask)[0]

        pre_mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.cast(tf.broadcast_to(pre_mask, tf.shape(last_hidden_state)), dtype=tf.float32)
        masked = tf.math.multiply(mask, last_hidden_state)
        mean_pooled = tf.math.divide(tf.reduce_sum(masked, 1),
                                     tf.clip_by_value(tf.reduce_sum(mask, 1),
                                                      clip_value_min=1e-9, clip_value_max=100))
        embedding_synopsis = mean_pooled

        concatenated_embedding = np.concatenate((embedding_synopsis, embedding_title), axis=1)

        prediction = self._trained_model.predict(concatenated_embedding)[0]
        result = self._get_genres(prediction)

        return prediction, result

    def _get_genres(self, data, tolerance=0.3):
        result = []
        for i in range(self._data_preprocessor.return_num_genres()):
            if data[i] > tolerance:
                result.append(self._data_preprocessor.GENRES[i])
        return result

    def _compute_metrics(self):
        threshold = 0.3
        print(self._trained_model.evaluate(self._X_test, self._y_test))
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
            print(f'Label {self._data_preprocessor.GENRES[i]}')
            print(f'Accuracy {label_accuracies[i]:.4f}')
            print(f'Precision = {label_precisions[i]:.4f}')
            print(f'Recall = {label_recalls[i]:.4f}')
            print(f'F1 Score = {label_f1_scores[i]:.4f}\n')
        print(f'Mean Precision = {np.mean(label_precisions):.4f}')
        print(f'Mean Recall = {np.mean(label_recalls):.4f}')
        print(f'Mean F1 Score = {np.mean(label_f1_scores):.4f}\n')


def do_majority_voting(synopsis_prediction, name_prediction, genres):
    threshold = 0.3
    ensemble_predictions = np.maximum(synopsis_prediction, name_prediction)
    ensemble_predictions = np.where(ensemble_predictions >= threshold, 1, 0)
    result = []
    for i in range(len(genres)):
        if ensemble_predictions[i] == 1:
            result.append(genres[i])
    return result


if __name__ == '__main__':
    # fcn = FCN()

    # with open('synopsis.txt', 'r') as f:
    #     synopsis_text = f.read()
    # synopsis = Synopsis('anime_embeddings.csv', synopsis_text)
    # print(synopsis.final_result, synopsis.final_prediction)

    with open('name.txt', 'r') as f:
        name_text = f.read()
    name = Name('anime_embeddings_name.csv', name_text)
    print(name.final_result, name.final_prediction)
    #
    # with open('synopsis.txt', 'r') as f:
    #     synopsis_text = f.read()
    # with open('name.txt', 'r') as f:
    #     name_text = f.read()
    #
    # synopsis_name = SynopsisName('anime_embeddings.csv', 'anime_embeddings_name.csv',
    #                              name_text, synopsis_text)
    # # print(synopsis_name.final_prediction, synopsis_name.final_result)
    #
    # print(do_majority_voting(synopsis.final_prediction, name.final_prediction, synopsis._data_preprocessor._GENRES))
