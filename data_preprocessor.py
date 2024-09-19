import pandas as pd
import matplotlib.pyplot as plt

import regex


class DataPreprocessor:
    def __init__(self, filepath):
        self._filepath = filepath
        self._df = self.read_format_csv()

        self._GENRES = ['Comedy', 'Fantasy', 'Action', 'Adventure', 'Sci-Fi', 'Drama', 'Romance', 'Slice of Life']
        self._filter_genres()

    @property
    def GENRES(self):
        return self._GENRES

    def return_num_genres(self):
        return len(self._GENRES)

    def genre_count(self):
        return self._create_genre_count(self._find_genres())

    def plot_genre_hist(self):
        return self._create_genre_hist(self.genre_count())

    def read_format_csv(self):
        return self._read_csv()

    def create_data_for_fcn(self):
        return self._create_data_for_fcn()

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

    def _create_data_for_fcn(self):
        df = self._df.drop(columns=['anime_id', 'Image URL', 'Title', 'Synopsis'])

        one_hot_encoded_anime_rating = df['Rating'].str.get_dummies(sep=', ').add_prefix('Rating ')
        rank_index = df.columns.get_loc('Rank')
        df = pd.concat([df.iloc[:, :rank_index], one_hot_encoded_anime_rating,
                        df.iloc[:, rank_index:]], axis=1)
        df = df.drop(columns=['Rating'])

        one_hot_encoded_anime_studio = df['Studios'].str.get_dummies(sep=', ').add_prefix('Studios ')
        rank_index = df.columns.get_loc('Rank')
        df = pd.concat([df.iloc[:, :rank_index], one_hot_encoded_anime_rating,
                        df.iloc[:, rank_index:]], axis=1)
        df = df.drop(columns=['Studios'])

        output_vectors = df['Genres'].str.get_dummies(sep=', ').values
        df = df.drop(columns=['Genres'])

        return df, output_vectors

    def create_synopsis_data(self):
        df = self._df[self._df['Synopsis'].apply(lambda x: len(str(x)) > 250)]
        df.loc[:, "Synopsis"] = df["Synopsis"].apply(lambda x: regex.sub(r'(?:\r\n \r\n)(([.])|((.)))', "", x))
        df.loc[:, "Synopsis"] = df["Synopsis"].apply(lambda x: regex.sub(r'\\u\w{4}', "", x))

        synopsis_vector = pd.DataFrame(df['Synopsis'])
        output_vectors = df['Genres'].str.get_dummies(sep=', ').values

        return synopsis_vector, output_vectors

    def create_title_data(self):
        title_vector = pd.DataFrame(self._df['Title'])
        output_vectors = self._df['Genres'].str.get_dummies(sep=', ').values

        return title_vector, output_vectors

    def create_synopsis_title_data(self):
        _, output_vectors = self.create_synopsis_data()

        return output_vectors


if __name__ == '__main__':
    data_preprocessor = DataPreprocessor('./dataset/anime-dataset-2023.csv')
    # data_preprocessor.plot_genre_hist()
