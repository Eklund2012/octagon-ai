class DatasetInfo( ):

    def dataset_overview(self, df):
        print("Dataset Overview:")
        print(df.head())
        print("\nData Types:")
        print(df.dtypes)
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nStatistical Summary:")
        print(df.describe())