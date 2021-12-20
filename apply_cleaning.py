from prepare import *
if __name__ == '__main__':
    dataset = pd.read_csv("virus_data.csv")
    # Selecting samples from the dataset while randomizing the samples
    train_df = dataset.sample(frac=0.8, random_state=16)

    # Selecting all the samples which are not in training_data
    test_df = dataset.drop(train_df.index)

    # Clean training set according to itself
    train_df_clean = prepare_data(train_df, train_df)
    # Clean test set according to the raw training set
    test_df_clean = prepare_data(test_df, train_df)

    train_df_clean.to_csv("train_df_clean.csv")
    test_df_clean.to_csv("test_df_clean.csv")
