def split_data(merged_df):
    train_size = int(len(merged_df) * 0.8)
    val_size = int(len(merged_df) * 0.1)

    train_df = merged_df.iloc[:train_size]
    val_df = merged_df.iloc[train_size : train_size + val_size]
    test_df = merged_df.iloc[train_size + val_size :]

    return train_df, val_df, test_df