import pandas as pd 

def preprocess_data(input_filepath, output_filepath):
    """
    This function preprocesses the music sentiment dataset. 
    
    Args: 
        input_filepath (str): Path to the input CSV file.
        output_filepath (str): The path to save the cleaned CSV file.
    """

    try: 
        df = pd.read_csv(input_filepath)
        print("Dataset loaded successfully. Initial shape:", df.shape)
        print("\nFirst 5 rows of the original dataset:")
        print(df.head())

        print("\nCleaning for missing values...")
        if df.isnull().sum().sum() > 0:
            print(df.isnull().sum())
            df.dropna(inplace=True)
            print("Rows with missing values have been removed.")
        else:
            print("No missing values found.")

        print("\nChecking for duplicate rows...")
        if df.duplicated().sum() > 0:
            df.drop_duplicates(inplace=True)
            print(f"{df.duplicated().sum()} duplicate rows have been removed.")
        else:
            print("No duplicae rows found.")

        print("\nDropping unnecessary columns ('User_ID', 'Recommended_Song_ID')...")
        columns_to_drop = ['User_ID', 'Recommended_Song_ID']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        print("\n--- Preprocessing Complete ---")
        print("Cleaned dataset shape:", df.shape)
        print("\nFirst 5 rows of the cleaned dataset:")
        print(df.head())

        print("\nDataset Info:")
        df.info()

        print("\nDescriptive Statistics:")
        print(df.describe(include='all'))

        df.to_csv(output_filepath, index=False)
        print(f"\nCleaned data has been saved to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: The file at {input_filepath} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    preprocess_data('music_sentiment_dataset.csv', 'cleaned_music_sentiment_dataset.csv')


    