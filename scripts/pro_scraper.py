from curl_cffi import requests
import pandas as pd
import json
import time


def get_tgju_history(item_slug):
    
    url = f"https://api.tgju.org/v1/market/indicator/summary-table-data/{item_slug}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': f'https://www.tgju.org/profile/{item_slug}/history',
        'Origin': 'https://www.tgju.org',
        'Accept': 'application/json, text/plain, */*',
    }

    print(f"Fetching data for {item_slug}...")

    try:
        session = requests.Session()
        response = session.get(url, headers=headers, impersonate="chrome110")

        if response.status_code == 200:
            json_data = response.json()

            if 'data' in json_data:
                df = pd.DataFrame(json_data['data'])

                return df
            else:
                print("Key 'data' not found in response.")
                print(json_data.keys())
                return None
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None



df_gold = get_tgju_history("geram18")


if df_gold is not None:
    print(f"Data fetched successfully! Rows: {len(df_gold)}")
    print("Original Columns: ", df_gold.columns.tolist())
    print("First row sample: ", df_gold.iloc[0].tolist())

    new_column_names = ['Open', 'Low', 'High', 'Close', 'Change', 'Percent', 'Date', 'Date_G']
    current_cols_count = len(df_gold.columns)
    if current_cols_count >= len(new_column_names):
        df_gold.columns = new_column_names + [f'col_{i}' for i in range(len(new_column_names), current_cols_count)]
    else:
        df_gold.columns = new_column_names[:current_cols_count]


    def clean_currency(x):
        if isinstance(x, str):
            return x.replace(',', '')
        return x


    cols_to_fix = ['Open', 'Low', 'High', 'Close', 'Change']
    for col in cols_to_fix:
        if col in df_gold.columns:
            try:
                df_gold[col] = df_gold[col].apply(clean_currency).astype(float)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to float. Error: {e}")


    df_gold.to_csv("new_gold_data.csv", index=False)
    print("\nDone! Data saved to csv file")
    print(df_gold.head())

else:
    print("Failed to fetch data.")