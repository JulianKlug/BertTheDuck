import json
import pandas as pd

def preprocess_input(input_file):
    ## Data Preparation
    with open(input_file) as fin:
        dat = json.load(fin)

    entries = []
    for i in dat['hits']['hits']:
        src = i['_source']
        if 'title' in src and 'snippet' in src:
            entries.append((src['title'], src['snippet'], src['company_codes'], src['industry_codes'],
                            src['region_codes'], src['publication_date']))
    # %%
    df_full = pd.DataFrame([{"title": x[0], "desc": x[1], "company_codes": x[2], 'industry_codes': x[3],
                             'region_codes': x[4], 'publication_date': x[5]} for x in entries])
    print(df_full.desc.str.len().describe())
    print(df_full.shape)
    print(df_full.shape[0], df_full[df_full.desc.str.len() >= 10].shape[0],
          df_full[df_full.desc.str.len() > 200].shape[0])

    # filter small entries
    df_full = df_full[df_full.desc.str.len() >= 10].copy()

    return df_full