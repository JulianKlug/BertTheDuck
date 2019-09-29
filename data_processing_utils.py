import json
import pandas as pd
import numpy as np

def preprocess_input(input_file):
    ## Data Preparation
    with open(input_file) as fin:
        dat = json.load(fin)

    number_of_input_articles = 5#128
    entries = []
    for i in dat['hits']['hits']:
        src = i['_source']
        if 'title' in src and 'snippet' in src:
            entries.append((src['title'], src['snippet'], src['company_codes'], src['industry_codes'],
                            src['region_codes'], src['publication_date']))
    # %%
    df_full = pd.DataFrame([{"title": x[0], "desc": x[1], "company_codes": x[2], 'industry_codes': x[3],
                             'region_codes': x[4], 'publication_date': x[5]} for x in entries[:number_of_input_articles]])
    print(df_full.desc.str.len().describe())
    print(df_full.shape)
    print(df_full.shape[0], df_full[df_full.desc.str.len() >= 10].shape[0],
          df_full[df_full.desc.str.len() > 200].shape[0])

    # filter small entries
    df_full = df_full[df_full.desc.str.len() >= 10].copy()

    return df_full

def load_company_code_map():
    h = True
    company_code_map = {}
    with open('./data/DNA_Taxonomy_Lookup_companies.csv','r') as f:
       for line in f:
           if h:
               h = False
               continue
           try:
               linestr = line.strip().replace(',','\t',1)
               code, company_name = linestr.split('\t')
               company_code_map[code] = company_name
           except ValueError:
               print(line)
    return company_code_map

def load_region_code_map():
    h = True
    region_code_map = {}
    with open('./data/DNA_Taxonomy_Lookup_regions.csv','r') as f:
       for line in f:
           if h:
               h = False
               continue
           try:
               linestr = line.strip().replace(',','\t',1)
               code, region = linestr.split('\t')
               region_code_map[code] = region
           except ValueError:
               print(line)
    return region_code_map


def translate_company_codes(dataframe):
    sorted_by_match = dataframe
    sorted_by_match.insert(0, 'Companies', 'none') #adds column Companies
    sorted_by_match.insert(5, 'Regions', 'none')

    region_code_map = load_region_code_map()
    company_code_map = load_company_code_map()

    for news_counter in range(len(sorted_by_match['company_codes'])):
        if not pd.isnull(sorted_by_match['company_codes'][news_counter]):
            string = sorted_by_match['company_codes'][news_counter].split(sep=',')
            string = string[1:-1] #remove first and last element
            for counter, i in enumerate(string):
                if i.isalnum():
                    try: string[counter] = company_code_map[i.upper()]
                    except: break
            print(string)
            
            count = []
            for i in string:

                count.append(sorted_by_match['desc'][0].lower().count(i.split()[0].lower()))
            break
            max_ind = np.where(count == np.amax(count))
            sorted_by_match['Companies'][news_counter] = string[max_ind[0][0]]


        if not pd.isnull(sorted_by_match['region_codes'][news_counter]):
            string = sorted_by_match['region_codes'][news_counter].split(sep=',')
            string = string[1:-1]
            for counter, i in enumerate(string):
                if i.isalnum():
                    try: string[counter] = region_code_map[i.upper()]
                    except: break
            sorted_by_match['Regions'][news_counter] = string
    return sorted_by_match

def select_text_idx(keyword ,df_full):
    df_full['wordcount'] = df_full['desc'].apply(lambda x: len(x.split()))
    df_full = df_full.sort_values(by=['wordcount'], ascending=False)
    for i, title in enumerate(df_full['title']):
        if keyword in title.lower():
            break
    return(df_full, i)
