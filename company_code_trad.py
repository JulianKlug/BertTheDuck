import pandas as pd
import numpy as np

h = True
company_code_map = {}
with open('/Users/Hendrik/Documents/HackZurich2019-master/data/DNA_Taxonomy_Lookup_companies.csv','r') as f:
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

h = True
region_code_map = {}
with open('/Users/Hendrik/Documents/HackZurich2019-master/data/DNA_Taxonomy_Lookup_regions.csv','r') as f:
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


sorted_by_match = pd.read_csv('/Users/Hendrik/Documents/HackZurich2019-master/data/sorted_by_match.csv', sep='\t')
sorted_by_match.insert(0, 'Companies', 'none') #adds column Companies
sorted_by_match.insert(5, 'Regions', 'none')



def translate_company_codes():
    for news_counter in range(len(sorted_by_match['company_codes'])):
        if not pd.isnull(sorted_by_match['company_codes'][news_counter]):
            string = sorted_by_match['company_codes'][news_counter].split(sep=',')
            string = string[1:-1] #remove first and last element
            for counter, i in enumerate(string):
                if i.isalnum():
                    try: string[counter] = company_code_map[i.upper()]
                    except: break
            sorted_by_match['Companies'][news_counter] = string

        if not pd.isnull(sorted_by_match['region_codes'][news_counter]):
            string = sorted_by_match['region_codes'][news_counter].split(sep=',')
            string = string[1:-1]
            for counter, i in enumerate(string):
                if i.isalnum():
                    try: string[counter] = region_code_map[i.upper()]
                    except: break
            sorted_by_match['Regions'][news_counter] = string
    return sorted_by_match




if __name__ == "__main__":
    sorted_by_match = translate_company_codes()
    return


