import requests, json, os, time
from elasticsearch import Elasticsearch
from ssl import create_default_context

context = create_default_context(cafile="client.cer")
es = Elasticsearch(['https://data.schnitzel.tech:9200'], http_auth=('hack_zurich', 'punctualunicorns'), ssl_context=context)

start = time.time()
# res_dj = es.search(index="dj", body={"query": {"match": {'industry_codes':'isocial'}}, "size":10000})
#res_dj = es.search(index="dj", body={"query": {"match": {'publication_date':{'gte':1538120000}}}, "size":1000})
# res_dj = es.search(index="dj", body={"query": {"match": {'company_codes':'snyco,onlnfr'}}, "size":10})


body_json = {
"size": 2000,
    "query": {
       "query_string" : {
           "query" : "(publication_datetime >= 1538120000 AND company_codes_about:onlnfr AND industry_codes:isocial)"
       }
   }
   
}



res_dj = es.search(index="dj", body=body_json)

print("Got %d Hits:" % len(res_dj['hits']['hits']))

print(time.time()-start)
# 

# print(res_dj['hits']['hits'][0]['_source'])
# with open('isocial_10k.json', 'w') as f:
with open('dj_fb.json', 'w') as f:

	json.dump(res_dj, f)
