import json, os, time
from flask import Flask, abort, jsonify, make_response
from BERT_Recommender import BERT_Recommender
from data_processing_utils import preprocess_input, translate_company_codes
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


input_file = os.path.expanduser('~/hackzurich/trial1/isocial.json')
output_dir = os.path.expanduser('~/hackzurich/asset_recommendations_app/public/recommendations')


app = Flask(__name__)

analyser = SentimentIntensityAnalyzer()
recommandation_system = BERT_Recommender()

@app.route('/get_recommendations')
def get_recommendations():
    seed_company = 'snapchat'
    try:
        df_full = preprocess_input(input_file)
        start = time.time()
        all_sorted_matches, company_code = recommandation_system.get_recommendations(df_full, seed_company)
        end = time.time()
        print('Prediction time:', end - start)

        # Postprocessing
        df_full = translate_company_codes(df_full)

        #  Saving Data
        
        
        id_rm = [company_code in i for i in df_full['company_codes']]
        best_matches = [i for i in all_sorted_matches if i not in id_rm]
        best_matches = best_matches[:20]
        df_sub = df_full.iloc[best_matches,:]
        
        df_sub['publication_date'].astype(int)
        df_sub = df_sub.sort_values(by=['publication_date'], ascending=False)
        df_sub = df_sub.iloc[:10,:]
        
        print('OUT', df_sub)

        article_list = []
        for _, row in df_sub.iterrows():
            article_list.append(row.to_dict())
        # %%
        
        for article in article_list:
            sentiment = analyser.polarity_scores(article['desc'])['compound']
            article['sentiment'] = sentiment

        with open(os.path.join(output_dir, 'recommendations_' + seed_company + '.json'), 'w') as outfile:
            json.dump(article_list, outfile)
        # %%

        df_sub.to_csv(os.path.join(output_dir, 'sorted_by_match.csv'), index=False)
        print(df_sub.head())

        data = {'message': 'Computed', 'code': 'SUCCESS'}
        return make_response(jsonify(data), 200)
    except Exception as i:
        print(i)
        return abort(404)

if __name__ == "__main__":
    # here is starting of the development HTTP server
    app.run()

