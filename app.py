import json, os, time
from flask import Flask, abort, jsonify, make_response
from BERT_Recommender import BERT_Recommender
from data_processing_utils import preprocess_input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()



input_file = '/Users/julian/hackzurich/trial1/isocial.json'
output_dir = '/Users/julian/hackzurich/banking_web_app/public/recommendations'

app = Flask(__name__)

@app.route('/get_recommendations')
def get_recommendations():
    try:
        df_full = preprocess_input(input_file)
        recommandation_system = BERT_Recommender()

        start = time.time()
        all_sorted_matches = recommandation_system.get_recommendations(df_full)
        best_matches = all_sorted_matches[:10]
        print('OUT', best_matches)
        end = time.time()
        print('Prediction time:', end - start)

        article_list = []
        for _, row in df_full.iloc[best_matches].iterrows():
            article_list.append(row.to_dict())
        # %%
        
        for article in article_list.items():
            sentiment = analyser.polarity_scores(article['desc'])['compound']
            article['sentiment'] = sentiment

        with open(os.path.join(output_dir, 'recommendations.json'), 'w') as outfile:
            json.dump(article_list, outfile)
        # %%

        df_full.iloc[all_sorted_matches].to_csv(os.path.join(output_dir, 'sorted_by_match.csv'), index=False)
        print(df_full.iloc[all_sorted_matches].head())

        data = {'message': 'Computed', 'code': 'SUCCESS'}
        return make_response(jsonify(data), 200)
    except:
        return abort(404)

if __name__ == "__main__":
    # here is starting of the development HTTP server
    app.run()

