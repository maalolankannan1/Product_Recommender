from flask import Flask, request, render_template
from recommender import recommend_items, top5_products

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    user_id = ""
    if request.method == 'POST':
        user_id = request.form['User Name']
        top20_products = recommend_items(user_id)
        if top20_products is None:
            return render_template('index.html', text='No recommendations found for this user ID.')
        print(top20_products.head())
        get_top5 = top5_products(top20_products)
        #return render_template('index.html',tables=[get_top5.to_html(classes='data',header=False,index=False)],text='Recommended products')
        return render_template('index.html',column_names=get_top5.columns.values, row_data=list(get_top5.values.tolist()), zip=zip,text='Recommended products')

    return render_template('index.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
