import flask
import csv
import os
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import difflib
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import random
from random import choice
from werkzeug.utils import secure_filename
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
from tensorflow import keras

nltk.download('vader_lexicon')

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/acer/Desktop/Sourcecode/Sourcecode/static/img'
if not os.path.exists(UPLOAD_FOLDER) :
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df2 = pd.read_csv('tmdb.csv')
count = CountVectorizer(stop_words='english') #
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']) #data cam movie recommendation
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    dat = df2['release_date'].iloc[movie_indices]
    rating = df2['vote_average'].iloc[movie_indices]
    moviedetails=df2['overview'].iloc[movie_indices]
    movietypes=df2['keywords'].iloc[movie_indices]
    movieid=df2['id'].iloc[movie_indices]

    return_df = pd.DataFrame(columns=['Title','Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return_df['Ratings'] = rating
    return_df['Overview']=moviedetails
    return_df['Types']=movietypes
    return_df['ID']=movieid
    return return_df

def get_suggestions():
    data = pd.read_csv('tmdb.csv')
    return list(data['title'].str.capitalize())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classification')
def classification():
    return render_template("classification.html" )

@app.route("/home") #HOME
def home():
    return render_template("home.html")

@app.route("/api", methods=["POST"])
def api():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            for imagefile in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, imagefile))

            filename = secure_filename(file.filename)

            filename = filename.split('.')[0] + str(random()) + "." + filename.split('.')[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            path = os.path.join(UPLOAD_FOLDER, filename)  # Use os.path.join for paths
            img = keras.utils.load_img(path, target_size=(350, 350, 3))
            img = np.array(img)
            img = img / 255

            img = img.reshape(1, 350, 350, 3)
            model = keras.models.load_model("C:/Users/acer/Desktop/Sourcecode/Sourcecode/model/PosterCNN.h5")
            y_prob = model.predict(img)

            classes = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                       'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
                       'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance',
                       'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

            y_prob = list(y_prob[0])

            sorted_y = sorted(y_prob, reverse=True)
            top3 = []
            for i in range(4):
                top3.append(classes[y_prob.index(sorted_y[i])])

            return render_template("classification.html",
                                   filename=str(filename),
                                   classes=classes,
                                   sorted_y=sorted_y,
                                   y_prob=y_prob,
                                   len=len(sorted_y) - 14,
                                   top3=top3)

    # Add a return statement for cases where none of the conditions above are met.
    return "Invalid request"

@app.route("/analysis", methods=["GET", "POST"])
def analysis():

    if request.method=="POST":
        inp =request.form.get("inp")
        sid =SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score["compound"] <= -0.05:
             with open('ReviewHistory.csv', 'a',newline='') as csv_file:
                fieldnames = ['Review','Polarity']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Review': inp,'Polarity':'Negative'})
                Neg_polarity = 0
                Neg_polarity = Neg_polarity + 1
                return (flask.render_template("result.html", message="Negative☹️☹️",review=inp))
        elif score["compound"] >= 0.05:
            with open('ReviewHistory.csv', 'a',newline='') as csv_file:
                fieldnames = ['Review','Polarity']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Review': inp,'Polarity':'Positive'})
                Pos_polarity = 0
                Pos_polarity = Pos_polarity + 1
                return flask.render_template("result.html", message="Positive🙂🙂",review=inp)
        else:
              with open('ReviewHistory.csv', 'a',newline='') as csv_file:
                fieldnames = ['Review','Polarity']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Review': inp,'Polarity':'Neutral'})
                Neu_polarity = 0
                Neu_polarity = Neu_polarity + 1
                return flask.render_template("result.html", message="Neutral😐😐",review=inp)
                
    return render_template("analysis.html")  

@app.route('/delete')
def delete():
    # open CSV file
    with open('ReviewHistory.csv') as rf:
        # updated data
        data = []
        
        # load data
        temp_data = []
        
        # create CSV dictionary reader
        reader = csv.DictReader(rf)
        
        # init CSV rows
        [temp_data.append(dict(row)) for row in reader]
        
        # create mew dataset but without a row to delete
        [
            data.append(temp_data[row]) 
            for row in range(0, len(temp_data))
            if row != int(request.args.get('id'))
        ]

        # update the CSV file
        with open('ReviewHistory.csv', 'w') as wf:
            # create CSV dictionary writer
            writer = csv.DictWriter(wf, fieldnames=data[0].keys())
            
            # write CSV column names
            writer.writeheader()
            
            # write CSV rows
            writer.writerows(data)

    # return to READ data page (to see the updated data)
    return redirect('/view')

@app.route('/view')
def read():
    # variable to hold CSV data
    data = []
    
    # read data from CSV file
    with open('ReviewHistory.csv') as f:
        
        # create CSV dictionary reader instance
        reader = csv.DictReader(f)
        # init CSV dataset
        [data.append(dict(row)) for row in reader]
    # render HTML page dynamically
    return render_template("view.html", data=data, list=list, len=len, str=str)

@app.route("/index")
def index():
    NewMovies=[]
    with open('SearchHistory.csv','r') as csvfile:
        readCSV = csv.reader(csvfile)
        NewMovies.append(choice(list(readCSV)))
    m_name = NewMovies[0][0]
    m_name = m_name.title()
    
    with open('SearchHistory.csv', 'a',newline='') as csv_file:
        fieldnames = ['Movie']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'Movie': m_name})
        result_final = get_recommendations(m_name)
        names = []
        dates = []
        ratings = []
        overview=[]
        types=[]
        mid=[]
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])
            dates.append(result_final.iloc[i][1])
            ratings.append(result_final.iloc[i][2])
            overview.append(result_final.iloc[i][3])
            types.append(result_final.iloc[i][4])
            mid.append(result_final.iloc[i][5])
    suggestions = get_suggestions()
    
    return render_template('index.html',suggestions=suggestions,movie_type=types[5:],movieid=mid,movie_overview=overview,movie_names=names,movie_date=dates,movie_ratings=ratings,search_name=m_name)

@app.route('/positive', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
        if m_name not in all_titles:
            return(flask.render_template('negative.html',name=m_name))
        else:
            with open('SearchHistory.csv', 'a',newline='') as csv_file:
                fieldnames = ['Movie']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Movie': m_name})
            result_final = get_recommendations(m_name)
            names = []
            dates = []
            ratings = []
            overview=[]
            types=[]
            mid=[]
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])
                ratings.append(result_final.iloc[i][2])
                overview.append(result_final.iloc[i][3])
                types.append(result_final.iloc[i][4])
                mid.append(result_final.iloc[i][5])
               
            return flask.render_template('positive.html',movie_type=types[5:],movieid=mid,movie_overview=overview,movie_names=names,movie_date=dates,movie_ratings=ratings,search_name=m_name)
            
if __name__ == '__main__':
    app.run(debug=True)
