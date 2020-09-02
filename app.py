from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import mysql.connector
from flaskext.mysql import MySQL
import MySQLdb.cursors
import re, os
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import num2words
import math
import operator
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_ngrok import run_with_ngrok
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from joblib import dump
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'ppt', 'doc', 'pptx', 'mp4'}
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#run_with_ngrok(app)  # Start ngrok when app is run
mydb=mysql.connector.connect(host='localhost',port='3306', user='root',passwd='1234' ,database='mydatabase')
app.secret_key = b'\xb8ti]aK\xd3;\x87\xd0&AS\x18\x9e$'
np.random.seed(500)

@app.route('/newBTP/<userid>/', methods=['GET', 'POST'])
def newBTP(userid):
    userid=userid
    return render_template('newBTP.html',userid=userid)
@app.route('/newBTP1/<userid>/', methods=['GET', 'POST'])
def newBTP1(userid):
    if request.method == 'POST' :
        userid=userid
        mycursor = mydb.cursor(buffered=True)
        mycursor.execute("SELECT * from accounts WHERE id = %(userid)s ",{'userid':userid})
        account = mycursor.fetchone()
        author=" ".join([account[1], account[2]])
        title = request.form['title']
        keywords=request.form['keywords']
        description = request.form['description']
        level =' , '.join( request.form.getlist('level[]'))
        courses = request.form['courses']
        file = request.files['file']
        rights = request.form['rights']
        if not title:
            msg = 'Please insert a title for your BTP!'
            return render_template('newBTP.html', msg=msg)
        elif not description:
            msg = 'Please insert a description for your BTP!'
            return render_template('newBTP.html', msg=msg)
        elif not keywords:
            msg = 'Please insert keywords for your BTP!'
            return render_template('newBTP.html', msg=msg)
        elif not level :
            msg = 'Please select at least one level for the students!'
            return render_template('newBTP.html', msg=msg)
        elif not courses:
            msg = 'Please insert courses in which your BTP can be applied!'
            return render_template('newBTP.html', msg=msg)
        else:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                filename=" "

            BTPclass=predictBTPclass(description)

            mycursor = mydb.cursor()
            mycursor.execute('INSERT INTO BTPs (title, keywords, description, level, courses, rights, filepath, BTPclass,author) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (title, keywords, description, level, courses, rights, filename, (BTPclass),author))
            mydb.commit()
            mycursor.execute("SELECT LAST_INSERT_ID()")
            BTPid=mycursor.fetchone()
            BTPid=BTPid[0]

            #return redirect('/')
            #return render_template('newBTP.html', msg=msg)
            return openBTP1(BTPid)

    return render_template('btp.html', BTPid=BTPid)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def predictBTPclass(description):
    description=description
    Final_description= []
    rootwords=WordNetLemmatizer()
    description= str(description)
    description=description.lower()
    description=description.split()
    description=[rootwords.lemmatize(word) for word in description]
    description=' '.join(description)
    Final_description.append(description)

    with open('vectorizer.pkl', 'rb') as vectorizer:
        vectorizer_model= pickle.load(vectorizer)
    Final_description = vectorizer_model.transform(Final_description)
    with open('classifier.pkl', 'rb') as classifier:
        classifier_model= pickle.load(classifier)
    BTPClass = classifier_model.predict(Final_description)
    p=classifier_model.label_binarizer_.inverse_transform(BTPClass)
    with open('binarizer.pk', 'rb') as le:
        le= pickle.load(le)
    BTPClass = le.inverse_transform(p)
    BTPClass =(str(BTPClass)).replace("'", '')
    BTPClass =BTPClass.replace("(", '')
    BTPClass =BTPClass.replace(")", '')
    BTPClass =BTPClass.replace("[", '')
    BTPClass =BTPClass.replace("]", '')
    return BTPClass

@app.route('/')
def home():
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute( "SELECT * FROM btps ORDER BY id  DESC LIMIT 3")
    rows=mycursor.fetchall()
    mycursor.close()
    x=0
    for x in range(3):
        if x==0:
            row1=rows[x]
        elif x==1:
            row2=rows[x]
        elif x==2:
            row3=rows[x]
    mydb.commit()
    return render_template('home.html',row1=row1,row2=row2,row3=row3)
@app.route('/home')
def home2():
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute( "SELECT * FROM btps ORDER BY id  DESC LIMIT 3")
    rows=mycursor.fetchall()
    mycursor.close()
    x=0
    for x in range(3):
        if x==0:
            row1=rows[x]
        elif x==1:
            row2=rows[x]
        elif x==2:
            row3=rows[x]
    mydb.commit()
    return render_template('home.html',row1=row1,row2=row2,row3=row3)
@app.route('/loggedHome/<BTPid>/<userid>/' , methods=['GET', 'POST'])
def loggedhome(BTPid,userid):
    if request.method == 'POST' :
        userid=userid
        BTPid=BTPid
        mycursor = mydb.cursor()
        mycursor.execute( "DELETE FROM userBTPS WHERE  BTPid = %(BTPid)s and userid=%(userid)s ",{'BTPid':BTPid,'userid':userid })
    return home1(userid)
def home1(userid):
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("SELECT * from accounts WHERE id = %(userid)s ",{'userid':userid})
    account = mycursor.fetchone()
    mycursor.execute("SELECT BTPs.id, BTPs.title, BTPs.description,BTPs.BTPclass, userBTPS.score, userBTPS.userid FROM BTPs RIGHT JOIN userBTPS ON BTPs.id=userBTPS.BTPid WHERE userid = %(userid)s ",{'userid':userid})
    BTPSs=mycursor.fetchall()
    mydb.commit()
    name=account[1]+account[2]
    recommendedBTP={}
    for BTP in BTPSs:
        recommendedBTP[BTP]= BTP[4]
    recommendations=sorted(recommendedBTP, key=operator.itemgetter(4), reverse=True)[:10]
    for x in recommendations:
        print(x[0])
    return render_template('loggedHome.html', name=name, recommendations=recommendations,userid=userid)

#@app.route('/loggedhome/<BTPid>/' , methods=['GET', 'POST'])
#def loggedhome(BTPid):
    #if request.method == 'POST' :
        #BTPid=BTPid
        #mycursor = mydb.cursor()
        #mycursor.execute( "DELETE FROM userBTPS WHERE  BTPid = %(BTPid)s ",{'BTPid':BTPid})
    #return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/media')
def media():
    return render_template('media.html')

@app.route('/login', methods=['GET', 'POST'])
def login1():
    if request.method == 'POST':
        username = request.form['username']
        mycursor = mydb.cursor(buffered=True)
        mycursor.execute("SELECT * FROM accounts WHERE username = %(username)s", { 'username' : username })
        account = mycursor.fetchone()

        if account:
            session['loggedin'] = True
            session['username'] = username
            userid=account[0]
            mycursor = mydb.cursor(buffered=True)
            #recommend()
            return home1(userid)
        else:
            msg='Wrong username or password!!'
        return render_template('login.html', msg=msg)
def recommend():
    mycursor = mydb.cursor()
    ds_BTPs=pd.read_sql('SELECT id,BTPclass FROM  btps',con=mydb)
    ds_users=pd.read_sql('SELECT id,interests FROM  accounts',con=mydb)
    tfidf = TfidfVectorizer()
    users_matrix = tfidf.fit_transform(ds_users['interests'])
    tfidf = TfidfVectorizer()
    BTPs_matrix = tfidf.fit_transform(ds_BTPs['BTPclass'])
    s = cosine_similarity(users_matrix,BTPs_matrix)
    results={}
    for index, row in ds_users.iterrows():
        indices =s[index].argsort()[::-1][:50]
        similarBTPs = [(ds_BTPs['id'][i],s[index][i]) for i in indices]
        results[row['id']] = similarBTPs
    for x in results:
        userid=x
        for pair in results[x]:
            BTPid=pair[0]
            score=pair[1]
            #mycursor.execute("INSERT INTO userBTPs (userid, BTPid, score) VALUES (%s, %s, %s)" % (userid,BTPid,score))
            #mycursor.execute("UPDATE userbtps SET score=%s  WHERE userid = %s and BTPid= %s " % (score,userid,BTPid))
            mydb.commit()
    return 0

@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('home2'))

@app.route('/register')
def register1():
    return render_template('registration.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' :
        # Create variables for easy access
        firstname = request.form['firstname']
        secondname=request.form['secondname']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        interest = request.form.getlist('interests')
        interests = ' , '.join(interest)
        bio = request.form['bio']
        level = request.form.getlist('levels')
        levels = ','.join(level)
        courses = request.form['courses']
       # Check if account exists using MySQL
        mycursor = mydb.cursor()
        checking1="""select * from accounts where username = %s"""
        mycursor.execute(checking1, (username,))
        account = mycursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
            return render_template('registration.html', msg=msg)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
            return render_template('registration.html', msg=msg)
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
            return render_template('registration.html', msg=msg)
        elif not firstname or not username or not password or not email or not interests or not courses :
            msg = 'Please fill out the form!'
            return render_template('registration.html', msg=msg)
        elif not levels:
            msg = 'Please select at least on level of students you teach!'
            return render_template('registration.html', msg=msg)
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            mycursor.execute("INSERT INTO accounts VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (firstname, secondname, username, password, email, interests, bio, levels, courses))
            mydb.commit()
            #recommend()
            return login1()

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST' :
        option=request.form.get("option")
        query1 = request.form['keyword']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM BTPs WHERE description LIKE %s", ("%" + query1 + "%",))
        rows = mycursor.fetchall()
        if len(rows)==0 :
            msg1='no match found for '
        else:
            msg1='Results for '

            all_descriptions=[]
            for row in rows:
                wordss=word_tokenize(str(itemgetter(4,row)))
                wordss = [entry.lower() for entry in wordss]
                cleaned_words = [word for word in wordss if word not in stopwords.words('english') and word.isalpha()]
                PS=PorterStemmer()
                stemmed_words=[PS.stem(word) for word in cleaned_words]
                all_descriptions.append(stemmed_words)
            query=word_tokenize(query1)
            scores=0
            my_rows={}
            i=0
            for description in all_descriptions:
                sc=score(query,description,all_descriptions)
                if sc != 0:
                    my_rows[rows[i]]=sc
                    i+=1
            #for x in my_rows:
                #print(x[0], my_rows[x])
            rows=sorted(my_rows, key=my_rows.get, reverse=True)[:20]
    return render_template('searchResults.html', msg1=msg1, keyword=query1, rows=rows)
def score(q,description,all_descriptions):
    scoree = 0
    for t in q:
        scoree += (tf(t, description) * idf(t,all_descriptions))
    return float(scoree)
def tf(term, description):
    n=1
    for word in description:
        if term in word:
            n +=1
    return( float(n) / float(len(description)))
def idf(term,all_descriptions ):
    n = 1
    for r in all_descriptions :
        if term in r:
            n += 1
    return  math.log(len(all_descriptions )/(float(n)))


@app.route('/rank/<ranktype>/<keyword>',  methods=['GET', 'POST'])
def rank(ranktype,keyword):
    #query1 = request.form['keyword']
    query1=keyword
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM BTPs WHERE description LIKE %s", ("%" + query1 + "%",))
    rows = mycursor.fetchall()
    if len(rows)==0 :
        msg1='no match found for '
    else:
        msg1='Results for '
        if ranktype == 'relevence' :
            all_descriptions=[]
            for row in rows:
                wordss=word_tokenize(str(itemgetter(4,row)))
                wordss = [entry.lower() for entry in wordss]
                cleaned_words = [word for word in wordss if word not in stopwords.words('english') and word.isalpha()]
                PS=PorterStemmer()
                stemmed_words=[PS.stem(word) for word in cleaned_words]
                all_descriptions.append(stemmed_words)
            query=word_tokenize(query1)
            scores=0
            my_rows={}
            i=0
            for description in all_descriptions:
                sc=score(query,description,all_descriptions)
                if sc != 0:
                    my_rows[rows[i]]=sc
                    i+=1
            rows=sorted(my_rows, key=my_rows.get, reverse=True)[:20]
        elif ranktype == 'similarity' :
            if session.get('loggedin'):
                if session['loggedin'] == True:
                    username=session['username']
                    mycursor.execute("SELECT interests FROM accounts WHERE username = %(username)s ", { 'username' : username} )
                    interests=mycursor.fetchone()
                    #interests = account[7]
                    interests = str(interests).split(",")
                    interestss=[]
                    for x in interests:
                        x =x.replace("'", '')
                        x =x.replace("(", '')
                        x =x.replace(")", '')
                        print(x)
                        interestss.append(x)

                    scores={}
                    for row in rows:
                        BTPid=row[0]
                        BTPclasses=row[8]
                        BTPclasses1 = str(BTPclasses).split(",")
                        BTPclasses=[]
                        for x in BTPclasses1:
                            x =x.replace("'", "")
                            x =x.replace("(", "")
                            x =x.replace(")", "")
                            print(x)
                            BTPclasses.append(x)
                        i=0
                        for x in interestss:
                            for BTPclass in BTPclasses:
                                if x!="" and BTPclass!="":
                                    if re.findall(x.strip(),BTPclass.strip()):
                                        i+=1
                        j=0
                        for BTPclass in BTPclasses:
                            if not(BTPclass in interestss):
                                j-=1
                        scores[row]=(.8*i)+(.2*j)
                    for x in scores:
                        print(x[0],scores[x])
                    rows=sorted(scores,key=scores.get,reverse=True)[:20]
            else:
                return render_template('login.html')
    return render_template('searchResults.html', msg1=msg1, keyword=query1, rows=rows)
def score(q,description,all_descriptions):
    scoree = 0
    for t in q:
        scoree += (tf(t, description) * idf(t,all_descriptions))
    return float(scoree)

def tf(term, description):
    n=1
    for word in description:
        if term in word:
            n +=1
    return( float(n) / float(len(description)))

def idf(term,all_descriptions ):
    n = 1
    for r in all_descriptions :
        if term in r:
            n += 1
    return  math.log(len(all_descriptions )/(float(n)))

@app.route('/likeRecommendedBTP/<BTPid>', methods=['GET','POST'])
def likeRecommendedBTP(BTPid):
    if session.get('loggedin'):
        if session['loggedin'] == True:
            BTPid=BTPid
            username=session['username']
            mycursor = mydb.cursor(buffered=True)
            mycursor.execute("SELECT * FROM accounts WHERE username = %(username)s ", { 'username' : username} )
            account=mycursor.fetchone()
            userid = int(account[0])
            ds_BTPs=pd.read_sql('SELECT id,BTPclass FROM  btps',con=mydb)
            tfidf = TfidfVectorizer()
            BTPs_matrix = tfidf.fit_transform(ds_BTPs['BTPclass'])
            s = cosine_similarity(BTPs_matrix,BTPs_matrix)
            similarBTPs=[]
            for index, row in ds_BTPs.iterrows():
                indices =s[index].argsort()[::-1][:10]
                for i in indices:
                    similarBTPs.append((row['id'],ds_BTPs['id'][i],s[index][i]))
            for x in similarBTPs:
                if x[0]==int(BTPid) :
                    similarBTPid=int(x[1])
                    mycursor.execute("SELECT * FROM userbtps WHERE userid = %s and BTPid= %s " % (userid,similarBTPid))
                    similarity=mycursor.fetchone()
                    if similarity !=None:
                        similarBTPscore=similarity[2]
                        similarBTPscore=similarBTPscore+0.8
                        mycursor.execute("UPDATE userbtps SET score=%s WHERE userid = %s and BTPid= %s " % (similarBTPscore,userid,similarBTPid))
                        mydb.commit()
            mycursor.execute( "DELETE FROM userBTPS WHERE  BTPid = %(BTPid)s and userid=%(userid)s ",{'BTPid':BTPid,'userid':userid })
            mydb.commit()
            return redirect(redirect_url())
        else:
            return redirect('login.html')
    return redirect(redirect_url())

def redirect_url(default='index'):
    return request.args.get('next') or \
           request.referrer or \
           url_for(default)
@app.route('/UnlikeRecommendedBTP/<BTPid>', methods=['GET','POST'])
def UnlikeRecommendedBTP(BTPid):
    if session.get('loggedin'):
        if session['loggedin'] == True:
            BTPid=BTPid
            username=session['username']
            mycursor = mydb.cursor(buffered=True)
            mycursor.execute("SELECT * FROM accounts WHERE username = %(username)s ", { 'username' : username} )
            account=mycursor.fetchone()
            userid = int(account[0])
            ds_BTPs=pd.read_sql('SELECT id,BTPclass FROM  btps',con=mydb)
            tfidf = TfidfVectorizer()
            BTPs_matrix = tfidf.fit_transform(ds_BTPs['BTPclass'])
            s = cosine_similarity(BTPs_matrix,BTPs_matrix)
            similarBTPs=[]
            for index, row in ds_BTPs.iterrows():
                indices =s[index].argsort()[::-1][:10]
                for i in indices:
                    similarBTPs.append((row['id'],ds_BTPs['id'][i],s[index][i]))
            for x in similarBTPs:
                if x[0]==int(BTPid) :
                    similarBTPid=int(x[1])
                    mycursor.execute("SELECT * FROM userbtps WHERE userid = %s and BTPid= %s " % (userid,similarBTPid))
                    similarity=mycursor.fetchone()
                    if similarity !=None:
                        similarBTPscore=float(similarity[2])
                        similarBTPscore=similarBTPscore-0.2
                        mycursor.execute("UPDATE userbtps SET score=%s WHERE userid = %s and BTPid= %s " % (similarBTPscore,userid,similarBTPid))
                        mydb.commit()
            mycursor.execute( "DELETE FROM userBTPS WHERE  BTPid = %(BTPid)s and userid=%(userid)s ",{'BTPid':BTPid,'userid':userid })
            mydb.commit()
            return redirect(redirect_url())
        else:
            return redirect('login.html')
    return redirect(redirect_url())
def redirect_url(default='index'):
    return request.args.get('next') or \
           request.referrer or \
           url_for(default)
@app.route('/openBTP/<BTPid>', methods=['GET','POST'])
def openBTP(BTPid):
    if session.get('loggedin'):
        if session['loggedin'] == True:
            BTPid=BTPid
            username=session['username']
            mycursor = mydb.cursor()
            mycursor.execute("SELECT * FROM accounts WHERE username = %(username)s ", { 'username' : username} )
            account=mycursor.fetchone()
            userid = account[0]
    return openBTP1(BTPid)

def openBTP1(BTPid):
    BTPid=BTPid
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("SELECT * from btps WHERE id = %(BTPid)s ", { 'BTPid' : BTPid } )
    row1=mycursor.fetchone()
    filename11 =row1[7]
    if filename11 != None:
        filename1="/static/uploads/"+ str(filename11)
    else:
        filename1=" "
    return render_template('btp.html',BTP=row1,filename1=filename1)
def display(filename1):
    return render_template('media.html', (send_file(filename1,as_attachment=True)))


@app.route('/teachingPractices/<name>/' , methods=['GET', 'POST'])
def teachingPractices(name):
    mycursor = mydb.cursor(buffered=True)
    if name == '1':
        title = 'General and Reference'
    elif name == '2':
        title = 'Hardware'
    elif name == '3':
        title = 'Computer Systems Organization'
    elif name == '4':
        title = 'Networks'
    elif name == '5':
        title = 'Software and its Engineering'
    elif name == '6':
        title = 'Theory of Computation'
    elif name == '7':
        title = 'Mathematics of Computing'
    elif name == '8':
        title = 'Information Systems'
    elif name == '9':
        title = 'Security and Privacy'
    elif name == '10':
        title = 'Human-centered Computing'
    elif name == '11':
        title = 'Computing Methodologies'
    elif name == '12':
        title = 'Applied Computing'
    elif name == '13':
        title = 'Social and Professional Topics'
    else :
        title = 'General '
    mycursor.execute("SELECT * from btps WHERE BTPclass LIKE %s", ("%" + title + "%",))
    BTPs = mycursor.fetchall()
    return render_template('teachingPractices.html', title=title,BTPs=BTPs)

if __name__ == "__main__":
    app.run(debug=True)
