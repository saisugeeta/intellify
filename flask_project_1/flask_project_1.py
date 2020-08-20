# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,flash,redirect
from datetime import datetime
from form import RegistrationForm,LoginForm
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
# USED import secrets 
#secrets.token_hex(16)
app.config["SECRET_KEY"]='d2961be86a5103f40c687effc4b0f341'
app.config["SQLALCHEMY_DATABASE_URI"]='sqlite:///site.db'
db=SQLAlchemy(app)

class User(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(20),unique=True,nullable=False)
    email=db.Column(db.String(120),unique=True,nullable=False)
    image_file=db.Column(db.String(20),nullable=False,default='default.jpg')
    password=db.Column(db.String(60),nullable=False)
    posts=db.relationship("Post",backref="author",lazy=True)
    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"
    
class Post(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    title=db.Column(db.String(100),nullable=False)
    dateposted=db.Column(db.DateTime,nullable=False,default=datetime.utcnow())
    content=db.Column(db.Text,nullable=False)
    user_id=db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    def __repr__(self):
        return f"Post('{self.title}','{self.dateposted}')"
posts=[
       {
        "author":"Ch Sai Sugeeta",
        "date_posted":datetime.now().strftime("%Y-%m-%d"),
        "title":"Post1",
        "content":"My first flask work"
        
        },
        {
        "author":"Prassnna",
        "date_posted":datetime.now().strftime("%Y-%m-%d"),
        "title":"Post2",
        "content":"I am intermidiate in  flask work"
        
        },
        
        
       
       ]

@app.route('/')
def hello():
    return render_template('home.html',posts=posts)
@app.route('/about')
def about():
    return render_template('about.html',title="About")
@app.route('/register',methods=['GET','POST'])
def register():
    form=RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account Created for {form.username.data}!','success')
        return redirect(url_for('hello'))#url_for ()-the function should be passed inside,like here if we wanna redirect to home hello() is passed.
        
    return render_template('register.html',title="Register",form=form)
@app.route('/login',methods=['GET','POST'])
def login():
    form=LoginForm()
    if form.validate_on_submit():
        if form.email.data=="sugeeta@gmail.com" and form.password.data=="1234":
            flash("You have logged in","success")
            return redirect(url_for('hello'))
        else:
            flash("Incorrect User name and Password Check Again",'danger')
    return render_template('login.html',title="Register",form=form)
if __name__=="__main__":
   
    app.run(debug=True)


