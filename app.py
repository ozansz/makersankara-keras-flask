from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home_template():
    return render_template('index.html')

@app.route('/name/<name>')
def hello_name(name):
    return '<h1>Hello ' + name + '</h1>'

@app.route('/signin')
def signin_page():
    return render_template('in.html')

if __name__ == '__main__':
    app.run()
