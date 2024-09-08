from flask import Flask
from routes import index, describe

app = Flask(__name__)

app.add_url_rule('/', view_func=index)
app.add_url_rule('/describe', view_func=describe, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
