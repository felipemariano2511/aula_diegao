from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({})

@app.route("/salas")
def salas():
    salas = ["c41", "c42", "c43"]
    return jsonify(salas)