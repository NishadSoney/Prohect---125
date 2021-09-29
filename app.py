from types import MethodDescriptorType
from flask import Flask,jsonify,request
from classifier import get_pre
app = Flask(__name__)

@app.route("/predict-digit",methods = ["post"])

def pre_data():
    image = request.files.get("digit")
    pre = get_pre(image)
    return jsonify({"prediction":pre}),200
    
if __name__ == "__main__":
    app.run(debug=True)