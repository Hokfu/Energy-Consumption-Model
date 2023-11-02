import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = "model_1.bin"

with open(model_file,"rb") as f_in:
    dv,model = pickle.load(f_in)

app = Flask("energy_consumption")
@app.route("/predict",methods=['POST'])

def predict():
    energy_consumption = request.get_json()
    X = dv.transform([energy_consumption])
    y_pred = model.predict(X)
    result = {
        "load_type" : str(y_pred)
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=9696)