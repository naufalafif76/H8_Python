import flask
import pandas as pd
import pickle

model = pickle.load(open('model/model_ensemble.pkl', 'rb'))
scaler = pickle.load(open('model/model_scaler.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
  return(flask.render_template('main.html'))

@app.route('/predict', methods=['POST'])
def predict():
  X_inp = [[int(x) for x in flask.request.form.values()]]
  X_sclr = scaler.transform(X_inp)
  feat_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_sodium', 'sex', 'smoking', 'time']
  X_inf = pd.DataFrame(X_sclr, columns=feat_cols)

  predict = model.predict(X_inf)
  output = {0: "survive", 1: 'died'}

  return flask.render_template('main.html', prediction_text="The patients likely {} during follow-up period".format(output[predict[0]]))

if __name__ == '__main__':
  app.run(debug=True)
