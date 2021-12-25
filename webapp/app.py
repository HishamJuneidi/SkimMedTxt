import numpy as np
from flask import Flask, request, render_template
import pickle
import sys
import Skimmedtext




app = Flask(__name__)


# model = pickle.load(open("../ufo-model_1.pkl", "rb"))
# filename = "../dog.pkl"
# infile = open(filename,'rb')
# new_dict = pickle.load(infile)
# infile.close()
# print(new_dict)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    label = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    results = {}
    for x in label:
        results[x] = ""
    str_features = [str(x) for x in request.form.values()]
    splitted_text = str(str_features).split(". ")
    splitted_text = [i + "." for i in splitted_text]
    for i in splitted_text:

  
        y_pred_Temp_text_proba = Skimmedtext.model_1.predict([i])
        # Temp_text=Skimmedtext.cv.transform([i])
    
        # # tf-idf scores 
        # tf_idf_vector_Temp_text=Skimmedtext.tfidf_transformer.transform(Temp_text)

        # y_pred_Temp_text = Skimmedtext.nb.predict(tf_idf_vector_Temp_text)
        # y_pred_Temp_text_proba = Skimmedtext.nb.predict_proba(tf_idf_vector_Temp_text)
        # y_pred_Temp_tex, label_encoder.classes_, y_pred_Temp_text_proba
        index1 = np.argmax(y_pred_Temp_text_proba, axis = 1)[0]
        temp = max(y_pred_Temp_text_proba[0])
        temp = "{:.2f}".format(temp)
        results[label[index1]] =  results[label[index1]]+ " " +  i

   
  
    return render_template(
         "index.html", BACKGROUND_text=results['BACKGROUND'],
         METHODS_text=results['METHODS'],
          OBJECTIVE_text=results['OBJECTIVE'], 
          RESULTS_text=results['RESULTS'],
          CONCLUSIONS_text=results['CONCLUSIONS']
    )


if __name__ == "__main__":
    app.run(debug=True)
