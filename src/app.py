import utils.preprocessing_data as ppd
import tensorflow as tf 

from flask import Flask, request, jsonify

model = tf.keras.models.load_model('src/model/modelo_ge.h5')

app = Flask(__name__)

@app.route("/")
def get_server():
    return "<h3>The server is up</h3>"

@app.route('/ge-prediction', methods=['POST'])
def ge_prediction():
    try: 
        if not request.is_json:
            return jsonify({"error": "Missing json in request"}),400
        body = request.get_json()
        input_model = ppd.create_input(body)
        predict = model.predict(input_model)
        response = {"returnPredition": "{}%".format(round(predict[0][0]*100,2)),
                    "inputVariables":{
                        "returnOneDayBack":round(input_model[0][0],4),
                        "returnOneWeekBack":round(input_model[0][1],4),
                        "returnOneMonthBack":round(input_model[0][2],4),
                        "movingAvaregeOneWeekBack":round(input_model[0][3],4),
                        "movingAvaregeOneMonthBack":round(input_model[0][4],)
                                     }
                    }
        return jsonify(response)
   
    except: 
        return jsonify({'error': 'Bad request'}),400

    
#if __name__ == "__main__":
#    app.run()
