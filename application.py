from flask import Flask,request,render_template,jsonify
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logger

application = Flask(__name__)

@application.get('/')
def index():
    return render_template('home.html') 


@application.post('/predict_data')
def predict_data_point():
    if request.method == 'POST':
        data=CustomData(
            gender=request.json["gender"],
            race_ethnicity=request.json['ethnicity'],
            parental_level_of_education=request.json['parental_level_of_education'],
            lunch=request.json['lunch'],
            test_preparation_course=request.json['test_preparation_course'],
            reading_score=float(request.json['writing_score']),
            writing_score=float(request.json['reading_score'])

        )
        pred_df=data.get_data_as_data_frame()
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        logger.info('Prediction Completed.')
        print(results)
        return jsonify(results[0])
    

if __name__=="__main__":
    application.run(host="0.0.0.0",debug=True,port=5100)   