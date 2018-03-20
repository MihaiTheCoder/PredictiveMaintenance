from flask import Flask, render_template
from flask import request
import csv
import json
from sklearn.externals import joblib
import pandas
import os
import io
from flask import make_response
from process_input import create_feature_records

app = Flask(__name__)
top_features = ""
train_df = None

def csv_to_json(file_path):
    data = []
    with open(file_path) as f:
        for row in csv.DictReader(f):
            data.append(row)

    json_data = json.dumps(data)
    return json_data


@app.route('/summary', methods=['GET'])
def summary():
    algorithm_type = request.args.get('algorithm_type')
    return csv_to_json("{}/{}".format(algorithm_type, "comparison.csv"))


def is_pikle_file(path):
    return path.endswith('pkl') and os.path.isfile(path)

@app.route('/')
def index():
    return render_template('PredictionHtml.html')

def normalize_relative_to_dataset(dataset, refence_dataset):
    normalized_dataset = (dataset - refence_dataset.min()) / (refence_dataset.max()-refence_dataset.min())
    return normalized_dataset


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    original_df = create_feature_records.create_dataset_from_input(file)
    df_with_features = original_df.copy()
    df_with_features = create_feature_records.add_moving_average_moving_std(df_with_features, 0)
    df_with_features = df_with_features.drop(labels=['id'], axis=1)
    df_with_features = normalize_relative_to_dataset(df_with_features, train_df)
    df_with_features = df_with_features[top_features]

    buffer = io.StringIO()

    ml_dirs = [ml_dir for ml_dir in os.listdir("step_2") if os.path.isdir("step_2/" + ml_dir)]
    for algorithm_type in ml_dirs:
        algorithm_dir = "step_2/{}/".format(algorithm_type)
        model_files = [model_file for model_file in os.listdir("step_2/{}".format(algorithm_type)) if
                       is_pikle_file(os.path.join(algorithm_dir, model_file))]
        for algorithm_pkl_file_name in model_files:
            algorithm_name = algorithm_type + "_" + os.path.splitext(algorithm_pkl_file_name)[0]
            algorithm_pkl_path = os.path.join(algorithm_dir, algorithm_pkl_file_name)
            model = joblib.load(algorithm_pkl_path)
            predictions = model.predict(df_with_features)
            original_df[algorithm_name] = pandas.Series(predictions)

    original_df.to_csv(buffer)
    buffer.seek(0)
    contents = buffer.getvalue()
    buffer.close()
    output = make_response(contents)
    output.headers["Content-Disposition"] = "attachment; filename=export.csv"
    output.headers["Content-type"] = "text/csv"
    return output


if __name__ == '__main__':
    with open("step_2/important_features.txt", "r") as text_file:
        top_features = text_file.read().split('\n')

    top_features = list(filter(None, top_features))

    train_df = pandas.DataFrame.from_csv("process_input/train_dataset_with_extra_features.csv")

    train_df = train_df[top_features]

    app.run(host='0.0.0.0', port=5000)







