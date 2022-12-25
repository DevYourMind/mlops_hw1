from flask import Flask
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage
import models
import json
import glob
from pathlib import Path
import pandas as pd
import os

app = Flask(__name__)
api = Api(app)


models_dir = Path('trained_models')
models_dir.mkdir(parents=True, exist_ok=True)


upload_parser = api.parser()
upload_parser.add_argument(
    'data',
    location='files',
    type=FileStorage,
    required=True
)
upload_parser.add_argument(
    'params',
    location='files',
    type=FileStorage,
    required=False
)
upload_parser.add_argument(
    'model',
    choices=['All', 'CatBoost', 'LightGBM'],
    location='args',
    required=True
)


@api.route(
    '/load_train',
    methods=['PUT', 'POST'],
    doc={'description': 'Загрузка параметров и обучение модели'})
@api.expect(upload_parser)
class FitModel(Resource):
    """
    Data generating and models' fitting.
    """

    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)

    @api.doc(params={
        'data': 'Загрузите данные для обучения',
        'model': 'Выберите модель',
        'params': 'Выберите .json файл с гиперпараметрами модели (доступно, если не выбрана опция "All")',
    })
    @api.response(200, 'Success')
    @api.response(400, 'Validation Error')
    @api.response(500, 'No such file or directory')
    def put(self):
        args = upload_parser.parse_args()
        model_name = args['model']
        train_data_target = pd.read_csv(args['data'], index_col=[0])
        train_data = train_data_target.drop('target', axis=1)
        train_target = train_data_target['target']
        if model_name == 'All':
            models.train_dump_model(
                model_name='CatBoost', models_dir=models_dir, X=train_data, y=train_target)
            models.train_dump_model(
                model_name='LightGBM', models_dir=models_dir, X=train_data, y=train_target)
            return 'All models have been fitted'
        else:
            params = args['params']
            if params is not None:
                params = json.load(params)
            try:
                models.train_dump_model(
                    model_name=model_name, models_dir=models_dir, X=train_data, y=train_target, params=params)
            except TypeError:
                return 'Incorrect parametres'
            return f"Model {model_name} has been fitted"


model_list_parser = api.parser()
model_list_parser.add_argument(
    'model_type',
    required=True,
    location='args',
    choices=['All', 'CatBoost', 'LGBM']
)


@api.route('/show_models', methods=['GET'],
           doc={'description': 'Список всех обученных моделей'})
@api.expect(model_list_parser)
class ShowModels(Resource):
    """
    Show list of models.
    """
    @api.doc(
        params={
            'model_type': 'Выберите семейство обученных моделей'
        }
    )
    def get(self):
        args = model_list_parser.parse_args()
        model_type = args['model_type']
        all_models = models.get_trained_models(models_dir)
        models_to_print = list()
        if model_type == 'All':
            models_to_print = [model for model in all_models]
        if model_type == 'CatBoost':
            models_to_print = [
                model for model in all_models if 'catboost' in model]
        if model_type == 'LGBM':
            models_to_print = [
                model for model in all_models if 'lightgbm' in model]
        if models_to_print == []:
            return 'No models'
        models_to_print = ', '.join(models_to_print)
        return models_to_print


model_delete_parser = api.parser()
model_delete_parser.add_argument(
    'model_type',
    required=True,
    location='args',
    choices=models.get_trained_models(Path('trained_models')) + ["All"]
)


@api.route('/delete_models', methods=['GET'],
           doc={'description': 'Удаление моделей'})
@api.expect(model_delete_parser)
class DeleteModels(Resource):
    """
    Delete chosen models.
    """
    @api.doc(
        params={
            'model_type': 'Выберите модель для удаления'
        }
    )
    @api.response(500, 'Model has been removed')
    def get(self):
        args = model_delete_parser.parse_args()
        model_to_delete = args['model_type']
        if model_to_delete == 'All':
            files = glob.glob('trained_models/*')
            for f in files:
                os.remove(f)
            return 'All models have been deleted'
        file_to_rem = Path(f"trained_models/{model_to_delete}.pkl")
        try:
            file_to_rem.unlink()
        except FileNotFoundError:
            return "Model doesn't exist"
        return f'Model {file_to_rem.name} has been removed'


predict_parser = api.parser()
predict_parser.add_argument(
    'data',
    location='files',
    type=FileStorage,
    required=True
)
predict_parser.add_argument(
    'model_type',
    required=True,
    location='args',
    choices=models.get_trained_models(models_dir)
)


@api.route('/predict', methods=['PUT', 'POST', 'GET'],
           doc={'description': 'Сделать предсказание для .csv файла'})
@api.expect(predict_parser)
class Predict(Resource):
    """
    Make a prediction for data.
    """

    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)

    @api.doc(
        params={
            'data': 'Введите данные',
            'model_type': 'Выберите модель для прогноза'
        }
    )
    def put(self):
        args = predict_parser.parse_args()
        model_to_predict = args['model_type']
        test_data_target = pd.read_csv(args['data'], index_col=[0])
        test_data = test_data_target.drop('target', axis=1)
        prediction = models.make_prediction(
            model_to_predict, test_data, models_dir)
        return 'Prediction: ' + ', '.join([str(x) for x in prediction])


if __name__ == '__main__':
    app.run(debug=True)
