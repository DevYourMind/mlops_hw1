from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import joblib
import uuid


def train_dump_model(model_name, models_dir, X, y, params=None):
    """
    Train and dump LightGBM or CatBoostRegressor model.
    input: model_name: str
           models_dir: Path object
           X, y: np.array
           params: dict
    """
    models = {
        'CatBoost': CatBoostRegressor,
        'LightGBM': LGBMRegressor
    }
    hyper_str = ''
    if params is None:
        params = {}
    else:
        hyper_str = '_hyperparams'
    model = models[model_name](**params)
    model.fit(X, y)
    model_id = str(uuid.uuid4()).split('-')[0]
    joblib.dump(model, models_dir /
                f'{model_name.lower()+hyper_str+"_"+model_id}.pkl')


def get_trained_models(models_dir):
    """
    Look for trained models in folder.
        input: models_dir: Path object
        return: list of models in folder with models
    """
    return [model.name.split('.')[0] for model in models_dir.glob('*.pkl')]


def make_prediction(model_name, X, models_dir):
    """
    Make a prediction by model.
        input: model_name: str
               X: np.array
               models_dir: Path object
        return: float prediction
    """
    model = joblib.load(models_dir / f'{model_name}.pkl')
    print('####################')
    print(model)
    return model.predict(X)
