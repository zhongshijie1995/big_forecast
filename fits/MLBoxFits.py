from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

from settings import Data_Val


def go(paths=[Data_Val.feature_matrix], target_name=Data_Val.tg, submit_csv=Data_Val.sc, result_csv=Data_Val.rc):
    
    rd = Reader(sep=',')
    df = rd.train_test_split(paths, target_name)

    dft = Drift_thresholder()
    df = dft.fit_transform(df)

    opt = Optimiser(scoring=Data_Val.scoring, n_folds=Data_Val.n_folds)

    space = Data_Val.space
    params = opt.optimise(space, df, 15)

    prd = Predictor()
    prd.fit_predict(params, df)

    submit = pd.read_csv(submit_csv, sep=',')
    preds = pd.read_csv("save/" + target_name + "_predictions.csv")

    submit[target_name] = preds[target_name + "_predicted"].values
    submit.to_csv(result_csv, index=False)


if __name__ == '__main__':
    # csv_list = [
    #     'D:\\99_Data\\02_home-credit-default-risk\\application_train.csv',
    #     'D:\\99_Data\\02_home-credit-default-risk\\application_test.csv'
    # ]
    # target = 'TARGET'
    # sub = 'D:\\99_Data\\02_home-credit-default-risk\\sample_submission.csv'
    # result = 'D:\\99_Data\\02_home-credit-default-risk\\result_just_app.csv'
    # go(csv_list, target, sub, result)
    pass
