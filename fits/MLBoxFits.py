from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

from settings import Data_Val


def go():
    paths, target_name, submit_csv, result_csv = ([Data_Val.feature_matrix, ], Data_Val.tg, Data_Val.sc, Data_Val.rc)

    rd = Reader(sep=',')
    df = rd.train_test_split(paths, target_name)

    dft = Drift_thresholder()
    df = dft.fit_transform(df)

    # opt = Optimiser(scoring=Data_Val.scoring, n_folds=Data_Val.n_folds)
    opt = Optimiser(scoring=Data_Val.scoring)

    space = Data_Val.space
    params = opt.optimise(space, df, 15)

    prd = Predictor()
    prd.fit_predict(params, df)

    submit = pd.read_csv(submit_csv, sep=',')
    preds = pd.read_csv("save/" + target_name + "_predictions.csv")

    submit[target_name] = preds[target_name + "_predicted"].values
    submit.to_csv(result_csv, index=False)


if __name__ == '__main__':
    pass
