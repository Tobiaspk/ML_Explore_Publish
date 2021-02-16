# Eplore ML in Sklearn Wrappers

## Usage

Easily explore parameters and models of sklearn, XGBoost, etc. using a group of helper functions. Run a simulation using the following steps

1. Load Data: i.e. `pd.read_csv(...)`
2. Define Regressors and Parameter Grids: `models = [Regressor("name", RegressorObject, [{'param1':value1},{'param2':value2}])` (see `src/DefineModels.py`)
3. Define DataSettings: `ds = DataSetting(y=data.y, x=data[regressors], models=models.copy(), loss_function=rmse, k=5)
4. Run `evaluate_all()`
5. Plot validation or losses: i.e. `ds.plot_model_validation_curves(path=output_path)

