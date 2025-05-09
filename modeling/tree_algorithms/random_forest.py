from datetime import datetime
import os

import joblib
from sklearn.pipeline import Pipeline
from evaluation.performance import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, KFold)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from db.queries import ForecastConfig
from dml.dml import convert_column, create_lag_features, prepare_lag_features, create_month_and_year_columns
from modeling.utilities import load_hyperparameter_grid_rf, regressor_grid_for_pipeline
from db.utilities import config

# Setup logger
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib._store_backends")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")

cache_dir = os.path.join(os.getcwd(), 'pipeline_cache')
memory = joblib.Memory(location=cache_dir, verbose=0)

def most_frequent_params(param_list):
    """Return the most frequent parameter configuration."""
    from collections import Counter
    counter = Counter(tuple(sorted(p.items())) for p in param_list)
    most_common = counter.most_common(1)[0][0]
    return dict(most_common)


def nested_cv_rf(X, y, param_grid=None, scoring='neg_mean_absolute_error', perform_validation=False,
                 stored_hyperparams=None, interpret: bool = False):
    """
    Perform nested cross-validation using a pipeline that includes:
      - Imputation
      - Scaling
      - Feature selection (RFECV)
      - Random Forest Regression

    If perform_validation is True, the function conducts nested CV (outer and inner loops)
    to determine the best hyperparameters and aggregates the outer fold metrics.
    If False, it will skip the CV process and use stored hyperparameters (if provided) or
    default hyperparameters from the configuration, then perform a quick evaluation.

    Returns:
      - final_pipeline: A pipeline refitted on the entire dataset with the chosen hyperparameters.
      - scores: A dictionary with evaluation metrics (MAE, RMSE, R²).
      - selected_features: Features selected by the RFECV step.
    """
    if perform_validation:
        # Load the hyperparameter grid if not provided.
        if param_grid is None:
            param_grid = load_hyperparameter_grid_rf(config)
            param_grid = regressor_grid_for_pipeline(param_grid)

        # Create stratified bins for regression using quantile binning (converted to integer codes)
        strat_bins = pd.qcut(y, q=10, duplicates='drop').cat.codes
        outer_cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

        outer_scores_list = []
        best_params_list = []

        # Define the pipeline with caching enabled.
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('feature_selection', RFECV(
                estimator=RandomForestRegressor(random_state=42),
                step=0.1, cv=3, scoring=scoring
            )),
            ('regressor', RandomForestRegressor(random_state=42))
        ], memory=memory)

        # Outer CV loop for hyperparameter tuning
        for train_idx, test_idx in outer_cv.split(X, strat_bins):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Inner CV: using KFold (non-stratified) to reduce computational overhead
            inner_cv = KFold(n_splits=3, random_state=42, shuffle=True)

            grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=-1, scoring=scoring)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            fold_scores = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred)
            }
            outer_scores_list.append(fold_scores)
            best_params_list.append(grid_search.best_params_)
            logging.info(f"Fold scores: {fold_scores}")

        # Determine the most frequent hyperparameters across folds
        best_overall_params = most_frequent_params(best_params_list)
        logging.info(f"Most frequent hyperparameters: {best_overall_params}")

    else:
        # Skip nested CV and use stored or default hyperparameters
        if stored_hyperparams is not None:
            best_overall_params = stored_hyperparams
            logging.info("Using stored hyperparameters.")
        else:
            # Load the grid and choose default values (e.g., first option in each list)
            default_grid = load_hyperparameter_grid_rf(config)
            default_grid = regressor_grid_for_pipeline(default_grid)
            best_overall_params = {k: v[0] for k, v in default_grid.items() if k.startswith('regressor__')}
            logging.info(f"Using default hyperparameters: {best_overall_params}.")
        # When skipping validation, we don't have outer CV fold scores.
        outer_scores_list = None

    # Build the final pipeline using the chosen hyperparameters.
    # Remove duplicate random_state if present in best_overall_params.
    rf_params = {k.split('__')[1]: v for k, v in best_overall_params.items()
                 if k.startswith('regressor__') and k.split('__')[1] != 'random_state'}

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=RandomForestRegressor(random_state=42),
            step=0.1, cv=3, scoring=scoring
        )),
        ('regressor', RandomForestRegressor(**rf_params))
    ], memory=memory)

    final_pipeline.fit(X, y)

    # Extract the features selected by RFECV.
    selector = final_pipeline.named_steps['feature_selection']
    selected_features = X.columns[selector.get_support()]

    # Compute evaluation scores on the full dataset for the final pipeline.
    y_pred = final_pipeline.predict(X)
    eval_scores = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred)
    }

    # If nested CV was performed, aggregate outer fold scores; otherwise, use the evaluation scores.
    if perform_validation and outer_scores_list:
        # Average the fold metrics.
        agg_scores = {metric: np.mean([fold[metric] for fold in outer_scores_list])
                      for metric in outer_scores_list[0].keys()}
        scores = agg_scores
        logging.info(f"Aggregated outer CV scores: {scores}")
    else:
        scores = eval_scores
        logging.info(f"Evaluation scores on training data: {scores}")

    if interpret:
        # Generate diagnostics (e.g., feature importance, SHAP analysis)
        generate_diagnostics(final_pipeline, X, y, selected_features)

    return final_pipeline, scores, y_pred, selected_features


def train_forecast_for_consumption_type(podel_df: pd.DataFrame, feature_columns: List[str], consumption_type: str, param_grid: dict=None, perform_validation=False):
    """
    Trains a forecast model for a specific consumption type.

    Parameters:
      - podel_df: DataFrame with the data.
      - feature_columns: List of feature column names.
      - consumption_type: The target consumption type.
      - param_grid: (Optional) Hyperparameter grid dictionary.

    Returns:
      - final_model: The trained pipeline.
      - scores: Performance metrics dictionary.
      - selected_features: Features selected by RFECV.
    """
    X = podel_df[feature_columns]
    y = podel_df[consumption_type]


    scoring = 'neg_mean_absolute_error'
    # final_model, scores, selected_features = nested_cv_rf(X, y, param_grid=param_grid, scoring=scoring, perform_validation=perform_validation)
    final_model, scores, y_pred, selected_features = nested_cv_rf(
        X, y, param_grid=param_grid, scoring=scoring, perform_validation=perform_validation
    )

    return final_model, y_pred, scores, selected_features


def train_forecast_per_pod(podel_df: pd.DataFrame, customer_id: str, pod_id: str, feature_columns: List[str], consumption_types: List[str],
                           ufm_config: ForecastConfig):
    """
    Trains models for each consumption type for a given pod and returns a performance summary.

    The trained models are saved to disk using joblib along with metadata.
    """
    logging.info("Training Random Forest for each pod_id.")
    metric_keys = {'RMSE', 'MAE', 'R2'}
    data = []
    for consumption_type in consumption_types:
        try:
            logging.info(f"Training for Customer: {customer_id}, POD: {pod_id}, Consumption Type: {consumption_type}")
            if podel_df[consumption_type].isnull().all():
                continue

            # Define model path using ufm_config settings and create directories if needed
            model_path = os.path.join("models", ufm_config.forecast_method_name,
                                      f"customer_{customer_id}", f"pod_{pod_id}")
            os.makedirs(model_path, exist_ok=True)
            model_file = os.path.join(model_path, f"{consumption_type}.pkl")

            final_model, y_pred, scores, selected_features = train_forecast_for_consumption_type(podel_df, feature_columns,
                                                                                         consumption_type)
            forecasts_by_type = {
                'pod_id': pod_id,
                'customer_id': customer_id,
                'consumption_type': consumption_type,
                'forecast': y_pred,
            }
            for key in metric_keys:
                forecasts_by_type[key] = scores.get(key, None)

            data.append(forecasts_by_type)

            # Save the model with metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": final_model.named_steps['regressor'].get_params()
            }
            joblib.dump({"model": final_model, "metadata": metadata}, model_file)
            logging.info(f"Saved model to {model_file} with metadata: {metadata}")

        except Exception as e:
            logging.exception(f"Error during training for POD {pod_id}, Consumption Type {consumption_type}: {e}")

    performance_data_frame = pd.DataFrame(data)
    pod_id_performance_data = PodIDPerformanceData(pod_id, ufm_config.forecast_method_name, customer_id,
                                                   ufm_config.user_forecast_method_id, performance_data_frame)
    return pod_id_performance_data

def train_random_forest_for_single_customer( df: pd.DataFrame,
        customer_id: str,
        ufm_config: ForecastConfig,
        column: str = 'CustomerID',
        lag_features: List[str] = None,
        selected_columns: List[str] = None,
        consumption_types: List[str] = None):
    if selected_columns is None:
        selected_columns = ["StandardConsumption","OffpeakConsumption" ,"PeakConsumption"]

    if consumption_types is None:
        consumption_types = [
            "PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
            "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"
        ]
    if lag_features is None:
        lag_features = ['StandardConsumption']

    df = create_month_and_year_columns(df)
    customer_data = df[df[column] == customer_id].sort_values('PodID')
    customer_data = convert_column(customer_data)

    # Initialize a CustomerPerformanceData object to store performance per pod.
    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)

    # Get unique PodIDs for this customer.
    unique_pod_ids: List[str] = list(customer_data["PodID"].unique())

    # Process each pod: forecast consumption for each consumption type.
    for pod_id in unique_pod_ids:
        # Sort pod data by reporting month.
        podel_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        podel_df = create_lag_features(podel_df, lag_features, lags=3)
        podel_df, feature_columns = prepare_lag_features(podel_df, lag_features)
        performance_data = train_forecast_per_pod(podel_df, customer_id, pod_id, feature_columns, consumption_types,ufm_config)
        consumer_performance_data.pod_by_id_performance.append(performance_data)


    all_performance_df: pd.DataFrame = consumer_performance_data.get_pod_performance_data()

    pod_id_performance_data = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)
    return pod_id_performance_data
