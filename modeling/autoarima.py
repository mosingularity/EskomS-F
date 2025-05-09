from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from dml.dml import *
import numpy as np
import joblib
import pandas as pd
from typing import List, Tuple, Union, Optional, NamedTuple
from db.queries import ForecastConfig
from evaluation.performance import *

# Setup logger
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Optionally, define a named tuple for forecast results
class ForecastResult(NamedTuple):
    forecast: pd.Series
    metrics: dict
    baseline_metrics: dict

def _apply_log(series: pd.Series, log: bool) -> pd.Series:
    """
    Optionally apply a log transformation to the series.
    Adds a small constant for numerical stability.
    """
    if log:
        return np.log(series + 1e-6)
    return series

def train_time_series_model(
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        log: bool = False
) -> Union[ARIMAResults, SARIMAXResults]:
    """
    Train a time series model (ARIMA or SARIMA) on the provided series.

    Parameters:
        series (pd.Series): The time series data.
        order (Tuple[int, int, int]): ARIMA order parameters (p, d, q).
        seasonal_order (Optional[Tuple[int, int, int, int]]): Seasonal order (P, D, Q, s) for SARIMA.
            If None, a non-seasonal ARIMA is trained.
        log (bool, optional): Whether to apply a log transform. Defaults to False.

    Returns:
        Union[ARIMAResults, SARIMAXResults]: The fitted model.
    """
    series = _apply_log(series, log)

    if seasonal_order is None:
        logging.info("Training non-seasonal ARIMA model")
        model = ARIMA(series, order=order)
        fitted_model = model.fit(method_kwargs={"maxiter": 200})
    else:
        logging.info("Training seasonal ARIMA (SARIMA) model with seasonal_order=%s", seasonal_order)
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(method_kwargs={"maxiter": 200})

    return fitted_model


def predict_time_series_model(
        model: Union[ARIMAResults, SARIMAXResults],
        order: Tuple[int, int, int],
        n_periods: int,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        log: bool = False
) -> pd.Series:
    """
    Generate forecasts using the fitted time series model.

    Parameters:
        model (Union[ARIMAResults, SARIMAXResults]): The fitted model.
        order (Tuple[int, int, int]): The ARIMA order parameters.
        n_periods (int): Number of periods to forecast.
        seasonal_order (Optional[Tuple[int, int, int, int]]): Seasonal order parameters if applicable.
        log (bool, optional): Whether to reverse a log transform (exponentiate results). Defaults to False.

    Returns:
        pd.Series: Forecasted values.
    """
    # Note: The seasonal_order parameter in predict may not be required for SARIMAX;
    # adjust if the API differs. Here, we assume a unified interface.
    if seasonal_order is None:
        results = model.predict(n_periods=n_periods, params=order)
    else:
        results = model.predict(n_periods=n_periods, params=order, seasonal_order=seasonal_order)

    if log:
        results = np.exp(results)
    return results


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def naive_forecast(series: pd.Series, n_periods: int) -> pd.Series:
    """Naive forecast using last observed value."""
    return np.full(n_periods, series.iloc[-1])

def mean_forecast(series: pd.Series, n_periods: int) -> pd.Series:
    """Mean forecast using average of training series."""
    return np.full(n_periods, series.mean())

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Evaluate forecast using RMSE and R2 metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def generate_forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    metrics = evaluate_forecast(y_true, y_pred)
    baseline = mean_forecast(pd.Series(y_true), len(y_true))
    baseline_metrics = evaluate_forecast(y_true, baseline)
    return metrics, baseline_metrics


def compare_with_baseline(actual, forecast, baseline) -> pd.DataFrame:
    """Compares model forecast vs. naive"""
    return pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'SARIMA': list(evaluate_forecast(actual, forecast).values()),
        'Baseline': list(evaluate_forecast(actual, baseline).values())
    })

def load_model(path):
    return joblib.load(path)


def evaluate_model_on_test(model_path: str, order: Tuple[int, int, int],  n_periods: int, seasonal_order: Tuple[int, int, int, int] = None, log: bool = False) -> pd.Series:
    model = load_model(model_path)
    forecast = predict_time_series_model(model, order, n_periods, seasonal_order, log=log)
    return forecast


def forecast_for_consumption_type(
        df: pd.DataFrame,
        order: Tuple[int, int, int],
        consumption_type: str,
        n_periods: int,
        model_path: str,
        mode: str = 'train',
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        log: bool = False
) -> object:
    """
    Forecast consumption for a specific consumption type and compute evaluation metrics.

    The function:
      - Converts the specified consumption column to numeric,
      - Ensures the index is datetime with monthly frequency,
      - Splits the series into training and test sets,
      - Trains an ARIMA/SARIMA model (or loads one in test mode),
      - Generates a forecast,
      - Evaluates the forecast against the training data.

    Parameters:
        df (pd.DataFrame): DataFrame with consumption data.
        order (Tuple[int, int, int]): ARIMA order parameters.
        consumption_type (str): The column name for the consumption type.
        n_periods (int): Number of periods to forecast.
        model_path (str): Path to save or load the model.
        mode (str, optional): 'train' to fit a new model, 'test' to evaluate an existing model.
            Defaults to 'train'.
        seasonal_order (Optional[Tuple[int, int, int, int]], optional): Seasonal order for SARIMA.
            If None, a non-seasonal ARIMA is used.
        log (bool, optional): Whether to apply a log transform. Defaults to False.

    Returns:
        ForecastResult: Named tuple with forecast, metrics, and baseline_metrics.
    """
    # Convert consumption to numeric, ensure datetime index with monthly frequency.
    series: pd.Series = pd.to_numeric(df[consumption_type], errors='coerce')
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('MS').fillna(0)

    # Split the series into train and test sets.
    train, test = time_series_train_test_split(series, n_periods)

    if mode == 'train':
        model = train_time_series_model(train, order, seasonal_order, log=log)
        forecast = predict_time_series_model(model, order, n_periods, seasonal_order, log=log)
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            logging.error("Failed to save model to %s: %s", model_path, e)
            raise
    elif mode == 'test':
        # For test mode, assume evaluate_model_on_test returns a tuple (forecast, metrics)
        forecast, metrics = evaluate_model_on_test(model_path, order, n_periods, seasonal_order, log=log)
    else:
        raise ValueError("Mode must be 'train' or 'test'")

    # Generate evaluation metrics and baseline forecasts.
    metrics, baseline_metrics = generate_forecast_metrics(train, forecast)
    return forecast, metrics, baseline_metrics

def forecast_arima_for_single_customer(
        df: pd.DataFrame,
        customer_id: str,
        ufm_config: ForecastConfig,
        order: Tuple[int, int, int] = (1, 1, 1),
        n_periods: int = 12,
        column: str = 'CustomerID',
        mode: str = 'train',
        selected_columns: List[str] = None,
        consumption_types: List[str] = None,
        seasonal_order: Tuple[int, int, int, int] = None
) -> pd.DataFrame:
    """
    Forecast ARIMA models for a single customer across all their pods and consumption types,
    then aggregate and convert performance data from wide to long format.

    The function filters the input DataFrame to retain only rows corresponding to the
    specified customer, then for each unique PodID in that subset it performs ARIMA forecasting
    on multiple consumption types. The performance metrics for each forecast are stored in a
    CustomerPerformanceData object. Finally, all pod-level performance data is aggregated into a
    single DataFrame and converted from a wide format (with multiple columns per consumption type)
    into a long format (with pod_id, customer_id, and consumption_type as indices, and metrics as columns).

    Parameters:
        df (pd.DataFrame): Input DataFrame containing customer data. Expected columns include:
            - 'CustomerID': Unique identifier for the customer.
            - 'PodID': Unique identifier for each pod.
            - 'ReportingMonth': Time period indicator for ordering.
            - Consumption metrics columns (e.g. 'PeakConsumption', 'StandardConsumption', etc.).
        customer_id (str): The unique identifier for the customer.
        ufm_config (ForecastConfig): Configuration object with forecasting parameters.
        order (Tuple[int, int, int], optional): ARIMA order (p, d, q). Defaults to (1, 1, 1).
        n_periods (int, optional): Number of periods to forecast. Defaults to 12.
        column (str, optional): Column name used to filter customer data. Defaults to 'CustomerID'.
        mode (str, optional): Forecasting mode, 'train' or 'test'. Defaults to 'train'.
        selected_columns (List[str], optional): List of consumption columns to include in performance data.
            Defaults to ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"].
        consumption_types (List[str], optional): List of consumption types to forecast.
            Defaults to ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                         "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"].

    Returns:
        pd.DataFrame: A long-format DataFrame containing combined performance metrics for each
                      unique combination of (pod_id, customer_id, consumption_type).
    """
    # Set default columns if not provided.
    if selected_columns is None:
        selected_columns = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]

    if consumption_types is None:
        consumption_types = [
            "PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
            "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"
        ]

    # Filter the DataFrame for the specific customer and sort by PodID.
    customer_data = df[df[column] == customer_id].sort_values('PodID')

    # Initialize a CustomerPerformanceData object to store performance per pod.
    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)

    # Get unique PodIDs for this customer.
    unique_pod_ids: List[str] = list(customer_data["PodID"].unique())

    # Process each pod: forecast consumption for each consumption type.
    for pod_id in unique_pod_ids:
        # Sort pod data by reporting month.
        podel_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        # Get the performance data for this pod.
        performance_data = forecast_for_podel_id(
            podel_df, order, customer_id, pod_id, consumption_types, n_periods, ufm_config, mode
        )
        consumer_performance_data.pod_by_id_performance.append(performance_data)

    # Combine all pod-level performance data into one DataFrame.
    all_performance_df: pd.DataFrame = consumer_performance_data.get_pod_performance_data()

    # Convert the combined wide-format DataFrame into long format.
    # Note: We now call convert_pod_id_performance_data on the performance container,
    # not on the DataFrame itself.
    pod_id_performance_data = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)

    return pod_id_performance_data


def arima(df: pd.DataFrame, ufm_config: ForecastConfig, order: Tuple[int ,int, int] = (1,1,1), n_periods: int = 12, column: str = 'CustomerID', mode: str = 'train', selected_columns: List[str] = None, consumption_types: List[str] = None):
    if selected_columns is None:
        selected_columns = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]

    if consumption_types is None:
        consumption_types = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                             "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"]
    feature_col: pd.Series = df[column].unique() # Chosen colum
    for customer_id in feature_col:
        customer_data = df[df['CustomerID'] == customer_id].sort_values('PodID')
        consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)
        unique_pod_ids: list = list(customer_data["PodID"].unique())
        for pod_id in unique_pod_ids:
            podel_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
            performance_data = forecast_for_podel_id(podel_df, order, customer_id, pod_id, consumption_types, n_periods, ufm_config, mode)
            consumer_performance_data.pod_by_id_performance.append(performance_data)


def forecast_for_podel_id(df: pd.DataFrame, order: Tuple[int, int, int], customer_id: str, pod_id: str, consumption_types: List[str], n_periods: int, ufm_config: ForecastConfig, mode: str = 'train') -> PodIDPerformanceData:
    """
     Forecasts consumption and computes performance metrics per consumption type.
     Returns a dic
    """
    logging.info("Predicting ARIMA for each pod_id.")
    metric_keys = {'RMSE', 'MAE', 'R2'}
    data = []
    for consumption_type in consumption_types:
        try:
            if df[consumption_type].isnull().all():
                continue
            model_path = f"models/{ufm_config.forecast_method_name}/customer_{customer_id}/pod_{pod_id}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path += f"/{consumption_type}.pkl"
            forecasts_by_type: dict = {'pod_id': pod_id, 'customer_id': customer_id}
            forecast, metrics, baseline_metrics = forecast_for_consumption_type(df, order, consumption_type, n_periods, model_path, mode)
            forecasts_by_type['consumption_type'] = consumption_type
            forecasts_by_type['forecast'] = forecast
            for key in metric_keys:
                forecasts_by_type[key] = metrics[key]
                forecasts_by_type[f'{key}_baseline'] = baseline_metrics[key]

            data.append(forecasts_by_type)

        except Exception as e:
            print(f"Error processing {consumption_type} for PodID {pod_id}: {e}")
            exit()
    performance_data_frame = pd.DataFrame(data)
    pod_id_performance_data = PodIDPerformanceData(pod_id, ufm_config.forecast_method_name, customer_id, ufm_config.user_forecast_method_id, performance_data_frame)
    return pod_id_performance_data


"""
Goal: Train a single model (or one per cluster) on all customer data. 
The model uses multiple consumption types as outputs (or features), and can generalize common patterns.

Design Considerations:
Data Preprocessing:
• Combine data from all customers and pods.
• Engineer features: customer_id, pod_id, temporal features (month, season), consumption types, etc.

Model Choices:
• Although ARIMA is univariate, you might train one ARIMA per consumption type across clusters.
• Alternatively, consider a multivariate model like VAR or even a deep learning model (LSTM, Transformer) if you want to capture cross-dependencies.
"""
# def train_global_arima_models(df: pd.DataFrame, consumption_types: List[str], order: Tuple[int, int, int]) -> Dict:
#     # Group by cluster or globally, then for each consumption type, train an ARIMA
#     global_models = {}
#     for ctype in consumption_types:
#         series = pd.to_numeric(df[ctype], errors='coerce').dropna()
#         if series.empty:
#             continue
#         model = train_arima(series, order=order)
#         global_models[ctype] = model
#     return global_models

# def fine_tune_model_for_customer(customer_series: pd.Series, order: Tuple[int,int,int]) -> ARIMAResults:
#     # Optionally, load global model weights and re-fit on customer data
#     # This is a simplified example; ARIMA doesn’t support fine-tuning out of the box,
#     # but you could use the global model's parameters as an initializer.
#     model = train_arima(customer_series, order)
#     return model


# def forecast_for_customer(df: pd.DataFrame, customer_id: str, consumption_type: str, order: Tuple[int, int, int],
#                           n_periods: int) -> pd.Series:
#     customer_data = df[df['CustomerID'] == customer_id]
#     series = pd.to_numeric(customer_data[consumption_type], errors='coerce').dropna()
#
#     # Fine-tune (or reuse) the model
#     model = fine_tune_model_for_customer(series, order)
#     forecast = predict_arima(model, order, n_periods)
#     return forecast