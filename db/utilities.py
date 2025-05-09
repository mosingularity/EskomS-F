# Databricks notebook source
import os
import logging
import yaml

from databricks.connect import DatabricksSession

from pyspark.dbutils import DBUtils

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark & DBUtils
spark = DatabricksSession.builder.getOrCreate()


# Load YAML config
def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# Determine environment
env = os.getenv("ENV", "QA")
config = load_config()

if env not in config:
    raise ValueError(f"Environment '{env}' not found in config.yaml.")

# Extract SQL Server connection settings
server_url = config[env]["server"]
database = config[env]["database"]
user = os.getenv("DB_USER", "fortrackSQL")
password = os.getenv("DB_PASSWORD", "vuxpapyvu@2024")

# Construct JDBC URL
jdbc_url = f"{server_url};databaseName={database}"
jdbc_properties = {
    "user": user,
    "password": password
}

# Generic function to run Spark SQL queries via JDBC
def read_sql_query(query: str):
    try:
        df = spark.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("query", query) \
            .options(**jdbc_properties) \
            .load()
        logging.info("Query executed successfully via Spark.")
        return df
    except Exception as e:
        logging.error(f"Failed to run Spark SQL query: {e}")
        return None

# Wrapper for parameterized table reads
def read_table(table_name: str, limit: int = 10):
    query = f"SELECT TOP {limit} * FROM {table_name}"
    return read_sql_query(query)
