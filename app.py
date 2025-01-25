from flask import Flask, render_template, jsonify
from threading import Thread, Lock
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
import time
import threading
from kafka import KafkaProducer
import datetime

app = Flask(__name__)

# Kafka configuration
kafka_broker = "127.0.0.1:9092"
kafka_topic = "Airfare_Prediction"

# Shared predictions list for Flask view
predictions_list = []
predictions_lock = Lock()  # Lock to ensure thread-safety when updating predictions_list

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4") \
    .getOrCreate()

# Define schema for incoming Kafka messages
schema = StructType([
    StructField("Journey_day", StringType(), True),
    StructField("Class", StringType(), True),
    StructField("Source", StringType(), True),
    StructField("Departure", StringType(), True),
    StructField("Total_stops", StringType(), True),
    StructField("Arrival", StringType(), True),
    StructField("Destination", StringType(), True),
    StructField("Duration_in_hours", FloatType(), True),
    StructField("Days_left", IntegerType(), True),
    StructField("Journey_month", IntegerType(), True),
    StructField("Weekend", IntegerType(), True),
])

# Load the trained model
model_path = "model"  # Path to the trained model
try:
    model = PipelineModel.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

# Kafka streaming to process and predict data
def process_kafka_stream():
    kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_broker) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "latest") \
        .load()

    parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as json") \
        .select(from_json(col("json"), schema).alias("data")) \
        .select("data.*")

    # Handle null values in the parsed stream
    parsed_stream = parsed_stream.na.drop()

    # Apply the trained model to the streaming data
    predictions = model.transform(parsed_stream)

    # Process results and store them for Flask to access
    def process_row(batch_df, batch_id):
        conversion_rate = 0.012  # Example: 1 INR = 0.012 USD
        # Collect the batch dataframe into a list of rows
        batch_df = batch_df.collect()  # Collect the data in each batch
        
        with predictions_lock:
            for idx, row in enumerate(batch_df):
                prediction = row['prediction']
                prediction_inr = row['prediction']  # Prediction in INR
                prediction_usd = prediction_inr * conversion_rate 
                prediction_data = {
                    "Index": batch_id + 1,  # Index starting from 1
                    "Departure": row['Source'],
                    "Destination": row['Destination'],
                    "DayOfWeek": row['Journey_day'],
                    "Time": row['Departure'],
                    "Predict_Price": round(prediction_usd, 2),
                    "Days_left": row['Days_left']
                }
                predictions_list.append(prediction_data)

    # Apply foreachBatch correctly within the streaming query
    query = predictions.writeStream \
        .outputMode("append") \
        .foreachBatch(process_row) \
        .trigger(processingTime="5 seconds") \
        .start()

    query.awaitTermination()

# Start Kafka stream in a separate thread
def start_kafka_stream():
    threading.Thread(target=process_kafka_stream, daemon=True).start()

@app.route('/')
def index():
    # Ensure data is processed before rendering
    with predictions_lock:
        return render_template('index.html', predictions=predictions_list)

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    with predictions_lock:
        return jsonify(predictions_list)

if __name__ == '__main__':
    # Start the Kafka consumer stream
    start_kafka_stream()

    # Run Flask application
    app.run(debug=True, host='localhost', port=5000)
