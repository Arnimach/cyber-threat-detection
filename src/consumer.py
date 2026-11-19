
import json
import pandas as pd
from confluent_kafka import Consumer, Producer

from src.inference import ThreatDetector

# Initialize the trained model
detector = ThreatDetector()

# Kafka consumer: reads incoming network events
consumer = Consumer({
    "bootstrap.servers": "kafka:29092",
    "group.id": "threat-detection-group",
    "auto.offset.reset": "earliest"
})

consumer.subscribe(["security-events"])

# Kafka producer: publishes model predictions
producer = Producer({
    "bootstrap.servers": "kafka:29092"
})

PREDICTION_TOPIC = "threat-predictions"


def delivery_report(err, msg):
    if err is not None:
        print(f"Prediction delivery failed: {err}")


print("Listening for security events...")

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        try:
            event = json.loads(msg.value().decode("utf-8"))

            # Convert event into DataFrame
            features = pd.DataFrame([event["features"]])

            # Run model prediction
            result = detector.predict(features)

            prediction = result.to_dict(orient="records")[0]

            print("Received event:")
            print(event["event_id"])

            print("Prediction:")
            print(prediction)

            # Build prediction message
            prediction_event = {
                "event_id": event["event_id"],
                "flow_id": event["flow_id"],
                "timestamp": event["timestamp"],
                "prediction": prediction
            }

            # Publish prediction to Kafka
            producer.produce(
                topic=PREDICTION_TOPIC,
                key=event["event_id"],
                value=json.dumps(
                    prediction_event,
                    default=lambda x: x.item() if hasattr(x, "item") else str(x)
                ),
                callback=delivery_report
            )

            producer.poll(0)

        except Exception as e:
            print(f"Skipping event due to prediction error: {e}")
            continue

except KeyboardInterrupt:
    print("Stopping consumer...")

finally:
    producer.flush()
    consumer.close()
