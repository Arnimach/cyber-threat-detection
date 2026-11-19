import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer


# Kafka configuration
producer = Producer({
    "bootstrap.servers": "kafka:29092"
})

TOPIC = "security-events"
DELAY = 0.1  # 100 ms between events


DATA_DIR = Path(__file__).resolve().parents[1]

test_folder = DATA_DIR/"data"/"processed"/"test_rev"

CSV_FILES = list(test_folder.glob("*.csv"))


print(f"Found {len(CSV_FILES)} CSV files:")
for file in CSV_FILES:
    print(file.name)


def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(
            f"Delivered to {msg.topic()} "
            f"[partition {msg.partition()}]"
        )


def stream_events():

    flow_id = 0

    for file_path in CSV_FILES:

        print(f"Reading: {file_path.name}")

        for chunk in pd.read_csv(file_path, chunksize=1000):

            # Remove target: it must never be sent as a model feature
            if "target" in chunk.columns:
                features = chunk.drop(columns=["target"])
            else:
                features = chunk

            for record in features.to_dict(orient="records"):
                if flow_id % 1000 == 0:
                    print(f"Sent {flow_id} events")

                flow_id += 1

                event = {
                    "event_id": str(uuid.uuid4()),
                    "flow_id": flow_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "features": record
                }

                producer.produce(
                    topic=TOPIC,
                    key=str(flow_id),
                    value=json.dumps(event, default=str),
                    callback=delivery_report
                )

                producer.poll(0)
                time.sleep(DELAY)

    producer.flush()
    print("Streaming completed.")


if __name__ == "__main__":
    stream_events()
