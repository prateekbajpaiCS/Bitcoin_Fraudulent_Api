from flask_restful import Resource, Api
from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

class TransactionClassifierApi(Resource):
    def get(self):
        json_payload = request.get_json()
        df_columns = ['amount', 'size', 'weight', 'version', 'lock_time',
       'is_coinbase', 'has_witness', 'input_count', 'output_count',
       'input_total_usd', 'output_total_usd', 'fee_usd', 'fee_per_kb_usd',
       'fee_per_kwu_usd', 'cdd_total']

        df = pd.DataFrame(columns = df_columns)
        df_row = []
        for key in df_columns:
            try:
                df_row.append(json_payload[key])
            except Exception as e:
                print(e)
                return jsonify({"error": "request body is empty or invalid"})
        
        df.loc[0] = df_row
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        transaction_classifier = load_model("bitcoin_fraudulent_identifier.h5")

        prediction = transaction_classifier.predict(X_scaled)
        
        response_payload_value = True if prediction > 0.5 else False

        response_payload = jsonify({"fraudulent": response_payload_value})

        return response_payload

api = Api(app)
api.add_resource(TransactionClassifierApi, "/getTransactionType")

if __name__ == "__main__":
    app.run()