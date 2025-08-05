import os
import shutil
import pandas as pd
import joblib

import hopsworks

from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from hsml.model_registry import ModelRegistry
from hsml.deployment import Deployment
from hsml.resources import PredictorResources, Resources
from hsml.model import Model

from src.utils import singleton
from src.common import IAQI_FEATURES

MODEL_NAME = "aqi_prediction_model"


@singleton
class HopsworksClient:

    def __init__(self):
        hopsworks_aqi_token = os.environ["HOPSWORKS_AQI_TOKEN"]
        self.project = hopsworks.login(api_key_value=hopsworks_aqi_token)

    def load_hourly_data(self):
        feature_store = self.project.get_feature_store()

        iaqi_fg = feature_store.get_feature_group(name="iaqi", version=1)

        iaqi_fg_df = iaqi_fg.select(
            ["event_timestamp", "pm25", "pm10", "no2", "so2", "co"]
        ).read()
        # Remove TimeZone info and reset to 0 hours so that it can be compared to historical data
        iaqi_fg_df["event_timestamp"] = (
            pd.to_datetime(iaqi_fg_df["event_timestamp"])
            .dt.tz_localize(None)
            .dt.normalize()
        )
        return iaqi_fg_df

    def save_model(
        self,
        project_root,
        model,
        metrics,
        input_example,
        output_example,
        feature_scaler,
    ) -> Model:
        # 0. Prepare temp folder for deployment
        DEPLOYMENT_FOLDER = "deployment"
        deployment_path = os.path.join(project_root, DEPLOYMENT_FOLDER)
        shutil.rmtree(deployment_path, ignore_errors=True)
        os.makedirs(deployment_path)

        # 1. Save Model
        model_path = os.path.join(deployment_path, f"{MODEL_NAME}.pkl")
        joblib.dump(model, model_path)

        # 2. Save Scaler
        # Model path cannot contain more than one model file (i.e. .pkl, .pickle, .joblib files)
        feature_scaler_path = os.path.join(deployment_path, f"feature_scaler.bin")
        joblib.dump(feature_scaler, feature_scaler_path)

        # 3. Save Predictor script
        source_predictor_path = os.path.join(project_root, "scripts", "predictor.py")
        # The model server explicitly looks for a predictor.py file within the root of the uploaded model artifact
        destination_predictor_path = os.path.join(deployment_path, "predictor.py")
        shutil.copy(source_predictor_path, destination_predictor_path)

        # 4. Save source code for predictor
        source_folder = os.path.join(project_root, "src")
        destination_folder = os.path.join(deployment_path, "src")
        shutil.copytree(
            source_folder,
            destination_folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

        # 5. Copy requirements.txt
        source_requiremnts_path = os.path.join(project_root, "requirements.txt")
        destination_requiremnts_path = os.path.join(deployment_path, "requirements.txt")
        shutil.copy(source_requiremnts_path, destination_requiremnts_path)

        # 6. Save everything
        model_registry: ModelRegistry = self.project.get_model_registry()

        input_schema = Schema(input_example)
        output_schema = Schema(output_example)
        model_schema = ModelSchema(
            input_schema=input_schema, output_schema=output_schema
        )

        # TODO: should I use pythong instead of sklearn (what about moder server)
        aqi_model: Model = model_registry.python.create_model(
            name=MODEL_NAME,
            description="Air Quality Index prediction model",
            metrics=metrics,
            input_example=input_example,
            model_schema=model_schema,
        )
        aqi_model.save(deployment_path)

        return aqi_model

    def format_metrics(self, prediction_metrics, target_std_devs):
        metrics = {}
        all_nrmse_scores = []
        all_r2_scores = []
        
        for column_index, aqi in enumerate(IAQI_FEATURES):
            aqi_metrics = {}
            aqi_nrmse_scores = []
            aqi_r2_scores = []

            for day_metrics in prediction_metrics:
                for (metric_name, metric_value) in day_metrics[column_index].items():
                    aqi_metrics.setdefault(metric_name, []).append(metric_value)
            
            for aqi_metric_name, aqi_metric_values in aqi_metrics.items():
                avg_value = sum(aqi_metric_values) / len(aqi_metric_values)
                
                if aqi_metric_name == 'rmse':
                    # Convert RMSE to NRMSE
                    nrmse = avg_value / target_std_devs[column_index]
                    metrics[f"{aqi}_nrmse"] = round(nrmse, 4)
                    aqi_nrmse_scores.append(nrmse)
                    all_nrmse_scores.extend([v / target_std_devs[column_index] for v in aqi_metric_values])
                elif aqi_metric_name == 'r2_score':
                    metrics[f"{aqi}_r2"] = round(avg_value, 4)
                    aqi_r2_scores.append(avg_value)
                    all_r2_scores.extend(aqi_metric_values)
                else:
                    # Keep other metrics as-is
                    metrics[f"{aqi}_{aqi_metric_name}"] = round(avg_value, 4)
        
        # Overall summary metrics
        metrics['overall_nrmse'] = round(sum(all_nrmse_scores) / len(all_nrmse_scores), 4)
        metrics['overall_r2'] = round(sum(all_r2_scores) / len(all_r2_scores), 4)
        
        return metrics

    def load_model(self, version=1):
        model_registry: ModelRegistry = self.project.get_model_registry()
        retrieved_model = model_registry.get_model(MODEL_NAME, version)
        download_path = retrieved_model.download()
        model_file_path = os.path.join(download_path, f"{MODEL_NAME}.pkl")
        model = joblib.load(model_file_path)
        return retrieved_model, model

    def deploy_model(self, hopsworks_model: Model, overwrite=False) -> Deployment:
        environment_api = self.project.get_environment_api()
        # Unfortunately, environment has to be created through UI for now
        environment = environment_api.get_environment("aqi-inference-pipeline-v1")
        if (not environment):
            raise Exception("Environment 'aqi-inference-pipeline-v1' has to be set up.")

        # 1. Install requirements based on requirements.txt
        requirements_path = os.path.join(hopsworks_model.version_path, "Files", "requirements.txt")
        environment.install_requirements(requirements_path, await_installation=True)

        # 2. Destroy previous deployment
        deployment_name = "aqipredictionmodeldeployment"
        model_serving = HopsworksClient().project.get_model_serving()

        deployment: Deployment = model_serving.get_deployment(deployment_name)
        if overwrite and deployment:
            deployment.stop()
            deployment.delete()

        predictor_script_path = os.path.join(hopsworks_model.version_path, "Files", "predictor.py")

        predictor_res = PredictorResources(
            num_instances=0,
            requests=Resources(cores=0.5, memory=512, gpus=0),
            limits=Resources(cores=0.5, memory=1024, gpus=0),
        )

        deployment = hopsworks_model.deploy(
            name=deployment_name,
            script_file=predictor_script_path,
            resources=predictor_res,
            environment="aqi-inference-pipeline-v1"
        )

        return deployment
