from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.models.nnUNet.nnUNet_pipeline import nnUNet

import pandas as pd
import os
from SAES.html.html_generator import notebook_no_fronts, notebook_bayesian
from SAES.latex_generation.stats_table import MeanMedian
from SAES.latex_generation.stats_table import Friedman
from SAES.latex_generation.stats_table import WilcoxonPivot
from SAES.latex_generation.stats_table import Wilcoxon 

def train_nnUNet_models():
    trainer = Trainer.EPOCHS_50
    csv_path = f"{os.getcwd()}/data.csv"
    
    for i in range(5):
        dataset_id = f"00{i}"     
        for configuration in Configuration:
            for fold in Fold:
                #nnUNet(dataset_id, configuration, fold, trainer).execute_pipeline(csv_path)   
                nnUNet(dataset_id, configuration, fold, trainer).write_results(csv_path)

def create_results_html(data: str, metrics: str) -> None:
    metrics_list = pd.read_csv(metrics)["MetricName"].unique().tolist()
    for metric in metrics_list:
        notebook_no_fronts(data, metrics, metric, os.getcwd())
        os.rename("no_fronts.html", f"{metric}.html")

        notebook_bayesian(data, metrics, metric, "nnUNet3D", f"{os.getcwd()}/outputs")
        os.rename("bayesian.html", f"{metric}_bayesian.html")
        
        # Create the LaTeX tables
        mean_median = MeanMedian(data, metrics, metric)
        friedman = Friedman(data, metrics, metric)
        wilcoxon_pivot = WilcoxonPivot(data, metrics, metric)
        wilcoxon = Wilcoxon(data, metrics, metric)

        # Save the LaTeX tables on disk
        mean_median.save(os.getcwd())
        friedman.save(os.getcwd())
        wilcoxon_pivot.save(os.getcwd())
        wilcoxon.save(os.getcwd())
        

if __name__ == "__main__":
    data = "outputs/data.csv"
    metrics = "outputs/metrics.csv"
    create_results_html(data, metrics)
    
    
