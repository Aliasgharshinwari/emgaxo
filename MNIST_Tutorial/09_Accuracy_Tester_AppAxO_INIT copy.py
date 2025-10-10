import os
import onnx
import csv
from emgaxo import check_accuracy
#import tensorflow as tf
#(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
import numpy as np

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")


WIDTH = 36

def count_zero_bits(n: int, width: int = WIDTH) -> int:
    """Count the number of 0‐bits in the `width`-bit representation of n."""
    ones = n.bit_count()
    return width - ones

def evaluate_all_models(model_dir, output_csv):
    """Loads all ONNX models from a folder, evaluates their accuracy, and writes results to a CSV file."""
    
    if not os.path.exists(model_dir):
        print(f"Error: Directory '{model_dir}' does not exist.")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
    
    if not model_files:
        print("No ONNX models found in the directory.")
        return
    
    accuracy_results = {}
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "Accuracy (%)", "Precision", "Recall", "F1 Score", "Number of LUTs disabled", "INIT Value"])

        for i, model_file in enumerate(model_files):
            model_path = os.path.join(model_dir, model_file)
            print(f"Loading model: {model_file}")

            try:
                modified_model = onnx.load(model_path)
                # onnx.checker.check_model(modified_model)  # Uncomment if needed
                 # parse the config‐integer from the filename prefix
                cfg_str = model_file.split("_", 1)[0]
                try:
                    cfg_int = int(cfg_str)
                except ValueError:
                    print(f"  ⚠️ could not parse config int from '{cfg_str}', skipping zero‐bit count")
                    zero_bits = ""
                else:
                    zero_bits = count_zero_bits(cfg_int)

                accuracy, precision, recall, f1  = check_accuracy(modified_model, True, 'test.customop', 
                                          "cuda_based_approximate_qgemm_uint_modified", x_test, y_test, 1000)
                accuracy_results[model_file] = accuracy
                
                 # Write full results to CSV
                writer.writerow([
                    model_file,
                    f"{accuracy * 100:.5f}",
                    f"{precision:.4f}",
                    f"{recall:.4f}",
                    f"{f1:.4f}",
                    zero_bits,
                    cfg_str
                ])
                print(f"Processing Model {i}")
                
            except Exception as e:
                print(f"Failed to evaluate {model_file}: {e}")
    
    return accuracy_results


# Example usage
model_directory = "./AppAxO_Models30k"  # Change this to your actual model directory
output_csv_file = "accuracy_results_30k.csv"
results = evaluate_all_models(model_directory, output_csv_file)

# Print final results
if results:
    print("\nFinal Accuracy Results:")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc * 100:.5f}%")
    print(f"Results saved to {output_csv_file}")