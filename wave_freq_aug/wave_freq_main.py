# ===========================================================================================================
# This main experiment script is developed by Md Abid Hasan as part of the WaveFreqAug project.
# It orchestrates the training and evaluation of time series forecasting models with Wave-Freq augmentation.
# ===========================================================================================================

import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import DLinear, SCINet
from train_eval import train, test
from dataloader import TimeSeriesDataset
from dataset_parameter import dataset_configs

# Main experiment
def main(epochs, learning_rate, patience, num_iterations, label_len):
    # Common parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)  # Limit to 50% of GPU memory
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True"  # Reduce memory fragmentation
        )
    models = ["DLinear", "SCINet"]
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    os.makedirs("/home/abid/a_c_p/wave_freq/results", exist_ok=True)

    # Run experiments for each dataset
    for dataset_name, config in dataset_configs.items():
        # Initialize per-iteration results CSV file
        iteration_csv_path = f"/home/abid/a_c_p/wave_freq/results/iteration_results_{dataset_name}.csv"
        with open(iteration_csv_path, "w") as f_iter:
            f_iter.write(
                "dataset,model,pred_len,aug_type,mask_rate,level,wavelet,iteration,val_loss,mae,mse,rse\n"
            )

        # Initialize average results CSV file
        average_csv_path = f"/home/abid/a_c_p/wave_freq/results/average_results_{dataset_name}.csv"
        with open(average_csv_path, "w") as f_avg:
            f_avg.write(
                "dataset,model,pred_len,aug_type,mask_rate,level,wavelet,val_loss,mae,mse,rse,mae_std,mse_std,rse_std\n"
            )
        for model_name in models:
            print(
                f"\n=== Processing dataset: {dataset_name} with model: {model_name} ==="
            )
            # Load datasets
            for pred_len in config["pred_lens"]:
                print(f"\nPrediction length: {pred_len}")
                train_dataset = TimeSeriesDataset(
                    data_path=config["data_path"],
                    data_name=config["data_name"],
                    flag="train",
                    seq_len=config["seq_len"],
                    label_len=label_len,
                    pred_len=pred_len,
                    enc_in=config["enc_in"],
                )
                val_dataset = TimeSeriesDataset(
                    data_path=config["data_path"],
                    data_name=config["data_name"],
                    flag="val",
                    seq_len=config["seq_len"],
                    label_len=label_len,
                    pred_len=pred_len,
                    enc_in=config["enc_in"],
                )
                test_dataset = TimeSeriesDataset(
                    data_path=config["data_path"],
                    data_name=config["data_name"],
                    flag="test",
                    seq_len=config["seq_len"],
                    label_len=label_len,
                    pred_len=pred_len,
                    enc_in=config["enc_in"],
                )
                print(f"Train dataset length: {len(train_dataset)}")

                batch_size = config["batch_size"]

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )

                # Grid search for Wave-Freq hyperparameters
                mask_rates = [0.1, 0.15, 0.2]
                levels = [1, 3, 5]
                wavelets = ["db2", "db4", "sym4"]
                aug_type = "Wave-Freq"
                sampling_rate = 0.2  # Fixed as in original grid search
                for mask_rate in mask_rates:
                    for level in levels:
                        for wavelet in wavelets:
                            print(
                                f"\nGrid Search: mask_rate={mask_rate}, level={level}, wavelet={wavelet}, sampling_rate={sampling_rate}"
                            )
                            checkpoint_dir = f"./checkpoints/{dataset_name}_{aug_type}_{pred_len}_{model_name}_{mask_rate}_{level}_{wavelet}"
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            mse_list, mae_list, rse_list, val_loss_list = [], [], [], []
                            for itr in range(num_iterations):
                                print(f"Iteration {itr+1}/{num_iterations}")
                                if model_name == "DLinear":
                                    model = DLinear(
                                        config["seq_len"],
                                        pred_len,
                                        enc_in=config["enc_in"],
                                        individual=False,
                                    ).to(device)
                                elif model_name == "SCINet":
                                    model = SCINet(
                                        input_len=config["seq_len"],
                                        output_len=pred_len,
                                        input_dim=config["enc_in"],
                                        hid_size=1,
                                        num_stacks=1,
                                        num_levels=3,
                                        kernel_size=5,
                                        dropout=0.2,
                                    ).to(device)
                                val_loss = train(
                                    model,
                                    train_loader,
                                    val_loader,
                                    device,
                                    aug_type,
                                    config["seq_len"],
                                    label_len,
                                    pred_len,
                                    dataset_name=dataset_name,
                                    mask_rate=mask_rate,  # Grid search parameter
                                    wavelet=wavelet,      # Grid search parameter
                                    level=level,          # Grid search parameter
                                    sampling_rate=sampling_rate,  # Grid search parameter
                                    epochs=epochs,
                                    lr=learning_rate,
                                    patience=patience,
                                    checkpoint_dir=checkpoint_dir,
                                )
                                try:
                                    model.load_state_dict(
                                        torch.load(f"{checkpoint_dir}/checkpoint.pth")
                                    )
                                except FileNotFoundError:
                                    print(
                                        f"Checkpoint not found at {checkpoint_dir}/checkpoint.pth. Skipping iteration {itr+1}."
                                    )
                                    continue
                                mae, mse, rse = test(
                                    model,
                                    test_loader,
                                    device,
                                    train_dataset.scaler,
                                    pred_len,
                                )
                                mse_list.append(mse)
                                mae_list.append(mae)
                                rse_list.append(rse)
                                val_loss_list.append(val_loss)
                                print(
                                    f"Iteration {itr+1} - Val Loss: {val_loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RSE: {rse:.6f}"
                                )
                                # Save iteration metrics to iteration_results CSV
                                with open(iteration_csv_path, "a") as f_iter:
                                    f_iter.write(
                                        f"{dataset_name},{model_name},{pred_len},{aug_type},{mask_rate},{level},{wavelet},{itr+1},"
                                        f"{val_loss:.6f},{mae:.6f},{mse:.6f},{rse:.6f}\n"
                                    )

                            # Save average and standard deviation metrics to average_results CSV
                            if mse_list:  # Only save if at least one iteration succeeded
                                with open(average_csv_path, "a") as f_avg:
                                    f_avg.write(
                                        f"{dataset_name},{model_name},{pred_len},{aug_type},{mask_rate},{level},{wavelet},"
                                        f"{np.mean(val_loss_list):.6f},{np.mean(mae_list):.6f},{np.mean(mse_list):.6f},"
                                        f"{np.mean(rse_list):.6f},{np.std(mae_list):.6f},{np.std(mse_list):.6f},{np.std(rse_list):.6f}\n"
                                    )

                                print(
                                    f"{aug_type} - Avg Val Loss: {np.mean(val_loss_list):.6f}, Avg MAE: {np.mean(mae_list):.6f}, "
                                    f"Avg MSE: {np.mean(mse_list):.6f}, Avg RSE: {np.mean(rse_list):.6f}, "
                                    f"MAE Std: {np.std(mae_list):.6f}, MSE Std: {np.std(mse_list):.6f}, RSE Std: {np.std(rse_list):.6f}"
                                )

                            # Plot predictions
                            try:
                                model.load_state_dict(
                                    torch.load(f"{checkpoint_dir}/checkpoint.pth")
                                )
                                model.eval()
                                with torch.no_grad():
                                    batch_x, batch_y = next(iter(test_loader))  # Expect 2 values
                                    batch_x = batch_x.float().to(device)
                                    outputs = model(batch_x)
                                    outputs = outputs[:, -pred_len:, :].cpu().numpy()
                                    batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
                                    outputs = train_dataset.scaler.inverse_transform(
                                        outputs[0]
                                    )
                                    batch_y = train_dataset.scaler.inverse_transform(
                                        batch_y[0]
                                    )
                                    plt.figure()
                                    plt.plot(batch_y[:, 0], label="Ground Truth")
                                    plt.plot(
                                        outputs[:, 0], label=f"Prediction ({aug_type}, {model_name})"
                                    )
                                    plt.legend()
                                    plt.savefig(
                                        f"./plots/prediction_{dataset_name}_{aug_type}_{pred_len}_{model_name}_{mask_rate}_{level}_{wavelet}.png"
                                    )
                                    plt.close()
                            except FileNotFoundError:
                                print(
                                    f"Checkpoint not found for plotting at {checkpoint_dir}/checkpoint.pth. Skipping plot."
                                )
                            # Clean up memory
                            del model
                            torch.cuda.empty_cache()
                            gc.collect()

            # Clean up datasets after pred_len
            del (
                train_dataset,
                val_dataset,
                test_dataset,
                train_loader,
                val_loader,
                test_loader,
            )
            torch.cuda.empty_cache()
            gc.collect()

    print("All experiments completed.")

if __name__ == "__main__":
    print("Starting main experiment...")
    main(
        epochs=20, learning_rate=0.01, patience=7, num_iterations=5, label_len=0
    )
    print("Main experiment finished.")
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleanup completed.")
    print("You can now check the results and plots in the respective directories.")