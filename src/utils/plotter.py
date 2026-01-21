# src/utils/plotter.py

import matplotlib
matplotlib.use('Agg') # Chạy trên server không màn hình
import matplotlib.pyplot as plt
import os
import json

class LogPlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "log_history.json")
        self.plot_file = os.path.join(output_dir, "loss_curve.png")
        self.history = []

    def update(self, epoch, loss, lr):
        # Lưu vào list
        self.history.append({
            "epoch": epoch,
            "loss": loss,
            "lr": lr
        })
        
        # Ghi file JSON dự phòng
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=4)
        
        # Vẽ biểu đồ
        self._plot()

    def _plot(self):
        if not self.history: return
        
        epochs = [x['epoch'] for x in self.history]
        losses = [x['loss'] for x in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o', label='Train Loss', color='b')
        
        plt.title("Training Progress (CLIP)")
        plt.xlabel("Epoch")
        plt.ylabel("Contrastive Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.savefig(self.plot_file)
        plt.close()