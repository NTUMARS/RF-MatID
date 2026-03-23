import torch
import torch.nn as nn

class RFVibeAdaptive(nn.Module):
    def __init__(self, num_classes):
        super(RFVibeAdaptive, self).__init__()

        self.freq_head = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 8, kernel_size=3, padding=1), nn.ReLU(), # 输出 8 通道
            nn.AdaptiveAvgPool1d(128) 
        )
        

        self.freq_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )


        self.power_input_adapter = nn.AdaptiveAvgPool1d(25)
        
        self.power_head = nn.Sequential(
            nn.Linear(25, 64), 
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.power_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )


    def forward(self, x_complex, x_power):

        f_feat = self.freq_head(x_complex)   # [Batch, 8, 128]
        f_feat = torch.sum(f_feat, dim=1)    # [Batch, 128] (Sum across channels)
        
        p_in = self.power_input_adapter(x_power).view(x_power.size(0), -1)
        p_feat = self.power_head(p_in)       # [Batch, 128]
        
        out_freq = self.freq_classifier(f_feat)   
        out_power = self.power_classifier(p_feat) 
        
        combined = torch.cat((f_feat, p_feat), dim=1) # [Batch, 256]
        out_final = self.final_classifier(combined)
        
        return out_freq, out_power, out_final

if __name__ == "__main__": 

    from torchinfo import summary
    import torch.nn as nn


    def analyze_model(model_class, num_classes, L=512, batch_size=1):

        model = model_class(num_classes)
        
        input_size_complex = (batch_size, 2, L)
        input_size_power = (batch_size, 1, L)
        
        model_summary = summary(
            model, 
            input_data=[
                torch.randn(input_size_complex), 
                torch.randn(input_size_power)
            ],
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
            ],
            verbose=0
        )
        
        print("="*50)
        print(f"L={L}, Classes={num_classes}")
        print("="*50)
        

        print(model_summary)
        print(f"\n Total Parameters: {model_summary.total_params:,}")
        print(f"Total FLOPs/Mult-Adds: {model_summary.total_mult_adds/1e6:.2f} M")
        print("="*50)



    NUM_CLASSES = 16 
    SEQUENCE_LENGTH = 2048 

    analyze_model(RFVibeAdaptive, NUM_CLASSES, L=SEQUENCE_LENGTH)