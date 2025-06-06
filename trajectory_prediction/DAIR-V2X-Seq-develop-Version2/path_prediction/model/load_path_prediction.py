from .QCNet import QCNet

def load_path_prediction_model():
    net = QCNet(
            dataset='argoverse_v2',
            input_dim=2,
            hidden_dim=128,
            output_dim=2,
            output_head=False,
            num_historical_steps=50,
            num_future_steps=60,
            num_modes=6,
            num_recurrent_steps=3,
            num_freq_bands=64,
            num_map_layers=1,
            num_agent_layers=2,
            num_dec_layers=2,
            num_heads=8,
            head_dim=16,
            dropout=0.1,
            pl2pl_radius=150.0,
            time_span=10,
            pl2a_radius=50.0,
            a2a_radius=50.0,
            num_t2m_steps=30,
            pl2m_radius=150.0,
            a2m_radius=150.0,
            lr=5e-4,
            weight_decay=1e-4,
            T_max=64,
            submission_dir='./',
            submission_file_name='submission'
        )
    
    return net