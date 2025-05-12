import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from torch_geometric.data import HeteroData

from model.data_handling import Map, qcnet_pred
from model.QCNet import QCNet

from model.utils import TargetBuilder

import pdb

if __name__ == "__main__":
    path_to_data = 'data/V2X-Seq-TFD'
    path_to_traj = '{}/single-infrastructure/trajectories/val'.format(path_to_data)
    path_to_maps = '{}/maps'.format(path_to_data)
    path_to_wts = '{}/checkpoints/DAIR.ckpt'.format(os.path.split(os.path.realpath(__file__))[0])
    device = 'cuda:0'

    net = QCNet(
        dataset='dair',
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

    transform = TargetBuilder(50, 60)
    transform2 = TargetBuilder(50, 0)

    net = net.load_from_checkpoint(path_to_wts).eval().to(device)
    traj_paths = sorted(glob("{}/*.csv".format(path_to_traj)), key=lambda x: int(x.split('/')[-1].split('.')[0]))

    map_fmt = path_to_maps+"/{}_hdmap{}.json"

    plt.rcParams.update({'font.size': 16})

    fig, axs = plt.subplots(3,5, figsize=(48, 32))
    axs = axs.flatten()

    # traj_paths[0] = 'data/V2X-Seq-TFD/single-infrastructure/trajectories/val/10.csv'
    for i in tqdm(range(len(traj_paths))[:15]):

        traj_path = traj_paths[i]
        df = pd.read_csv(traj_path)
        t = np.array(sorted(list(set(df['timestamp'].values))))
        
        map_path = map_fmt.format(df['intersect_id'][0].split('#')[0], df['intersect_id'][0].split('#')[1].split('-')[0])
        ## map_path='data/V2X-Seq-TFD/maps/yizhuang_hdmap14.json'
        map_api = Map(map_path)

        ade = {'vehicle':[], 'pedestrian':[], 'cyclist':[]}
        for j in range(0, max(1,len(t)-110)):
            # dfはtimestamp, id, x,yの値を持っている必要がある

            agent_features = map_api.get_agent_features(df, j, 50)
            data = transform2(HeteroData({**{'agent': agent_features}, **map_features}))



            agent_features = map_api.get_agent_features(df, j, 110)
            agent_xy = np.median(agent_features['position'][:,:,:2].cpu().numpy().reshape(-1,2), axis=0)
            if np.sum(agent_xy) == 0:
                for k in range(0, len(agent_features['position'])):
                    agent_xy = np.median(agent_features['position'][k,:,:2].cpu().numpy(), axis=0)
                    if np.sum(agent_xy) != 0:
                        break
            map_features = map_api.get_map_features(agent_xy)
            
            # Plot map
            polygons = map_api.get_polygons_within(agent_xy, max_dist=100.0)        
            for polygon,polygon_type in polygons.values():
                if polygon_type == 'crosswalk':
                    color = (0,0,0)
                    linewidth = 1
                    zorder=1
                else:
                    color = (0.8,0.8,0.8)
                    linewidth = 7
                    zorder=0
                axs[i].plot(polygon[:,0], polygon[:,1], color=color, zorder=zorder, linewidth=linewidth)

            
            # Compute prediction
            data = transform(HeteroData({**{'agent': agent_features}, **map_features}))

            traj, pi, pred = qcnet_pred(net, data, device=device)

            for k in range(len(pi)):
                agent_id = data['agent']['id'][k]     
                agent_type = map_api._agent_types[data['agent']['type'][k]]
                past_x = data['agent']['position'][k,:50,0].cpu().numpy()
                past_y = data['agent']['position'][k,:50,1].cpu().numpy()
                future_x = data['agent']['position'][k,50:,0].cpu().numpy()
                future_y = data['agent']['position'][k,50:,1].cpu().numpy()
                future_xy = data['agent']['position'][k,50:,:2].cpu().numpy()
                
                mask_past = np.logical_or(past_x != 0, past_y != 0)
                past_x = past_x[mask_past]
                past_y = past_y[mask_past]
                mask_future = np.logical_or(future_x != 0, future_y != 0)
                future_x = future_x[mask_future]
                future_y = future_y[mask_future]                
                
                if np.sum(mask_future) == 0 or np.sum(mask_past) == 0:
                    continue
                
                dist = np.mean(np.linalg.norm(traj[k] - future_xy[None], axis=2)[:,mask_future],axis=1)
                
                pidx = np.argsort(pi[k])[::-1][0]

                ade[agent_type].append(dist[pidx])                

                axs[i].plot(traj[k,pidx,:,0], traj[k,pidx,:,1], 'r', zorder=5, linewidth=2)
                axs[i].plot(future_x, future_y, 'b', zorder=4, linewidth=4)
                axs[i].plot(past_x, past_y, 'g', zorder=3, linewidth=4)
                    

        ade['vehicle'] = np.nan if len(ade['vehicle']) == 0 else np.mean(ade['vehicle'])
        ade['pedestrian'] = np.nan if len(ade['pedestrian']) == 0 else np.mean(ade['pedestrian'])
        ade['cyclist'] = np.nan if len(ade['cyclist']) == 0 else np.mean(ade['cyclist'])

        axs[i].axis('scaled')
        colors = ['g', 'r', 'b']
        lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
        labels = ['Past', 'Pred', 'GT']
        axs[i].legend(lines, labels, loc="upper left")
        axs[i].title.set_text('{}\nmADE(car, cyc, ped) = {:.2f}, {:.2f}, {:.2f}'.format('-'.join(traj_path.split('/')[-2:]).split('.')[0], ade['vehicle'], ade['cyclist'], ade['pedestrian']))
        axs[i].axis('off')

        polygons = np.stack([v[0] for v in polygons.values()],axis=0).reshape(-1,2)
        x_mid = np.median(polygons[:,0])
        y_mid = np.median(polygons[:,1])
        plt_size = 200
        axs[i].set_xlim([x_mid - plt_size//2, x_mid + plt_size//2])
        axs[i].set_ylim([y_mid - plt_size//2, y_mid + plt_size//2])


    plt.savefig('out_path_prediction.jpg')
