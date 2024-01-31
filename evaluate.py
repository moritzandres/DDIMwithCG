import cv2
import logging
import lpips
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from skimage.metrics import structural_similarity as ssim
from torch.nn import L1Loss


matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Evaluator:
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, model, device, test_data_dict):
        # super(Evaluator, self).__init__(model, device, test_data_dict)
        super(Evaluator, self).__init__()

        self.model = model
        self.device = model.device
        self.test_data_dict = test_data_dict
        self.criterion_rec = L1Loss().to(device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze',
                                     use_dropout=True, eval_mode=True,
                                     spatial=True, lpips=True).to(device)

    def evaluate(self):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("############### Object Localzation TEST ################")
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        # self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'L1': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'L1': [],
                'LPIPS': [],
                'SSIM': [],
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
            }
            print('******************* DATASET: {} ****************'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                inputs, masks, neg_masks = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                nr_batches, nr_slices, width, height = inputs.shape
                neg_masks[neg_masks > 0.5] = 1
                neg_masks[neg_masks < 1] = 0
                results = self.model.detect_anomaly(inputs)
                reconstructions = results['reconstruction']
                anomaly_maps = results['anomaly_map']

                for i in range(nr_batches):
                    count = str(idx * nr_batches + i)
                    x_i = inputs[i][0]
                    x_rec_i = reconstructions[i][0] if reconstructions is not None else None
                    ano_map_i = anomaly_maps[i][0].cpu().detach().numpy()
                    mask_i = masks[i][0].cpu().detach().numpy()
                    neg_mask_i = neg_masks[i][0].cpu().detach().numpy()
                    bboxes = cv2.cvtColor(neg_mask_i * 255, cv2.COLOR_GRAY2RGB)
                    cnts_gt = cv2.findContours((mask_i * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []

                    for c_gt in cnts_gt:
                        x, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([x, y, x + w, y + h])
                        cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    if x_rec_i is not None:
                        loss_l1 = self.criterion_rec(x_rec_i, x_i)
                        test_metrics['L1'].append(loss_l1.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        x_rec_i = x_rec_i.cpu().detach().numpy()
                        ssim_ = ssim(x_rec_i, x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

                    x_i = x_i.cpu().detach().numpy()
                    x_pos = ano_map_i * mask_i
                    x_neg = ano_map_i * neg_mask_i
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_i)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0  # 10% overlap due to large bboxes e.g. for enlarged ventricles
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = int(res_healthy / max(res_anomaly, 1))
                    fps.append(fp)
                    precision = tp / max((tp + fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    if int(count) == 0:
                        if x_rec_i is None:
                            x_rec_i = np.zeros(x_i.shape)
                        elements = [x_i, x_rec_i, ano_map_i, bboxes.astype(np.int64), x_pos, x_neg]
                        v_maxs = [1, 1, 0.99, 1, np.max(ano_map_i), np.max(ano_map_i)]

                        titles = ['Input', 'Reconstruction', 'Anomaly Map', 'GT',
                                  str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp),
                                  str(np.round(res_healthy, 2)) + ', FP: ' + str(fp)]

                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

            for metric in test_metrics:
                print('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                  np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    print(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    print(f'FP: {np.sum(test_metrics[metric])} missed')
                metrics[metric].append(test_metrics[metric])

        print('Writing plots...')
        fig_bps = dict()
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
            fig_bps[metric] = fig_bp
        return metrics, fig_bps, diffp
