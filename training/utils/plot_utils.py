import pickle

import numpy as np
import torch
from tqdm import tqdm


def _plot_prediction_with_target_simple(output, target, ax, row_idx, data_idx):
    import seaborn as sns
    assert output.shape[0] == 3
    assert target.shape[0] == 3
    for col_idx in range(3):
        output_t = output[col_idx, ...]
        target_t = target[col_idx, ...]

        sns.heatmap(output_t, ax=ax[row_idx, 2 * col_idx], vmax=50)
        sns.heatmap(target_t, ax=ax[row_idx, 2 * col_idx + 1], vmax=50)
        ax[row_idx, 2 * col_idx].set_title(f'Pred: {data_idx}-{col_idx}', )
        ax[row_idx, 2 * col_idx + 1].set_title(f'Tar: {data_idx}-{col_idx}')

        ax[row_idx, 2 * col_idx].tick_params(left=False, right=False, top=False, bottom=False)
        ax[row_idx, 2 * col_idx].axis('off')
        ax[row_idx, 2 * col_idx + 1].tick_params(left=False, right=False, top=False, bottom=False)
        ax[row_idx, 2 * col_idx + 1].axis('off')


def _plot_prediction_with_target_continuous(output, target, ax, col_idx, ts):
    import seaborn as sns
    assert output.shape[0] == 3
    assert target.shape[0] == 3
    ts = ts.strftime('%Y%m%d-%H%M')

    target_t = target[0, ...]
    sns.heatmap(target_t, ax=ax[0, col_idx], vmax=50)
    ax[0, col_idx].set_title(f'{ts}')
    ax[0, col_idx].tick_params(left=False, right=False, top=False, bottom=False)
    ax[0, col_idx].axis('off')

    for row_idx in range(3):
        output_t = output[row_idx, ...]
        sns.heatmap(output_t, ax=ax[row_idx + 1, col_idx], vmax=50)

        # ax[row_idx + 1, col_idx].set_title(f'P: {col_idx}', )
        ax[row_idx + 1, col_idx].tick_params(left=False, right=False, top=False, bottom=False)
        ax[row_idx + 1, col_idx].axis('off')


def plot_prediction_with_target(model,
                                data_loader,
                                count,
                                with_prior=False,
                                img_sz=3,
                                downsample_factor=1,
                                continuous_plot=False,
                                post_processing_fn=None,
                                index_list=None):
    import matplotlib.pyplot as plt
    if continuous_plot:
        _, ax = plt.subplots(figsize=(img_sz * count, img_sz * 4), ncols=count, nrows=4)
    else:
        _, ax = plt.subplots(figsize=(img_sz * 6, img_sz * count), ncols=6, nrows=count)

    if index_list is None:
        index_list = np.random.choice(np.arange(len(data_loader)), size=count, replace=False)
    print(index_list)
    with torch.no_grad():
        for i, idx in tqdm(enumerate(index_list)):
            data = data_loader[idx]
            data = [torch.Tensor(elem).cuda() for elem in data]
            inp, target, mask = data[:3]
            prior_inp = [elem[None, ...] for elem in data[3:]]
            output = model(inp[None, ...])
            if post_processing_fn is not None:
                output = post_processing_fn(output)

            if with_prior:
                prior = model.module._prior_model(*prior_inp)
                prior = prior.permute(1, 0, 2, 3)
                output = output * prior
            output = output.cpu().numpy()[:, 0, ...]
            target = target.cpu().numpy()
            if downsample_factor > 1:
                output = output[:, ::downsample_factor, ::downsample_factor]
                target = target[:, ::downsample_factor, ::downsample_factor]
            if continuous_plot:
                _plot_prediction_with_target_continuous(output, target, ax, i, data_loader.target_ts(idx))
            else:
                _plot_prediction_with_target_simple(output, target, ax, i, idx)


def plot_prediction_around_ts(
        model,
        data_loader,
        ts,
        window=4,
        with_prior=False,
        downsample_factor=5,
        img_sz=2,
):
    assert with_prior == False
    assert data_loader._sampling_rate == 1

    index = data_loader.get_index_from_target_ts(ts)
    start_index = max(0, index - window)
    end_index = min(len(data_loader), index + window + 1)
    index_list = list(range(start_index, end_index))
    plot_prediction_with_target(
        model,
        data_loader,
        len(index_list),
        with_prior=with_prior,
        img_sz=img_sz,
        downsample_factor=downsample_factor,
        index_list=index_list,
        continuous_plot=True,
    )
