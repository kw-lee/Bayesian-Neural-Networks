"""utils for bayesian network
"""

import scipy.stats as _stats
import matplotlib.pyplot as _plt

def plot_sample_func(
    x_grid, meanftn, sdftn, _outputs, _sigmas, title, sd=1.0, 
    train_data=None, show=True, path=None
):
    # true 
    y_grid_u = meanftn(x_grid) + _stats.norm.ppf(0.975) * sdftn(x_grid) * sd
    y_grid_l = meanftn(x_grid) + _stats.norm.ppf(0.025) * sdftn(x_grid) * sd
    y_grid = meanftn(x_grid)

    # sampled function
    samp_size = _outputs.shape[0]
    for i in range(samp_size):
        _plt.plot(x_grid[:, 0], _outputs[i, :].reshape(-1), c='gray', alpha=0.5)
        _plt.fill_between(
            x_grid[:, 0], 
            _outputs[i, :].reshape(-1) + _stats.norm.ppf(0.025) * _sigmas[i, :].reshape(-1), 
            _outputs[i, :].reshape(-1) + _stats.norm.ppf(0.975) * _sigmas[i, :].reshape(-1), 
            alpha=0.1, 
            color='gray')

    # train_data
    if train_data is not None:
        _plt.scatter(train_data[0], train_data[1], c='red', alpha=0.5, label="training data")
    _plt.plot(x_grid[:, 0], y_grid, label="true mean")
    _plt.fill_between(x_grid[:, 0], y_grid_l, y_grid_u, alpha=0.5, label="true 95% PI")
    handles, labels = _plt.gca().get_legend_handles_labels()
    handles.append([
        _plt.plot([0], [0], color='gray', label='sampled mean'),
        _plt.fill_between([0], [0], [0], color='gray', label='sampled CI')
    ])

    _plt.legend(loc='upper right')
    _plt.title(title)
    if show:
        _plt.show()
    if path is not None:
        _plt.savefig(path)
    _plt.clf()

# def plot_expected_func(
#     x_grid, meanftn, sdftn, _outputs, _sigmas, title, sd=1.0, show=True, path=None
# ):
#     y_grid_u = meanftn(x_grid) + _stats.norm.ppf(0.975) * sdftn(x_grid) * sd
#     y_grid_l = meanftn(x_grid) + _stats.norm.ppf(0.025) * sdftn(x_grid) * sd
#     y_grid = meanftn(x_grid)
#     pred_mean = _outputs.mean(1)
#     alea_std = _np.nanmean(nan_if(_sigmas, _np.inf), 1)
#     epis_std = _outputs.std(1) + _np.nanstd(nan_if(_sigmas, _np.inf), 1)
#     conf_grid_u = pred_mean + _stats.norm.ppf(0.975) * alea_std
#     conf_grid_l = pred_mean + _stats.norm.ppf(0.025) * alea_std
#     pred_grid_u = pred_mean + _stats.norm.ppf(0.975) * (alea_std + epis_std)
#     pred_grid_l = pred_mean + _stats.norm.ppf(0.025) * (alea_std + epis_std)
#     pred_grid = pred_mean
#     _plt.plot(x_grid[:, 0], y_grid, label="true mean")
#     _plt.fill_between(x_grid[:, 0], y_grid_l, y_grid_u, alpha=0.5, label="true 95% PI")
#     _plt.plot(x_grid[:, 0], pred_grid, label="predicted mean")
#     _plt.fill_between(x_grid[:, 0], conf_grid_l, conf_grid_u, alpha=0.5, label="predicted 95% CI")
#     _plt.fill_between(x_grid[:, 0], pred_grid_l, pred_grid_u, alpha=0.5, label="predicted 95% PI")
#     _plt.legend(loc='upper right')
#     _plt.title(title)
#     if show:
#         _plt.show()
#     if path is not None:
#         _plt.savefig(path)
#     _plt.clf()
