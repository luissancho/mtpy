import math
import numpy as np
import pandas as pd

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy import stats
from sklearn.cluster import estimate_bandwidth, KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from xgboost.sklearn import XGBClassifier, XGBRegressor

from .helpers import to_pandas


def pca(x, n=None, svd_solver='full', model=None, seed=None, **kwargs):
    index = x.index
    x = np.array(x)

    if not model:
        model = PCA(
            n_components=n,
            svd_solver=svd_solver,
            random_state=seed,
            **kwargs
        )
        d = model.fit_transform(x)
    else:
        d = model.transform(x)

    d = pd.DataFrame(d, index=index)
    exp_var = model.explained_variance_ratio_

    return d, exp_var, model


def umap(x, n=2, n_neighbors=15, min_dist=0.1, metric='euclidean', init='spectral', densmap=False, seed=None, **kwargs):
    from umap import UMAP  # Lazy load library due to slow importing

    if not isinstance(init, (np.ndarray, str)):
        init = np.array(init)

    index = x.index
    x = np.array(x)

    model = UMAP(
        n_components=n,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        init=init,
        densmap=densmap,
        random_state=seed,
        transform_seed=seed,
        **kwargs
    )

    d = model.fit_transform(x)
    d = pd.DataFrame(d, index=index)

    return d, model


def tsne(x, n=2, perplexity=30, lr=200, epochs=1000, init='random', seed=None, **kwargs):
    from sklearn.manifold import TSNE  # Lazy load library due to slow importing

    if not isinstance(init, (np.ndarray, str)):
        init = np.array(init)

    index = x.index
    x = np.array(x)

    model = TSNE(
        n_components=n,
        perplexity=perplexity,
        learning_rate=lr,
        n_iter=epochs,
        init=init,
        n_jobs=-1,
        random_state=seed,
        **kwargs
    )

    d = model.fit_transform(x)
    d = pd.DataFrame(d, index=index)

    return d, model


def som(x, size=None, sigma=1.0, lr=0.01, epochs=None, pca_init=False, verbose=False, seed=None, **kwargs):
    from minisom import MiniSom  # Lazy load library due to slow importing

    size = size or int(math.ceil(math.sqrt(5 * math.sqrt(x.shape[0]))))
    sigma = sigma or size / 2
    epochs = epochs or x.shape[0] * 10

    index = x.index
    x = np.array(x)

    model = MiniSom(
        size,
        size,
        x.shape[1],
        sigma=sigma,
        learning_rate=lr,
        random_seed=seed,
        **kwargs
    )

    if pca_init:
        model.pca_weights_init(x)
    else:
        model.random_weights_init(x)

    model.train_batch(x, epochs, verbose=verbose)

    d = [model.winner(r) for r in x]
    d = pd.DataFrame(d, index=index)

    return d, model


def kmeans(x, n_clusters=8, seed=None, **kwargs):
    index = x.index
    columns = x.columns
    x = np.array(x)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        **kwargs
    ).fit(x)

    c = pd.Series(model.labels_, index=index, name='c')
    p = pd.DataFrame(model.cluster_centers_, columns=columns, index=sorted(set(model.labels_)))

    return c, p, model


def mshift(x, quantile=0.3, n_samples=None, bin_seeding=False, cluster_all=True, seed=None, **kwargs):
    index = x.index
    columns = x.columns
    x = np.array(x)

    bandwidth = estimate_bandwidth(
        x,
        quantile=quantile,
        n_samples=n_samples,
        random_state=seed,
        n_jobs=-1
    )

    model = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=bin_seeding,
        cluster_all=cluster_all,
        n_jobs=-1,
        **kwargs
    ).fit(x)

    p_labels = sorted(set(model.labels_))
    if -1 in p_labels:
        p_labels = p_labels[1:]

    c = pd.Series(model.labels_, index=index, name='c')
    p = pd.DataFrame(model.cluster_centers_, columns=columns, index=p_labels)

    return c, p, model


def hdbscan(x, min_cluster_size=5, min_samples=None, gen_min_span_tree=False, **kwargs):
    from hdbscan import HDBSCAN  # Lazy load library due to slow importing
    
    index = x.index
    x = np.array(x)

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=gen_min_span_tree,
        core_dist_n_jobs=-1,
        **kwargs
    ).fit(x)

    c = pd.Series(model.labels_, index=index, name='c')

    return c, model


def clusters_silhouette(x, c):
    # Exclude outliers
    x_val = x[c >= 0]
    c_val = c[c >= 0]

    score = metrics.silhouette_score(x_val, c_val)
    samples = metrics.silhouette_samples(x, c)

    return score, samples


def shap_values(x, model):
    from shap import TreeExplainer  # Lazy load library due to slow importing

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    return shap_values


def feature_correlations(x, y=None, method='pearson'):
    d = to_pandas(x, pdtype='DataFrame')

    if y is not None:
        if not isinstance(y, str):
            d['y'] = to_pandas(y, pdtype='Series')
            y = 'y'

        corr = d.corr(method=method)[y].drop(index=[y])
    else:
        corr = d.corr(method=method)

    return corr


def feature_stat_test(x, y):
    """
    Perform Univariate T-test / ANOVA Analysis on each column (null hypothesis: all samples with identical average).
    """

    if isinstance(y, str):
        y = x[y]
        x = x.drop(columns=[y])

    x = to_pandas(x, pdtype='DataFrame')
    y = to_pandas(y, pdtype='Series')

    names = list(x.columns)
    splits = sorted(y.unique())
    n_samples = len(splits)

    result = pd.Series(index=names, dtype=float)

    if n_samples < 2:
        return result

    for name in names:
        samples = [
            x[y == s][name].dropna().values
            for s in splits
        ]

        if n_samples > 2:
            _, p_value = stats.f_oneway(*samples)
        else:
            _, p_value = stats.ttest_ind(*samples)

        result.loc[name] = p_value

    return result


def feature_regression(x, y, conf_int=0.95):
    """
    Fit a Multivariate Regression Model (null hypothesis: feature independent of target, does not affect outcome).
    """

    if isinstance(y, str):
        y = x[y]
        x = x.drop(columns=[y])

    x = to_pandas(x, pdtype='DataFrame')
    y = to_pandas(y, pdtype='Series')

    names = list(x.columns)
    splits = sorted(y.unique())
    n_samples = len(splits)

    result = pd.DataFrame(columns=['p_value', 'odds_ratio', 'or_lo', 'or_up'], index=names, dtype=float)

    if n_samples < 2:
        return result

    x_ = sm.add_constant(x.values)
    y_ = y.values
    if n_samples > 2:
        reg = sm.OLS(y_, x_, missing='drop').fit()
    else:
        reg = sm.GLM(y_, x_, missing='drop', family=sm.families.Binomial()).fit()

    result['p_value'] = reg.pvalues[1:]
    result['odds_ratio'] = np.exp(reg.params[1:])
    result[['or_lo', 'or_up']] = np.exp(reg.conf_int(alpha=(1 - conf_int))[1:])

    return result


def feature_collinearity(x, thr=None):
    features = list(x.columns)
    corr = x.corr()

    w, v = np.linalg.eig(corr)
    det = np.linalg.det(corr)
    cond = np.linalg.cond(corr)

    w = pd.Series(w, index=features)
    v = pd.DataFrame(v, columns=features, index=features)

    if thr is not None:
        w = w.loc[abs(w) < thr]
        v = v.where(v.abs() > thr, np.nan)[w.index]

    return w, v, det, cond


def feature_vif(x, thr=None):
    d = pd.Series([VIF(x.values, i) for i in range(x.shape[1])], index=x.columns)

    if thr is not None:
        d = d.loc[d > thr]

    return d


def cv_model_train(x, y, cv=5, estimator=None, scoring=None, fit_params=None, n_jobs=-1, verbose=0, seed=None, **kwargs):
    if estimator is None:
        xgb_params = dict(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            verbosity=0,
            n_jobs=n_jobs,
            random_state=seed
        )

        if len(y.unique()) > 2:
            xgb_params['eval_metric'] = 'rmse'

            estimator = XGBRegressor(**xgb_params)
            scoring = scoring or 'neg_root_mean_squared_error'
        else:
            xgb_params['eval_metric'] = 'auc'

            yt = len(y[y == 1])
            yf = len(y[y == 0])
            if yt < 0.2 * len(y):
                xgb_params['scale_pos_weight'] = float(round(yf / yt))

            estimator = XGBClassifier(**xgb_params)
            scoring = scoring or 'roc_auc'

        fit_params = fit_params or dict(
            verbose=False
        )

    cv_model = cross_validate(
        estimator,
        x, y,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        return_train_score=True,
        return_estimator=True,
        fit_params=fit_params,
        **kwargs
    )

    return cv_model


def sfs(x, y, cv=None, estimator=None, scoring=None, fit_params=None, n_jobs=-1, verbose=0, seed=None, **kwargs):
    if estimator is None:
        xgb_params = dict(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            verbosity=0,
            n_jobs=n_jobs,
            random_state=seed
        )

        if len(y.unique()) > 2:
            xgb_params['eval_metric'] = 'rmse'

            estimator = XGBRegressor(**xgb_params)
            scoring = scoring or 'neg_root_mean_squared_error'
        else:
            xgb_params['eval_metric'] = 'auc'

            yt = len(y[y == 1])
            yf = len(y[y == 0])
            if yt < 0.2 * len(y):
                xgb_params['scale_pos_weight'] = float(round(yf / yt))

            estimator = XGBClassifier(**xgb_params)
            scoring = scoring or 'roc_auc'

        fit_params = fit_params or dict(
            verbose=False
        )
    elif fit_params is None:
        fit_params = {}

    sfs_model = SFS(
        estimator,
        k_features='parsimonious',  # best or parsimonious
        forward=True,
        floating=False,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
        **kwargs
    ).fit(
        x, y,
        **fit_params
    )

    return sfs_model


def sfs_select(sfs_model):
    dmetric = pd.DataFrame.from_dict(sfs_model.get_metric_dict()).T.astype({
        'avg_score': float,
        'ci_bound': float,
        'std_dev': float,
        'std_err': float
    })

    names = dmetric.feature_names.values
    flist = [
        names[0][0] if len(names[0]) == 1 else None
    ]
    for i in np.arange(1, len(names)):
        flist.append(
            np.setdiff1d(names[i], names[i - 1])[0]
        )

    dmetric['feature'] = flist

    k_par = k_max = np.argmax(dmetric.avg_score)
    k_max_score, k_max_std, k_max_cv = dmetric[['avg_score', 'std_dev', 'cv_scores']].iloc[k_max]

    for k in np.arange(k_max):
        if dmetric.avg_score.iloc[k] >= k_max_score - (k_max_std / len(k_max_cv)):
            k_par = k
            break

    dmetric['optimal'] = [k <= k_par for k in np.arange(dmetric.shape[0])]
    dmetric['best'] = [k <= k_max for k in np.arange(dmetric.shape[0])]

    return dmetric
