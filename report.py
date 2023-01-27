import pandas as pd
import numpy as np
import abc
# import utils
import config as cfg
from itertools import product
from metric_builder import Metric, CalculateMetric
from stattests import TTestFromStats, UTest, PropTestFromStats, Statistics, calculate_statistics, calculate_linearization
from collections import namedtuple


class Report:
    def __init__(self, report):
        self.report = report


class BuildMetricReport:
    def __call__(self, calculated_metric, metric_items) -> Report:
        cfg.logger.info(f"{metric_items.name}")

        ratio_metrics_type = ['ratio', 'continuous']
        prop_metrics_type = ['proportion']

        ratio_estimators = ['t_test_ratio_lin', 't_test_ratio_del', 'mann_whitney_test']
        prop_estimators = ['prop_test']

        estimator_name = metric_items.estimator
        metric_type = metric_items.type

        try:
            if (estimator_name in ratio_estimators and metric_type not in ratio_metrics_type) or \
                    (estimator_name in prop_estimators and metric_type not in prop_metrics_type):
                raise ValueError(f"`{estimator_name}` does not support `{metric_type}` metrics type.")

            if estimator_name == 'mann_whitney_test':
                estimator = UTest()
                criteria_res = estimator(calculated_metric)
                stats = calculate_statistics(calculated_metric, metric_items.type)
            elif estimator_name == 't_test_ratio_lin':
                estimator = TTestFromStats()
                df_ = calculate_linearization(calculated_metric)
                stats = calculate_statistics(df_, metric_items.type)
                criteria_res = estimator(stats)
            elif estimator_name == 't_test_ratio_del':
                raise NotImplementedError
            elif estimator_name == 'prop_test':
                estimator = PropTestFromStats()
                stats = calculate_statistics(calculated_metric, metric_items.type)
                criteria_res = estimator(stats)
            else:
                raise ValueError(f"`BuildMetricReport` supports only estimators in `{ratio_estimators + prop_estimators}`, "
                                 f"got `{estimator_name}`.")

            delta = stats.mean_1 - stats.mean_0
            if stats.mean_0 == 0:
                lift = None
                cfg.logger.warning('Mean value of one group equals 0, can not calculate lift.')
            else:
                lift = (stats.mean_1 - stats.mean_0) / stats.mean_0

        except Exception as e:
            cfg.logger.error(e)
            CriteriaRes = namedtuple('criteria_res', 'statistic pvalue')
            criteria_res = CriteriaRes(None, None)
            stats = Statistics()
            delta = None
            lift = None

        report_items = pd.DataFrame({
            "estimator": metric_items.estimator,
            "metric_name": metric_items.name,
            "mean_0": stats.mean_0,
            "mean_1": stats.mean_1,
            "var_0": stats.var_0,
            "var_1": stats.var_1,
            "delta": delta,
            "lift": lift,
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)


def build_experiment_report(df, metric_config):
    build_metric_report = BuildMetricReport()
    reports = []

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = build_metric_report(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)


def build_mc_report(df, metric_config, mc_config):
    # estimator | metric | %
    mc_config = mc_config[0]
    lifts_params = mc_config['lifts']
    build_metric_report = BuildMetricReport()
    reports = []

    for lift in tqdm(np.arange(lifts_params['start'], lifts_params['end'], lifts_params['by'])):
        for i in range(mc_config['iters']):
            df_ = df.copy()
            df_[cfg.VARIANT_COL] = np.random.choice(2, len(df))
            df_.loc[df_[cfg.VARIANT_COL] == 1, metric_parsed.numerator_aggregation_field] = df_.loc[df_[
                                                                                                        cfg.VARIANT_COL] == 1, metric_parsed.numerator_aggregation_field] * lift
            report = build_experiment_report(
                df=df_,
                metric_config=metric_config
            )
            report.loc[:, 'is_reject'] = report['pvalue'] < mc_config['alpha']
            report.loc[:, 'real_lift'] = lift
            report.loc[:, 'iter'] = i

            reports.append(report)

    main = pd.concat(reports)
    main = main.groupby(['estimator', 'metric_name', 'real_lift'])['is_reject'].mean().unstack().reset_index()

    return main
