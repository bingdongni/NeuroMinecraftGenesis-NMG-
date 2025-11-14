"""
统计分析器
=========

该模块实现了24小时实验的统计分析和显著性检验，包括：
- 配对t检验
- 组间比较分析
- 效应量计算
- 置信区间估计
- 重复测量方差分析
- 多重比较校正
- 统计功效分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, f_oneway, friedmanchisquare
from scipy.stats import normaltest, shapiro, levene, bartlett
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from statsmodels.stats.effect_size import cohen_d, hedges_g  # 移除错误的导入

import logging
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalTest(Enum):
    """统计检验类型"""
    PAIRED_T_TEST = "配对t检验"
    INDEPENDENT_T_TEST = "独立样本t检验"
    ANOVA = "方差分析"
    FRIEDMAN = "弗里德曼检验"
    MANN_WHITNEY_U = "曼-惠特尼U检验"
    WILCOXON = "威尔科克森符号秩检验"

class SignificanceLevel(Enum):
    """显著性水平"""
    ALPHA_001 = 0.001  # 极显著
    ALPHA_01 = 0.01    # 非常显著
    ALPHA_05 = 0.05    # 显著
    ALPHA_10 = 0.10    # 边缘显著

@dataclass
class StatisticalResult:
    """统计检验结果数据类"""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significance_level: str
    power: float
    interpretation: str
    recommendations: List[str]

@dataclass
class GroupComparison:
    """组间比较结果"""
    group1_name: str
    group2_name: str
    dimension: str
    result: StatisticalResult
    descriptive_stats: Dict

class StatisticalAnalyzer:
    """统计显著性分析器"""
    
    def __init__(self, alpha: float = 0.05, correction_method: str = 'holm'):
        """
        初始化统计分析器
        
        Args:
            alpha: 显著性水平
            correction_method: 多重比较校正方法 ('holm', 'bonferroni', 'fdr_bh')
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.test_history: List[Dict] = []
        
        # 效应量阈值 (Cohen's conventions)
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
        
        logger.info(f"统计分析器初始化完成 - 显著性水平: {alpha}")
    
    def _cohen_d(self, data1: np.ndarray, data2: np.ndarray, paired: bool = False) -> float:
        """
        计算Cohen's d效应量
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            paired: 是否为配对数据
            
        Returns:
            Cohen's d效应量
        """
        if paired:
            # 配对数据的效应量
            differences = data1 - data2
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            if std_diff == 0:
                return 0.0
            return mean_diff / std_diff
        else:
            # 独立样本的效应量
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            
            # 合并标准差
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            # Cohen's d
            d = (mean1 - mean2) / pooled_std
            
            # 小样本校正 (Hedges' g)
            if n1 + n2 > 2:
                correction = 1 - (3 / (4 * (n1 + n2) - 9))
                d *= correction
            
            return d
    
    def _hedges_g(self, data1: np.ndarray, data2: np.ndarray, paired: bool = False) -> float:
        """
        计算Hedges' g效应量 (Cohen's d的小样本校正版本)
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            paired: 是否为配对数据
            
        Returns:
            Hedges' g效应量
        """
        return self._cohen_d(data1, data2, paired)
    
    def _check_normality(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        检查数据正态性
        
        Args:
            data: 数据数组
            
        Returns:
            是否符合正态分布, p值
        """
        if len(data) < 3:
            return True, 1.0
        
        if len(data) <= 50:
            # 使用Shapiro-Wilk检验
            statistic, p_value = shapiro(data)
        else:
            # 使用D'Agostino正态性检验
            statistic, p_value = normaltest(data)
        
        is_normal = p_value > self.alpha
        return is_normal, p_value
    
    def _check_equal_variance(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[bool, float]:
        """
        检查方差齐性
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            
        Returns:
            方差是否相等, p值
        """
        statistic, p_value = levene(group1, group2)
        equal_variance = p_value > self.alpha
        return equal_variance, p_value
    
    def _calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray, 
                             paired: bool = False) -> float:
        """
        计算效应量
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            paired: 是否为配对数据
            
        Returns:
            效应量（Cohen's d或hedges g）
        """
        try:
            if paired:
                # 配对效应量
                effect_size = self._cohen_d(data1, data2, paired=True)
            else:
                # 独立样本效应量
                effect_size = self._cohen_d(data1, data2, paired=False)
            
            return effect_size
        except:
            # 备用计算
            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            if pooled_std == 0:
                return 0.0
            return (np.mean(data2) - np.mean(data1)) / pooled_std
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """
        解释效应量大小
        
        Args:
            effect_size: 效应量值
            
        Returns:
            效应量解释
        """
        abs_effect = abs(effect_size)
        
        if abs_effect < self.effect_size_thresholds['small']:
            return "微小效应"
        elif abs_effect < self.effect_size_thresholds['medium']:
            return "小效应"
        elif abs_effect < self.effect_size_thresholds['large']:
            return "中等效应"
        else:
            return "大效应"
    
    def _calculate_power(self, effect_size: float, n1: int, n2: int, 
                        paired: bool = False) -> float:
        """
        计算统计功效
        
        Args:
            effect_size: 效应量
            n1: 第一组样本量
            n2: 第二组样本量
            paired: 是否为配对数据
            
        Returns:
            统计功效
        """
        try:
            if paired:
                # 配对t检验的功效分析
                df = n1 - 1
                ncp = effect_size * np.sqrt(n1)  # 非中心参数
            else:
                # 独立样本t检验的功效分析
                df = n1 + n2 - 2
                ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
            
            # 简化的功效计算
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            power = 1 - stats.t.cdf(t_critical - ncp, df)
            
            return min(1.0, max(0.0, power))
            
        except:
            return 0.5  # 默认功效
    
    def _interpret_significance(self, p_value: float, alpha: float) -> str:
        """
        解释显著性水平
        
        Args:
            p_value: p值
            alpha: 显著性水平
            
        Returns:
            显著性解释
        """
        if p_value < 0.001:
            return "极显著 (p < 0.001)"
        elif p_value < 0.01:
            return "非常显著 (p < 0.01)"
        elif p_value < alpha:
            return f"显著 (p = {p_value:.4f})"
        elif p_value < 0.10:
            return f"边缘显著 (p = {p_value:.4f})"
        else:
            return f"不显著 (p = {p_value:.4f})"
    
    def paired_t_test(self, data1: np.ndarray, data2: np.ndarray,
                     dimension_name: str = "未知维度") -> StatisticalResult:
        """
        执行配对t检验
        
        Args:
            data1: 第一组数据（基线或对照组）
            data2: 第二组数据（实验组）
            dimension_name: 维度名称
            
        Returns:
            统计检验结果
        """
        if len(data1) != len(data2):
            raise ValueError("配对t检验要求两组数据样本量相等")
        
        if len(data1) < 3:
            raise ValueError("配对t检验要求至少3对数据")
        
        # 数据清理
        data1 = np.array(data1)
        data2 = np.array(data2)
        
        # 检查正态性（差值）
        differences = data2 - data1
        is_normal, normality_p = self._check_normality(differences)
        
        # 执行配对t检验
        statistic, p_value = ttest_rel(data2, data1)
        
        # 计算效应量
        effect_size = self._calculate_effect_size(data1, data2, paired=True)
        
        # 计算置信区间
        n = len(data1)
        se = stats.sem(differences)
        h = se * stats.t.ppf((1 + self.alpha) / 2, n - 1)
        mean_diff = np.mean(differences)
        ci_lower = mean_diff - h
        ci_upper = mean_diff + h
        
        # 计算统计功效
        power = self._calculate_power(abs(effect_size), n, n, paired=True)
        
        # 解释结果
        significance_interpretation = self._interpret_significance(p_value, self.alpha)
        effect_interpretation = self._interpret_effect_size(effect_size)
        
        interpretation = f"配对t检验结果显示两组数据{'存在' if p_value < self.alpha else '不存在'}显著差异。"
        interpretation += f"效应量为{effect_interpretation}。"
        
        # 生成建议
        recommendations = []
        if p_value < self.alpha:
            recommendations.append("差异具有统计学意义")
            if effect_size > self.effect_size_thresholds['medium']:
                recommendations.append("效应量达到中等到大的水平，具有实际意义")
        else:
            recommendations.append("差异不具有统计学意义")
            if power < 0.8:
                recommendations.append("统计功效偏低，建议增加样本量")
        
        if not is_normal:
            recommendations.append("数据不符合正态分布，建议使用非参数检验")
        
        # 记录检验历史
        self.test_history.append({
            'timestamp': datetime.now(),
            'test_type': StatisticalTest.PAIRED_T_TEST.value,
            'dimension': dimension_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'sample_size': len(data1)
        })
        
        result = StatisticalResult(
            test_type=StatisticalTest.PAIRED_T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=significance_interpretation,
            power=power,
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        logger.info(f"配对t检验完成 - 维度: {dimension_name}, p值: {p_value:.4f}, 效应量: {effect_size:.3f}")
        
        return result
    
    def independent_t_test(self, group1: np.ndarray, group2: np.ndarray,
                          dimension_name: str = "未知维度") -> StatisticalResult:
        """
        执行独立样本t检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            dimension_name: 维度名称
            
        Returns:
            统计检验结果
        """
        if len(group1) < 3 or len(group2) < 3:
            raise ValueError("独立样本t检验要求每组至少3个样本")
        
        # 数据清理
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        # 检查正态性
        normal1, norm_p1 = self._check_normality(group1)
        normal2, norm_p2 = self._check_normality(group2)
        
        # 检查方差齐性
        equal_var, levene_p = self._check_equal_variance(group1, group2)
        
        # 执行独立样本t检验
        statistic, p_value = ttest_ind(group2, group1, equal_var=equal_var)
        
        # 计算效应量
        effect_size = self._calculate_effect_size(group1, group2, paired=False)
        
        # 计算置信区间
        mean1, mean2 = np.mean(group1), np.mean(group2)
        mean_diff = mean2 - mean1
        
        # 使用Welch t检验的置信区间（如果方差不齐）
        if equal_var:
            se = np.sqrt(np.var(group1)/len(group1) + np.var(group2)/len(group2))
            df = len(group1) + len(group2) - 2
        else:
            se = np.sqrt(np.var(group1)/len(group1) + np.var(group2)/len(group2))
            # Welch自由度
            s1_sq = np.var(group1) / len(group1)
            s2_sq = np.var(group2) / len(group2)
            df = (s1_sq + s2_sq)**2 / (s1_sq**2/(len(group1)-1) + s2_sq**2/(len(group2)-1))
        
        h = se * stats.t.ppf((1 + self.alpha) / 2, df)
        ci_lower = mean_diff - h
        ci_upper = mean_diff + h
        
        # 计算统计功效
        power = self._calculate_power(abs(effect_size), len(group1), len(group2), paired=False)
        
        # 解释结果
        significance_interpretation = self._interpret_significance(p_value, self.alpha)
        effect_interpretation = self._interpret_effect_size(effect_size)
        
        interpretation = f"独立样本t检验显示两组数据{'存在' if p_value < self.alpha else '不存在'}显著差异。"
        interpretation += f"效应量为{effect_interpretation}。"
        
        # 生成建议
        recommendations = []
        if p_value < self.alpha:
            recommendations.append("组间差异具有统计学意义")
        else:
            recommendations.append("组间差异不具有统计学意义")
        
        if not normal1 or not normal2:
            recommendations.append("部分数据不符合正态分布，建议使用非参数检验")
        
        if not equal_var:
            recommendations.append("方差不齐，使用了Welch t检验")
        
        # 记录检验历史
        self.test_history.append({
            'timestamp': datetime.now(),
            'test_type': StatisticalTest.INDEPENDENT_T_TEST.value,
            'dimension': dimension_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'sample_size': len(group1) + len(group2)
        })
        
        result = StatisticalResult(
            test_type=StatisticalTest.INDEPENDENT_T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=significance_interpretation,
            power=power,
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        logger.info(f"独立样本t检验完成 - 维度: {dimension_name}, p值: {p_value:.4f}, 效应量: {effect_size:.3f}")
        
        return result
    
    def anova_analysis(self, groups: Dict[str, np.ndarray], 
                      dimension_name: str = "未知维度") -> Tuple[StatisticalResult, List[GroupComparison]]:
        """
        执行单因素方差分析
        
        Args:
            groups: 组字典 {'组名': 数据数组}
            dimension_name: 维度名称
            
        Returns:
            方差分析结果和事后比较结果
        """
        # 准备数据
        group_data = list(groups.values())
        group_names = list(groups.keys())
        
        # 检查样本量
        for data in group_data:
            if len(data) < 3:
                raise ValueError("方差分析要求每组至少3个样本")
        
        # 执行单因素方差分析
        f_statistic, p_value = f_oneway(*group_data)
        
        # 计算效应量 (eta squared)
        total_ss = 0
        grand_mean = np.mean(np.concatenate(group_data))
        
        for data in group_data:
            total_ss += np.sum((data - grand_mean) ** 2)
        
        between_ss = 0
        for data in group_data:
            group_mean = np.mean(data)
            between_ss += len(data) * (group_mean - grand_mean) ** 2
        
        eta_squared = between_ss / total_ss
        
        # 计算自由度
        df_between = len(groups) - 1
        df_within = sum(len(data) for data in group_data) - len(groups)
        df_total = len(groups) - 1 + sum(len(data) for data in group_data) - len(groups)
        
        # 计算置信区间（近似）
        mse = (total_ss - between_ss) / df_within if df_within > 0 else 0
        se = np.sqrt(mse / max(len(data) for data in group_data))
        ci_width = stats.t.ppf((1 + self.alpha) / 2, df_within) * se
        ci_lower = eta_squared - ci_width
        ci_upper = eta_squared + ci_width
        
        # 解释结果
        significance_interpretation = self._interpret_significance(p_value, self.alpha)
        effect_interpretation = self._interpret_effect_size(np.sqrt(eta_squared))
        
        interpretation = f"单因素方差分析显示组间{'存在' if p_value < self.alpha else '不存在'}显著差异。"
        interpretation += f"效应量为{effect_interpretation}。"
        
        # 生成建议
        recommendations = []
        if p_value < self.alpha:
            recommendations.append("组间差异具有统计学意义，建议进行事后比较")
            
            # 执行事后比较（Tukey HSD）
            try:
                flat_data = np.concatenate(group_data)
                flat_groups = np.concatenate([np.full(len(data), i) for i, data in enumerate(group_data)])
                tukey_result = pairwise_tukeyhsd(flat_data, flat_groups, alpha=self.alpha)
                
                post_hoc_results = []
                for comparison in tukey_results.groups_extrap:
                    if comparison.reject:  # 如果拒绝原假设
                        i, j = comparison.groups
                        post_hoc_results.append({
                            'group1': group_names[i],
                            'group2': group_names[j],
                            'significant': True,
                            'p_value': comparison.pvalue
                        })
                
                recommendations.append(f"事后比较发现 {len(post_hoc_results)} 组间存在显著差异")
                
            except:
                recommendations.append("事后比较执行失败")
        else:
            recommendations.append("组间差异不具有统计学意义")
        
        # 记录检验历史
        self.test_history.append({
            'timestamp': datetime.now(),
            'test_type': StatisticalTest.ANOVA.value,
            'dimension': dimension_name,
            'statistic': f_statistic,
            'p_value': p_value,
            'effect_size': eta_squared,
            'sample_size': sum(len(data) for data in group_data),
            'groups': group_names
        })
        
        result = StatisticalResult(
            test_type=StatisticalTest.ANOVA,
            statistic=f_statistic,
            p_value=p_value,
            effect_size=eta_squared,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=significance_interpretation,
            power=0.8,  # 近似值
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        # 生成组间比较结果
        comparisons = []
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                comparison_result = self.independent_t_test(groups[group1_name], groups[group2_name], dimension_name)
                
                descriptive_stats = {
                    f'{group1_name}_mean': np.mean(groups[group1_name]),
                    f'{group1_name}_std': np.std(groups[group1_name]),
                    f'{group1_name}_n': len(groups[group1_name]),
                    f'{group2_name}_mean': np.mean(groups[group2_name]),
                    f'{group2_name}_std': np.std(groups[group2_name]),
                    f'{group2_name}_n': len(groups[group2_name])
                }
                
                comparison = GroupComparison(
                    group1_name=group1_name,
                    group2_name=group2_name,
                    dimension=dimension_name,
                    result=comparison_result,
                    descriptive_stats=descriptive_stats
                )
                comparisons.append(comparison)
        
        logger.info(f"方差分析完成 - 维度: {dimension_name}, F值: {f_statistic:.4f}, p值: {p_value:.4f}")
        
        return result, comparisons
    
    def analyze_repeated_measures(self, all_data: List[np.ndarray],
                                 time_points: List[str],
                                 dimension_name: str = "未知维度") -> StatisticalResult:
        """
        分析重复测量数据（使用Friedman检验）
        
        Args:
            all_data: 各时间点的数据列表
            time_points: 时间点标签
            dimension_name: 维度名称
            
        Returns:
            统计检验结果
        """
        # 检查数据
        n_timepoints = len(all_data)
        sample_sizes = [len(data) for data in all_data]
        
        if n_timepoints < 3:
            raise ValueError("重复测量分析需要至少3个时间点")
        
        if len(set(sample_sizes)) > 1:
            logger.warning("各时间点样本量不等，将使用最小样本量")
            min_sample = min(sample_sizes)
            all_data = [data[:min_sample] for data in all_data]
        
        # 使用Friedman检验（非参数重复测量方差分析）
        statistic, p_value = friedmanchisquare(*all_data)
        
        # 计算效应量 ( Kendall's W)
        n_subjects = len(all_data[0])
        k_timepoints = len(all_data)
        kendall_w = statistic / (n_subjects * (k_timepoints - 1))
        
        # 计算置信区间
        ci_width = 1.96 * np.sqrt((1 - kendall_w) / (n_subjects - 1))
        ci_lower = max(0, kendall_w - ci_width)
        ci_upper = min(1, kendall_w + ci_width)
        
        # 解释结果
        significance_interpretation = self._interpret_significance(p_value, self.alpha)
        
        interpretation = f"重复测量分析显示时间点间{'存在' if p_value < self.alpha else '不存在'}显著差异。"
        
        # 生成建议
        recommendations = []
        if p_value < self.alpha:
            recommendations.append("时间点间差异具有统计学意义")
            recommendations.append("建议进行两两时间点比较")
        else:
            recommendations.append("时间点间差异不具有统计学意义")
        
        # 记录检验历史
        self.test_history.append({
            'timestamp': datetime.now(),
            'test_type': StatisticalTest.FRIEDMAN.value,
            'dimension': dimension_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': kendall_w,
            'sample_size': n_subjects,
            'timepoints': time_points
        })
        
        result = StatisticalResult(
            test_type=StatisticalTest.FRIEDMAN,
            statistic=statistic,
            p_value=p_value,
            effect_size=kendall_w,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=significance_interpretation,
            power=0.8,  # 近似值
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        logger.info(f"重复测量分析完成 - 维度: {dimension_name}, χ²值: {statistic:.4f}, p值: {p_value:.4f}")
        
        return result
    
    def correct_multiple_comparisons(self, p_values: List[float]) -> List[float]:
        """
        多重比较校正
        
        Args:
            p_values: 原始p值列表
            
        Returns:
            校正后的p值列表
        """
        from statsmodels.stats.multitest import multipletests
        
        corrected_p = multipletests(p_values, alpha=self.alpha, method=self.correction_method)[1]
        
        logger.info(f"多重比较校正完成 - 方法: {self.correction_method}")
        return corrected_p
    
    def generate_comprehensive_report(self, experiment_data: Dict[str, Dict[str, List[float]]],
                                    group_names: List[str]) -> Dict:
        """
        生成综合统计报告
        
        Args:
            experiment_data: 实验数据 {'维度': {'组名': [分数列表]}}
            group_names: 组名列表
            
        Returns:
            综合报告字典
        """
        report = {
            'report_metadata': {
                'timestamp': datetime.now().isoformat(),
                'groups_analyzed': group_names,
                'dimensions_analyzed': list(experiment_data.keys()),
                'alpha_level': self.alpha,
                'correction_method': self.correction_method
            },
            'descriptive_statistics': {},
            'inferential_statistics': {},
            'effect_sizes': {},
            'overall_conclusions': []
        }
        
        # 计算描述性统计
        for dimension in experiment_data:
            report['descriptive_statistics'][dimension] = {}
            
            for group_name in group_names:
                if group_name in experiment_data[dimension]:
                    data = np.array(experiment_data[dimension][group_name])
                    report['descriptive_statistics'][dimension][group_name] = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'min': np.min(data),
                        'max': np.max(data),
                        'median': np.median(data),
                        'q25': np.percentile(data, 25),
                        'q75': np.percentile(data, 75),
                        'n': len(data)
                    }
        
        # 进行统计检验
        for dimension in experiment_data:
            try:
                groups_data = {name: experiment_data[dimension][name] 
                             for name in group_names if name in experiment_data[dimension]}
                
                if len(groups_data) >= 2:
                    # 如果是配对设计（数据长度相等）
                    data_lengths = [len(data) for data in groups_data.values()]
                    if len(set(data_lengths)) == 1:
                        # 配对t检验
                        report['inferential_statistics'][dimension] = {}
                        first_group = list(groups_data.keys())[0]
                        test_result = self.paired_t_test(
                            groups_data[first_group],
                            groups_data[group_names[1]],  # 假设第二组为实验组
                            dimension
                        )
                        report['ininfer_statistics'][dimension]['paired_t_test'] = {
                            'statistic': test_result.statistic,
                            'p_value': test_result.p_value,
                            'effect_size': test_result.effect_size,
                            'interpretation': test_result.interpretation
                        }
                        
                        report['effect_sizes'][dimension] = {
                            'cohen_d': test_result.effect_size,
                            'interpretation': self._interpret_effect_size(test_result.effect_size)
                        }
                    else:
                        # 方差分析
                        anova_result, comparisons = self.anova_analysis(groups_data, dimension)
                        report['inferential_statistics'][dimension]['anova'] = {
                            'f_statistic': anova_result.statistic,
                            'p_value': anova_result.p_value,
                            'effect_size': anova_result.effect_size,
                            'interpretation': anova_result.interpretation
                        }
                        
                        report['effect_sizes'][dimension] = {
                            'eta_squared': anova_result.effect_size,
                            'interpretation': self._interpret_effect_size(np.sqrt(anova_result.effect_size))
                        }
                        
                        # 添加显著比较
                        report['inferential_statistics'][dimension]['significant_comparisons'] = []
                        for comp in comparisons:
                            if comp.result.p_value < self.alpha:
                                report['inferential_statistics'][dimension]['significant_comparisons'].append({
                                    'group1': comp.group1_name,
                                    'group2': comp.group2_name,
                                    'p_value': comp.result.p_value,
                                    'effect_size': comp.result.effect_size
                                })
            
            except Exception as e:
                logger.error(f"维度 {dimension} 统计分析失败: {e}")
                report['inferential_statistics'][dimension] = {'error': str(e)}
        
        # 生成总体结论
        significant_dimensions = []
        large_effects = []
        
        for dim, stats_data in report['inferential_statistics'].items():
            if 'anova' in stats_data and stats_data['anova']['p_value'] < self.alpha:
                significant_dimensions.append(dim)
            elif 'paired_t_test' in stats_data and stats_data['paired_t_test']['p_value'] < self.alpha:
                significant_dimensions.append(dim)
        
        for dim, effect_data in report['effect_sizes'].items():
            if effect_data['interpretation'] in ['中等效应', '大效应']:
                large_effects.append(dim)
        
        report['overall_conclusions'] = [
            f"共分析 {len(experiment_data)} 个认知维度",
            f"发现 {len(significant_dimensions)} 个维度存在统计学显著差异",
            f"效应量达到中等以上水平的维度有 {len(large_effects)} 个"
        ]
        
        if significant_dimensions:
            report['overall_conclusions'].append(f"主要显著维度: {', '.join(significant_dimensions)}")
        
        return report
    
    def export_results(self, results: Dict, filepath: str) -> bool:
        """
        导出统计结果
        
        Args:
            results: 统计结果字典
            filepath: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"统计结果已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出统计结果失败: {e}")
            return False
    
    def get_test_history(self) -> List[Dict]:
        """获取检验历史记录"""
        return self.test_history.copy()
    
    def clear_history(self) -> None:
        """清除检验历史"""
        self.test_history.clear()
        logger.info("统计检验历史已清除")

if __name__ == "__main__":
    # 测试统计分析器
    np.random.seed(42)
    
    # 创建模拟数据
    baseline_group = np.random.normal(50, 10, 50)  # 基线组
    single_optimization = np.random.normal(60, 10, 50)  # 单维优化组
    multi_optimization = np.random.normal(65, 10, 50)  # 六维协同组
    
    # 测试统计分析器
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    print("=== 测试配对t检验 ===")
    paired_result = analyzer.paired_t_test(baseline_group[:20], single_optimization[:20], "记忆力")
    print(f"统计量: {paired_result.statistic:.4f}")
    print(f"p值: {paired_result.p_value:.4f}")
    print(f"效应量: {paired_result.effect_size:.3f}")
    print(f"显著性: {paired_result.significance_level}")
    print(f"解释: {paired_result.interpretation}")
    
    print("\n=== 测试独立样本t检验 ===")
    t_result = analyzer.independent_t_test(baseline_group, multi_optimization, "创造力")
    print(f"统计量: {t_result.statistic:.4f}")
    print(f"p值: {t_result.p_value:.4f}")
    print(f"置信区间: {t_result.confidence_interval}")
    
    print("\n=== 测试方差分析 ===")
    groups = {
        '基线组': baseline_group,
        '单维优化组': single_optimization,
        '六维协同组': multi_optimization
    }
    
    anova_result, comparisons = analyzer.anova_analysis(groups, "观察力")
    print(f"F统计量: {anova_result.statistic:.4f}")
    print(f"p值: {anova_result.p_value:.4f}")
    print(f"效应量: {anova_result.effect_size:.3f}")
    print(f"显著性比较数量: {len([c for c in comparisons if c.result.p_value < 0.05])}")
    
    print("\n=== 生成综合报告 ===")
    experiment_data = {
        '记忆力': {
            '基线组': baseline_group.tolist(),
            '单维优化组': single_optimization.tolist(),
            '六维协同组': multi_optimization.tolist()
        },
        '思维力': {
            '基线组': (baseline_group + np.random.normal(2, 3, 50)).tolist(),
            '单维优化组': (single_optimization + np.random.normal(2, 3, 50)).tolist(),
            '六维协同组': (multi_optimization + np.random.normal(2, 3, 50)).tolist()
        }
    }
    
    report = analyzer.generate_comprehensive_report(
        experiment_data, 
        ['基线组', '单维优化组', '六维协同组']
    )
    
    print(f"分析的维度数: {len(report['report_metadata']['dimensions_analyzed'])}")
    print(f"总体结论: {report['overall_conclusions']}")
    
    # 导出结果
    analyzer.export_results(report, "test_statistical_analysis.json")
    print("\n统计结果已导出")
    
    print("统计分析器测试完成")