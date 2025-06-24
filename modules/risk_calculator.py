"""
リスク計算モジュール
ポートフォリオのリスク指標計算機能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    ボラティリティ計算
    
    Args:
        returns: 日次リターン系列
        annualize: 年率換算するかどうか
    
    Returns:
        float: ボラティリティ（年率）
    """
    try:
        if returns.empty or returns.isna().all():
            return 0.0
        
        daily_vol = returns.std()
        
        if annualize:
            return daily_vol * np.sqrt(252)  # 年間営業日数で換算
        return daily_vol
        
    except Exception as e:
        logger.error(f"ボラティリティ計算エラー: {str(e)}")
        return 0.0


def calculate_portfolio_risk(
    returns: pd.DataFrame, 
    weights: np.ndarray
) -> Dict[str, any]:
    """
    ポートフォリオ全体のリスク計算（日次ベース）
    
    Args:
        returns: 各銘柄の日次リターンDataFrame
        weights: 各銘柄の重み（時価総額比率）
    
    Returns:
        dict: ポートフォリオリスク指標（日次ベース）
    """
    try:
        if returns.empty or len(weights) == 0:
            return {}
        
        # 日次共分散行列計算（年率換算しない）
        daily_cov_matrix = returns.cov()
        
        # 日次ポートフォリオボラティリティ
        daily_portfolio_variance = np.dot(weights.T, np.dot(daily_cov_matrix, weights))
        daily_portfolio_volatility = np.sqrt(daily_portfolio_variance)
        
        # 相関行列
        correlation_matrix = returns.corr()
        
        # 平均相関
        n = len(correlation_matrix)
        avg_correlation = (correlation_matrix.sum().sum() - n) / (n * (n - 1)) if n > 1 else 0
        
        # 銘柄別日次ボラティリティ
        daily_individual_volatilities = returns.std()
        
        # 重み付き平均ボラティリティ（日次）
        weighted_avg_vol = (daily_individual_volatilities * weights).sum()
        
        # 分散効果
        diversification_ratio = weighted_avg_vol / daily_portfolio_volatility if daily_portfolio_volatility > 0 else 1
        
        result = {
            'portfolio_volatility': daily_portfolio_volatility,  # 日次ボラティリティ
            'correlation_matrix': correlation_matrix,
            'covariance_matrix': daily_cov_matrix,
            'average_correlation': avg_correlation,
            'individual_volatilities': daily_individual_volatilities,  # 日次ボラティリティ
            'weighted_avg_volatility': weighted_avg_vol,
            'diversification_ratio': diversification_ratio
        }
        
        logger.info(f"ポートフォリオリスク計算完了: 日次ボラティリティ {daily_portfolio_volatility:.3%}, 年率 {daily_portfolio_volatility * np.sqrt(252):.1%}")
        return result
        
    except Exception as e:
        logger.error(f"ポートフォリオリスク計算エラー: {str(e)}")
        return {}


def calculate_var_cvar(
    portfolio_returns: pd.Series, 
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    VaRとCVaRを計算（ヒストリカル法）
    
    Args:
        portfolio_returns: ポートフォリオ日次リターン系列
        confidence_levels: 信頼水準のリスト
    
    Returns:
        dict: VaR・CVaR値
    """
    try:
        if portfolio_returns.empty or portfolio_returns.isna().all():
            return {}
        
        results = {}
        
        for confidence in confidence_levels:
            # VaR計算
            var_percentile = (1 - confidence) * 100
            var = np.percentile(portfolio_returns.dropna(), var_percentile)
            
            # CVaR計算（VaRを下回る損失の平均）
            tail_returns = portfolio_returns[portfolio_returns <= var]
            cvar = tail_returns.mean() if not tail_returns.empty else var
            
            results[f'VaR_{int(confidence*100)}'] = var
            results[f'CVaR_{int(confidence*100)}'] = cvar
        
        logger.info(f"VaR/CVaR計算完了: VaR95% {results.get('VaR_95', 0):.2%}")
        return results
        
    except Exception as e:
        logger.error(f"VaR/CVaR計算エラー: {str(e)}")
        return {}


def stress_test_scenario(
    returns: pd.DataFrame,
    weights: np.ndarray,
    stress_factor: float = 2.0,
    correlation_shock: float = 0.9
) -> Dict[str, float]:
    """
    ストレスシナリオ分析（日次ベース）
    
    Args:
        returns: 銘柄リターンデータ
        weights: ポートフォリオ重み
        stress_factor: ボラティリティ増加倍率
        correlation_shock: ストレス時の相関係数
    
    Returns:
        dict: ストレステスト結果（日次ボラティリティベース）
    """
    try:
        if returns.empty or len(weights) == 0:
            return {}
        
        # 通常時の日次共分散行列（年率換算しない）
        daily_normal_cov = returns.cov()
        
        # ストレス時の共分散行列構築
        daily_normal_vol = np.sqrt(np.diag(daily_normal_cov))
        daily_stressed_vol = daily_normal_vol * stress_factor
        
        # 相関行列をストレス値に変更
        n = len(returns.columns)
        stressed_corr = np.full((n, n), correlation_shock)
        np.fill_diagonal(stressed_corr, 1.0)
        
        # ストレス時共分散行列
        daily_stressed_cov = np.outer(daily_stressed_vol, daily_stressed_vol) * stressed_corr
        
        # 通常時ポートフォリオボラティリティ（日次）
        daily_normal_portfolio_var = np.dot(weights.T, np.dot(daily_normal_cov, weights))
        daily_normal_portfolio_vol = np.sqrt(daily_normal_portfolio_var)
        
        # ストレス時ポートフォリオボラティリティ（日次）
        daily_stressed_portfolio_var = np.dot(weights.T, np.dot(daily_stressed_cov, weights))
        daily_stressed_portfolio_vol = np.sqrt(daily_stressed_portfolio_var)
        
        # ストレス倍率
        stress_multiplier = daily_stressed_portfolio_vol / daily_normal_portfolio_vol if daily_normal_portfolio_vol > 0 else 1
        
        result = {
            'normal_portfolio_vol': daily_normal_portfolio_vol,  # 日次ボラティリティ
            'stressed_portfolio_vol': daily_stressed_portfolio_vol,  # 日次ボラティリティ
            'stress_multiplier': stress_multiplier,
            'stress_factor': stress_factor,
            'correlation_shock': correlation_shock
        }
        
        logger.info(f"ストレステスト完了: 日次通常時 {daily_normal_portfolio_vol:.3%}, ストレス倍率 {stress_multiplier:.2f}x")
        return result
        
    except Exception as e:
        logger.error(f"ストレステストエラー: {str(e)}")
        return {}


def calculate_risk_contribution(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> Dict[str, any]:
    """
    リスク寄与度分析（日次ベース）
    
    Args:
        returns: 銘柄リターンデータ
        weights: ポートフォリオ重み
    
    Returns:
        dict: リスク寄与度分析結果（日次ボラティリティベース）
    """
    try:
        if returns.empty or len(weights) == 0:
            return {}
        
        # 日次共分散行列（年率換算しない）
        daily_cov_matrix = returns.cov()
        
        # ポートフォリオ分散（日次）
        daily_portfolio_variance = np.dot(weights.T, np.dot(daily_cov_matrix, weights))
        daily_portfolio_volatility = np.sqrt(daily_portfolio_variance)
        
        # 各銘柄のリスク寄与度（偏微分）
        marginal_risk = np.dot(daily_cov_matrix, weights) / daily_portfolio_volatility if daily_portfolio_volatility > 0 else np.zeros_like(weights)
        
        # リスク寄与額
        risk_contribution = weights * marginal_risk
        
        # リスク寄与率
        risk_contribution_pct = risk_contribution / daily_portfolio_variance * 100 if daily_portfolio_variance > 0 else np.zeros_like(weights)
        
        result = {
            'tickers': returns.columns.tolist(),
            'weights': weights,
            'marginal_risk': marginal_risk,
            'risk_contribution': risk_contribution,
            'risk_contribution_pct': risk_contribution_pct,
            'portfolio_volatility': daily_portfolio_volatility  # 日次ボラティリティ
        }
        
        logger.info(f"リスク寄与度分析完了: 日次ボラティリティ {daily_portfolio_volatility:.3%}, 最大寄与 {risk_contribution_pct.max():.1f}%")
        return result
        
    except Exception as e:
        logger.error(f"リスク寄与度分析エラー: {str(e)}")
        return {}


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    トラッキングエラー計算
    
    Args:
        portfolio_returns: ポートフォリオリターン
        benchmark_returns: ベンチマークリターン
    
    Returns:
        dict: トラッキングエラー指標
    """
    try:
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}
        
        # アクティブリターン
        active_returns = portfolio_returns - benchmark_returns
        
        # トラッキングエラー（年率）
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # 情報比率
        average_active_return = active_returns.mean() * 252  # 年率
        information_ratio = average_active_return / tracking_error if tracking_error > 0 else 0
        
        # 最大アクティブリターン
        max_active_return = active_returns.max()
        min_active_return = active_returns.min()
        
        result = {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'average_active_return': average_active_return,
            'max_active_return': max_active_return,
            'min_active_return': min_active_return,
            'active_return_volatility': active_returns.std()
        }
        
        logger.info(f"トラッキングエラー計算完了: TE {tracking_error:.2%}")
        return result
        
    except Exception as e:
        logger.error(f"トラッキングエラー計算エラー: {str(e)}")
        return {}


def calculate_maximum_drawdown(returns: pd.Series) -> Dict[str, float]:
    """
    最大ドローダウン計算
    
    Args:
        returns: リターン系列
    
    Returns:
        dict: ドローダウン指標
    """
    try:
        if returns.empty:
            return {}
        
        # 累積リターンを計算
        cumulative_returns = (1 + returns).cumprod()
        
        # 過去の最高値を記録
        rolling_max = cumulative_returns.expanding().max()
        
        # ドローダウンを計算
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        # 最大ドローダウン
        max_drawdown = drawdown.min()
        
        # ドローダウン期間
        dd_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                dd_periods.append(i - start_idx)
        
        # 最長ドローダウン期間
        max_drawdown_duration = max(dd_periods) if dd_periods else 0
        
        result = {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'current_drawdown': drawdown.iloc[-1],
            'drawdown_periods': len(dd_periods)
        }
        
        logger.info(f"最大ドローダウン計算完了: {max_drawdown:.2%}")
        return result
        
    except Exception as e:
        logger.error(f"最大ドローダウン計算エラー: {str(e)}")
        return {}


def calculate_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """
    ベータ値計算
    
    Args:
        stock_returns: 個別株リターン
        market_returns: 市場リターン
    
    Returns:
        float: ベータ値
    """
    try:
        if stock_returns.empty or market_returns.empty:
            return 1.0
        
        # 共分散を計算
        covariance = np.cov(stock_returns.dropna(), market_returns.dropna())[0, 1]
        
        # 市場の分散を計算
        market_variance = market_returns.var()
        
        # ベータ計算
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        return beta
        
    except Exception as e:
        logger.error(f"ベータ計算エラー: {str(e)}")
        return 1.0