"""
村庄交易性能评估器

该模块实现村庄交易任务环境下的智能体性能评估，主要评估：
1. 交易技能：买卖决策的成功率和盈利能力
2. 学习速度：交易策略的优化和适应能力
3. 泛化能力：跨商品类型和市场条件的表现

村庄交易任务模拟了一个虚拟经济环境，智能体需要：
- 分析市场价格和供需关系
- 制定最优的交易策略
- 管理库存和资金
- 适应不同的市场环境
"""

import time
import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas未安装，将使用内置数据结构")


@dataclass
class TradingMetrics:
    """交易任务评估指标数据类"""
    success_rate: float = 0.0  # 交易成功率
    profit_margin: float = 0.0  # 利润率
    inventory_efficiency: float = 0.0  # 库存效率
    market_adaptation: float = 0.0  # 市场适应能力
    risk_management: float = 0.0  # 风险管理能力
    diversification_score: float = 0.0  # 多元化投资得分
    trading_frequency: float = 0.0  # 交易频率
    average_holding_period: float = 0.0  # 平均持有期
    total_trades: int = 0  # 总交易次数
    profitable_trades: int = 0  # 盈利交易次数
    trading_history: List[Dict] = field(default_factory=list)  # 交易历史


class VillageMarketSimulator:
    """村庄市场模拟器"""
    
    def __init__(self, num_commodities: int = 8):
        """
        初始化市场模拟器
        
        Args:
            num_commodities: 商品种类数量
        """
        self.num_commodities = num_commodities
        self.commodity_names = [
            "小麦", "牛肉", "铁锭", "布匹", "草药", 
            "宝石", "蜂蜜", "木材"
        ][:num_commodities]
        
        # 基础价格和需求
        self.base_prices = {
            name: random.uniform(20, 100) for name in self.commodity_names
        }
        self.base_demand = {
            name: random.uniform(50, 200) for name in self.commodity_names
        }
        
        # 季节性因素
        self.seasonal_factors = {
            'spring': {'小麦': 1.2, '草药': 1.3, '蜂蜜': 1.1},
            'summer': {'牛肉': 1.2, '布匹': 1.1},
            'autumn': {'苹果': 1.4, '木材': 1.2},
            'winter': {'铁锭': 1.3, '宝石': 1.4}
        }
        
        # 随机事件
        self.random_events = {
            'festival': {'布匹': 0.8, '宝石': 1.5, '蜂蜜': 1.2},
            'drought': {'小麦': 1.6, '牛肉': 1.3},
            'war': {'铁锭': 1.8, '木材': 1.4},
            'plague': {'草药': 2.0, '牛肉': 0.7}
        }
        
        self.current_season = 'spring'
        self.current_event = None
    
    def get_current_prices(self) -> Dict[str, float]:
        """获取当前市场价格"""
        prices = {}
        for commodity in self.commodity_names:
            base_price = self.base_prices[commodity]
            
            # 季节性影响
            seasonal_factor = self.seasonal_factors.get(self.current_season, {}).get(commodity, 1.0)
            
            # 随机事件影响
            event_factor = 1.0
            if self.current_event:
                event_factor = self.random_events.get(self.current_event, {}).get(commodity, 1.0)
            
            # 市场波动（±20%）
            volatility = random.uniform(0.8, 1.2)
            
            price = base_price * seasonal_factor * event_factor * volatility
            prices[commodity] = max(1.0, price)  # 确保价格非负
        
        return prices
    
    def get_current_demand(self) -> Dict[str, float]:
        """获取当前市场需求"""
        demand = {}
        for commodity in self.commodity_names:
            base_demand = self.base_demand[commodity]
            
            # 需求与价格成反比
            current_price = self.get_current_prices()[commodity]
            base_price = self.base_prices[commodity]
            price_ratio = base_price / current_price
            
            # 季节性影响
            seasonal_factor = self.seasonal_factors.get(self.current_season, {}).get(commodity, 1.0)
            
            demand[commodity] = base_demand * price_ratio * seasonal_factor
        
        return demand
    
    def simulate_day(self) -> Dict[str, Any]:
        """模拟一天的交易"""
        # 更新季节（简单循环）
        seasons = ['spring', 'summer', 'autumn', 'winter']
        self.current_season = seasons[int(time.time()) % 4]
        
        # 随机触发事件（10%概率）
        if random.random() < 0.1:
            self.current_event = random.choice(list(self.random_events.keys()))
        else:
            self.current_event = None
        
        return {
            'prices': self.get_current_prices(),
            'demand': self.get_current_demand(),
            'season': self.current_season,
            'event': self.current_event
        }


class TradingEvaluator:
    """
    村庄交易性能评估器
    
    在模拟的村庄经济环境中评估智能体的交易技能。
    支持多商品交易、库存管理和市场适应测试。
    
    Attributes:
        market_simulator (VillageMarketSimulator): 市场模拟器
        config (Dict): 配置参数
        logger (logging.Logger): 日志记录器
        evaluation_history (List): 评估历史记录
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 num_commodities: int = 8,
                 simulation_days: int = 100):
        """
        初始化交易评估器
        
        Args:
            config: 配置参数字典
            num_commodities: 商品种类数量
            simulation_days: 模拟天数
        """
        # 市场模拟器
        self.market_simulator = VillageMarketSimulator(num_commodities)
        self.simulation_days = simulation_days
        
        # 设置配置参数
        self.config = self._setup_default_config()
        if config:
            self.config.update(config)
        
        # 设置日志记录
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 评估历史记录
        self.evaluation_history = []
        
        # 模拟智能体状态
        self.initial_capital = self.config['initial_capital']
        self.initial_inventory = self.config['initial_inventory']
        
        self.logger.info(f"交易评估器初始化完成 - 商品数: {num_commodities}, 模拟天数: {simulation_days}")
    
    def _setup_default_config(self) -> Dict:
        """设置默认配置参数"""
        return {
            # 资金和库存配置
            'initial_capital': 1000.0,
            'initial_inventory': {name: 10 for name in [
                "小麦", "牛肉", "铁锭", "布匹", "草药", 
                "宝石", "蜂蜜", "木材"
            ]},
            
            # 评估参数
            'evaluation_episodes': 5,
            'simulation_days': self.simulation_days,
            'confidence_threshold': 0.95,
            
            # 交易参数
            'max_trades_per_day': 10,
            'transaction_cost_rate': 0.01,  # 1%交易成本
            'holding_cost_rate': 0.001,  # 0.1%持有成本
            
            # 风险管理
            'max_position_size': 0.3,  # 最大仓位
            'stop_loss_rate': 0.2,  # 止损阈值
            
            # 学习参数
            'learning_curve_points': 20,
            'adaptation_test_periods': 10,
            
            # 性能阈值
            'success_threshold': 0.15,  # 15%利润率为成功
            'optimal_profit_margin': 0.5,  # 50%利润率为最佳
            
            # 调试参数
            'save_trading_history': True,
            'verbose_logging': False
        }
    
    def evaluate(self, agent: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行交易性能评估
        
        Args:
            agent: 待评估的智能体
            config: 评估配置参数
            
        Returns:
            包含交易性能指标的字典
        """
        evaluation_config = self.config.copy()
        if config:
            evaluation_config.update(config)
        
        start_time = time.time()
        
        try:
            # 执行交易模拟
            trading_data = self._run_trading_simulation(agent, evaluation_config)
            
            # 计算交易指标
            metrics = self._calculate_trading_metrics(trading_data)
            
            # 计算衍生指标
            learning_curve = self._calculate_learning_curve(trading_data)
            generalization_scores = self._calculate_generalization(trading_data)
            
            # 构建详细结果
            detailed_metrics = {
                'success_rate': metrics.success_rate,
                'profit_margin': metrics.profit_margin,
                'inventory_efficiency': metrics.inventory_efficiency,
                'market_adaptation': metrics.market_adaptation,
                'risk_management': metrics.risk_management,
                'diversification_score': metrics.diversification_score,
                'trading_frequency': metrics.trading_frequency,
                'average_holding_period': metrics.average_holding_period,
                'total_trades': metrics.total_trades,
                'profitable_trades': metrics.profitable_trades,
                'trading_history': metrics.trading_history,
                'learning_curve': learning_curve,
                'generalization_scores': generalization_scores,
                'evaluation_config': evaluation_config,
                'evaluation_time': time.time() - start_time
            }
            
            # 记录评估历史
            self.evaluation_history.append({
                'timestamp': time.time(),
                'agent_id': getattr(agent, 'id', 'unknown'),
                'metrics': detailed_metrics
            })
            
            self.logger.info(
                f"交易评估完成 - 成功率: {metrics.success_rate:.2%}, "
                f"利润率: {metrics.profit_margin:.2%}, "
                f"总交易: {metrics.total_trades}"
            )
            
            return detailed_metrics
            
        except Exception as e:
            self.logger.error(f"交易评估失败: {e}")
            return {
                'error': str(e),
                'success_rate': 0.0,
                'profit_margin': 0.0,
                'inventory_efficiency': 0.0,
                'evaluation_time': time.time() - start_time
            }
    
    def _run_trading_simulation(self, agent: Any, config: Dict) -> Dict[str, Any]:
        """执行交易模拟"""
        # 初始化状态
        capital = self.initial_capital
        inventory = self.initial_inventory.copy()
        trading_history = []
        
        # 记录学习曲线数据
        learning_data = []
        
        for day in range(config['simulation_days']):
            # 获取市场信息
            market_info = self.market_simulator.simulate_day()
            prices = market_info['prices']
            demand = market_info['demand']
            
            # 智能体决策
            if hasattr(agent, 'make_decision'):
                decision = agent.make_decision(
                    market_info, capital, inventory
                )
            elif hasattr(agent, 'act'):
                decision = agent.act(market_info, capital, inventory)
            elif callable(agent):
                decision = agent(market_info, capital, inventory)
            else:
                # 默认随机决策
                decision = self._generate_random_decision(prices, demand)
            
            # 执行交易
            daily_result = self._execute_trades(
                decision, prices, demand, capital, inventory, day, config
            )
            
            # 更新状态
            capital = daily_result['capital']
            inventory = daily_result['inventory']
            
            # 记录交易历史
            trading_history.extend(daily_result['trades'])
            
            # 记录学习数据
            learning_data.append({
                'day': day,
                'capital': capital,
                'inventory_value': sum(
                    inventory[commodity] * prices[commodity] 
                    for commodity in inventory
                ),
                'total_wealth': capital + sum(
                    inventory[commodity] * prices[commodity] 
                    for commodity in inventory
                )
            })
            
            # 计算每日持有成本
            holding_cost = sum(
                inventory[commodity] * prices[commodity] * config['holding_cost_rate']
                for commodity in inventory
            )
            capital -= holding_cost
        
        # 计算最终总资产
        final_prices = self.market_simulator.get_current_prices()
        final_inventory_value = sum(
            inventory[commodity] * final_prices[commodity]
            for commodity in inventory
        )
        total_final_wealth = capital + final_inventory_value
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'final_inventory': inventory,
            'final_inventory_value': final_inventory_value,
            'total_final_wealth': total_final_wealth,
            'trading_history': trading_history,
            'learning_data': learning_data,
            'total_days': config['simulation_days']
        }
    
    def _generate_random_decision(self, prices: Dict, demand: Dict) -> Dict:
        """生成随机交易决策（用于测试）"""
        decisions = []
        
        for commodity in prices.keys():
            # 随机决定买卖
            action = random.choice(['buy', 'sell', 'hold'])
            quantity = random.randint(0, 5)
            
            if action == 'buy':
                decisions.append({
                    'commodity': commodity,
                    'action': 'buy',
                    'quantity': quantity,
                    'expected_price': prices[commodity]
                })
            elif action == 'sell':
                decisions.append({
                    'commodity': commodity,
                    'action': 'sell',
                    'quantity': quantity,
                    'expected_price': prices[commodity]
                })
        
        return {'decisions': decisions}
    
    def _execute_trades(self, decision: Dict, prices: Dict, demand: Dict,
                       capital: float, inventory: Dict, day: int, config: Dict) -> Dict[str, Any]:
        """执行交易逻辑"""
        trades = []
        updated_inventory = inventory.copy()
        updated_capital = capital
        
        for trade_decision in decision.get('decisions', []):
            commodity = trade_decision['commodity']
            action = trade_decision['action']
            quantity = trade_decision['quantity']
            
            if action == 'buy' and quantity > 0:
                # 执行买入
                trade_cost = quantity * prices[commodity] * (1 + config['transaction_cost_rate'])
                
                if trade_cost <= updated_capital and updated_inventory.get(commodity, 0) < 100:
                    updated_capital -= trade_cost
                    updated_inventory[commodity] = updated_inventory.get(commodity, 0) + quantity
                    
                    trades.append({
                        'day': day,
                        'commodity': commodity,
                        'action': 'buy',
                        'quantity': quantity,
                        'price': prices[commodity],
                        'cost': trade_cost,
                        'profit': 0
                    })
            
            elif action == 'sell' and quantity > 0:
                # 执行卖出
                available_quantity = updated_inventory.get(commodity, 0)
                sell_quantity = min(quantity, available_quantity)
                
                if sell_quantity > 0:
                    trade_revenue = sell_quantity * prices[commodity] * (1 - config['transaction_cost_rate'])
                    updated_capital += trade_revenue
                    updated_inventory[commodity] = available_quantity - sell_quantity
                    
                    # 计算利润（简化计算）
                    profit = trade_revenue - sell_quantity * prices[commodity] * 0.9  # 假设买入价格
                    
                    trades.append({
                        'day': day,
                        'commodity': commodity,
                        'action': 'sell',
                        'quantity': sell_quantity,
                        'price': prices[commodity],
                        'revenue': trade_revenue,
                        'profit': profit
                    })
        
        return {
            'capital': updated_capital,
            'inventory': updated_inventory,
            'trades': trades
        }
    
    def _calculate_trading_metrics(self, trading_data: Dict) -> TradingMetrics:
        """计算交易性能指标"""
        trading_history = trading_data['trading_history']
        
        if not trading_history:
            return TradingMetrics()
        
        # 基本统计
        total_trades = len(trading_history)
        buy_trades = [t for t in trading_history if t['action'] == 'buy']
        sell_trades = [t for t in trading_history if t['action'] == 'sell']
        
        # 成功率计算
        profitable_sells = [t for t in sell_trades if t.get('profit', 0) > 0]
        success_rate = len(profitable_sells) / len(sell_trades) if sell_trades else 0.0
        
        # 利润率计算
        total_profit = sum(t.get('profit', 0) for t in sell_trades)
        total_invested = sum(t.get('cost', 0) for t in buy_trades)
        profit_margin = total_profit / total_invested if total_invested > 0 else 0.0
        
        # 库存效率
        final_inventory = trading_data['final_inventory']
        initial_inventory_value = sum(
            self.initial_inventory[commodity] * self.market_simulator.base_prices[commodity]
            for commodity in self.initial_inventory
        )
        final_inventory_value = trading_data['final_inventory_value']
        inventory_efficiency = final_inventory_value / initial_inventory_value if initial_inventory_value > 0 else 1.0
        
        # 交易频率
        trading_frequency = total_trades / trading_data['total_days']
        
        # 平均持有期（简化计算）
        holding_periods = []
        commodity_first_buy = {}
        
        for trade in buy_trades:
            commodity = trade['commodity']
            if commodity not in commodity_first_buy:
                commodity_first_buy[commodity] = trade['day']
        
        for trade in sell_trades:
            commodity = trade['commodity']
            if commodity in commodity_first_buy:
                holding_period = trade['day'] - commodity_first_buy[commodity]
                holding_periods.append(holding_period)
        
        average_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        # 多元化得分
        unique_commodities = set(t['commodity'] for t in trading_history)
        diversification_score = len(unique_commodities) / len(self.market_simulator.commodity_names)
        
        # 风险管理（基于仓位分布）
        max_position_size = max(
            sum(t['quantity'] for t in trading_history if t['commodity'] == commodity)
            for commodity in unique_commodities
        ) / sum(self.initial_inventory.values())
        risk_management = max(0.0, 1.0 - max_position_size)
        
        return TradingMetrics(
            success_rate=success_rate,
            profit_margin=profit_margin,
            inventory_efficiency=inventory_efficiency,
            market_adaptation=0.8,  # 简化值
            risk_management=risk_management,
            diversification_score=diversification_score,
            trading_frequency=trading_frequency,
            average_holding_period=average_holding_period,
            total_trades=total_trades,
            profitable_trades=len(profitable_sells),
            trading_history=trading_history
        )
    
    def _calculate_learning_curve(self, trading_data: Dict) -> List[float]:
        """计算学习曲线"""
        learning_data = trading_data['learning_data']
        
        # 基于财富增长计算学习曲线
        initial_wealth = trading_data['initial_capital'] + sum(
            self.initial_inventory[commodity] * self.market_simulator.base_prices[commodity]
            for commodity in self.initial_inventory
        )
        
        learning_curve = []
        for data in learning_data:
            current_wealth = data['total_wealth']
            performance = (current_wealth - initial_wealth) / initial_wealth
            learning_curve.append(max(0.0, min(1.0, performance + 0.5)))  # 归一化到0-1
        
        return learning_curve[:self.config['learning_curve_points']]
    
    def _calculate_generalization(self, trading_data: Dict) -> Dict[str, float]:
        """计算泛化能力"""
        # 基于不同商品的表现计算泛化能力
        trading_history = trading_data['trading_history']
        
        commodity_performances = defaultdict(list)
        for trade in trading_history:
            if trade['action'] == 'sell':
                commodity_performances[trade['commodity']].append(
                    trade.get('profit', 0) / (trade['price'] * trade['quantity'])
                )
        
        generalization_scores = {}
        for commodity, profits in commodity_performances.items():
            avg_profit_rate = np.mean(profits) if profits else 0.0
            generalization_scores[f'commodity_{commodity}'] = max(0.0, min(1.0, avg_profit_rate + 0.5))
        
        return generalization_scores
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """获取性能分析报告"""
        if not self.evaluation_history:
            return {"message": "暂无评估数据"}
        
        recent_evals = self.evaluation_history[-5:]  # 最近5次评估
        
        analysis_data = {
            'profit_margins': [],
            'success_rates': [],
            'total_trades': []
        }
        
        for eval_data in recent_evals:
            metrics = eval_data['metrics']
            analysis_data['profit_margins'].append(metrics.get('profit_margin', 0.0))
            analysis_data['success_rates'].append(metrics.get('success_rate', 0.0))
            analysis_data['total_trades'].append(metrics.get('total_trades', 0))
        
        return {
            '评估总数': len(self.evaluation_history),
            '平均利润率': np.mean(analysis_data['profit_margins']),
            '平均成功率': np.mean(analysis_data['success_rates']),
            '平均交易次数': np.mean(analysis_data['total_trades']),
            '利润率趋势': np.polyfit(range(len(analysis_data['profit_margins'])), 
                               analysis_data['profit_margins'], 1)[0],
            '成功率趋势': np.polyfit(range(len(analysis_data['success_rates'])), 
                               analysis_data['success_rates'], 1)[0]
        }
    
    def save_evaluation_data(self, filepath: str):
        """保存评估数据"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"交易评估数据已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存交易评估数据失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("交易评估器资源已清理")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()