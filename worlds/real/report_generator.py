#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器模块
=============

这个模块负责生成专业格式的性能测试报告。
支持多种输出格式、自动化报告生成、
可视化图表和数据导出功能。

核心功能：
- 多格式报告生成（HTML、PDF、JSON）
- 数据可视化和图表生成
- 自动化报告调度
- 报告模板和样式定制
- 邮件通知和分享功能

作者：AI研究团队
日期：2025-11-13
"""

import base64
import json
import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jinja2 import Template
import markdown
from weasyprint import HTML, CSS


class ReportGenerator:
    """
    报告生成器类
    
    负责将性能分析结果转换为专业的测试报告。
    支持多种格式输出、丰富的可视化图表和自定义模板。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化报告生成器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 输出目录
        self.output_dir = Path(self.config.get('output_dir', '/workspace/worlds/real/reports'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 模板配置
        self.template_dir = Path(self.config.get('template_dir', '/workspace/worlds/real/templates'))
        self.template_dir.mkdir(exist_ok=True)
        
        # 图表配置
        self.chart_config = {
            'figsize': self.config.get('chart_figsize', (12, 8)),
            'dpi': self.config.get('chart_dpi', 100),
            'style': self.config.get('chart_style', 'seaborn'),
            'color_palette': self.config.get('color_palette', 'husl')
        }
        
        # 邮件配置
        self.email_config = self.config.get('email', {})
        
        # 报告模板
        self.templates = self._load_templates()
        
        # 生成统计
        self.generation_stats = {
            'total_reports': 0,
            'html_reports': 0,
            'pdf_reports': 0,
            'json_reports': 0,
            'email_sent': 0
        }
        
        # 初始化matplotlib中文支持
        self._setup_matplotlib()
        
        self.logger.info("报告生成器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'output_dir': '/workspace/worlds/real/reports',
            'template_dir': '/workspace/worlds/real/templates',
            'chart_figsize': (12, 8),
            'chart_dpi': 100,
            'chart_style': 'seaborn',
            'color_palette': 'husl',
            'auto_email': False,
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': '',
                'to_addresses': []
            },
            'report_formats': ['html', 'pdf', 'json'],
            'include_charts': True,
            'include_recommendations': True,
            'chart_styles': {
                'time_series': 'seaborn-v0_8',
                'bar_chart': 'seaborn-v0_8',
                'heatmap': 'RdYlBu_r'
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('ReportGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_dir = Path('/workspace/worlds/real/logs')
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f'report_generator_{datetime.now().strftime("%Y%m%d")}.log',
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _setup_matplotlib(self):
        """设置matplotlib中文支持"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use(self.chart_config['style'])
        
        # 设置seaborn风格
        try:
            sns.set_style("whitegrid")
            sns.set_palette(self.chart_config['color_palette'])
        except Exception:
            pass  # 如果seaborn不可用，使用默认设置
    
    def _load_templates(self) -> Dict[str, str]:
        """加载报告模板"""
        templates = {}
        
        # HTML模板
        html_template = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #007acc; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #007acc; margin: 0; font-size: 2.5em; }
        .header .subtitle { color: #666; font-size: 1.2em; margin-top: 10px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-left: 4px solid #007acc; padding-left: 15px; font-size: 1.8em; }
        .section h3 { color: #555; font-size: 1.4em; margin-top: 25px; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 10px 0; display: inline-block; min-width: 200px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .metric-label { font-size: 1.1em; opacity: 0.9; }
        .chart-container { text-align: center; margin: 20px 0; }
        .chart-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .table th { background-color: #007acc; color: white; font-weight: bold; }
        .table tr:nth-child(even) { background-color: #f9f9f9; }
        .alert { padding: 15px; margin: 15px 0; border-radius: 5px; }
        .alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .alert-success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .alert-danger { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .recommendation { background-color: #e8f4fd; border-left: 4px solid #007acc; padding: 15px; margin: 15px 0; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="subtitle">{{ subtitle }}</div>
            <div style="margin-top: 10px; color: #888;">生成时间: {{ generate_time }}</div>
        </div>
        
        {{ content }}
        
        <div class="footer">
            <p>© 2025 智能体性能测试系统 | 报告由AI自动生成</p>
        </div>
    </div>
</body>
</html>
        '''
        
        # JSON模板
        json_template = {
            'title': '{{ title }}',
            'subtitle': '{{ subtitle }}',
            'generate_time': '{{ generate_time }}',
            'data': '{{ data }}'
        }
        
        templates['html'] = html_template
        templates['json'] = json.dumps(json_template, ensure_ascii=False, indent=2)
        
        return templates
    
    def generate_weekly_report(self, test_results: Dict[str, Any],
                             performance_data: Dict[str, Any],
                             trend_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        生成周报
        
        这是报告生成器的核心方法，生成完整的每周测试报告。
        
        Args:
            test_results: 测试结果数据
            performance_data: 性能数据
            trend_analysis: 趋势分析结果
            
        Returns:
            生成的报告文件路径字典
        """
        try:
            self.logger.info("开始生成每周测试报告")
            start_time = datetime.now()
            
            # 生成报告内容
            report_data = self._prepare_report_data(test_results, performance_data, trend_analysis)
            
            # 生成各种格式的报告
            report_files = {}
            
            if 'html' in self.config['report_formats']:
                html_file = self._generate_html_report(report_data)
                if html_file:
                    report_files['html'] = html_file
                    self.generation_stats['html_reports'] += 1
            
            if 'pdf' in self.config['report_formats']:
                pdf_file = self._generate_pdf_report(report_data)
                if pdf_file:
                    report_files['pdf'] = pdf_file
                    self.generation_stats['pdf_reports'] += 1
            
            if 'json' in self.config['report_formats']:
                json_file = self._generate_json_report(report_data)
                if json_file:
                    report_files['json'] = json_file
                    self.generation_stats['json_reports'] += 1
            
            # 生成图表
            if self.config.get('include_charts', True):
                chart_files = self._generate_report_charts(test_results, performance_data, trend_analysis)
                report_files['charts'] = chart_files
            
            # 发送邮件通知
            if self.config.get('auto_email', False):
                self._send_email_notification(report_files, report_data)
            
            self.generation_stats['total_reports'] += 1
            
            generation_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"报告生成完成，耗时 {generation_time:.2f} 秒")
            
            return report_files
            
        except Exception as e:
            self.logger.error(f"生成周报失败: {e}")
            return {'error': str(e)}
    
    def _prepare_report_data(self, test_results: Dict[str, Any],
                           performance_data: Dict[str, Any],
                           trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """准备报告数据"""
        current_time = datetime.now()
        
        # 提取关键指标
        summary_metrics = test_results.get('summary_metrics', {})
        environment_scores = test_results.get('environment_scores', {})
        individual_tests = test_results.get('individual_tests', {})
        
        # 计算总体评分
        total_score = 0
        successful_tests = 0
        total_tests = len(individual_tests)
        
        for env_name, score in environment_scores.items():
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                total_score += score
                successful_tests += 1
        
        avg_score = total_score / max(successful_tests, 1)
        success_rate = summary_metrics.get('success_rate', 0)
        
        # 生成报告标题和摘要
        title = f"智能体真实世界任务测试周报 - {current_time.strftime('%Y年%m月%d日')}"
        subtitle = f"平均性能评分: {avg_score:.3f} | 测试成功率: {success_rate:.3f} | 环境覆盖: {successful_tests}/{total_tests}"
        
        # 性能亮点和警告
        highlights = self._extract_performance_highlights(individual_tests)
        warnings = self._extract_performance_warnings(individual_tests, trend_analysis)
        
        # 趋势洞察
        trend_insights = self._extract_trend_insights(trend_analysis)
        
        # 建议行动
        recommendations = trend_analysis.get('recommendations', []) if trend_analysis else []
        
        return {
            'title': title,
            'subtitle': subtitle,
            'generate_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_score': avg_score,
                'success_rate': success_rate,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'test_coverage': successful_tests / max(total_tests, 1)
            },
            'environment_scores': environment_scores,
            'highlights': highlights,
            'warnings': warnings,
            'trend_insights': trend_insights,
            'recommendations': recommendations,
            'test_results': test_results,
            'performance_data': performance_data,
            'trend_analysis': trend_analysis
        }
    
    def _extract_performance_highlights(self, test_results: Dict[str, Any]) -> List[str]:
        """提取性能亮点"""
        highlights = []
        
        for env_name, env_result in test_results.items():
            if isinstance(env_result, dict) and env_result.get('status') == 'completed':
                # 检查高分指标
                if 'accuracy' in env_result and env_result['accuracy'] > 0.9:
                    highlights.append(f"🌟 {env_name} 环境准确性达到 {env_result['accuracy']:.3f}")
                
                if 'f1_score' in env_result and env_result['f1_score'] > 0.85:
                    highlights.append(f"🎯 {env_name} 环境F1分数优异: {env_result['f1_score']:.3f}")
                
                if 'mAP' in env_result and env_result['mAP'] > 0.8:
                    highlights.append(f"🎪 {env_name} 环境mAP表现优秀: {env_result['mAP']:.3f}")
        
        return highlights
    
    def _extract_performance_warnings(self, test_results: Dict[str, Any], 
                                    trend_analysis: Dict[str, Any]) -> List[str]:
        """提取性能警告"""
        warnings = []
        
        # 检查低分环境
        for env_name, env_result in test_results.items():
            if isinstance(env_result, dict) and env_result.get('status') == 'completed':
                if 'accuracy' in env_result and env_result['accuracy'] < 0.7:
                    warnings.append(f"⚠️ {env_name} 环境准确性偏低: {env_result['accuracy']:.3f}")
                
                if 'adaptation_time' in env_result and env_result['adaptation_time'] > 25:
                    warnings.append(f"⏱️ {env_name} 环境适应时间较长: {env_result['adaptation_time']:.1f}秒")
        
        # 检查趋势警告
        if trend_analysis and 'error' not in trend_analysis:
            trend_data = trend_analysis.get('trend_analysis', {})
            for metric, analysis in trend_data.items():
                if isinstance(analysis, dict):
                    direction = analysis.get('overall_direction', 'stable')
                    if direction == 'declining':
                        warnings.append(f"📉 {metric} 指标呈下降趋势")
        
        return warnings
    
    def _extract_trend_insights(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """提取趋势洞察"""
        insights = []
        
        if not trend_analysis or 'error' in trend_analysis:
            return ["暂无趋势分析数据"]
        
        # 整体趋势洞察
        trend_data = trend_analysis.get('trend_analysis', {})
        
        improving_metrics = []
        declining_metrics = []
        stable_metrics = []
        
        for metric, analysis in trend_data.items():
            if isinstance(analysis, dict):
                direction = analysis.get('overall_direction', 'stable')
                r_squared = analysis.get('linear_trend', {}).get('r_squared', 0)
                
                if direction == 'improving' and r_squared > 0.5:
                    improving_metrics.append(metric)
                elif direction == 'declining' and r_squared > 0.5:
                    declining_metrics.append(metric)
                elif direction == 'stable':
                    stable_metrics.append(metric)
        
        if improving_metrics:
            insights.append(f"📈 性能持续改善的指标: {', '.join(improving_metrics)}")
        
        if declining_metrics:
            insights.append(f"📉 需要关注的指标: {', '.join(declining_metrics)}")
        
        if stable_metrics:
            insights.append(f"⚖️ 表现稳定的指标: {', '.join(stable_metrics[:3])}...")
        
        # 季节性洞察
        seasonal_data = trend_analysis.get('seasonal_analysis', {})
        seasonal_metrics = [metric for metric, data in seasonal_data.items() 
                          if isinstance(data, dict) and data.get('has_seasonality', False)]
        
        if seasonal_metrics:
            insights.append(f"🔄 存在季节性模式的指标: {', '.join(seasonal_metrics)}")
        
        return insights if insights else ["趋势分析显示整体性能稳定"]
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        try:
            # 渲染HTML内容
            content = self._render_html_content(report_data)
            
            # 替换模板变量
            html_content = Template(self.templates['html']).render(
                title=report_data['title'],
                subtitle=report_data['subtitle'],
                generate_time=report_data['generate_time'],
                content=content
            )
            
            # 保存文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.output_dir / f'weekly_report_{timestamp}.html'
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML报告已生成: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"生成HTML报告失败: {e}")
            return ""
    
    def _render_html_content(self, report_data: Dict[str, Any]) -> str:
        """渲染HTML内容"""
        content_parts = []
        
        # 执行摘要
        content_parts.append(self._render_summary_section(report_data))
        
        # 环境性能详情
        content_parts.append(self._render_environment_section(report_data))
        
        # 趋势分析
        content_parts.append(self._render_trend_section(report_data))
        
        # 性能亮点和警告
        content_parts.append(self._render_highlights_warnings_section(report_data))
        
        # 建议行动
        content_parts.append(self._render_recommendations_section(report_data))
        
        return '\n'.join(content_parts)
    
    def _render_summary_section(self, report_data: Dict[str, Any]) -> str:
        """渲染摘要部分"""
        summary = report_data['summary']
        
        return f'''
        <div class="section">
            <h2>📊 执行摘要</h2>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value">{summary['total_score']:.3f}</div>
                    <div class="metric-label">平均性能评分</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['success_rate']:.1%}</div>
                    <div class="metric-label">测试成功率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['successful_tests']}/{summary['total_tests']}</div>
                    <div class="metric-label">完成测试数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['test_coverage']:.1%}</div>
                    <div class="metric-label">环境覆盖率</div>
                </div>
            </div>
        </div>
        '''
    
    def _render_environment_section(self, report_data: Dict[str, Any]) -> str:
        """渲染环境部分"""
        env_scores = report_data['environment_scores']
        
        if not env_scores:
            return '<div class="section"><h2>🏢 环境性能</h2><p>暂无环境数据</p></div>'
        
        # 生成环境表格
        env_rows = []
        for env_name, score in env_scores.items():
            status_class = "alert-success" if score > 0.8 else "alert-warning" if score > 0.6 else "alert-danger"
            status_text = "优秀" if score > 0.8 else "良好" if score > 0.6 else "需改进"
            
            env_rows.append(f'''
            <tr>
                <td>{env_name}</td>
                <td>{score:.3f}</td>
                <td><div class="alert {status_class}">{status_text}</div></td>
            </tr>
            ''')
        
        env_table = '\n'.join(env_rows)
        
        return f'''
        <div class="section">
            <h2>🏢 环境性能详情</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>环境名称</th>
                        <th>性能评分</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    {env_table}
                </tbody>
            </table>
        </div>
        '''
    
    def _render_trend_section(self, report_data: Dict[str, Any]) -> str:
        """渲染趋势部分"""
        trend_insights = report_data['trend_insights']
        
        if not trend_insights:
            return '<div class="section"><h2>📈 趋势分析</h2><p>暂无趋势数据</p></div>'
        
        insights_html = '\n'.join([f'<li>{insight}</li>' for insight in trend_insights])
        
        return f'''
        <div class="section">
            <h2>📈 趋势分析洞察</h2>
            <ul>
                {insights_html}
            </ul>
        </div>
        '''
    
    def _render_highlights_warnings_section(self, report_data: Dict[str, Any]) -> str:
        """渲染亮点和警告部分"""
        highlights = report_data['highlights']
        warnings = report_data['warnings']
        
        content = []
        
        if highlights:
            highlights_html = '\n'.join([f'<div class="alert alert-success">{highlight}</div>' for highlight in highlights])
            content.append(f'''
            <div class="section">
                <h2>🌟 性能亮点</h2>
                {highlights_html}
            </div>
            ''')
        
        if warnings:
            warnings_html = '\n'.join([f'<div class="alert alert-danger">{warning}</div>' for warning in warnings])
            content.append(f'''
            <div class="section">
                <h2>⚠️ 需要关注</h2>
                {warnings_html}
            </div>
            ''')
        
        return '\n'.join(content) if content else '<div class="section"><h2>📋 性能概览</h2><p>本周期内未检测到特殊性能亮点或警告</p></div>'
    
    def _render_recommendations_section(self, report_data: Dict[str, Any]) -> str:
        """渲染建议部分"""
        recommendations = report_data.get('recommendations', [])
        
        if not recommendations:
            return '<div class="section"><h2>💡 改进建议</h2><p>暂无具体建议</p></div>'
        
        recommendations_html = []
        for rec in recommendations:
            priority_class = "high" if rec.get('priority') == 'high' else "medium"
            actions_html = '\n'.join([f'<li>{action}</li>' for action in rec.get('suggested_actions', [])])
            
            recommendations_html.append(f'''
            <div class="recommendation">
                <h4>[{rec.get('priority', 'normal').upper()}] {rec.get('message', '无描述')}</h4>
                <ul>
                    {actions_html}
                </ul>
            </div>
            ''')
        
        recommendations_content = '\n'.join(recommendations_html)
        
        return f'''
        <div class="section">
            <h2>💡 改进建议</h2>
            {recommendations_content}
        </div>
        '''
    
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> str:
        """生成PDF报告"""
        try:
            # 首先生成HTML
            html_file = self._generate_html_report(report_data)
            if not html_file:
                return ""
            
            # 转换为PDF
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_file = self.output_dir / f'weekly_report_{timestamp}.pdf'
            
            # 读取HTML内容
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 添加PDF样式
            css_content = '''
            @page { margin: 2cm; }
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; }
            .metric-card { page-break-inside: avoid; }
            .section { page-break-inside: avoid; }
            '''
            
            # 生成PDF
            HTML(string=html_content).write_pdf(
                str(pdf_file),
                stylesheets=[CSS(string=css_content)]
            )
            
            self.logger.info(f"PDF报告已生成: {pdf_file}")
            return str(pdf_file)
            
        except Exception as e:
            self.logger.error(f"生成PDF报告失败: {e}")
            return ""
    
    def _generate_json_report(self, report_data: Dict[str, Any]) -> str:
        """生成JSON报告"""
        try:
            # 序列化数据
            json_data = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
            
            # 保存文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.output_dir / f'weekly_report_{timestamp}.json'
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            
            self.logger.info(f"JSON报告已生成: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"生成JSON报告失败: {e}")
            return ""
    
    def _generate_report_charts(self, test_results: Dict[str, Any],
                              performance_data: Dict[str, Any],
                              trend_analysis: Dict[str, Any]) -> Dict[str, str]:
        """生成报告图表"""
        chart_files = {}
        
        try:
            # 环境性能对比图
            env_chart = self._create_environment_performance_chart(test_results)
            if env_chart:
                chart_files['environment_performance'] = env_chart
            
            # 趋势分析图
            trend_chart = self._create_trend_analysis_chart(trend_analysis)
            if trend_chart:
                chart_files['trend_analysis'] = trend_chart
            
            # 性能指标分布图
            metrics_chart = self._create_metrics_distribution_chart(performance_data)
            if metrics_chart:
                chart_files['metrics_distribution'] = metrics_chart
            
            self.logger.info(f"报告图表已生成: {len(chart_files)} 个")
            
        except Exception as e:
            self.logger.error(f"生成报告图表失败: {e}")
        
        return chart_files
    
    def _create_environment_performance_chart(self, test_results: Dict[str, Any]) -> str:
        """创建环境性能对比图"""
        try:
            env_scores = test_results.get('environment_scores', {})
            
            if not env_scores:
                return ""
            
            # 准备数据
            environments = list(env_scores.keys())
            scores = list(env_scores.values())
            
            # 创建图表
            fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])
            
            bars = ax.bar(environments, scores, color=plt.cm.viridis(np.linspace(0, 1, len(environments))))
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            ax.set_title('各环境性能评分对比', fontsize=16, fontweight='bold')
            ax.set_xlabel('环境', fontsize=12)
            ax.set_ylabel('性能评分', fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f'environment_performance_{timestamp}.png'
            plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"环境性能图表已生成: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"创建环境性能图表失败: {e}")
            return ""
    
    def _create_trend_analysis_chart(self, trend_analysis: Dict[str, Any]) -> str:
        """创建趋势分析图"""
        try:
            if not trend_analysis or 'error' in trend_analysis:
                return ""
            
            trend_data = trend_analysis.get('trend_analysis', {})
            
            if not trend_data:
                return ""
            
            # 准备数据
            metrics = []
            r_squared_values = []
            slope_values = []
            
            for metric, analysis in trend_data.items():
                if isinstance(analysis, dict) and 'linear_trend' in analysis:
                    metrics.append(metric)
                    r_squared_values.append(analysis['linear_trend'].get('r_squared', 0))
                    slope_values.append(analysis['linear_trend'].get('slope', 0))
            
            if not metrics:
                return ""
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])
            
            # R²值图
            bars1 = ax1.bar(range(len(metrics)), r_squared_values, color='skyblue', alpha=0.7)
            ax1.set_title('各指标趋势拟合度 (R²)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('R² 值', fontsize=12)
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, r2 in zip(bars1, r_squared_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{r2:.3f}', ha='center', va='bottom')
            
            # 斜率图
            colors = ['red' if slope < 0 else 'green' for slope in slope_values]
            bars2 = ax2.bar(range(len(metrics)), slope_values, color=colors, alpha=0.7)
            ax2.set_title('各指标趋势斜率', fontsize=14, fontweight='bold')
            ax2.set_xlabel('指标', fontsize=12)
            ax2.set_ylabel('斜率', fontsize=12)
            ax2.set_xticks(range(len(metrics)))
            ax2.set_xticklabels(metrics, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f'trend_analysis_{timestamp}.png'
            plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"趋势分析图表已生成: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"创建趋势分析图表失败: {e}")
            return ""
    
    def _create_metrics_distribution_chart(self, performance_data: Dict[str, Any]) -> str:
        """创建指标分布图"""
        try:
            env_details = performance_data.get('environment_details', {})
            
            if not env_details:
                return ""
            
            # 收集所有指标数据
            all_metrics = {}
            for env_name, env_data in env_details.items():
                if isinstance(env_data, dict) and 'metrics' in env_data:
                    for metric_name, value in env_data['metrics'].items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        if isinstance(value, (int, float)):
                            all_metrics[metric_name].append(value)
            
            if not all_metrics:
                return ""
            
            # 创建箱线图
            metrics_data = list(all_metrics.values())
            metrics_labels = list(all_metrics.keys())
            
            fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])
            
            box_plot = ax.boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_data)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('性能指标分布图', fontsize=16, fontweight='bold')
            ax.set_ylabel('指标值', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f'metrics_distribution_{timestamp}.png'
            plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"指标分布图表已生成: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"创建指标分布图表失败: {e}")
            return ""
    
    def _send_email_notification(self, report_files: Dict[str, str], report_data: Dict[str, Any]):
        """发送邮件通知"""
        try:
            email_config = self.email_config
            
            if not email_config.get('to_addresses'):
                self.logger.warning("未配置邮件地址，跳过邮件发送")
                return
            
            # 创建邮件消息
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"智能体性能测试周报 - {datetime.now().strftime('%Y年%m月%d日')}"
            
            # 邮件正文
            body = f"""
智能体性能测试系统

本周测试摘要：
- 平均性能评分: {report_data['summary']['total_score']:.3f}
- 测试成功率: {report_data['summary']['success_rate']:.1%}
- 完成测试数: {report_data['summary']['successful_tests']}/{report_data['summary']['total_tests']}

详细报告请查看附件。

---
系统自动发送
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 添加附件
            for format_type, file_path in report_files.items():
                if format_type != 'charts' and os.path.exists(file_path):
                    with open(file_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(file_path)}'
                    )
                    msg.attach(part)
            
            # 发送邮件
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.generation_stats['email_sent'] += 1
            self.logger.info("邮件通知发送成功")
            
        except Exception as e:
            self.logger.error(f"发送邮件通知失败: {e}")
    
    def generate_custom_report(self, data: Dict[str, Any], template_name: str = "default",
                             format_type: str = "html") -> str:
        """生成自定义报告"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type.lower() == "html":
                # 渲染自定义HTML模板
                content = Template(self.templates.get('html', '')).render(**data)
                file_path = self.output_dir / f'custom_report_{template_name}_{timestamp}.html'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return str(file_path)
            
            elif format_type.lower() == "json":
                # 生成自定义JSON报告
                json_data = json.dumps(data, ensure_ascii=False, indent=2, default=str)
                file_path = self.output_dir / f'custom_report_{template_name}_{timestamp}.json'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                
                return str(file_path)
            
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
            
        except Exception as e:
            self.logger.error(f"生成自定义报告失败: {e}")
            return ""
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        return {
            **self.generation_stats,
            'output_directory': str(self.output_dir),
            'supported_formats': list(self.config['report_formats']),
            'email_enabled': self.config.get('auto_email', False)
        }
    
    def cleanup_old_reports(self, days: int = 30):
        """清理旧报告"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            for file_path in self.output_dir.glob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
                    cleaned_count += 1
            
            self.logger.info(f"清理了 {cleaned_count} 个旧报告文件")
            
        except Exception as e:
            self.logger.error(f"清理旧报告失败: {e}")


if __name__ == "__main__":
    # 示例用法
    generator = ReportGenerator()
    
    # 模拟测试数据
    test_results = {
        'summary_metrics': {
            'average_environment_score': 0.85,
            'success_rate': 0.90,
            'total_tests': 5,
            'successful_tests': 4
        },
        'environment_scores': {
            'image_classification': 0.92,
            'object_detection': 0.88,
            'scene_analysis': 0.85,
            'cross_domain_transfer': 0.80,
            'adaptation_test': 0.82
        },
        'individual_tests': {
            'image_classification': {'status': 'completed', 'accuracy': 0.92},
            'object_detection': {'status': 'completed', 'mAP': 0.88},
            'scene_analysis': {'status': 'completed', 'scene_understanding_score': 0.85},
            'cross_domain_transfer': {'status': 'completed', 'transfer_efficiency': 0.80},
            'adaptation_test': {'status': 'completed', 'adaptation_time': 20.5}
        }
    }
    
    performance_data = {
        'timestamp': datetime.now().isoformat(),
        'environment_details': {
            'image_classification': {'metrics': {'accuracy': 0.92, 'precision': 0.89}},
            'object_detection': {'metrics': {'mAP': 0.88, 'precision': 0.85}}
        }
    }
    
    trend_analysis = {
        'trend_analysis': {
            'accuracy': {
                'overall_direction': 'improving',
                'linear_trend': {'r_squared': 0.75, 'slope': 0.02}
            }
        },
        'recommendations': [
            {
                'type': 'optimization_opportunity',
                'priority': 'medium',
                'message': '图像分类性能持续改善，建议推广配置',
                'suggested_actions': ['记录当前配置', '扩展到其他场景']
            }
        ]
    }
    
    # 生成报告
    report_files = generator.generate_weekly_report(test_results, performance_data, trend_analysis)
    print(f"报告生成结果: {json.dumps(report_files, indent=2, ensure_ascii=False)}")
    
    # 获取统计信息
    stats = generator.get_generation_statistics()
    print(f"生成器统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")