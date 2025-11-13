#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体抽取模块

本模块实现知识图谱的实体抽取器，支持多种抽取算法和模式。
能够从文本数据中识别和提取实体，支持自定义实体类型和规则。

主要功能：
- 基于规则的实体抽取
- 基于统计学习的实体识别
- 支持自定义实体模式
- 实体归一化和去重
- 多语言实体抽取
- 置信度评估

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import jieba
import jieba.analyse
import spacy
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime


class EntityType(Enum):
    """实体类型枚举"""
    PERSON = "PERSON"           # 人名
    LOCATION = "LOCATION"       # 地点
    ORGANIZATION = "ORGANIZATION"  # 组织机构
    EVENT = "EVENT"             # 事件
    CONCEPT = "CONCEPT"         # 概念
    PRODUCT = "PRODUCT"         # 产品
    TIME = "TIME"               # 时间
    NUMBER = "NUMBER"           # 数字
    CUSTOM = "CUSTOM"           # 自定义类型


@dataclass
class EntityMatch:
    """实体匹配结果"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    normalized_form: str = None
    aliases: Set[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()
        if self.normalized_form is None:
            self.normalized_form = self.text


@dataclass
class ExtractionRule:
    """抽取规则定义"""
    name: str
    entity_type: EntityType
    pattern: str
    confidence: float = 1.0
    context_window: int = 50
    normalize: bool = True
    aliases: Set[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()


class EntityExtractor:
    """
    实体抽取器
    
    支持多种实体抽取方法：
    - 基于正则表达式的规则匹配
    - 基于词典的实体识别
    - 基于统计的实体抽取
    - 基于上下文的实体消歧
    
    特性：
    - 高性能批量处理
    - 可配置的抽取规则
    - 多语言支持
    - 实体归一化和去重
    - 置信度评估
    """
    
    def __init__(self, 
                 language: str = 'zh',
                 enable_nlp: bool = True,
                 enable_jieba: bool = True,
                 max_entities: int = 10000,
                 min_confidence: float = 0.5):
        """
        初始化实体抽取器
        
        Args:
            language: 语言设置 ('zh', 'en')
            enable_nlp: 是否启用NLP处理
            enable_jieba: 是否启用中文分词
            max_entities: 最大实体抽取数量
            min_confidence: 最小置信度阈值
        """
        self.language = language
        self.enable_nlp = enable_nlp
        self.enable_jieba = enable_jieba and language == 'zh'
        self.max_entities = max_entities
        self.min_confidence = min_confidence
        
        # 初始化组件
        self.nlp_model = None
        self.dictionary = set()
        self.extraction_rules = {}
        self.entity_patterns = defaultdict(list)
        
        # 实体缓存和统计
        self.entity_cache = {}  # 文本 -> 实体列表
        self.entity_stats = defaultdict(int)  # 实体类型统计
        self.normalization_map = {}  # 归一化映射
        
        # 性能统计
        self.processing_stats = {
            'total_texts': 0,
            'total_entities': 0,
            'processing_time': 0.0,
            'cache_hits': 0
        }
        
        self.logger = logging.getLogger("EntityExtractor")
        
        # 初始化
        self._initialize_nlp()
        self._load_default_rules()
        self._load_dictionary()
        
        self.logger.info(f"实体抽取器初始化完成，语言: {language}")
    
    def add_extraction_rule(self, rule: ExtractionRule):
        """
        添加抽取规则
        
        Args:
            rule: 抽取规则
        """
        self.extraction_rules[rule.name] = rule
        self.entity_patterns[rule.entity_type].append(rule.pattern)
        
        self.logger.debug(f"添加抽取规则: {rule.name}, 类型: {rule.entity_type}")
    
    def add_dictionary_entities(self, entities: List[Tuple[str, EntityType]], 
                              source: str = 'user_dict'):
        """
        添加词典实体
        
        Args:
            entities: 实体列表，格式: [(实体文本, 实体类型)]
            source: 词典来源
        """
        for entity_text, entity_type in entities:
            self.dictionary.add(entity_text)
            
            # 如果启用jieba，添加到用户词典
            if self.enable_jieba:
                jieba.add_word(entity_text)
        
        self.logger.info(f"添加词典实体: {len(entities)} 个来自 {source}")
    
    def extract_entities(self, 
                        text: Union[str, List[str]], 
                        return_context: bool = True,
                        normalize: bool = True) -> List[EntityMatch]:
        """
        从文本中抽取实体
        
        Args:
            text: 输入文本（单文本或文本列表）
            return_context: 是否返回上下文
            normalize: 是否进行实体归一化
            
        Returns:
            List[EntityMatch]: 实体匹配结果列表
        """
        start_time = datetime.now()
        
        # 处理单文本
        if isinstance(text, str):
            return self._extract_from_single_text(text, return_context, normalize)
        
        # 处理文本列表
        all_entities = []
        for text_item in text:
            entities = self._extract_from_single_text(text_item, return_context, normalize)
            all_entities.extend(entities)
        
        # 更新统计
        self.processing_stats['total_texts'] += len(text) if isinstance(text, list) else 1
        self.processing_stats['total_entities'] += len(all_entities)
        self.processing_stats['processing_time'] += (datetime.now() - start_time).total_seconds()
        
        return all_entities
    
    def extract_entities_batch(self, 
                              texts: List[str], 
                              batch_size: int = 100,
                              return_context: bool = True,
                              normalize: bool = True) -> List[List[EntityMatch]]:
        """
        批量抽取实体（优化大批量处理）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            return_context: 是否返回上下文
            normalize: 是否进行实体归一化
            
        Returns:
            List[List[EntityMatch]]: 每文本的实体列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                entities = self._extract_from_single_text(text, return_context, normalize)
                batch_results.append(entities)
            
            results.extend(batch_results)
        
        self.logger.info(f"批量处理完成: {len(texts)} 个文本")
        return results
    
    def find_entity_contexts(self, entity_text: str, corpus: List[str]) -> List[Dict[str, Any]]:
        """
        查找实体在语料库中的上下文
        
        Args:
            entity_text: 实体文本
            corpus: 语料库文本列表
            
        Returns:
            List[Dict[str, Any]]: 上下文信息列表
        """
        contexts = []
        
        for text_idx, text in enumerate(corpus):
            if entity_text in text:
                # 找到所有出现位置
                for match in re.finditer(re.escape(entity_text), text):
                    start_pos = max(0, match.start() - 50)
                    end_pos = min(len(text), match.end() + 50)
                    context = text[start_pos:end_pos]
                    
                    contexts.append({
                        'text_index': text_idx,
                        'entity_text': entity_text,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'context': context,
                        'full_text': text
                    })
        
        return contexts
    
    def disambiguate_entities(self, 
                            entities: List[EntityMatch], 
                            context_window: int = 100) -> List[EntityMatch]:
        """
        实体消歧（简化实现）
        
        Args:
            entities: 实体列表
            context_window: 上下文窗口大小
            
        Returns:
            List[EntityMatch]: 消歧后的实体列表
        """
        disambiguated = []
        seen_entities = set()
        
        for entity in entities:
            # 检查是否重复
            entity_key = (entity.normalized_form, entity.entity_type)
            if entity_key in seen_entities:
                continue
            
            # 简单的消歧逻辑：基于上下文相似度
            similar_entities = [e for e in entities 
                              if e.normalized_form == entity.normalized_form 
                              and e.entity_type == entity.entity_type]
            
            if len(similar_entities) > 1:
                # 选择置信度最高的实体
                best_entity = max(similar_entities, key=lambda x: x.confidence)
                disambiguated.append(best_entity)
            else:
                disambiguated.append(entity)
            
            seen_entities.add(entity_key)
        
        self.logger.info(f"实体消歧完成: {len(entities)} -> {len(disambiguated)}")
        return disambiguated
    
    def normalize_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """
        实体归一化
        
        Args:
            entities: 实体列表
            
        Returns:
            List[EntityMatch]: 归一化后的实体列表
        """
        normalized = []
        
        for entity in entities:
            # 应用归一化规则
            normalized_text = self._normalize_entity_text(entity.text)
            
            # 更新实体信息
            entity.normalized_form = normalized_text
            normalized.append(entity)
        
        # 更新归一化映射
        for entity in normalized:
            if entity.text != entity.normalized_form:
                self.normalization_map[entity.text] = entity.normalized_form
        
        return normalized
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """
        获取实体抽取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.processing_stats.copy()
        
        # 实体类型分布
        type_distribution = dict(self.entity_stats)
        
        # 词典统计
        stats.update({
            'dictionary_size': len(self.dictionary),
            'extraction_rules': len(self.extraction_rules),
            'entity_type_distribution': type_distribution,
            'normalization_map_size': len(self.normalization_map),
            'cache_size': len(self.entity_cache)
        })
        
        return stats
    
    def save_rules(self, filepath: str) -> bool:
        """
        保存抽取规则到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            rules_data = {}
            for name, rule in self.extraction_rules.items():
                rules_data[name] = {
                    'name': rule.name,
                    'entity_type': rule.entity_type.value,
                    'pattern': rule.pattern,
                    'confidence': rule.confidence,
                    'context_window': rule.context_window,
                    'normalize': rule.normalize,
                    'aliases': list(rule.aliases)
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"抽取规则保存成功: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存抽取规则失败: {str(e)}")
            return False
    
    def load_rules(self, filepath: str) -> bool:
        """
        从文件加载抽取规则
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            for name, rule_data in rules_data.items():
                rule = ExtractionRule(
                    name=rule_data['name'],
                    entity_type=EntityType(rule_data['entity_type']),
                    pattern=rule_data['pattern'],
                    confidence=rule_data['confidence'],
                    context_window=rule_data['context_window'],
                    normalize=rule_data['normalize'],
                    aliases=set(rule_data['aliases'])
                )
                self.add_extraction_rule(rule)
            
            self.logger.info(f"抽取规则加载成功: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载抽取规则失败: {str(e)}")
            return False
    
    def _initialize_nlp(self):
        """初始化NLP模型"""
        if self.enable_nlp:
            try:
                if self.language == 'en':
                    self.nlp_model = spacy.load("en_core_web_sm")
                elif self.language == 'zh':
                    self.nlp_model = spacy.load("zh_core_web_sm")
                else:
                    # 尝试加载多语言模型
                    self.nlp_model = spacy.load("xx_sent_ud_sm")
                
                self.logger.info("NLP模型加载成功")
                
            except OSError:
                self.logger.warning("NLP模型未安装，将使用基础功能")
                self.nlp_model = None
    
    def _load_default_rules(self):
        """加载默认抽取规则"""
        if self.language == 'zh':
            self._load_chinese_rules()
        else:
            self._load_english_rules()
    
    def _load_chinese_rules(self):
        """加载中文抽取规则"""
        # 人名规则（姓氏 + 名字）
        person_pattern = r'(?:[李王张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏锺汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文][a-zA-Z一-鿿]{1,3})'
        
        # 地名规则
        location_pattern = r'(?:北京|上海|广州|深圳|杭州|南京|苏州|武汉|成都|重庆|西安|青岛|大连|厦门|天津|重庆|中国香港|中国台湾|省|市|县|区|镇|村|路|街|道|区|县)'
        
        # 时间规则
        time_pattern = r'(?:[一二三四五六七八九十百千万\d]+年|[一二三四五六七八九十百千万\d]+月|[一二三四五六七八九十百千万\d]+日|[一二三四五六七八九十百千万\d]+时|[一二三四五六七八九十百千万\d]+分|[一二三四五六七八九十百千万\d]+秒|昨天|今天|明天|上午|下午|晚上|深夜|早晨|中午|傍晚)'
        
        # 添加规则
        self.add_extraction_rule(ExtractionRule(
            name="chinese_person",
            entity_type=EntityType.PERSON,
            pattern=person_pattern,
            confidence=0.8
        ))
        
        self.add_extraction_rule(ExtractionRule(
            name="chinese_location", 
            entity_type=EntityType.LOCATION,
            pattern=location_pattern,
            confidence=0.7
        ))
        
        self.add_extraction_rule(ExtractionRule(
            name="chinese_time",
            entity_type=EntityType.TIME,
            pattern=time_pattern,
            confidence=0.9
        ))
    
    def _load_english_rules(self):
        """加载英文抽取规则"""
        # 人名规则（大写字母开头）
        person_pattern = r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
        
        # 地名规则
        location_pattern = r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|El Paso|Nashville|Detroit|Oklahoma City|Portland|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Mesa|Kansas City|Atlanta|Long Beach|Colorado Springs|Raleigh|Miami|Virginia Beach|Omaha|Oakland|Minneapolis|Tulsa|Arlington|Tampa|New Orleans|Wichita)\b'
        
        # 时间规则
        time_pattern = r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December|202[0-9]|201[0-9]|200[0-9]|yesterday|today|tomorrow|morning|afternoon|evening|night|noon)\b'
        
        self.add_extraction_rule(ExtractionRule(
            name="english_person",
            entity_type=EntityType.PERSON,
            pattern=person_pattern,
            confidence=0.8
        ))
        
        self.add_extraction_rule(ExtractionRule(
            name="english_location",
            entity_type=EntityType.LOCATION, 
            pattern=location_pattern,
            confidence=0.7
        ))
        
        self.add_extraction_rule(ExtractionRule(
            name="english_time",
            entity_type=EntityType.TIME,
            pattern=time_pattern,
            confidence=0.9
        ))
    
    def _load_dictionary(self):
        """加载词典实体"""
        # 加载默认词典
        default_entities = []
        
        if self.language == 'zh':
            # 中文常用实体
            common_persons = ['张三', '李四', '王五', '赵六', '钱七']
            common_locations = ['北京', '上海', '广州', '深圳', '杭州']
            common_orgs = ['清华大学', '北京大学', '阿里巴巴', '腾讯', '华为']
            
            default_entities.extend([(name, EntityType.PERSON) for name in common_persons])
            default_entities.extend([(loc, EntityType.LOCATION) for loc in common_locations])
            default_entities.extend([(org, EntityType.ORGANIZATION) for org in common_orgs])
        
        else:
            # 英文常用实体
            common_persons = ['John Smith', 'Mary Johnson', 'David Brown', 'Sarah Wilson', 'Michael Davis']
            common_locations = ['New York', 'London', 'Tokyo', 'Paris', 'Beijing']
            common_orgs = ['Microsoft', 'Google', 'Apple', 'Amazon', 'Facebook']
            
            default_entities.extend([(name, EntityType.PERSON) for name in common_persons])
            default_entities.extend([(loc, EntityType.LOCATION) for loc in common_locations])
            default_entities.extend([(org, EntityType.ORGANIZATION) for org in common_orgs])
        
        self.add_dictionary_entities(default_entities, 'default_dict')
    
    def _extract_from_single_text(self, 
                                 text: str, 
                                 return_context: bool,
                                 normalize: bool) -> List[EntityMatch]:
        """
        从单个文本抽取实体
        
        Args:
            text: 输入文本
            return_context: 是否返回上下文
            normalize: 是否归一化
            
        Returns:
            List[EntityMatch]: 实体匹配结果列表
        """
        entities = []
        
        # 检查缓存
        cache_key = f"{text}_{return_context}_{normalize}"
        if cache_key in self.entity_cache:
            self.processing_stats['cache_hits'] += 1
            return self.entity_cache[cache_key]
        
        # 方法1：基于NLP模型抽取
        if self.nlp_model is not None:
            nlp_entities = self._extract_with_nlp(text)
            entities.extend(nlp_entities)
        
        # 方法2：基于规则抽取
        rule_entities = self._extract_with_rules(text)
        entities.extend(rule_entities)
        
        # 方法3：基于词典抽取
        dict_entities = self._extract_with_dictionary(text)
        entities.extend(dict_entities)
        
        # 合并和去重
        entities = self._merge_entities(entities)
        
        # 消歧和归一化
        if normalize:
            entities = self.normalize_entities(entities)
        
        entities = self.disambiguate_entities(entities)
        
        # 过滤低置信度实体
        entities = [e for e in entities if e.confidence >= self.min_confidence]
        
        # 限制实体数量
        if len(entities) > self.max_entities:
            entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:self.max_entities]
        
        # 添加上下文信息
        if return_context:
            entities = self._add_context_info(entities, text)
        
        # 缓存结果
        self.entity_cache[cache_key] = entities
        
        # 更新统计
        for entity in entities:
            self.entity_stats[entity.entity_type] += 1
        
        return entities
    
    def _extract_with_nlp(self, text: str) -> List[EntityMatch]:
        """使用NLP模型抽取实体"""
        if self.nlp_model is None:
            return []
        
        try:
            doc = self.nlp_model(text)
            entities = []
            
            for ent in doc.ents:
                # 映射NLP实体类型到我们的枚举类型
                entity_type = self._map_spacy_entity_type(ent.label_)
                
                entity = EntityMatch(
                    text=ent.text,
                    entity_type=entity_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.8,
                    context="",
                    normalized_form=ent.text.lower() if self.language == 'en' else ent.text
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"NLP抽取失败: {str(e)}")
            return []
    
    def _extract_with_rules(self, text: str) -> List[EntityMatch]:
        """基于规则抽取实体"""
        entities = []
        
        for rule_name, rule in self.extraction_rules.items():
            try:
                matches = re.finditer(rule.pattern, text)
                for match in matches:
                    entity = EntityMatch(
                        text=match.group(),
                        entity_type=rule.entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=rule.confidence,
                        context="",
                        aliases=rule.aliases.copy()
                    )
                    entities.append(entity)
                    
            except Exception as e:
                self.logger.error(f"规则 {rule_name} 匹配失败: {str(e)}")
        
        return entities
    
    def _extract_with_dictionary(self, text: str) -> List[EntityMatch]:
        """基于词典抽取实体"""
        entities = []
        
        # 使用jieba进行中文分词（如果启用）
        if self.enable_jieba:
            words = jieba.cut(text)
            word_positions = []
            pos = 0
            for word in words:
                if word.strip():
                    word_positions.append((pos, pos + len(word), word.strip()))
                    pos += len(word)
            
            # 匹配词典中的实体
            for start_pos, end_pos, word in word_positions:
                if word in self.dictionary:
                    # 根据词典确定实体类型（这里简化处理）
                    entity_type = EntityType.CONCEPT  # 默认类型
                    
                    entity = EntityMatch(
                        text=word,
                        entity_type=entity_type,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=0.9,
                        context=""
                    )
                    entities.append(entity)
        
        # 简单字符串匹配
        else:
            for dict_entity in self.dictionary:
                if dict_entity in text:
                    start_pos = text.find(dict_entity)
                    end_pos = start_pos + len(dict_entity)
                    
                    entity = EntityMatch(
                        text=dict_entity,
                        entity_type=EntityType.CONCEPT,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=0.9,
                        context=""
                    )
                    entities.append(entity)
        
        return entities
    
    def _merge_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """合并和去重实体"""
        if not entities:
            return []
        
        # 按位置排序
        entities.sort(key=lambda x: (x.start_pos, -x.confidence))
        
        merged = []
        used_positions = set()
        
        for entity in entities:
            # 检查是否与已选择的实体重叠
            overlap = False
            for used_start, used_end in used_positions:
                if (entity.start_pos < used_end and entity.end_pos > used_start):
                    overlap = True
                    break
            
            if not overlap:
                merged.append(entity)
                used_positions.add((entity.start_pos, entity.end_pos))
        
        return merged
    
    def _add_context_info(self, entities: List[EntityMatch], text: str) -> List[EntityMatch]:
        """添加上下文信息"""
        for entity in entities:
            # 提取上下文
            context_start = max(0, entity.start_pos - 50)
            context_end = min(len(text), entity.end_pos + 50)
            entity.context = text[context_start:context_end]
        
        return entities
    
    def _normalize_entity_text(self, text: str) -> str:
        """归一化实体文本"""
        # 移除多余的空格和标点
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # 转换为小写（英文）
        if self.language == 'en':
            normalized = normalized.lower()
        
        return normalized
    
    def _map_spacy_entity_type(self, spacy_type: str) -> EntityType:
        """映射spaCy实体类型到我们的枚举类型"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'NORP': EntityType.ORGANIZATION,
            'FAC': EntityType.LOCATION,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'EVENT': EntityType.EVENT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'PRODUCT': EntityType.PRODUCT,
            'DATE': EntityType.TIME,
            'TIME': EntityType.TIME,
            'MONEY': EntityType.NUMBER,
            'QUANTITY': EntityType.NUMBER
        }
        
        return mapping.get(spacy_type, EntityType.CONCEPT)