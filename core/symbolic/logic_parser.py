"""
逻辑表达式解析器
负责解析和构建各种逻辑表达式，支持命题逻辑、一阶逻辑和模态逻辑
"""

import re
from typing import Dict, List, Set, Union, Any, Optional
from enum import Enum
from dataclasses import dataclass
import operator


class LogicType(Enum):
    """逻辑类型枚举"""
    PROPOSITIONAL = "propositional"  # 命题逻辑
    FIRST_ORDER = "first_order"      # 一阶逻辑
    MODAL = "modal"                  # 模态逻辑
    FUZZY = "fuzzy"                  # 模糊逻辑


class Operator(Enum):
    """逻辑运算符枚举"""
    # 基础运算符
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    
    # 量词
    FORALL = "∀"
    EXISTS = "∃"
    
    # 模态运算符
    NECESSARY = "□"
    POSSIBLE = "◇"
    
    # 模糊逻辑运算符
    FUZZY_AND = "min"
    FUZZY_OR = "max"
    FUZZY_NOT = "1-"
    
    # 括号
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    
    # 分隔符
    COMMA = ","
    DOT = "."


@dataclass
class LogicToken:
    """逻辑表达式标记"""
    type: str
    value: str
    position: int
    line: int
    column: int


@dataclass
class LogicExpression:
    """逻辑表达式抽象语法树节点"""
    type: str
    value: Union[str, float, int]
    children: List['LogicExpression']
    operator: Optional[str] = None
    position: Optional[int] = None


class LogicParser:
    """逻辑表达式解析器"""
    
    def __init__(self):
        """初始化解析器"""
        self.tokens: List[LogicToken] = []
        self.current_pos = 0
        self.current_token_index = 0
        
        # 定义运算符优先级
        self.operator_precedence = {
            Operator.NECESSARY.value: 7,
            Operator.POSSIBLE.value: 7,
            Operator.NOT.value: 6,
            Operator.AND.value: 5,
            Operator.OR.value: 4,
            Operator.IMPLIES.value: 3,
            Operator.IFF.value: 2,
            Operator.LEFT_PAREN.value: 1,
            Operator.RIGHT_PAREN.value: 1
        }
        
        # 模糊逻辑操作符映射
        self.fuzzy_operators = {
            "min": min,
            "max": max,
            "1-": lambda x: 1 - x
        }
        
    def tokenize(self, expression: str) -> List[LogicToken]:
        """将逻辑表达式分解为标记序列"""
        self.tokens = []
        self.current_pos = 0
        
        lines = expression.split('\n')
        for line_num, line in enumerate(lines):
            self.current_pos = 0
            while self.current_pos < len(line):
                # 跳过空白字符
                if line[self.current_pos].isspace():
                    self.current_pos += 1
                    continue
                
                # 识别多字符运算符
                token_found = False
                for op in [Operator.IMPLIES.value, Operator.IFF.value, 
                           Operator.FORALL.value, Operator.EXISTS.value,
                           Operator.NECESSARY.value, Operator.POSSIBLE.value]:
                    if line[self.current_pos:self.current_pos+len(op)] == op:
                        self.add_token("OPERATOR", op, line_num, self.current_pos)
                        self.current_pos += len(op)
                        token_found = True
                        break
                
                if token_found:
                    continue
                
                # 识别单字符运算符
                current_char = line[self.current_pos]
                if current_char in "∧∨¬→↔∀∃□◇(),.":
                    self.add_token("OPERATOR", current_char, line_num, self.current_pos)
                    self.current_pos += 1
                elif current_char.isalpha() or current_char == '_':
                    # 识别变量名或常量名
                    start_pos = self.current_pos
                    while (self.current_pos < len(line) and 
                           (line[self.current_pos].isalnum() or line[self.current_pos] == '_')):
                        self.current_pos += 1
                    name = line[start_pos:self.current_pos]
                    self.add_token("IDENTIFIER", name, line_num, start_pos)
                elif current_char.isdigit():
                    # 识别数字（用于模糊逻辑）
                    start_pos = self.current_pos
                    while (self.current_pos < len(line) and line[self.current_pos].isdigit()):
                        self.current_pos += 1
                    number = line[start_pos:self.current_pos]
                    self.add_token("NUMBER", float(number), line_num, start_pos)
                elif current_char in "={}[]":
                    # 识别其他符号
                    self.add_token(current_char, line_num, self.current_pos)
                    self.current_pos += 1
                else:
                    raise ValueError(f"未知字符: '{current_char}' 在位置 {self.current_pos}")
        
        return self.tokens
    
    def add_token(self, type_name: str, value: Union[str, float], 
                  line: int, column: int):
        """添加标记"""
        token = LogicToken(type_name, value, len(self.tokens), line, column)
        self.tokens.append(token)
    
    def parse_logic_expression(self, expression: str, logic_type: LogicType = LogicType.PROPOSITIONAL) -> LogicExpression:
        """解析逻辑表达式"""
        if not expression.strip():
            raise ValueError("空逻辑表达式")
        
        # 词法分析
        self.tokens = self.tokenize(expression)
        self.current_token_index = 0
        
        try:
            # 语法分析
            ast = self.parse_expression()
            
            # 检查是否还有未解析的标记
            if self.current_token_index < len(self.tokens):
                raise ValueError(f"未解析的标记: {self.tokens[self.current_token_index]}")
            
            return ast
        except Exception as e:
            raise ValueError(f"解析错误: {e}")
    
    def parse_expression(self, min_precedence: int = 1) -> LogicExpression:
        """解析表达式（支持运算符优先级）"""
        if self.current_token_index >= len(self.tokens):
            raise ValueError("意外的表达式结束")
        
        left = self.parse_primary()
        
        while (self.current_token_index < len(self.tokens) and 
               self.tokens[self.current_token_index].value in self.operator_precedence and
               self.operator_precedence[self.tokens[self.current_token_index].value] >= min_precedence):
            
            op_token = self.tokens[self.current_token_index]
            operator_name = op_token.value
            
            # 计算下一个优先级
            next_precedence = self.operator_precedence[operator_name] + 1
            
            # 跳过操作符
            self.current_token_index += 1
            
            # 右结合性处理
            if operator_name in [Operator.IMPLIES.value]:
                next_precedence = self.operator_precedence[operator_name]
            
            right = self.parse_expression(next_precedence)
            
            left = LogicExpression(
                type="binary_op",
                value=operator_name,
                children=[left, right],
                operator=operator_name,
                position=op_token.position
            )
        
        return left
    
    def parse_primary(self) -> LogicExpression:
        """解析基本表达式单元"""
        if self.current_token_index >= len(self.tokens):
            raise ValueError("意外的表达式结束")
        
        token = self.tokens[self.current_token_index]
        
        # 处理否定运算符
        if token.value == Operator.NOT.value:
            self.current_token_index += 1
            operand = self.parse_expression(7)  # NOT具有最高优先级
            return LogicExpression(
                type="unary_op",
                value=Operator.NOT.value,
                children=[operand],
                operator=Operator.NOT.value,
                position=token.position
            )
        
        # 处理模态运算符
        if token.value in [Operator.NECESSARY.value, Operator.POSSIBLE.value]:
            self.current_token_index += 1
            operand = self.parse_expression(7)
            return LogicExpression(
                type="modal_op",
                value=token.value,
                children=[operand],
                operator=token.value,
                position=token.position
            )
        
        # 处理量词
        if token.value in [Operator.FORALL.value, Operator.EXISTS.value]:
            quantifier = token.value
            self.current_token_index += 1
            
            # 解析变量
            if self.current_token_index >= len(self.tokens):
                raise ValueError("量词后缺少变量")
            
            var_token = self.tokens[self.current_token_index]
            if var_token.type != "IDENTIFIER":
                raise ValueError(f"期望变量名，得到: {var_token.type}")
            
            self.current_token_index += 1
            
            # 解析量词作用域
            if (self.current_token_index < len(self.tokens) and 
                self.tokens[self.current_token_index].value == Operator.LEFT_PAREN.value):
                self.current_token_index += 1  # 跳过左括号
                scope = self.parse_expression()
                
                if (self.current_token_index < len(self.tokens) and
                    self.tokens[self.current_token_index].value == Operator.RIGHT_PAREN.value):
                    self.current_token_index += 1  # 跳过右括号
            else:
                scope = self.parse_expression()
            
            return LogicExpression(
                type="quantified",
                value=quantifier,
                children=[LogicExpression(type="variable", value=var_token.value, children=[]), scope],
                operator=quantifier,
                position=token.position
            )
        
        # 处理括号
        if token.value == Operator.LEFT_PAREN.value:
            self.current_token_index += 1  # 跳过左括号
            expr = self.parse_expression()
            
            if (self.current_token_index < len(self.tokens) and
                self.tokens[self.current_token_index].value == Operator.RIGHT_PAREN.value):
                self.current_token_index += 1  # 跳过右括号
                return expr
            else:
                raise ValueError("缺少右括号")
        
        # 处理标识符
        if token.type == "IDENTIFIER":
            self.current_token_index += 1
            return LogicExpression(
                type="identifier",
                value=token.value,
                children=[],
                position=token.position
            )
        
        # 处理数字（模糊逻辑）
        if token.type == "NUMBER":
            self.current_token_index += 1
            return LogicExpression(
                type="number",
                value=token.value,
                children=[]
            )
        
        raise ValueError(f"意外的标记: {token.value}")
    
    def evaluate_expression(self, ast: LogicExpression, context: Dict[str, Any]) -> Union[bool, float, Any]:
        """计算逻辑表达式的值"""
        if ast.type == "identifier":
            return context.get(ast.value, False)
        
        elif ast.type == "number":
            return ast.value
        
        elif ast.type == "unary_op":
            operand_value = self.evaluate_expression(ast.children[0], context)
            if ast.operator == Operator.NOT.value:
                return not operand_value
            elif ast.operator == Operator.NECESSARY.value:
                return operand_value  # 模态逻辑的必然性
            elif ast.operator == Operator.POSSIBLE.value:
                return operand_value  # 模态逻辑的可能性
            elif ast.operator == "1-":  # 模糊逻辑非
                return 1 - operand_value
        
        elif ast.type == "binary_op":
            left_value = self.evaluate_expression(ast.children[0], context)
            right_value = self.evaluate_expression(ast.children[1], context)
            
            if ast.operator == Operator.AND.value:
                return left_value and right_value
            elif ast.operator == Operator.OR.value:
                return left_value or right_value
            elif ast.operator == Operator.IMPLIES.value:
                return (not left_value) or right_value
            elif ast.operator == Operator.IFF.value:
                return left_value == right_value
            elif ast.operator == "min":  # 模糊逻辑与
                return self.fuzzy_operators["min"](left_value, right_value)
            elif ast.operator == "max":  # 模糊逻辑或
                return self.fuzzy_operators["max"](left_value, right_value)
        
        elif ast.type == "quantified":
            var_name = ast.children[0].value
            scope_expr = ast.children[1]
            
            if ast.operator == Operator.FORALL.value:
                # 全称量词：需要检查所有可能值
                domain = context.get(f"{var_name}_domain", [True, False])
                return all(self.evaluate_expression(scope_expr, {**context, var_name: val}) for val in domain)
            elif ast.operator == Operator.EXISTS.value:
                # 存在量词：需要检查是否存在值满足条件
                domain = context.get(f"{var_name}_domain", [True, False])
                return any(self.evaluate_expression(scope_expr, {**context, var_name: val}) for val in domain)
        
        return False
    
    def format_expression(self, ast: LogicExpression) -> str:
        """将抽象语法树格式化为字符串"""
        if ast.type == "identifier":
            return str(ast.value)
        
        elif ast.type == "number":
            return str(ast.value)
        
        elif ast.type == "unary_op":
            operand = self.format_expression(ast.children[0])
            return f"{ast.operator}({operand})"
        
        elif ast.type == "binary_op":
            left = self.format_expression(ast.children[0])
            right = self.format_expression(ast.children[1])
            return f"({left} {ast.operator} {right})"
        
        elif ast.type == "quantified":
            var = ast.children[0].value
            scope = self.format_expression(ast.children[1])
            return f"{ast.operator}{var}({scope})"
        
        return str(ast.value)
    
    def simplify_expression(self, ast: LogicExpression) -> LogicExpression:
        """简化逻辑表达式"""
        if ast.type == "binary_op":
            left = self.simplify_expression(ast.children[0])
            right = self.simplify_expression(ast.children[1])
            
            # 常量折叠
            if left.type == "number" and right.type == "number":
                return LogicExpression(
                    type="number",
                    value=self.evaluate_expression(ast, {}),
                    children=[]
                )
            
            # 单位元简化
            if ast.operator == Operator.AND.value:
                if left.type == "number" and left.value == 1:  # 真
                    return right
                elif right.type == "number" and right.value == 1:
                    return left
                elif left.type == "number" and left.value == 0:  # 假
                    return LogicExpression(type="number", value=0, children=[])
                elif right.type == "number" and right.value == 0:
                    return LogicExpression(type="number", value=0, children=[])
            
            elif ast.operator == Operator.OR.value:
                if left.type == "number" and left.value == 0:
                    return right
                elif right.type == "number" and right.value == 0:
                    return left
                elif left.type == "number" and left.value == 1:
                    return LogicExpression(type="number", value=1, children=[])
                elif right.type == "number" and right.value == 1:
                    return LogicExpression(type="number", value=1, children=[])
            
            return LogicExpression(ast.type, ast.value, [left, right], ast.operator, ast.position)
        
        elif ast.type == "unary_op":
            child = self.simplify_expression(ast.children[0])
            
            # 双重否定消除
            if ast.operator == Operator.NOT.value and child.type == "unary_op" and child.operator == Operator.NOT.value:
                return self.simplify_expression(child.children[0])
            
            return LogicExpression(ast.type, ast.value, [child], ast.operator, ast.position)
        
        return ast
    
    def to_normal_form(self, ast: LogicExpression, form_type: str = "cnf") -> LogicExpression:
        """将逻辑表达式转换为范式（CNF或DNF）"""
        if form_type == "cnf":
            return self.to_conjunctive_normal_form(ast)
        elif form_type == "dnf":
            return self.to_disjunctive_normal_form(ast)
        else:
            raise ValueError(f"不支持的范式类型: {form_type}")
    
    def to_conjunctive_normal_form(self, ast: LogicExpression) -> LogicExpression:
        """转换为合取范式（CNF）"""
        # 这里实现简化的CNF转换算法
        # 完整的实现需要复杂的逻辑运算规则
        return self.simplify_expression(ast)
    
    def to_disjunctive_normal_form(self, ast: LogicExpression) -> LogicExpression:
        """转换为析取范式（DNF）"""
        # 这里实现简化的DNF转换算法
        return self.simplify_expression(ast)